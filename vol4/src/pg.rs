//! 本 9章「方策勾配法」— 最も単純な方策勾配法(9.1)・REINFORCE(9.2)・Actor-Critic(9.4)。
//! 9.3(ベースライン)は数式の節で、その思想は Actor-Critic の TD 誤差 δ に実装として合流する。
//! 環境は `cart_pole`、3 手法の比較実験は `tests/ch09.rs`。

use crate::cart_pole::CartPoleAction;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::StdRng;
use vol3::functions::{mean_squared_error, relu, softmax_simple};
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Adam, Optimizer};
use vol3::variable::Variable;

/// Actor ネットワークの出力(ロジット)を受け取り、softmax を適用してサンプリングし、
/// (選択された行動, その行動を選ぶ確率の計算グラフノード) のペアを返す。
pub fn sample_action_from_logits(
    logits: &Variable,
    rng: &mut StdRng,
) -> (CartPoleAction, Variable) {
    let probs = softmax_simple(logits, 1);

    let probs_data = probs.data().into_dimensionality::<ndarray::Ix2>().unwrap();
    let probs_vec = probs_data.row(0).to_vec();
    let dist = WeightedIndex::new(&probs_vec).unwrap();
    let action_idx = dist.sample(rng);
    let action = CartPoleAction::from_usize(action_idx);
    let prob = probs.gather(&[action_idx]); // グラフを持ったまま返す
    (action, prob)
}

/// 本 9.1.3「方策勾配法の実装」: 方策 π を MLP(logits 出力)で表すエージェント。
/// 本家の Policy クラスは作らず、softmax は `sample_action_from_logits` 側で適用する設計。
/// 更新は 2 変種 — `update_simple`(9.1)と `update`(9.2 REINFORCE)。
pub struct PGAgent {
    pub pi_net: MLP,
    pub optimizer: Adam,
    pub gamma: f32,
    pub action_size: usize,
    pub state_size: usize,
    pub memory: Vec<(f32, Variable)>, // (報酬, 選択した行動の確率) を保存
}
impl PGAgent {
    pub fn new(
        state_size: usize,
        hidden_sizes: &[usize],
        action_size: usize,
        gamma: f32,
        lr: f32,
        rng: &mut StdRng,
    ) -> Self {
        let mut sizes = Vec::with_capacity(hidden_sizes.len() + 2);
        sizes.push(state_size);
        sizes.extend_from_slice(hidden_sizes);
        sizes.push(action_size);
        let pi_net = MLP::new(&sizes, relu, rng);
        let mut optimizer = Adam::new(lr);
        optimizer.setup(&pi_net);
        Self {
            pi_net,
            optimizer,
            gamma,
            action_size,
            state_size,
            memory: Vec::new(),
        }
    }

    pub fn get_action(&self, state: &[f32; 4], rng: &mut StdRng) -> (CartPoleAction, Variable) {
        let state_var = crate::utils::to_batch_var(state);
        let logits = self.pi_net.forward(&state_var);
        sample_action_from_logits(&logits, rng)
    }

    pub fn add(&mut self, reward: f32, prob: Variable) {
        self.memory.push((reward, prob));
    }

    /// 本 9.1.3: 最も単純な方策勾配法 — 全ステップに同一の G(t=0 の総収益)を掛ける。
    /// 勾配推定の分散が大きく、学習は遅い(9.2 との比較実験の対照群)。
    pub fn update_simple(&mut self) {
        self.pi_net.cleargrads();
        // 1. まずエピソード全体の収益 G (t=0 での割引報酬和) を計算する
        let mut g = 0.0;
        for (reward, _prob) in self.memory.iter().rev() {
            g = *reward + self.gamma * g;
        }
        // 2. 計算したただ1つの G を、すべてのステップの損失計算に使い回す
        let mut loss: Option<Variable> = None;
        for (_reward, prob) in self.memory.iter() {
            let term = &prob.ln() * (-g);

            loss = match loss {
                Some(l) => Some(&l + &term),
                None => Some(term),
            };
        }
        if let Some(l) = loss {
            l.backward(false, false);
            self.optimizer.update();
        }
        self.memory.clear();
    }

    /// 本 9.2.2「REINFORCE の実装」: 各ステップの重みを G_t(そのステップ以降の収益)にして
    /// 分散を下げる。9.1 との差分は「項の構築が G の畳み込みループの中に居る」ことだけ。
    pub fn update(&mut self) {
        self.pi_net.cleargrads();
        // 各ステップに「そのステップ以降の収益 G_t」を掛ける — 行動は過去の報酬に影響しないため、分散低減
        let mut g = 0.0;
        let loss = self
            .memory
            .iter()
            .rev()
            .map(|(reward, prob)| {
                g = *reward + self.gamma * g;
                &prob.ln() * (-g)
            })
            .reduce(|acc, t| &acc + &t);

        if let Some(l) = loss {
            l.backward(false, false);
            self.optimizer.update();
        }
        self.memory.clear();
    }
}

/// 本 9.4.2「Actor-Critic の実装」: 方策 π(Actor)と状態価値 V(Critic)の 2 ネット構成。
/// エピソード一括ではなくステップごとの TD 更新に戻る。切断点 2 箇所(V のターゲットと δ)は
/// f32 側で計算することで表現 — 本家の `unchain()` 2 回を「忘れられない」形に翻訳。
pub struct ActorCriticAgent {
    pub pi_net: MLP,
    pub v_net: MLP,
    pub optimizer_pi: Adam,
    pub optimizer_v: Adam,
    pub gamma: f32,
    pub action_size: usize,
    pub state_size: usize,
}
impl ActorCriticAgent {
    pub fn new(
        state_size: usize,
        hidden_sizes: &[usize],
        action_size: usize,
        gamma: f32,
        lr_pi: f32,
        lr_v: f32,
        rng: &mut StdRng,
    ) -> Self {
        // Actor(π) は行動確率を出力
        let mut pi_sizes = Vec::with_capacity(hidden_sizes.len() + 2);
        pi_sizes.push(state_size);
        pi_sizes.extend_from_slice(hidden_sizes);
        pi_sizes.push(action_size);

        // Critic(V) は状態価値をスカラー出力 (1 ユニット)
        let mut v_sizes = Vec::with_capacity(hidden_sizes.len() + 2);
        v_sizes.push(state_size);
        v_sizes.extend_from_slice(hidden_sizes);
        v_sizes.push(1);

        let pi_net = MLP::new(&pi_sizes, relu, rng);
        let v_net = MLP::new(&v_sizes, relu, rng);

        // オプティマイザは独立して保持 (学習率も別々)
        let mut optimizer_pi = Adam::new(lr_pi);
        optimizer_pi.setup(&pi_net);

        let mut optimizer_v = Adam::new(lr_v);
        optimizer_v.setup(&v_net);

        Self {
            pi_net,
            v_net,
            optimizer_pi,
            optimizer_v,
            gamma,
            action_size,
            state_size,
        }
    }

    pub fn get_action(&self, state: &[f32; 4], rng: &mut StdRng) -> (CartPoleAction, Variable) {
        let state_var = crate::utils::to_batch_var(state);
        let logits = self.pi_net.forward(&state_var);
        sample_action_from_logits(&logits, rng)
    }

    /// 本 9.4.2(update): loss_v = MSE(V(s), r + γV(s'))、loss_pi = −δ·ln π(a|s)。
    /// δ = target − V(s) はベースライン付き方策勾配(9.3)の実装形。
    pub fn update(
        &mut self,
        state: &[f32; 4],
        prob: &Variable,
        reward: f32,
        next_state: &[f32; 4],
        done: bool,
    ) {
        let state_var = crate::utils::to_batch_var(state);
        let next_state_var = crate::utils::to_batch_var(next_state);

        // ========== (1) V-net (Critic) の更新 ==========
        let v_next = self.v_net.forward(&next_state_var);
        // f32 に落とすことで次状態への逆伝播を完全に断つ (切断点1: DQN のターゲット網同族)
        let v_next_val = v_next.data().into_dimensionality::<ndarray::Ix2>().unwrap()[[0, 0]];

        // (1 - done) マスク適用と同等
        let target_val = if done {
            reward
        } else {
            reward + self.gamma * v_next_val
        };
        // 損失計算用にスカラーから再び Variable に包み直す（親ノードは持たない）
        let target_var = Variable::new(ndarray::array![[target_val]].into_dyn());

        let v_curr = self.v_net.forward(&state_var);
        let loss_v = mean_squared_error(&v_curr, &target_var);

        // ========== (2) pi-net (Actor) の更新 ==========
        let v_curr_val = v_curr.data().into_dimensionality::<ndarray::Ix2>().unwrap()[[0, 0]];

        // f32 同士で引き算することで、Critic の学習信号が Actor 側に逆伝播するのを防ぐ (切断点2)
        // これによって delta はただの定数重みとなる
        let delta = target_val - v_curr_val;
        let loss_pi = &(prob.ln()) * (-delta);

        // ========== (3) 逆伝播と重み更新 ==========
        self.v_net.cleargrads();
        self.pi_net.cleargrads();

        // グラフは完全に分離されているため、それぞれ個別に backward して問題ない
        loss_v.backward(false, false);
        loss_pi.backward(false, false);

        self.optimizer_v.update();
        self.optimizer_pi.update();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_pg_agent_update_weights_isolation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut agent = PGAgent::new(4, &[16], 2, 0.99, 0.01, &mut rng);

        // 疑似的な連続ステップを生成して get_action と add を実行
        let state1 = [0.1, 0.2, 0.3, 0.4];
        let state2 = [0.2, 0.3, 0.4, 0.5];

        let (_action1, prob1) = agent.get_action(&state1, &mut rng);
        agent.add(1.0, prob1);

        let (_action2, prob2) = agent.get_action(&state2, &mut rng);
        agent.add(1.0, prob2);

        // update() を呼ぶ前の全レイヤの重みをスナップショット保存
        let net_before: Vec<_> = agent
            .pi_net
            .params()
            .into_iter()
            .map(|p| p.data())
            .collect();

        // 逆伝播と重み更新
        agent.update();

        // 更新後の重みを取得
        let net_after: Vec<_> = agent
            .pi_net
            .params()
            .into_iter()
            .map(|p| p.data())
            .collect();

        // グラフが正しく繋がっていて、Adam によって重みが更新されたことを保証
        assert_ne!(
            net_before, net_after,
            "pi_net parameters MUST change after update (gradient flow check)"
        );

        // update 後はメモリが空になっていること
        assert!(
            agent.memory.is_empty(),
            "Memory must be cleared after update"
        );

        // update_simple() でも同様のチェックを行う
        let (_action1, prob1) = agent.get_action(&state1, &mut rng);
        agent.add(1.0, prob1);
        agent.update_simple();

        let net_after_simple: Vec<_> = agent
            .pi_net
            .params()
            .into_iter()
            .map(|p| p.data())
            .collect();

        assert_ne!(
            net_after, net_after_simple,
            "pi_net parameters MUST change after update_simple (gradient flow check)"
        );

        assert!(
            agent.memory.is_empty(),
            "Memory must be cleared after update_simple"
        );
    }

    #[test]
    fn test_actor_critic_agent_update_weights_isolation() {
        let mut rng = StdRng::seed_from_u64(42);

        // テスト用なので学習率は高め (0.01) にして重みの変化を検知しやすくする
        let mut agent = ActorCriticAgent::new(4, &[16], 2, 0.99, 0.01, 0.01, &mut rng);
        // update() 前の両ネットワークの重みをスナップショット保存
        let pi_net_before: Vec<_> = agent
            .pi_net
            .params()
            .into_iter()
            .map(|p| p.data())
            .collect();
        let v_net_before: Vec<_> = agent.v_net.params().into_iter().map(|p| p.data()).collect();
        // 1ステップ分のダミーデータを作成して疑似実行
        let state = [0.1, 0.2, 0.3, 0.4];
        let next_state = [0.2, 0.3, 0.4, 0.5];
        let reward = 1.0;
        let done = false;
        let (_action, prob) = agent.get_action(&state, &mut rng);

        // メモリへの add なしで、直接 update を呼ぶ
        agent.update(&state, &prob, reward, &next_state, done);
        // update() 後の重みを取得
        let pi_net_after: Vec<_> = agent
            .pi_net
            .params()
            .into_iter()
            .map(|p| p.data())
            .collect();
        let v_net_after: Vec<_> = agent.v_net.params().into_iter().map(|p| p.data()).collect();
        // Actor(π) の重みが更新されていること（= 損失 loss_pi からの勾配が流れた）
        assert_ne!(
            pi_net_before, pi_net_after,
            "pi_net parameters MUST change after update (Actor gradient flow check)"
        );

        // Critic(V) の重みが更新されていること（= 損失 loss_v からの勾配が流れた）
        assert_ne!(
            v_net_before, v_net_after,
            "v_net parameters MUST change after update (Critic gradient flow check)"
        );
    }
}
