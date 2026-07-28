use crate::cart_pole::CartPoleAction;
use crate::utils::argmax_f32;
use ndarray::Array2;
use rand::RngExt;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::seq::index::sample;
use std::collections::VecDeque;
use vol3::functions::{mean_squared_error, relu};
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Adam, Optimizer};
use vol3::variable::Variable;

pub type Experience = ([f32; 4], CartPoleAction, f32, [f32; 4], bool);

pub struct ReplayBuffer<T> {
    buffer: VecDeque<T>,
    buffer_size: usize,
    pub batch_size: usize,
}
impl<T: Clone> ReplayBuffer<T> {
    pub fn new(buffer_size: usize, batch_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            batch_size,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.buffer_size
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn add(&mut self, item: T) {
        // maxlen の模倣: 満杯なら一番古いもの（front）を捨てる
        if self.buffer.len() == self.buffer_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    // 本家の random.sample 相当
    // RNG の参照を受け取ることでシードを確実に引き回す
    pub fn get_batch(&self, rng: &mut StdRng) -> Vec<T> {
        assert!(
            self.buffer.len() >= self.batch_size,
            "Buffer size must be at least batch size. Current buffer size: {}, batch size: {}",
            self.buffer.len(),
            self.batch_size
        );

        // rand::seq::index::sample は O(batch_size) で重複のないインデックスを生成
        let indices = sample(rng, self.buffer.len(), self.batch_size);
        indices
            .into_iter()
            .map(|i| self.buffer[i].clone())
            .collect()
    }
}

pub struct DQNAgent {
    pub q_net: MLP,
    pub target_q_net: MLP,
    pub optimizer: Adam,
    pub gamma: f32,
    pub epsilon: f32,
    pub action_size: usize,
    pub state_size: usize,
    pub use_double_dqn: bool, // 追加
}
impl DQNAgent {
    pub fn new(
        state_size: usize,
        hidden_sizes: &[usize], // 任意の層数・ユニット数のスライスを受け取る
        action_size: usize,
        gamma: f32,
        lr: f32,
        epsilon: f32,
        rng: &mut StdRng,
    ) -> Self {
        // [state_size, hidden_sizes..., action_size] を結合して sizes を構築
        let mut sizes = Vec::with_capacity(hidden_sizes.len() + 2);
        sizes.push(state_size);
        sizes.extend_from_slice(hidden_sizes);
        sizes.push(action_size);

        let q_net = MLP::new(&sizes, relu, rng);
        let target_q_net = MLP::new(&sizes, relu, rng);
        let mut optimizer = Adam::new(lr);
        optimizer.setup(&q_net);
        let mut agent = Self {
            q_net,
            target_q_net,
            optimizer,
            gamma,
            epsilon,
            action_size,
            state_size,
            use_double_dqn: false, // デフォルトは false
        };
        agent.sync_qnet();
        agent
    }

    /// q_net の重みを target_q_net に丸写しする
    pub fn sync_qnet(&mut self) {
        assert!(
            self.q_net.params().len() == self.target_q_net.params().len(),
            "q_net and target_q_net must have the same number of parameters."
        );
        self.q_net
            .params()
            .into_iter()
            .zip(self.target_q_net.params())
            .for_each(|(src, dst)| dst.set_data(src.data()));
    }

    pub fn get_action(&self, state: &[f32; 4], rng: &mut StdRng) -> CartPoleAction {
        let actions = CartPoleAction::all();
        if rng.random::<f32>() < self.epsilon {
            *actions.choose(rng).unwrap()
        } else {
            // CartPole の状態 [f32; 4] を [1, 4] の Variable に変換
            let state_var = Variable::new(
                Array2::from_shape_vec((1, self.state_size), state.to_vec())
                    .unwrap()
                    .into_dyn(),
            );

            let qs = self.q_net.forward(&state_var);
            let qs_data = qs.data().into_dimensionality::<ndarray::Ix2>().unwrap();

            let max_a = argmax_f32(qs_data.row(0));
            CartPoleAction::from_usize(max_a)
        }
    }

    /// 特定の状態における最大のQ値を返す (過大評価の測定用)
    pub fn max_q(&self, state: &[f32; 4]) -> f32 {
        let state_var = Variable::new(
            Array2::from_shape_vec((1, self.state_size), state.to_vec())
                .unwrap()
                .into_dyn(),
        );
        let qs = self.q_net.forward(&state_var);
        let qs_data = qs.data().into_dimensionality::<ndarray::Ix2>().unwrap();
        qs_data
            .row(0)
            .into_iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// ReplayBuffer からサンプリングしたバッチで学習を行う
    pub fn update(&mut self, batch: &[Experience]) {
        let batch_size = batch.len();

        // 1. バッチを ndarray 用に平坦化
        let mut states_vec = Vec::with_capacity(batch_size * self.state_size);
        let mut next_states_vec = Vec::with_capacity(batch_size * self.state_size);
        let mut actions = Vec::with_capacity(batch_size);

        for (s, a, _r, ns, _d) in batch {
            states_vec.extend_from_slice(s);
            next_states_vec.extend_from_slice(ns);
            actions.push(a.to_usize()); // Action -> usize の変換 (7章の資産)
        }
        let states_var = Variable::new(
            Array2::from_shape_vec((batch_size, self.state_size), states_vec)
                .unwrap()
                .into_dyn(),
        );
        let next_states_var = Variable::new(
            Array2::from_shape_vec((batch_size, self.state_size), next_states_vec)
                .unwrap()
                .into_dyn(),
        );
        // 2. 現在のQ値を予測し、選択した行動の列だけを Gather で抽出 (qs[:, actions] 相当)
        let qs = self.q_net.forward(&states_var);
        let q = qs.gather(&actions);

        // 3. 次のQ値を予測し、ターゲットを計算
        // ここで .data() を取り出すことで、DeZero の .unchain() と完全に同じグラフ切断を実現。
        let next_qs_target = self.target_q_net.forward(&next_states_var);
        let next_qs_target_data = next_qs_target
            .data()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let next_q_max_vec: Vec<f32> = if self.use_double_dqn {
            // Double DQN: アクション選択は q_net、価値の評価は target_q_net
            let next_qs_online = self.q_net.forward(&next_states_var);
            let next_qs_online_data = next_qs_online
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            next_qs_online_data
                .rows()
                .into_iter()
                .zip(next_qs_target_data.rows())
                .map(|(online_row, target_row)| {
                    let max_a = argmax_f32(online_row);
                    target_row[max_a]
                })
                .collect()
        } else {
            // 通常の DQN: アクション選択も価値の評価も target_q_net (単純な max)
            next_qs_target_data
                .rows()
                .into_iter()
                .map(|target_row| {
                    target_row
                        .into_iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .collect()
        };
        // 報酬 + γ * 次のQ値 を計算
        let target_vec: Vec<f32> = batch
            .iter()
            .zip(next_q_max_vec)
            .map(|((_s, _a, reward, _ns, done), next_q_max)| {
                let next_q = if *done { 0.0 } else { next_q_max };
                reward + self.gamma * next_q
            })
            .collect();

        let target_var = Variable::new(ndarray::Array1::from_vec(target_vec).into_dyn());
        // 4. 損失計算と逆伝播
        self.q_net.cleargrads();
        let loss = mean_squared_error(&q, &target_var);
        loss.backward(false, false);
        self.optimizer.update();
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_replay_buffer() {
        // 正常にバッチが取得できるかのテスト
        let mut rng = StdRng::seed_from_u64(42);
        let mut buffer = ReplayBuffer::new(3, 3);
        buffer.add(1);
        buffer.add(2);
        buffer.add(3);
        buffer.add(4); // 1 が捨てられる
        assert_eq!(buffer.len(), 3);

        let batch = buffer.get_batch(&mut rng);
        for &item in &batch {
            assert!(item == 2 || item == 3 || item == 4);
        }
    }

    #[test]
    #[should_panic(expected = "at least batch size")]
    // 学習初期はバッファが溜まるまで get_batch を呼んではいけない
    fn test_replay_buffer_underfill() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut buffer = ReplayBuffer::new(3, 3);
        buffer.add(1);
        buffer.add(2);
        let _ = buffer.get_batch(&mut rng);
    }

    #[test]
    fn test_dqn_agent_sync_qnet() {
        let mut rng = StdRng::seed_from_u64(42);
        let agent = DQNAgent::new(4, &[16], 2, 0.98, 0.0005, 0.1, &mut rng);
        // q_net と target_q_net のパラメータが同じであること
        for (q_param, target_param) in agent
            .q_net
            .params()
            .into_iter()
            .zip(agent.target_q_net.params())
        {
            assert_eq!(q_param.data(), target_param.data());
        }
    }

    #[test]
    #[should_panic(expected = "same number of parameters")]
    fn test_dqn_agent_sync_qnet_mismatch() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut agent = DQNAgent::new(4, &[16], 2, 0.98, 0.0005, 0.1, &mut rng);
        // q_net のパラメータを意図的に増やす
        agent.q_net = MLP::new(&[4, 16, 8, 2], relu, &mut rng);
        agent.sync_qnet(); // panic する
    }

    #[test]
    fn test_dqn_agent_update_weights_isolation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut agent = DQNAgent::new(4, &[16], 2, 0.99, 0.01, 0.1, &mut rng);

        // update を呼ぶ前の全レイヤの重みを ArrayD のままスナップショット保存
        let q_net_before: Vec<_> = agent.q_net.params().into_iter().map(|p| p.data()).collect();
        let target_net_before: Vec<_> = agent
            .target_q_net
            .params()
            .into_iter()
            .map(|p| p.data())
            .collect();
        // 擬似的な経験（ミニバッチサイズ2）を作成。done=true の分岐も踏ませる
        let batch = vec![
            (
                [0.1, 0.2, 0.3, 0.4],
                CartPoleAction::Left,
                1.0,
                [0.2, 0.3, 0.4, 0.5],
                false,
            ),
            (
                [0.5, 0.6, 0.7, 0.8],
                CartPoleAction::Right,
                -1.0,
                [0.0, 0.0, 0.0, 0.0],
                true,
            ),
        ];
        // メインの学習処理を実行
        agent.update(&batch);
        // 更新後の全レイヤの重みを取得
        let q_net_after: Vec<_> = agent.q_net.params().into_iter().map(|p| p.data()).collect();
        let target_net_after: Vec<_> = agent
            .target_q_net
            .params()
            .into_iter()
            .map(|p| p.data())
            .collect();
        // 1. メインのネットワークは「どこかの層」が必ず更新されていること
        assert_ne!(
            q_net_before, q_net_after,
            "At least some q_net parameters MUST change after update"
        );
        // 2. ターゲットネットワークは「全層にわたって」一切更新されていないこと
        assert_eq!(
            target_net_before, target_net_after,
            "All target_q_net parameters MUST remain exactly the same"
        );
    }
}
