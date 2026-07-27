use crate::bandit::{update_q, update_q_step};
use crate::dp::Policy;
use crate::grid_world::Action;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use std::collections::HashMap;

pub struct RandomAgent {
    gamma: f32,
    // (状態, 行動, 報酬) の履歴
    memory: Vec<((usize, usize), Action, f32)>,
    // 状態価値 V と訪問回数 N の記録
    pub v: HashMap<(usize, usize), f32>,
    pub n: HashMap<(usize, usize), usize>,
}
impl RandomAgent {
    pub fn new(gamma: f32) -> Self {
        RandomAgent {
            gamma,
            memory: Vec::new(),
            v: HashMap::new(),
            n: HashMap::new(),
        }
    }

    // 1章と同じく RNG は外から渡す。RandomAgent なので純粋なランダム
    pub fn get_action(&self, _state: (usize, usize), rng: &mut StdRng) -> Action {
        let actions = Action::all();
        // スライスのランダム選択 (rand::seq::IndexedRandom の機能)
        *actions.choose(rng).unwrap()
    }

    // 経験を記録する
    pub fn add(&mut self, state: (usize, usize), action: Action, reward: f32) {
        self.memory.push((state, action, reward));
    }

    // 次のエピソードに向けて記憶をリセット
    pub fn reset(&mut self) {
        self.memory.clear();
    }

    pub fn eval(&mut self) {
        let mut g = 0.0;

        // 記憶を逆順（後ろから）辿る
        for &(state, _action, reward) in self.memory.iter().rev() {
            // 収益 G = r + γG
            g = reward + self.gamma * g;

            // 訪問回数をインクリメント
            let count = self.n.entry(state).or_insert(0);
            *count += 1;

            // 1章の update_q を使って逐次平均を更新（引数の reward 枠に g を渡す）
            let val = self.v.entry(state).or_insert(0.0);
            *val = update_q(*val, g, *count);
        }
    }
}

/// ε-greedy 方策の確率分布を計算する関数
fn greedy_probs(
    q_s: &HashMap<Action, f32>, // ある状態 s における各行動のQ値
    epsilon: f32,
) -> HashMap<Action, f32> {
    let actions = Action::all();
    let action_size = actions.len() as f32;
    let base_prob = epsilon / action_size; // 探索としてランダムに選ばれる確率
    // Q値が最大の行動を探す（1章や4章と同じ max_by + partial_cmp）
    let best_action = *actions
        .iter()
        .max_by(|&&a1, &&a2| {
            let q1 = q_s.get(&a1).unwrap_or(&0.0);
            let q2 = q_s.get(&a2).unwrap_or(&0.0);
            q1.partial_cmp(q2).unwrap()
        })
        .unwrap();
    let mut probs = HashMap::new();
    for &action in &actions {
        let prob = if action == best_action {
            1.0 - epsilon + base_prob // 活用(1-ε) ＋ 探索の割り当て
        } else {
            base_prob // 探索の割り当てのみ
        };
        probs.insert(action, prob);
    }
    probs
}
pub struct McAgent {
    gamma: f32,
    epsilon: f32,
    alpha: f32,
    memory: Vec<((usize, usize), Action, f32)>,
    // 状態 s における行動価値 Q(s,a)
    pub q: HashMap<(usize, usize), HashMap<Action, f32>>,
    // 状態 s における行動確率分布 π(a|s)
    pub pi: Policy,
}
impl McAgent {
    pub fn new(gamma: f32, epsilon: f32, alpha: f32) -> Self {
        McAgent {
            gamma,
            epsilon,
            alpha,
            memory: Vec::new(),
            q: HashMap::new(),
            pi: HashMap::new(),
        }
    }
    pub fn get_action(&self, state: (usize, usize), rng: &mut StdRng) -> Action {
        let actions = Action::all();

        // 確率分布の取得（未知の状態なら均等確率）
        let action_probs = if let Some(probs) = self.pi.get(&state) {
            actions
                .iter()
                .map(|a| *probs.get(a).unwrap_or(&0.0))
                .collect::<Vec<f32>>()
        } else {
            vec![1.0 / actions.len() as f32; actions.len()]
        };
        // 確率の重み付けに従ってランダム選択
        let dist = WeightedIndex::new(&action_probs).unwrap();
        actions[dist.sample(rng)]
    }
    pub fn add(&mut self, state: (usize, usize), action: Action, reward: f32) {
        self.memory.push((state, action, reward));
    }
    pub fn reset(&mut self) {
        self.memory.clear();
    }
    pub fn eval(&mut self) {
        let mut g = 0.0;

        for &(state, action, reward) in self.memory.iter().rev() {
            g = reward + self.gamma * g;

            // 1. Q(s, a) の更新（固定ステップサイズ α を使う）
            let q_s = self.q.entry(state).or_default();
            let q_val = q_s.entry(action).or_insert(0.0);
            *q_val = update_q_step(*q_val, g, self.alpha);

            // 2. π(s) の更新（更新された Q から ε-greedy 分布を作り直す）
            let new_probs = greedy_probs(q_s, self.epsilon);
            self.pi.insert(state, new_probs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_eq;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn test_random_agent_memory_ops() {
        let mut agent = RandomAgent::new(0.9);
        // 1. 初期状態の確認
        assert_eq!(agent.memory.len(), 0);
        // 2. add の確認: 経験が順番に保存されるか
        agent.add((2, 0), Action::Up, 0.0);
        agent.add((1, 0), Action::Right, 1.0);

        assert_eq!(agent.memory.len(), 2);
        // (state, action, reward) のタプルが一致するか
        assert_eq!(agent.memory[0], ((2, 0), Action::Up, 0.0));
        assert_eq!(agent.memory[1], ((1, 0), Action::Right, 1.0));
        // 3. reset の確認: メモリが空になるか
        agent.reset();
        assert_eq!(agent.memory.len(), 0);
    }

    #[test]
    fn test_random_agent_get_action() {
        let agent = RandomAgent::new(0.9);
        let mut rng = StdRng::seed_from_u64(42); // シード固定
        let mut seen_actions = HashSet::new();

        // 100回行動を引く
        for _ in 0..100 {
            let action = agent.get_action((2, 0), &mut rng);
            seen_actions.insert(action);
        }

        // 100回引けば、一様分布なら確実に4種類すべてが1回は出現するはず
        // （もし 4 未満なら、固定値を返すなどの実装バグが疑われる）
        assert_eq!(
            seen_actions.len(),
            4,
            "Agent should return all 4 possible actions over 100 random draws."
        );
    }

    #[test]
    fn test_random_agent_eval() {
        let mut agent = RandomAgent::new(0.9);
        // --- エピソード1 ---
        // (0,1) -> (0,2)[GOAL, 報酬1.0] という軌跡を手動で作る
        agent.add((0, 1), Action::Right, 0.0);
        agent.add((0, 2), Action::Right, 1.0);

        agent.eval();

        // 逆順計算の確認:
        // (0,2) の G = 1.0 + 0.9 * 0 = 1.0
        // (0,1) の G = 0.0 + 0.9 * 1.0 = 0.9
        assert!(approx_eq(agent.v[&(0, 2)], 1.0, 1e-6));
        assert!(approx_eq(agent.v[&(0, 1)], 0.9, 1e-6));
        assert_eq!(agent.n[&(0, 2)], 1); // (0,2) は1回訪問
        assert_eq!(agent.n[&(0, 1)], 1); // (0,1) は1回訪問
        // --- エピソード2 ---
        agent.reset();
        agent.add((0, 1), Action::Down, 0.0);
        agent.add((1, 0), Action::Down, 0.0);

        agent.eval();
        // 逆順計算の確認:
        // (1,0) の G = 0.0, 訪問1回目 -> V = 0.0
        // (0,1) の G = 0.0, 訪問2回目!
        // 1章の update_q(現在のQ: 0.9, 今回のG: 0.0, 回数: 2) -> (0.9 + 0.0) / 2 = 0.45 になるはず

        assert_eq!(agent.v[&(1, 0)], 0.0);
        assert!((agent.v[&(0, 1)] - 0.45).abs() < 1e-6);
        assert_eq!(agent.n[&(0, 1)], 2);
    }
}
