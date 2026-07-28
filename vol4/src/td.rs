use crate::bandit::update_q_step;
use crate::dp::Policy;
use crate::grid_world::Action;
use crate::utils::{greedy_probs, sample_action};
use rand::RngExt;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use std::collections::HashMap;

pub struct TdAgent {
    gamma: f32,
    alpha: f32,
    pub pi: Policy, // 評価対象の固定方策（今回は方策制御しないので不変）
    pub v: HashMap<(usize, usize), f32>,
}
impl TdAgent {
    pub fn new(gamma: f32, alpha: f32, pi: Policy) -> Self {
        TdAgent {
            gamma,
            alpha,
            pi,
            v: HashMap::new(),
        }
    }

    pub fn get_action(&self, state: (usize, usize), rng: &mut StdRng) -> Action {
        sample_action(&self.pi, state, rng)
    }

    // ★ TD(0) の真骨頂：1ステップごとに呼ばれる更新関数
    pub fn update(
        &mut self,
        state: (usize, usize),
        reward: f32,
        next_state: (usize, usize),
        done: bool,
    ) {
        // 実装の急所：done なら次状態の価値 V(S') は 0
        let next_v = if done {
            0.0
        } else {
            *self.v.get(&next_state).unwrap_or(&0.0)
        };

        // TDターゲット = 今回の報酬 + 割引された次状態の価値（ブートストラップ）
        let target = reward + self.gamma * next_v;

        // V(S) ← V(S) + α(target - V(S))
        // 1章の update_q_step がまたしても全く同じ形で使えます！
        let v = self.v.entry(state).or_insert(0.0);
        *v = update_q_step(*v, target, self.alpha);
    }
}

pub struct SarsaAgent {
    gamma: f32,
    alpha: f32,
    epsilon: f32,
    pub q: HashMap<(usize, usize), HashMap<Action, f32>>,
    pub pi: Policy,
    // 本家の deque(maxlen=2) の正体：「1歩前の記録」
    prev_step: Option<((usize, usize), Action, f32)>,
}
impl SarsaAgent {
    pub fn new(gamma: f32, alpha: f32, epsilon: f32) -> Self {
        SarsaAgent {
            gamma,
            alpha,
            epsilon,
            q: HashMap::new(),
            pi: HashMap::new(),
            prev_step: None,
        }
    }
    pub fn get_action(&self, state: (usize, usize), rng: &mut StdRng) -> Action {
        sample_action(&self.pi, state, rng)
    }

    pub fn reset(&mut self) {
        // エピソード途中での強制リセット用
        self.prev_step = None;
    }

    pub fn update(&mut self, state: (usize, usize), action: Action, reward: f32, done: bool) {
        // 1. もし「1歩前の記録」があれば、SARSAの更新式を回す
        if let Some((prev_s, prev_a, prev_r)) = self.prev_step {
            // Target = 前回の報酬 + γ * 今回のQ値
            let next_q = self
                .q
                .get(&state)
                .map_or(0.0, |q_s| *q_s.get(&action).unwrap_or(&0.0));
            let target = prev_r + self.gamma * next_q;
            let q_s = self.q.entry(prev_s).or_default();
            let q_val = q_s.entry(prev_a).or_insert(0.0);
            *q_val = update_q_step(*q_val, target, self.alpha);
            self.pi.insert(prev_s, greedy_probs(q_s, self.epsilon));
        }
        // 2. 終端状態（エピソード終了）の特別処理
        if done {
            // 今回のステップは次がないため、即座に next_q = 0 として更新する
            let target = reward; // reward + gamma * 0.0

            let q_s = self.q.entry(state).or_default();
            let q_val = q_s.entry(action).or_insert(0.0);
            *q_val = update_q_step(*q_val, target, self.alpha);
            self.pi.insert(state, greedy_probs(q_s, self.epsilon));
            // エピソード終了なので記憶をリセット
            self.prev_step = None;
        } else {
            // 終端でなければ、今回のステップを「次回の更新」のために保存
            self.prev_step = Some((state, action, reward));
        }
    }
}

pub struct OffPolicySarsaAgent {
    gamma: f32,
    alpha: f32,
    epsilon: f32,
    pub q: HashMap<(usize, usize), HashMap<Action, f32>>,
    pub pi: Policy, // 学習の目標とする方策（純粋な greedy）
    pub b: Policy,  // 実際に環境を探索する方策（ε-greedy）
    prev_step: Option<((usize, usize), Action, f32)>,
}
impl OffPolicySarsaAgent {
    pub fn new(gamma: f32, alpha: f32, epsilon: f32) -> Self {
        OffPolicySarsaAgent {
            gamma,
            alpha,
            epsilon,
            q: HashMap::new(),
            pi: HashMap::new(),
            b: HashMap::new(),
            prev_step: None,
        }
    }
    pub fn get_action(&self, state: (usize, usize), rng: &mut StdRng) -> Action {
        // 環境を行動するのは「挙動方策 b」の役目
        sample_action(&self.b, state, rng)
    }
    pub fn reset(&mut self) {
        self.prev_step = None;
    }
    pub fn update(&mut self, state: (usize, usize), action: Action, reward: f32, done: bool) {
        if let Some((prev_s, prev_a, prev_r)) = self.prev_step {
            let next_q = self
                .q
                .get(&state)
                .map_or(0.0, |q_s| *q_s.get(&action).unwrap_or(&0.0));

            // まだ訪問しておらず HashMap に無い状態のデフォルト確率は一様(1/4 = 0.25)
            let pi_prob = self
                .pi
                .get(&state)
                .map_or(0.25, |p| *p.get(&action).unwrap_or(&0.0));
            let b_prob = self
                .b
                .get(&state)
                .map_or(0.25, |p| *p.get(&action).unwrap_or(&0.0));

            // 重点サンプリングの重み ρ = π(a'|s') / b(a'|s')
            let rho = pi_prob / b_prob;

            // TDターゲット全体に ρ を掛ける
            let target = rho * (prev_r + self.gamma * next_q);
            let q_s = self.q.entry(prev_s).or_default();
            let q_val = q_s.entry(prev_a).or_insert(0.0);
            *q_val = update_q_step(*q_val, target, self.alpha);

            // ★ greedy_probs を使って π と b の両方を作り直す
            self.pi.insert(prev_s, greedy_probs(q_s, 0.0));
            self.b.insert(prev_s, greedy_probs(q_s, self.epsilon));
        }
        if done {
            // 終端状態の場合は next_q=0, 次の行動が存在しないため ρ=1 と同等
            let target = reward;

            let q_s = self.q.entry(state).or_default();
            let q_val = q_s.entry(action).or_insert(0.0);
            *q_val = update_q_step(*q_val, target, self.alpha);

            self.pi.insert(state, greedy_probs(q_s, 0.0));
            self.b.insert(state, greedy_probs(q_s, self.epsilon));
            self.prev_step = None;
        } else {
            self.prev_step = Some((state, action, reward));
        }
    }
}

pub struct QLearningAgent {
    gamma: f32,
    alpha: f32,
    epsilon: f32,
    pub q: HashMap<(usize, usize), HashMap<Action, f32>>,
    // 確率分布を実体として持つことは完全にやめました（モデルフリー）
}
impl QLearningAgent {
    pub fn new(gamma: f32, alpha: f32, epsilon: f32) -> Self {
        QLearningAgent {
            gamma,
            alpha,
            epsilon,
            q: HashMap::new(),
        }
    }
    pub fn get_action(&self, state: (usize, usize), rng: &mut StdRng) -> Action {
        let actions = Action::all();

        // 確率 ε でランダム、または未訪問状態ならランダム
        if rng.random::<f32>() < self.epsilon || !self.q.contains_key(&state) {
            *actions.choose(rng).unwrap()
        } else {
            // さもなくば、Q値からその場で argmax を計算 (on-the-fly)
            let q_s = self.q.get(&state).unwrap();
            *actions
                .iter()
                .max_by(|&&a1, &&a2| {
                    let q1 = q_s.get(&a1).unwrap_or(&0.0);
                    let q2 = q_s.get(&a2).unwrap_or(&0.0);
                    q1.partial_cmp(q2).unwrap()
                })
                .unwrap()
        }
    }
    pub fn update(
        &mut self,
        state: (usize, usize),
        action: Action,
        reward: f32,
        next_state: (usize, usize),
        done: bool,
    ) {
        let next_q_max = if done {
            0.0
        } else {
            self.q.get(&next_state).map_or(0.0, |q_next| {
                Action::all()
                    .iter()
                    .map(|a| *q_next.get(a).unwrap_or(&0.0))
                    .max_by(|v1, v2| v1.partial_cmp(v2).unwrap())
                    .unwrap()
            })
        };
        let target = reward + self.gamma * next_q_max;
        let q_s = self.q.entry(state).or_default();
        let q_val = q_s.entry(action).or_insert(0.0);
        *q_val = update_q_step(*q_val, target, self.alpha);

        // ★ 分布の実体化（pi への保存）を削除
    }
}
