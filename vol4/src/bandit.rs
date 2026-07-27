use crate::utils::argmax;
use ndarray::Array1;
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, Normal};

pub struct Agent {
    epsilon: f32,
    q: Array1<f32>,
    n: Array1<usize>,
}
impl Agent {
    pub fn new(epsilon: f32, n_actions: usize) -> Self {
        Agent {
            epsilon,
            q: Array1::zeros(n_actions),
            n: Array1::zeros(n_actions),
        }
    }

    pub fn update(&mut self, action: usize, reward: f32) {
        self.n[action] += 1;
        self.q[action] = update_q(self.q[action], reward, self.n[action]);
    }

    pub fn get_action(&self, rng: &mut StdRng) -> usize {
        if rng.random::<f32>() < self.epsilon {
            rng.random_range(0..self.q.len())
        } else {
            argmax(&self.q).unwrap()
        }
    }
}

pub struct Bandit {
    rates: Array1<f32>,
}
impl Bandit {
    pub fn new(rates: Array1<f32>) -> Self {
        Bandit { rates }
    }

    pub fn play(&mut self, action: usize, rng: &mut StdRng) -> f32 {
        let rate = self.rates[action];
        if rng.random::<f32>() < rate { 1.0 } else { 0.0 }
    }
}

pub struct NonStatBandit {
    rates: Array1<f32>,
}
impl NonStatBandit {
    pub fn new(rates: Array1<f32>) -> Self {
        NonStatBandit { rates }
    }
    pub fn play(&mut self, action: usize, rng: &mut StdRng) -> f32 {
        let rate = self.rates[action];
        let reward = if rng.random::<f32>() < rate { 1.0 } else { 0.0 };
        let normal = Normal::new(0.0, 0.1).unwrap();
        for r in self.rates.iter_mut() {
            *r += normal.sample(rng);
            *r = r.clamp(0.0, 1.0);
        }

        reward
    }
}

pub struct AlphaAgent {
    epsilon: f32,
    alpha: f32,
    qs: Array1<f32>,
}
impl AlphaAgent {
    pub fn new(epsilon: f32, alpha: f32, action_size: usize) -> Self {
        AlphaAgent {
            epsilon,
            alpha,
            qs: Array1::zeros(action_size),
        }
    }
    pub fn update(&mut self, action: usize, reward: f32) {
        // n が不要になり、固定の alpha を step_size として渡すだけ！
        self.qs[action] = update_q_step(self.qs[action], reward, self.alpha);
    }
    pub fn get_action(&self, rng: &mut StdRng) -> usize {
        if rng.random::<f32>() < self.epsilon {
            rng.random_range(0..self.qs.len())
        } else {
            argmax(&self.qs).unwrap()
        }
    }
}

pub fn average_q(rewards: &Array1<f32>) -> f32 {
    let sum: f32 = rewards.iter().sum();
    sum / rewards.len() as f32
}

pub fn update_q_step(q: f32, reward: f32, step_size: f32) -> f32 {
    q + (reward - q) * step_size
}

pub fn update_q(q: f32, reward: f32, n: usize) -> f32 {
    let step_size = 1.0 / n as f32;
    update_q_step(q, reward, step_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_eq;
    use rand::SeedableRng;

    #[test]
    fn test_update_vs_average() {
        let rewards = Array1::from(vec![1.0, 0.0, 1.0, 1.0, 0.0]);
        let average = average_q(&rewards);
        let mut updated_q = 0.0;
        for (i, &reward) in rewards.iter().enumerate() {
            updated_q = update_q(updated_q, reward, i + 1);
        }
        assert!(approx_eq(average, updated_q, 1e-6));
    }

    #[test]
    fn test_bandit_play_rate0() {
        let rates = Array1::from(vec![0.0]);
        let mut bandit = Bandit::new(rates);
        let mut wins = 0;
        let trials = 10000;
        let mut rng = StdRng::seed_from_u64(101);
        for _ in 0..trials {
            wins += bandit.play(0, &mut rng) as usize; // Play the first arm
        }
        let win_rate = wins as f32 / trials as f32;
        assert!(approx_eq(win_rate, 0.0, 1e-6));
    }
    #[test]
    fn test_bandit_play_rate1() {
        let rates = Array1::from(vec![1.0]);
        let mut bandit = Bandit::new(rates);
        let mut wins = 0;
        let trials = 10000;
        let mut rng = StdRng::seed_from_u64(202);
        for _ in 0..trials {
            wins += bandit.play(0, &mut rng) as usize; // Play the first arm
        }
        let win_rate = wins as f32 / trials as f32;
        assert!(approx_eq(win_rate, 1.0, 1e-6));
    }

    #[test]
    fn test_bandit_play_rate05() {
        let mut bandit = Bandit::new(Array1::from(vec![0.5]));
        let trials = 100000;
        let mut updated_q = 0.0;
        let mut rewards = Array1::zeros(trials);
        let mut rng = StdRng::seed_from_u64(303);
        for i in 0..trials {
            let reward = bandit.play(0, &mut rng);
            updated_q = update_q(updated_q, reward, i + 1);
            rewards[i] = reward;
        }
        let average = average_q(&rewards);
        assert!(
            approx_eq(average, 0.5, 1e-2),
            "Average: {}, Expected: 0.5",
            average
        );
        assert!(
            approx_eq(average, updated_q, 1e-4),
            "Average: {}, Updated Q: {}",
            average,
            updated_q
        );
    }

    #[test]
    fn test_agent_ep0() {
        // --- 1. ε = 0 (常に greedy) のテスト ---
        let mut agent_greedy = Agent::new(0.0, 3); // 3本腕
        let mut rng = StdRng::seed_from_u64(42);

        // 同率時の max_by 挙動(最後が選ばれる)を避けるため、特定の腕に報酬を入れて明確な最大を作る
        agent_greedy.update(1, 1.0); // 腕1の Q = 1.0, それ以外は 0.0

        // ε=0 なので、常に Q値が最大の腕1が選ばれるはず
        assert_eq!(agent_greedy.get_action(&mut rng), 1);
        assert_eq!(agent_greedy.get_action(&mut rng), 1);
    }

    #[test]
    fn test_agent_ep1() {
        // --- 2. ε = 1 (常にランダム) のテスト ---
        let mut agent_random = Agent::new(1.0, 3);
        let mut rng = StdRng::seed_from_u64(42);

        // 腕1に報酬を与えて Q値を最大にしておくが...
        agent_random.update(1, 1.0);

        // ε=1 なので、最大値(腕1)に固定されずランダムに選ばれるはず
        // (複数回実行して、一度でも 1 以外の腕が選ばれればランダム探索しているとみなす)
        let mut picked_other = false;
        for _ in 0..10 {
            if agent_random.get_action(&mut rng) != 1 {
                picked_other = true;
                break;
            }
        }
        assert!(
            picked_other,
            "ε=1 should pick randomly, not just the max Q arm"
        );
    }

    #[test]
    fn test_agent_update() {
        // --- 3. update の値検証 ---
        let mut agent_update = Agent::new(0.1, 2);
        // 腕0に対して 1.0, 0.0, 1.0, 1.0, 0.0 を与える (平均 0.6)
        let rewards = vec![1.0, 0.0, 1.0, 1.0, 0.0];
        for &r in &rewards {
            agent_update.update(0, r);
        }

        // 腕0は 5回選ばれ、Q値は 0.6 になるはず
        assert!(approx_eq(agent_update.q[0], 0.6, 1e-6));
        assert_eq!(agent_update.n[0], 5);

        // 腕1は手つかずなので 0 のまま
        assert!(approx_eq(agent_update.q[1], 0.0, 1e-6));
        assert_eq!(agent_update.n[1], 0);
    }
}
