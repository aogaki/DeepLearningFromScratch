use crate::grid_world::{Action, GridWorld};
use std::collections::HashMap;

/// 方策の型エイリアス: 状態 → (行動 → 確率)
pub type Policy = HashMap<(usize, usize), HashMap<Action, f32>>;

/// ランダム方策（全状態で均等確率）
/// 型が変わった: 状態ごとの行動分布を返すようになった
pub fn random_policy(env: &GridWorld) -> Policy {
    let prob = 1.0 / Action::all().len() as f32;
    let action_probs: HashMap<Action, f32> = Action::all().iter().map(|&a| (a, prob)).collect();
    env.states().map(|s| (s, action_probs.clone())).collect()
}

/// 方策評価の1ステップ（インプレース更新）
/// pi が状態ごとの分布を持つようになった
pub fn eval_onestep(
    pi: &Policy,
    v: &mut HashMap<(usize, usize), f32>,
    env: &GridWorld,
    gamma: f32,
) {
    for state in env.states() {
        if env.is_goal(state) {
            v.insert(state, 0.0);
            continue;
        }
        let action_probs = &pi[&state];
        let mut new_v = 0.0;
        for (&action, &action_prob) in action_probs.iter() {
            let next_state = env.next_state(state, action);
            let r = env.reward(state, action, next_state);
            new_v += action_prob * (r + gamma * v[&next_state]);
        }
        v.insert(state, new_v);
    }
}

/// 方策評価（収束まで繰り返す）
pub fn policy_eval(
    pi: &Policy,
    env: &GridWorld,
    gamma: f32,
    threshold: f32,
) -> HashMap<(usize, usize), f32> {
    let mut v: HashMap<(usize, usize), f32> = HashMap::new();
    for state in env.states() {
        v.insert(state, 0.0);
    }
    loop {
        let old_v = v.clone();
        eval_onestep(pi, &mut v, env, gamma);
        let max_delta = env
            .states()
            .map(|s| (v[&s] - old_v[&s]).abs())
            .fold(0.0_f32, f32::max);
        if max_delta < threshold {
            break;
        }
    }
    v
}

/// 状態 s における全行動の行動価値 q(s,a) = r + γV(s') を返す
pub fn action_values(
    state: (usize, usize),
    v: &HashMap<(usize, usize), f32>,
    env: &GridWorld,
    gamma: f32,
) -> Vec<(Action, f32)> {
    Action::all()
        .iter()
        .map(|&action| {
            let next_state = env.next_state(state, action);
            let r = env.reward(state, action, next_state);
            (action, r + gamma * v[&next_state])
        })
        .collect()
}

/// greedy方策: V から各状態の argmax 行動を選び、確定方策を作る
pub fn greedy_policy(v: &HashMap<(usize, usize), f32>, env: &GridWorld, gamma: f32) -> Policy {
    let mut pi = Policy::new();

    for state in env.states() {
        if env.is_goal(state) {
            let prob = 1.0 / Action::all().len() as f32;
            pi.insert(state, Action::all().iter().map(|&a| (a, prob)).collect());
            continue;
        }

        // ★ 事前計算したペアの argmax に変わった
        let qs = action_values(state, v, env, gamma);
        let best_action = qs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(a, _)| *a)
            .unwrap();

        let action_probs: HashMap<Action, f32> = Action::all()
            .iter()
            .map(|&a| (a, if a == best_action { 1.0 } else { 0.0 }))
            .collect();
        pi.insert(state, action_probs);
    }
    pi
}

/// 方策反復法: 評価 ↔ 改善 を方策が変化しなくなるまで交互に回す
pub fn policy_iter(
    env: &GridWorld,
    gamma: f32,
    threshold: f32,
) -> (Policy, HashMap<(usize, usize), f32>) {
    let mut pi = random_policy(env);
    loop {
        let v = policy_eval(&pi, env, gamma, threshold);
        let new_pi = greedy_policy(&v, env, gamma);
        // 方策が変化しなくなったら収束
        if pi == new_pi {
            return (new_pi, v);
        }
        pi = new_pi;
    }
}

/// 価値反復法: V(s) ← max_a [r + γV(s')] を収束まで繰り返す
pub fn value_iter(
    env: &GridWorld,
    gamma: f32,
    threshold: f32,
) -> (Policy, HashMap<(usize, usize), f32>) {
    let mut v: HashMap<(usize, usize), f32> = HashMap::new();
    for state in env.states() {
        v.insert(state, 0.0);
    }

    loop {
        let old_v = v.clone();

        for state in env.states() {
            if env.is_goal(state) {
                v.insert(state, 0.0);
                continue;
            }

            // ★ 同じ action_values の max（greedy_policy では argmax）
            let qs = action_values(state, &v, env, gamma);
            let max_q = qs.iter().map(|(_, q)| *q).fold(f32::NEG_INFINITY, f32::max);
            v.insert(state, max_q);
        }

        let max_delta = env
            .states()
            .map(|s| (v[&s] - old_v[&s]).abs())
            .fold(0.0_f32, f32::max);

        if max_delta < threshold {
            break;
        }
    }

    // 収束した V から最適方策を抽出
    let pi = greedy_policy(&v, env, gamma);
    (pi, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_policy() {
        // まず random_policy で評価した V から greedy_policy を作る
        let env = GridWorld::make_default();
        let pi_random = random_policy(&env);
        let v = policy_eval(&pi_random, &env, 0.9, 0.001);
        let pi_greedy = greedy_policy(&v, &env, 0.9);
        // greedy方策は確定的: 選ばれた行動の確率が 1.0
        for state in env.states() {
            if env.is_goal(state) {
                continue;
            }
            let action_probs = &pi_greedy[&state];
            let max_prob = action_probs.values().cloned().fold(0.0_f32, f32::max);
            assert_eq!(
                max_prob, 1.0,
                "Greedy policy should be deterministic at {:?}",
                state
            );
        }
    }

    #[test]
    fn test_policy_iter_optimal_actions() {
        let env = GridWorld::make_default();
        let (pi, _v) = policy_iter(&env, 0.9, 0.001);
        // 最適方策の表駆動テスト: (状態, 期待される最適行動)
        let expected_actions = vec![
            ((0, 0), Action::Right),
            ((0, 1), Action::Right),
            ((0, 2), Action::Right),
            ((1, 0), Action::Up),
            ((1, 2), Action::Up),
            ((1, 3), Action::Up),    // 爆弾上でもゴールへ直行
            ((2, 0), Action::Right), // Up と Right は同値だが順番的に Right が選ばれる
            ((2, 1), Action::Right), // 上は壁なので右へ迂回
            ((2, 2), Action::Up),
            ((2, 3), Action::Left), // 上は爆弾なので左へ回避
        ];
        for (state, expected_action) in expected_actions {
            let action_probs = &pi[&state];
            // 確率 1.0 の行動 = 選ばれた行動
            let best_action = action_probs
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(&a, _)| a)
                .unwrap();
            assert_eq!(
                best_action, expected_action,
                "Optimal action at {:?} should be {:?}, got {:?}",
                state, expected_action, best_action
            );
        }
    }

    #[test]
    fn test_policy_iter_optimal_values() {
        let env = GridWorld::make_default();
        let (_pi, v) = policy_iter(&env, 0.9, 0.001);
        // 最適価値の表駆動テスト
        let expected_values = vec![
            ((0, 3), 0.0), // ゴール（終端）
            ((0, 2), 1.0), // ゴール隣
            ((1, 3), 1.0), // 爆弾（ゴールへ直行）
            ((0, 1), 0.9),
            ((1, 2), 0.9),
            ((0, 0), 0.81),
            ((2, 2), 0.81),
            ((1, 0), 0.729),
            ((2, 1), 0.729),
            ((2, 3), 0.729), // 爆弾を避けて左→(2,2)経由
            ((2, 0), 0.6561),
        ];
        for (state, expected) in expected_values {
            assert!(
                (v[&state] - expected).abs() < 0.01,
                "V({:?}) = {}, expected {}",
                state,
                v[&state],
                expected
            );
        }
    }

    #[test]
    fn test_value_iter_cross_check() {
        let env = GridWorld::make_default();
        let (pi_policy, v_policy) = policy_iter(&env, 0.9, 0.001);
        let (pi_value, v_value) = value_iter(&env, 0.9, 0.001);

        // 両アルゴリズムが同じ最適価値に収束すること
        for state in env.states() {
            assert!(
                (v_policy[&state] - v_value[&state]).abs() < 0.01,
                "V mismatch at {:?}: policy_iter={}, value_iter={}",
                state,
                v_policy[&state],
                v_value[&state]
            );
        }

        // 両アルゴリズムが同じ最適方策を返すこと
        for state in env.states() {
            if env.is_goal(state) {
                continue;
            }
            let best_policy = pi_policy[&state]
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(a, _)| *a)
                .unwrap();
            let best_value = pi_value[&state]
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(a, _)| *a)
                .unwrap();
            assert_eq!(best_policy, best_value, "Policy mismatch at {:?}", state);
        }
    }

    // 既存の手計算オラクル表もそのまま流用
    #[test]
    fn test_value_iter_optimal_values() {
        let env = GridWorld::make_default();
        let (_pi, v) = value_iter(&env, 0.9, 0.001);

        // policy_iter のテストと全く同じ表
        let expected_values = vec![
            ((0, 3), 0.0),
            ((0, 2), 1.0),
            ((1, 3), 1.0),
            ((0, 1), 0.9),
            ((1, 2), 0.9),
            ((0, 0), 0.81),
            ((2, 2), 0.81),
            ((1, 0), 0.729),
            ((2, 1), 0.729),
            ((2, 3), 0.729),
            ((2, 0), 0.6561),
        ];

        for (state, expected) in expected_values {
            assert!(
                (v[&state] - expected).abs() < 0.01,
                "V({:?}) = {}, expected {}",
                state,
                v[&state],
                expected
            );
        }
    }
}
