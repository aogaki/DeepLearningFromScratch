use crate::dp::Policy;
use crate::grid_world::Action;
use ndarray::Array1;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::StdRng;
use std::collections::HashMap;

pub fn argmax(arr: &Array1<f32>) -> Option<usize> {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}

pub fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// 方策（Policy）と現在の状態から、確率的に行動をサンプリングする。
/// 未知の状態（方策に未登録）の場合は、一様ランダムに選ぶ。
pub fn sample_action(pi: &Policy, state: (usize, usize), rng: &mut StdRng) -> Action {
    let actions = Action::all();

    // 方策に状態が存在すればその確率分布を、なければ均等確率(1/4)を採用
    let action_probs = if let Some(probs) = pi.get(&state) {
        actions
            .iter()
            .map(|a| *probs.get(a).unwrap_or(&0.0))
            .collect::<Vec<f32>>()
    } else {
        vec![1.0 / actions.len() as f32; actions.len()]
    };
    let dist = WeightedIndex::new(&action_probs).unwrap();
    actions[dist.sample(rng)]
}

/// 状態 s における各行動のQ値のマップから、ε-greedy な確率分布を作る
pub fn greedy_probs(q_s: &HashMap<Action, f32>, epsilon: f32) -> HashMap<Action, f32> {
    let actions = Action::all();
    let action_size = actions.len() as f32;
    let base_prob = epsilon / action_size; // 探索として割り当てられる均等な確率
    // Q値が最大の行動を探す
    let best_action = *actions
        .iter()
        .max_by(|&&a1, &&a2| {
            let q1 = q_s.get(&a1).unwrap_or(&0.0);
            let q2 = q_s.get(&a2).unwrap_or(&0.0);
            q1.partial_cmp(q2).unwrap()
        })
        .unwrap();
    // 全ての行動に確率を割り当てる
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
