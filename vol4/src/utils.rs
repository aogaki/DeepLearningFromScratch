//! 巻内で共有する小道具(章をまたいで再利用されるもの)。

use crate::dp::Policy;
use crate::grid_world::Action;
use ndarray::Array1;
use ndarray::Array2;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::StdRng;
use std::collections::HashMap;
use vol3::variable::Variable;

/// 本 1.4.2 で導入(np.argmax 相当)。空配列は None。
/// 同率時は max_by の仕様で「最後」の要素が勝つ(np.argmax は「最初」— 差異に注意)。
pub fn argmax(arr: &Array1<f32>) -> Option<usize> {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}

/// テスト用の誤差付き比較(プロジェクト規約: 浮動小数の `==` は使わない)。
pub fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// 8章で追加した ArrayView1 版 argmax(NN が出す Q 値の行向け)。
/// 同率時は「最後」、空ビューは 0 を返す。
pub fn argmax_f32(row: ndarray::ArrayView1<'_, f32>) -> usize {
    row.into_iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// 本 5.3.2(get_action)の共有部品: 方策分布から行動をサンプリングする(MC/TD で再利用)。
/// 未知の状態(方策に未登録)の場合は、一様ランダムに選ぶ。
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

/// 本 5.4.3「ε-greedy 法(1つ目の修正)」: 状態 s の Q 値マップから ε-greedy 分布を作る
/// (本家 common/utils.py の greedy_probs 相当。ε=0 なら純 greedy)。
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

/// 1次元の f32 スライスを、バッチサイズ 1 の Variable (形状 [1, N]) に変換する。
/// (DQN, PG, ActorCritic すべてで共通して使える入力前処理)
pub fn to_batch_var(state: &[f32]) -> Variable {
    Variable::new(
        Array2::from_shape_vec((1, state.len()), state.to_vec())
            .unwrap()
            .into_dyn(),
    )
}
