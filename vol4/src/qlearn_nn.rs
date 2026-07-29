//! 本 7.4「Q 学習とニューラルネットワーク」— GridWorld の Q 関数を vol3 の MLP で近似。
//! vol3 フレームワークの初実戦投入(path 依存)。実験は `tests/ch07.rs`。

use crate::grid_world::Action;
use ndarray::Array;
use rand::RngExt;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use vol3::functions::{mean_squared_error, relu};
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Optimizer, SGD};
use vol3::variable::Variable;

/// 本 7.4.1「ニューラルネットワークの前処理」: 状態 (y, x) を one-hot の [1, H·W] Variable に変換。
pub fn one_hot(state: (usize, usize), width: usize, height: usize) -> Variable {
    let size = width * height;
    let mut vec = vec![0.0_f32; size];
    // state.0 が y(行), state.1 が x(列) の想定
    let idx = state.0 * width + state.1;
    vec[idx] = 1.0;
    Variable::new(Array::from_shape_vec((1, size), vec).unwrap().into_dyn())
}

/// 本 7.4.3「ニューラルネットワークと Q 学習」: NN 版 Q 学習エージェント。
/// update は `qs.gather` で q[:, a] を切り出し(本家 p.228 の `qs[:, action]`)、
/// next_q は `.data()` の f32 取り出し = 本家 `unchain()` 相当(勾配を流さない)。
pub struct QLearningNNAgent {
    pub q_net: MLP,
    pub optimizer: SGD,
    pub gamma: f32,
    pub epsilon: f32,
    pub action_size: usize,
}

impl QLearningNNAgent {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        action_size: usize,
        gamma: f32,
        lr: f32,
        epsilon: f32,
        rng: &mut StdRng,
    ) -> Self {
        let q_net = MLP::new(&[input_size, hidden_size, action_size], relu, rng);
        let mut optimizer = SGD::new(lr);
        optimizer.setup(&q_net);

        Self {
            q_net,
            optimizer,
            gamma,
            epsilon,
            action_size,
        }
    }

    pub fn get_action(&self, state_var: &Variable, rng: &mut StdRng) -> Action {
        let actions = Action::all();
        if rng.random::<f32>() < self.epsilon {
            *actions.choose(rng).unwrap()
        } else {
            // q_net.forward は Immutable reference (&self) で可能 (内部可変性の恩恵)
            let qs = self.q_net.forward(state_var);
            let qs_data = qs.data();

            let mut max_q = f32::NEG_INFINITY;
            let mut max_a = 0;
            for i in 0..self.action_size {
                let q = qs_data[[0, i]];
                if q > max_q {
                    max_q = q;
                    max_a = i;
                }
            }
            Action::from_usize(max_a) // 以前の actions[max_a] から変更
        }
    }

    pub fn update(
        &mut self,
        state: &Variable,
        action: Action,
        reward: f32,
        next_state: &Variable,
        done: bool,
    ) {
        // 現在のQ値を予測
        let qs = self.q_net.forward(state);

        // Enum -> usize
        let action_idx = action.to_usize();

        // 選択した行動のQ値を切り出し ([1, 4] -> [1])
        let q = qs.gather(&[action_idx]);

        // 次の最大のQ値を予測 (done なら 0)
        let next_q_max = if done {
            0.0
        } else {
            let next_qs = self.q_net.forward(next_state);
            next_qs
                .data()
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        };

        // ターゲットを計算
        let target_val = reward + self.gamma * next_q_max;
        let target_var = Variable::new(
            Array::from_shape_vec(vec![1], vec![target_val])
                .unwrap()
                .into_dyn(),
        );

        // 勾配リセット -> 損失計算 -> 逆伝播 -> 重み更新
        self.q_net.cleargrads();
        let loss = mean_squared_error(&q, &target_var);
        loss.backward(false, false);
        self.optimizer.update();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_eq;

    #[test]
    fn test_one_hot_encoding() {
        let width = 4;
        let height = 3;

        // 状態 (2, 0) をエンコード
        // state.0 が y(行), state.1 が x(列) の想定なので、
        // インデックスは y * width + x = 2 * 4 + 0 = 8
        let state = (2, 0);
        let x = one_hot(state, width, height);
        let data = x.data();

        // 1. 形状の確認
        assert_eq!(
            data.shape(),
            &[1, 12],
            "Shape should be [1, width * height]"
        );

        // 2. 正しい位置に 1.0 が立っているかの確認
        // 演算してないのでそのまま 1.0 であることを確認
        assert_eq!(data[[0, 8]], 1.0, "Index 8 should be exactly 1.0");

        // 3. 他がすべて 0.0 であることの確認 (全要素の和が 1.0)
        // 演算してるので浮動小数点誤差を考慮して近似比較
        let sum: f32 = data.sum();
        assert!(
            approx_eq(sum, 1.0, 1e-6),
            "The sum of all elements should be exactly 1.0 (one-hot)"
        );
    }
}
