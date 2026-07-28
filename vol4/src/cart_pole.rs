//! 本 8.1: OpenAI Gym の CartPole-v0 相当の環境。
//!
//! 本は `gym.make('CartPole-v0')` と外部ライブラリを使うだけ(実装は学習対象外)
//! なので、この移植は「一度きりの下準備」として Claude が作成した。
//! 物理は gymnasium 1.3.0 の classic_control/cartpole.py に忠実
//! (陽的 Euler、τ=0.02s。原典は Barto, Sutton & Anderson 1983)。
//! 転記の正しさは自己一致では検証できないため(ch4 の教訓)、本家 gymnasium を
//! 実際に走らせた golden 軌道との突き合わせで担保する(tools/gen_cartpole_golden.py)。
//!
//! # 使い方
//!
//! 本の `env.reset()` / `env.step(action)` に対応(`info` は本も使わないため省略)。
//!
//! ```
//! use rand::{SeedableRng, rngs::StdRng};
//! use vol4::cart_pole::{CartPole, CartPoleAction};
//!
//! let mut rng = StdRng::seed_from_u64(0);
//! let mut env = CartPole::v0(); // 200 ステップ上限の CartPole-v0 仕様
//!
//! // エピソード開始: 状態は [カート位置, カート速度, 棒角度, 棒角速度] の [f32; 4]
//! let state = env.reset(&mut rng);
//! assert_eq!(state.len(), vol4::cart_pole::STATE_SIZE);
//!
//! // 1 ステップ進める。報酬はバランス維持中つねに 1.0。
//! // done が true になったら reset で次のエピソードへ(done 後の step は未定義)。
//! let (next_state, reward, done) = env.step(CartPoleAction::Left);
//! assert!(!done); // 初期状態(全要素 ±0.05)から 1 ステップでは終了し得ない
//! ```
//!
//! NN の入力にするときは `[f32; 4]` を形状 [1, 4] の `Variable` に詰め替える
//! (GridWorld の one_hot と同じ役どころの前処理。8.2 で書く)。

use rand::RngExt;
use rand::rngs::StdRng;

/// カートを押す方向。本の対応: 0 = 左、1 = 右。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CartPoleAction {
    Left,
    Right,
}

impl CartPoleAction {
    pub fn all() -> [CartPoleAction; 2] {
        [Self::from_usize(0), Self::from_usize(1)]
    }

    pub fn to_usize(self) -> usize {
        self as usize
    }

    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => CartPoleAction::Left,
            1 => CartPoleAction::Right,
            _ => panic!("Invalid action index"),
        }
    }
}

/// 状態ベクトルの要素数: [カート位置 x, カート速度 ẋ, 棒角度 θ, 棒角速度 θ̇]
pub const STATE_SIZE: usize = 4;

const GRAVITY: f32 = 9.8;
const MASS_CART: f32 = 1.0;
const MASS_POLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASS_CART + MASS_POLE;
/// 棒の半分の長さ(重心までの距離)
const HALF_POLE_LENGTH: f32 = 0.5;
const POLE_MASS_LENGTH: f32 = MASS_POLE * HALF_POLE_LENGTH;
const FORCE_MAG: f32 = 10.0;
/// 1 ステップの時間刻み [s]
const TAU: f32 = 0.02;
/// 終了条件: 棒の傾き ±12°(≈0.2094 rad)
const THETA_THRESHOLD: f32 = 12.0 * 2.0 * std::f32::consts::PI / 360.0;
/// 終了条件: カート位置 ±2.4
const X_THRESHOLD: f32 = 2.4;

/// CartPole 環境。`reset` → `step` ループで使う(GridWorld と同じ形)。
/// `step` は (次状態, 報酬, done) を返し、報酬はバランス維持中は常に 1.0。
/// done 後にさらに step を呼んだときの挙動は未定義(本家は警告を出す)。
pub struct CartPole {
    state: [f32; STATE_SIZE],
    steps: usize,
    max_steps: usize,
}

impl CartPole {
    /// CartPole-v0 仕様(200 ステップ上限)。
    pub fn v0() -> Self {
        Self::with_max_steps(200)
    }

    pub fn with_max_steps(max_steps: usize) -> Self {
        CartPole {
            state: [0.0; STATE_SIZE],
            steps: 0,
            max_steps,
        }
    }

    /// 全要素を一様乱数 [-0.05, 0.05) で初期化し、ステップ数を 0 に戻す。
    pub fn reset(&mut self, rng: &mut StdRng) -> [f32; STATE_SIZE] {
        self.steps = 0;
        for v in &mut self.state {
            *v = rng.random_range(-0.05..0.05);
        }
        self.state
    }

    pub fn state(&self) -> [f32; STATE_SIZE] {
        self.state
    }

    /// 状態の直接設定(テスト・再現実験用。ステップ数は変更しない)。
    pub fn set_state(&mut self, state: [f32; STATE_SIZE]) {
        self.state = state;
    }

    /// 1 ステップ進める。返り値は (次状態, 報酬, done)。
    pub fn step(&mut self, action: CartPoleAction) -> ([f32; STATE_SIZE], f32, bool) {
        let [x, x_dot, theta, theta_dot] = self.state;
        let force = match action {
            CartPoleAction::Left => -FORCE_MAG,
            CartPoleAction::Right => FORCE_MAG,
        };
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // 運動方程式(gymnasium 準拠。導出は Florian 2007 "Correct equations
        // for the dynamics of the cart-pole system")
        let temp = (force + POLE_MASS_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (HALF_POLE_LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta * cos_theta / TOTAL_MASS));
        let x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        // 陽的 Euler(位置系は「古い」速度で先に更新 — 本家の kinematics_integrator="euler")
        self.state = [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ];
        self.steps += 1;

        let out_of_bounds =
            self.state[0].abs() > X_THRESHOLD || self.state[2].abs() > THETA_THRESHOLD;
        let done = out_of_bounds || self.steps >= self.max_steps;
        // 報酬は終了ステップも含めて常に 1.0(CartPole-v0 仕様)
        (self.state, 1.0, done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_eq;
    use rand::SeedableRng;

    #[test]
    fn test_action_usize_roundtrip() {
        for &action in &CartPoleAction::all() {
            assert_eq!(
                action,
                CartPoleAction::from_usize(action.to_usize()),
                "Roundtrip failed. Check enum declaration order and from_usize match arms."
            );
        }
    }

    /// golden 値との比較。f32(こちら)vs float64(gymnasium)の丸め差が
    /// ステップごとに蓄積するので誤差付き比較(倒立振子は不安定系で誤差は指数的に
    /// 増幅されるが、10〜18 ステップでは 1e-5 程度に収まる)。
    /// 式の転記ミス(符号・係数・項の欠落)は 1 ステップ目から 1e-2 超のずれを
    /// 生むため、tol 1e-4 で十分な検出力がある。
    fn assert_matches_golden(
        env: &mut CartPole,
        golden: &[([f32; STATE_SIZE], f32, bool)],
        actions: &[usize],
    ) {
        for (i, (&(expected_state, expected_reward, expected_done), &a)) in
            golden.iter().zip(actions).enumerate()
        {
            let (state, reward, done) = env.step(CartPoleAction::from_usize(a));
            for (j, (&s, &e)) in state.iter().zip(&expected_state).enumerate() {
                assert!(
                    approx_eq(s, e, 1e-4),
                    "step {}, component {}: got {}, expected {}",
                    i + 1,
                    j,
                    s,
                    e
                );
            }
            assert!(approx_eq(reward, expected_reward, 1e-6));
            assert_eq!(done, expected_done, "done flag mismatch at step {}", i + 1);
        }
    }

    /// 物理パリティ: 非対称な初期状態+左右混合の行動列 10 ステップが
    /// 本家 gymnasium と一致する(golden は tools/gen_cartpole_golden.py で生成)。
    #[test]
    fn test_physics_parity_with_gymnasium() {
        const ACTIONS: [usize; 10] = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1];
        const GOLDEN_PHYSICS: [([f32; 4], f32, bool); 10] = [
            ([0.0096, 0.17467919, 0.0292, -0.3230687], 1.0, false),
            (
                [0.013093584, 0.36937344, 0.022738626, -0.60640204],
                1.0,
                false,
            ),
            (
                [0.020481052, 0.17394105, 0.010610585, -0.30664462],
                1.0,
                false,
            ),
            (
                [0.023959873, 0.3689102, 0.004477693, -0.59596246],
                1.0,
                false,
            ),
            (
                [0.031338077, 0.17372587, -0.007441556, -0.30187246],
                1.0,
                false,
            ),
            (
                [0.034812596, -0.021289226, -0.013479006, -0.0115456935],
                1.0,
                false,
            ),
            (
                [0.03438681, 0.17402342, -0.0137099195, -0.30845076],
                1.0,
                false,
            ),
            (
                [0.037867278, -0.020900534, -0.019878933, -0.020122891],
                1.0,
                false,
            ),
            (
                [0.037449267, 0.17450078, -0.020281391, -0.31901097],
                1.0,
                false,
            ),
            (
                [0.040939283, 0.36990562, -0.026661612, -0.6180203],
                1.0,
                false,
            ),
        ];

        let mut env = CartPole::v0();
        env.set_state([0.01, -0.02, 0.03, -0.04]);
        assert_matches_golden(&mut env, &GOLDEN_PHYSICS, &ACTIONS);
    }

    /// 角度終了の境界パリティ: 右へ押し続けると θ はステップ 5 で +0.2029 まで上がり
    /// 閾値 +0.2094 を「かすめて終了しない」。その後カートの加速で棒は反対側へ倒れ、
    /// ステップ 18 で -0.2094 を越えて終了する。閾値の両側を 1 本で検証。
    #[test]
    fn test_angle_termination_parity_with_gymnasium() {
        const ACTIONS: [usize; 18] = [1; 18];
        const GOLDEN_ANGLE_TERMINATION: [([f32; 4], f32, bool); 18] = [
            ([0.0, 0.19283356, 0.17, 0.7579324], 1.0, false),
            (
                [0.003856671, 0.38525596, 0.18515866, 0.52319914],
                1.0,
                false,
            ),
            ([0.0115617905, 0.5773555, 0.19562264, 0.2941013], 1.0, false),
            (
                [0.023108901, 0.76922894, 0.20150466, 0.06892754],
                1.0,
                false,
            ),
            ([0.03849348, 0.9609775, 0.20288321, -0.15403345], 1.0, false),
            (
                [0.057713028, 1.1527041, 0.19980253, -0.37648547],
                1.0,
                false,
            ),
            ([0.08076711, 1.3445108, 0.19227283, -0.60011995], 1.0, false),
            ([0.10765733, 1.536497, 0.18027043, -0.8266118], 1.0, false),
            ([0.13838726, 1.7287565, 0.16373819, -1.0576149], 1.0, false),
            ([0.1729624, 1.9213754, 0.1425859, -1.2947545], 1.0, false),
            ([0.2113899, 2.114427, 0.11669081, -1.5396152], 1.0, false),
            ([0.25367844, 2.3079681, 0.0858985, -1.7937229], 1.0, false),
            ([0.2998378, 2.502029, 0.050024044, -2.058518], 1.0, false),
            ([0.3498784, 2.6966057, 0.008853688, -2.3353171], 1.0, false),
            ([0.4038105, 2.8916466, -0.037852652, -2.6252642], 1.0, false),
            ([0.46164343, 3.0870361, -0.09035794, -2.9292643], 1.0, false),
            ([0.52338415, 3.282575, -0.14894322, -3.2479053], 1.0, false),
            ([0.58903563, 3.4779596, -0.21390133, -3.5813646], 1.0, true),
        ];

        let mut env = CartPole::v0();
        env.set_state([0.0, 0.0, 0.15, 1.0]);
        assert_matches_golden(&mut env, &GOLDEN_ANGLE_TERMINATION, &ACTIONS);
    }

    /// カート位置による終了(両側)。x の更新は x + τ·ẋ の運動学だけなので
    /// 期待値は手計算: 2.39 + 0.02×1.0 = 2.41 > 2.4。
    #[test]
    fn test_position_termination_both_sides() {
        let mut env = CartPole::v0();
        env.set_state([2.39, 1.0, 0.0, 0.0]);
        let (state, reward, done) = env.step(CartPoleAction::Right);
        assert!(approx_eq(state[0], 2.41, 1e-6));
        assert!(done, "x=2.41 > 2.4 should terminate");
        assert!(
            approx_eq(reward, 1.0, 1e-6),
            "terminal step still rewards 1.0"
        );

        let mut env = CartPole::v0();
        env.set_state([-2.39, -1.0, 0.0, 0.0]);
        let (state, _, done) = env.step(CartPoleAction::Left);
        assert!(approx_eq(state[0], -2.41, 1e-6));
        assert!(done, "x=-2.41 < -2.4 should terminate");
    }

    /// ステップ数上限(v0 の TimeLimit 相当)と、reset によるカウンタのリセット。
    #[test]
    fn test_max_steps_and_reset_clears_counter() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = CartPole::with_max_steps(3);

        // 原点付近からの 3 ステップでは物理的な終了条件には掛からない
        env.reset(&mut rng);
        let (_, _, done) = env.step(CartPoleAction::Left);
        assert!(!done);
        let (_, _, done) = env.step(CartPoleAction::Right);
        assert!(!done);
        let (_, _, done) = env.step(CartPoleAction::Left);
        assert!(done, "3rd step must hit the max_steps limit");

        // reset でカウンタが戻る(戻らなければ 1 ステップ目から done になる)
        env.reset(&mut rng);
        let (_, _, done) = env.step(CartPoleAction::Right);
        assert!(!done, "reset must clear the step counter");
    }

    /// reset の初期状態: 全 4 要素が [-0.05, 0.05) に入り、シード固定で再現する。
    #[test]
    fn test_reset_range_and_determinism() {
        let mut env = CartPole::v0();
        let state = env.reset(&mut StdRng::seed_from_u64(42));
        for (i, &v) in state.iter().enumerate() {
            assert!(
                (-0.05..0.05).contains(&v),
                "component {} out of init range: {}",
                i,
                v
            );
        }
        let state2 = env.reset(&mut StdRng::seed_from_u64(42));
        assert_eq!(
            state, state2,
            "same seed must reproduce the same init state"
        );
    }
}
