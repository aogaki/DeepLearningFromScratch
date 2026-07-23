//! 本 第1ステージ(ステップ1〜10)「微分を自動で求める」
//!
//! DeZero の動的計算グラフの最小核。Python 版との対応:
//! - `Variable` … 箱としての変数(ステップ1)+ grad(6)+ creator(7)
//! - `Function` trait … 関数の基底クラス(2)。`call` が `__call__` に相当
//! - `Node` … `__call__` が覚える「入力と関数」の組(7)を独立させたグラフノード
//! - `numerical_diff` … 数値微分(4)
//!
//! Python の「全てが共有参照」を、Rust では `Rc<RefCell<…>>` で明示する。
//! `Variable` はその薄いハンドルで、clone は参照カウントの増加のみ(データは複製されない)。

use ndarray::ArrayD;
use std::cell::RefCell;
use std::rc::Rc;

/// `Variable` の実体。グラフ上で共有されるため、常に `Rc<RefCell>` 越しに触る。
struct VariableInner {
    data: ArrayD<f32>,
    grad: Option<ArrayD<f32>>,
    creator: Option<Box<dyn Creator>>,
}

/// 本 ステップ1「箱としての変数」(grad はステップ6、creator はステップ7で追加)。
///
/// 実体への薄いハンドルで、`clone` しても中身は共有される(Python の変数の意味論)。
/// 勾配は `backward`(ステップ7・8)がグラフを遡って書き込み、`grad()` で読む。
/// `square`/`exp` のメソッドチェーンはステップ9「関数をより便利に」に対応。
#[derive(Clone)]
pub struct Variable(Rc<RefCell<VariableInner>>);

impl Variable {
    pub fn new(data: ArrayD<f32>) -> Self {
        Variable(Rc::new(RefCell::new(VariableInner {
            data,
            grad: None,
            creator: None,
        })))
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0.borrow().grad.clone()
    }

    pub fn set_grad(&self, grad: ArrayD<f32>) {
        self.0.borrow_mut().grad = Some(grad);
    }

    pub fn set_creator(&self, func: Box<dyn Creator>) {
        self.0.borrow_mut().creator = Some(func);
    }

    pub fn backward(&self) {
        if self.grad().is_none() {
            self.set_grad(ArrayD::from_elem(self.data().shape(), 1.0f32));
        }

        let mut queue = vec![self.clone()];

        while let Some(var) = queue.pop() {
            let (gx, input) = {
                let borrow = var.0.borrow();
                if let Some(creator) = &borrow.creator {
                    let grad = borrow.grad.as_ref().unwrap();
                    let gx = creator.backward(grad);
                    let input = creator.get_input();
                    (Some(gx), Some(input))
                } else {
                    (None, None)
                }
            };

            if let (Some(gx), Some(input)) = (gx, input) {
                input.set_grad(gx);
                queue.push(input);
            }
        }
    }

    pub fn square(&self) -> Variable {
        Square.call(self)
    }

    pub fn exp(&self) -> Variable {
        Exp.call(self)
    }
}

/// 逆伝播がグラフ遡行に必要とする最小の界面(ステップ7の creator 参照の Rust 版)。
///
/// 出力 Variable が `Box<dyn Creator>` として所有する。参照は常に過去向き
/// (出力 → 関数 → 入力)なので、現状のグラフに Rc の循環は存在しない。
pub trait Creator {
    fn backward(&self, gy: &ArrayD<f32>) -> ArrayD<f32>;
    fn get_input(&self) -> Variable;
}

/// 「関数と、その呼び出し時の入力」を束ねた計算グラフのノード。
///
/// `Function::call` が構築して出力の creator に渡すため、
/// 「入力が未設定の関数」という不正状態は型の上で存在しない。
pub struct Node<F> {
    input: Variable,
    func: F,
}

impl<F: Function> Creator for Node<F> {
    fn backward(&self, gy: &ArrayD<f32>) -> ArrayD<f32> {
        let x_data = &self.input.0.borrow().data;
        self.func.backward(x_data, gy)
    }

    fn get_input(&self) -> Variable {
        self.input.clone()
    }
}

/// 順伝播だけの能力。数値微分(ステップ4)が関数に要求するのはここまで。
///
/// クロージャにはブランケット実装でこれだけを与える — `numerical_diff` には
/// 渡せるが、backward を持たないため `call` で計算グラフには入れない
/// (書こうとするとコンパイルエラーになる)。
pub trait Forward {
    fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32>;
}

/// 本 ステップ2「変数を生み出す関数」。
///
/// `call` が Python 版 `__call__` に相当するテンプレートメソッド。self を消費して
/// `Node` に移し、出力 Variable が creator として所有する(ステップ7)。
/// `where Self: Sized` により `call` は vtable から外れ、trait は dyn 互換のまま。
/// `backward` は「入力 x と gy から gx」の純関数(ステップ6)。
pub trait Function: Forward {
    fn call(self, x: &Variable) -> Variable
    where
        Self: Sized + 'static,
    {
        let result_data = self.forward(&x.0.borrow().data);
        let result = Variable::new(result_data);

        let node = Node {
            input: x.clone(),
            func: self,
        };

        result.set_creator(Box::new(node));
        result
    }

    fn backward(&self, x: &ArrayD<f32>, gy: &ArrayD<f32>) -> ArrayD<f32>;
}

impl<T> Forward for T
where
    T: Fn(&ArrayD<f32>) -> ArrayD<f32>,
{
    fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        self(x)
    }
}

/// 本 ステップ2: y = x²。backward は gx = 2x·gy(ステップ6)。
pub struct Square;
impl Forward for Square {
    fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        x.mapv(|v| v * v)
    }
}
impl Function for Square {
    fn backward(&self, x: &ArrayD<f32>, gy: &ArrayD<f32>) -> ArrayD<f32> {
        gy * (2.0 * x)
    }
}

/// 本 ステップ3: y = eˣ。backward は gx = eˣ·gy(ステップ6)。
pub struct Exp;
impl Forward for Exp {
    fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        x.mapv(|v| v.exp())
    }
}
impl Function for Exp {
    fn backward(&self, x: &ArrayD<f32>, gy: &ArrayD<f32>) -> ArrayD<f32> {
        gy * x.mapv(|v| v.exp())
    }
}

/// 本 ステップ4「数値微分」— 中心差分 (f(x+h) − f(x−h)) / 2h。
///
/// 注意: 本の eps=1e-4 は float64 用の値。f32 では丸め誤差 O(ε/h) が支配的に
/// なるため、h ≈ ∛ε_f32 ≈ 5e-3 が目安(テストの EPSILON_FOR_DIFF がこれ)。
pub fn numerical_diff<F>(f: F, x: &Variable, eps: f32) -> Variable
where
    F: Forward,
{
    let mut x0 = x.data();
    let mut x1 = x.data();

    x0 += eps;
    x1 -= eps;

    let y0 = f.forward(&x0);
    let y1 = f.forward(&x1);

    let diff_data = (y0 - y1) / (2.0 * eps);
    Variable::new(diff_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // f32 の中心差分の最適幅 h* ≈ ∛ε_machine ≈ 5e-3。
    // 本の 1e-4 は float64 用で、f32 では丸め誤差(桁落ち)が支配的になる。
    const EPSILON_FOR_DIFF: f32 = 5e-3;

    // 形の一致 + 全要素の絶対誤差で比較(浮動小数に == は使わない)
    fn approx_equal_arrayd(a: &ArrayD<f32>, b: &ArrayD<f32>, tol: f32) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (*x - *y).abs() < tol)
    }

    // ステップ1: Variable がデータを保持する
    #[test]
    fn test_variable_creation() {
        let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data = ArrayD::from_shape_vec(vec![2, 3], original_data.clone()).unwrap();
        let variable = Variable::new(data);
        assert_eq!(
            variable.data(),
            Variable::new(ArrayD::from_shape_vec(vec![2, 3], original_data).unwrap()).data()
        );
    }

    // ステップ2: Square の順伝播(呼び方はステップ9のメソッド記法)
    #[test]
    fn test_square_function() {
        let original_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let variable = Variable::new(original_data.clone().into_dyn());
        let result_variable = variable.square();
        let expected_data = array![[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]];
        assert!(approx_equal_arrayd(
            &result_variable.data(),
            &expected_data.clone().into_dyn(),
            1e-6
        ));
    }

    // ステップ3: Exp 単体と、Square→Exp→Square の合成
    #[test]
    fn test_exp_function() {
        let original_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let variable = Variable::new(original_data.clone().into_dyn());
        let result_variable = variable.exp();
        let expected_data = array![
            [1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()],
            [4.0f32.exp(), 5.0f32.exp(), 6.0f32.exp()]
        ];
        assert!(approx_equal_arrayd(
            &result_variable.data(),
            &expected_data.clone().into_dyn(),
            1e-6
        ));

        let final_variable = variable.square().exp().square();
        let expected_final_data = array![
            [
                1.0f32.powi(2).exp().powi(2),
                2.0f32.powi(2).exp().powi(2),
                3.0f32.powi(2).exp().powi(2)
            ],
            [
                4.0f32.powi(2).exp().powi(2),
                5.0f32.powi(2).exp().powi(2),
                6.0f32.powi(2).exp().powi(2)
            ]
        ];
        assert!(approx_equal_arrayd(
            &final_variable.data(),
            &expected_final_data.clone().into_dyn(),
            1e-6
        ));
    }

    // ステップ4: 数値微分。単一関数(解析解 2x)と、合成関数(4x·e^{2x²}、本と同じ x=0.5)。
    // 合成側は値が急伸するので小さな x のみ(f32 の絶対誤差比較が成立する範囲)
    #[test]
    fn test_numerical_diff() {
        let original_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let variable = Variable::new(original_data.clone().into_dyn());
        let diff_variable = numerical_diff(Square, &variable, EPSILON_FOR_DIFF);
        let expected_diff_data = array![[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]];
        assert!(approx_equal_arrayd(
            &diff_variable.data(),
            &expected_diff_data.clone().into_dyn(),
            1e-3
        ));

        let simple_data = array![[0.5f32]];
        let variable = Variable::new(simple_data.clone().into_dyn());

        let f = |x: &ArrayD<f32>| {
            let y1 = Square.forward(x);
            let y2 = Exp.forward(&y1);
            Square.forward(&y2)
        };

        let diff_variable = numerical_diff(f, &variable, EPSILON_FOR_DIFF);
        let expected_diff_data = simple_data.mapv(|x| 4.0 * x * (2.0 * x.powi(2)).exp());
        assert!(approx_equal_arrayd(
            &diff_variable.data(),
            &expected_diff_data.clone().into_dyn(),
            1e-3
        ));
    }

    // ステップ6: 手作業の逆伝播(creator を1段ずつ辿る)。
    // 末尾で backward の冪等性(同じ creator で2回計算しても同じ答え — 決定性の
    // 検証なのでここだけビット一致の assert_eq が正しい)も確認する
    #[test]
    fn test_manual_backward_chain() {
        let x_data = array![[0.5f32]].into_dyn();
        let x = Variable::new(x_data);

        let a = x.square();
        let b = a.exp();
        let y = b.square();

        // y.grad = 1.0 (shapeは揃える)
        y.set_grad(array![[1.0f32]].into_dyn());

        let gy = y.grad().unwrap();
        b.set_grad(y.0.borrow().creator.as_ref().unwrap().backward(&gy));

        let gb = b.grad().unwrap();
        a.set_grad(b.0.borrow().creator.as_ref().unwrap().backward(&gb));

        let ga = a.grad().unwrap();
        x.set_grad(a.0.borrow().creator.as_ref().unwrap().backward(&ga));

        let expected_grad = array![[3.2974426f32]].into_dyn();
        assert!(approx_equal_arrayd(
            &x.grad().unwrap(),
            &expected_grad,
            1e-5 // 誤差の許容範囲
        ));

        let ga2 = a.grad().unwrap();
        let x_grad_2 = a.0.borrow().creator.as_ref().unwrap().backward(&ga2);
        assert_eq!(x.grad().unwrap(), x_grad_2);
    }

    // ステップ7〜8: y.backward() 一発でグラフを遡る(ループ実装)。
    // grad の 1 初期化(ステップ9の先取り)もここで効いている
    #[test]
    fn test_auto_backward() {
        let x_data = array![[0.5f32]].into_dyn();
        let x = Variable::new(x_data);

        let y = x.square().exp().square();

        // ステップ7の自動逆伝播！
        y.backward();

        let expected_grad = array![[3.2974426f32]].into_dyn();
        assert!(approx_equal_arrayd(
            &x.grad().unwrap(),
            &expected_grad,
            1e-5
        ));
    }

    // ステップ10: 勾配チェック(backprop の解析勾配 vs 数値微分)。
    // 以後のステップで backward を書き換えても、この2本が正しさの安全網になる
    #[test]
    fn test_gradient_check() {
        let x = Variable::new(array![[0.5f32]].into_dyn());

        let y = x.square();
        y.backward();
        let analytical_grad = x.grad().unwrap();

        let num_grad_var = numerical_diff(Square, &x, EPSILON_FOR_DIFF);
        let numerical_grad = num_grad_var.data();

        assert!(approx_equal_arrayd(&analytical_grad, &numerical_grad, 1e-3));
    }

    // ステップ10: 合成関数(Square→Exp→Square)版の勾配チェック
    #[test]
    fn test_gradient_check_composite() {
        let x = Variable::new(array![[0.5f32]].into_dyn());

        let y = x.square().exp().square();
        y.backward();
        let analytical_grad = x.grad().unwrap();

        let f = |x_arr: &ArrayD<f32>| {
            let y1 = Square.forward(x_arr);
            let y2 = Exp.forward(&y1);
            Square.forward(&y2)
        };
        let num_grad_var = numerical_diff(f, &x, EPSILON_FOR_DIFF);
        let numerical_grad = num_grad_var.data();

        assert!(approx_equal_arrayd(&analytical_grad, &numerical_grad, 1e-3));
    }
}
