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
    generation: usize,
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
            generation: 0,
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

    pub fn add_grad(&self, gx: ArrayD<f32>) {
        let mut borrow = self.0.borrow_mut();
        if let Some(grad) = &mut borrow.grad {
            *grad += &gx;
        } else {
            borrow.grad = Some(gx);
        }
    }

    pub fn cleargrad(&self) {
        self.0.borrow_mut().grad = None;
    }

    pub fn set_creator(&self, func: Box<dyn Creator>) {
        self.0.borrow_mut().creator = Some(func);
    }

    pub fn backward(&self) {
        if self.grad().is_none() {
            self.set_grad(ArrayD::from_elem(self.data().shape(), 1.0f32));
        }

        let mut queue = vec![];
        let mut seen_set = std::collections::HashSet::new();

        let ptr = Rc::as_ptr(&self.0) as usize;
        seen_set.insert(ptr);
        queue.push(self.clone());

        while !queue.is_empty() {
            queue.sort_by_key(|v| v.0.borrow().generation);
            let var = queue.pop().unwrap();

            let computed_gradients = {
                let borrow = var.0.borrow();
                if let Some(creator) = &borrow.creator {
                    let grad = borrow.grad.as_ref().unwrap();
                    let gxs = creator.backward(grad);
                    let inputs = creator.get_inputs();
                    Some((gxs, inputs))
                } else {
                    None
                }
            };

            if let Some((gxs, inputs)) = computed_gradients {
                for (gx, input) in gxs.into_iter().zip(inputs.into_iter()) {
                    input.add_grad(gx);
                    let ptr = Rc::as_ptr(&input.0) as usize;
                    if !seen_set.contains(&ptr) {
                        seen_set.insert(ptr);
                        queue.push(input);
                    }
                }
            }
        }
    }

    pub fn square(&self) -> Variable {
        Square.call(std::slice::from_ref(self))
    }

    pub fn exp(&self) -> Variable {
        Exp.call(std::slice::from_ref(self))
    }

    pub fn add(&self, other: &Variable) -> Variable {
        Add.call(&[self.clone(), other.clone()])
    }
}

/// 逆伝播がグラフ遡行に必要とする最小の界面(ステップ7の creator 参照の Rust 版)。
///
/// 出力 Variable が `Box<dyn Creator>` として所有する。参照は常に過去向き
/// (出力 → 関数 → 入力)なので、現状のグラフに Rc の循環は存在しない。
pub trait Creator {
    fn backward(&self, gy: &ArrayD<f32>) -> Vec<ArrayD<f32>>;
    fn get_inputs(&self) -> Vec<Variable>;
}

/// 「関数と、その呼び出し時の入力」を束ねた計算グラフのノード。
///
/// `Function::call` が構築して出力の creator に渡すため、
/// 「入力が未設定の関数」という不正状態は型の上で存在しない。
pub struct Node<F> {
    inputs: Vec<Variable>,
    func: F,
}

impl<F: Function> Creator for Node<F> {
    fn backward(&self, gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let xs: Vec<ArrayD<f32>> = self.inputs.iter().map(|v| v.data()).collect();
        self.func.backward(&xs, gy)
    }

    fn get_inputs(&self) -> Vec<Variable> {
        self.inputs.clone()
    }
}

/// 順伝播だけの能力。数値微分(ステップ4)が関数に要求するのはここまで。
///
/// クロージャにはブランケット実装でこれだけを与える — `numerical_diff` には
/// 渡せるが、backward を持たないため `call` で計算グラフには入れない
/// (書こうとするとコンパイルエラーになる)。
pub trait Forward {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32>;
}

/// 本 ステップ2「変数を生み出す関数」。
///
/// `call` が Python 版 `__call__` に相当するテンプレートメソッド。self を消費して
/// `Node` に移し、出力 Variable が creator として所有する(ステップ7)。
/// `where Self: Sized` により `call` は vtable から外れ、trait は dyn 互換のまま。
/// `backward` は「入力 x と gy から gx」の純関数(ステップ6)。
pub trait Function: Forward {
    fn call(self, inputs: &[Variable]) -> Variable
    where
        Self: Sized + 'static,
    {
        let xs: Vec<ArrayD<f32>> = inputs.iter().map(|x| x.data()).collect();
        let result_data = self.forward(&xs);
        let result = Variable::new(result_data);

        let max_gen = inputs
            .iter()
            .map(|x| x.0.borrow().generation)
            .max()
            .unwrap_or(0);
        result.0.borrow_mut().generation = max_gen + 1;

        let node = Node {
            inputs: inputs.to_vec(),
            func: self,
        };

        result.set_creator(Box::new(node));
        result
    }

    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>>;
}

impl<T> Forward for T
where
    T: Fn(&[ArrayD<f32>]) -> ArrayD<f32>,
{
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        self(xs)
    }
}

/// 本 ステップ2: y = x²。backward は gx = 2x·gy(ステップ6)。
pub struct Square;
impl Forward for Square {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Square expects 1 input")
        };
        x.mapv(|v| v * v)
    }
}
impl Function for Square {
    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let [x] = xs else {
            panic!("Square expects 1 input")
        };
        vec![gy * (2.0 * x)]
    }
}

/// 本 ステップ3: y = eˣ。backward は gx = eˣ·gy(ステップ6)。
pub struct Exp;
impl Forward for Exp {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Exp expects 1 input")
        };
        x.mapv(|v| v.exp())
    }
}
impl Function for Exp {
    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let [x] = xs else {
            panic!("Exp expects 1 input")
        };
        vec![gy * x.mapv(|v| v.exp())]
    }
}

pub struct Add;
impl Forward for Add {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0, x1] = xs else {
            panic!("Add expects 2 inputs")
        };
        x0 + x1
    }
}
impl Function for Add {
    fn backward(&self, _xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        vec![gy.clone(), gy.clone()]
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

    let y0 = f.forward(&[x0]);
    let y1 = f.forward(&[x1]);

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

        let f = |xs: &[ArrayD<f32>]| {
            let y1 = Square.forward(xs);
            let y2 = Exp.forward(&[y1]);
            Square.forward(&[y2])
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

        fn get_gx(var: &Variable, gy: &ArrayD<f32>) -> ArrayD<f32> {
            var.0
                .borrow()
                .creator
                .as_ref()
                .unwrap()
                .backward(gy)
                .into_iter()
                .next()
                .unwrap()
        }

        let gy = y.grad().unwrap();
        b.set_grad(get_gx(&y, &gy));

        let gb = b.grad().unwrap();
        a.set_grad(get_gx(&b, &gb));

        let ga = a.grad().unwrap();
        x.set_grad(get_gx(&a, &ga));

        let expected_grad = array![[3.2974426f32]].into_dyn();
        assert!(approx_equal_arrayd(
            &x.grad().unwrap(),
            &expected_grad,
            1e-5 // 誤差の許容範囲
        ));

        let ga2 = a.grad().unwrap();
        let x_grad_2 = get_gx(&a, &ga2);
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

        let f = |xs: &[ArrayD<f32>]| {
            let y1 = Square.forward(xs);
            let y2 = Exp.forward(&[y1]);
            Square.forward(&[y2])
        };
        let num_grad_var = numerical_diff(f, &x, EPSILON_FOR_DIFF);
        let numerical_grad = num_grad_var.data();

        assert!(approx_equal_arrayd(&analytical_grad, &numerical_grad, 1e-3));
    }

    #[test]
    fn test_add_function() {
        let x0 = Variable::new(array![[2.0f32]].into_dyn());
        let x1 = Variable::new(array![[3.0f32]].into_dyn());
        let y = x0.add(&x1);

        assert!(approx_equal_arrayd(
            &y.data(),
            &array![[5.0f32]].into_dyn(),
            1e-5
        ));
    }

    #[test]
    fn test_add_backward() {
        let x = Variable::new(array![[2.0f32]].into_dyn());
        let y = Variable::new(array![[3.0f32]].into_dyn());

        // z = x^2 + y^2
        let z = x.square().add(&y.square());
        z.backward();

        assert!(approx_equal_arrayd(
            &x.grad().unwrap(),
            &array![[4.0f32]].into_dyn(),
            1e-5
        ));
        assert!(approx_equal_arrayd(
            &y.grad().unwrap(),
            &array![[6.0f32]].into_dyn(),
            1e-5
        ));
    }

    #[test]
    fn test_add_same_variable() {
        let x = Variable::new(array![[3.0f32]].into_dyn());
        let y = x.add(&x);
        y.backward();

        assert!(approx_equal_arrayd(
            &x.grad().unwrap(),
            &array![[2.0f32]].into_dyn(),
            1e-5
        ));

        x.cleargrad();

        let y2 = x.add(&x).add(&x);
        y2.backward();

        assert!(approx_equal_arrayd(
            &x.grad().unwrap(),
            &array![[3.0f32]].into_dyn(),
            1e-5
        ));
    }

    // 本 ステップ16: 複雑な計算グラフ(実装編)
    // 世代 (トポロジカルソート)の実装により、同じ変数が複数回 backward されず、
    // 正しい値(64.0)になることを確認する。
    #[test]
    fn test_complex_graph_step16() {
        let x = Variable::new(array![[2.0f32]].into_dyn());
        let a = x.square();
        let y = a.square().add(&a.square());

        y.backward();

        // 正解は dy/dx = 8x^3 = 64.0。世代管理により正しく計算される。
        assert!(approx_equal_arrayd(
            &x.grad().unwrap(),
            &array![[64.0f32]].into_dyn(),
            1e-5
        ));
    }
}
