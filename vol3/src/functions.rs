use crate::function::{Forward, Function};
use crate::variable::Variable;
use ndarray::ArrayD;

/// 本 ステップ2: y = x²。backward は gx = 2x·gy(ステップ6、32で Variable 演算化)。
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
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Square expects 1 input")
        };
        vec![gy * (2.0 * x)]
    }
}

/// 本 ステップ3: y = eˣ。backward は gx = eˣ·gy(x から再計算する — 出力は保存しない設計)。
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
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Exp expects 1 input")
        };
        vec![gy * x.exp()]
    }
}

/// 本 ステップ11「可変長の引数」で登場した初の2入力関数。勾配は両入力へそのまま分配(ステップ13)。
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
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x0, x1] = xs else {
            panic!("Add expects 2 inputs")
        };
        vec![gy.sum_to(&x0.shape()), gy.sum_to(&x1.shape())]
    }
}

/// 本 ステップ20: y = x0·x1。backward は相手側の値を掛ける(gx0 = gy·x1、gx1 = gy·x0)。
pub struct Mul;
impl Forward for Mul {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0, x1] = xs else {
            panic!("Mul expects 2 inputs")
        };
        x0 * x1
    }
}
impl Function for Mul {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x0, x1] = xs else {
            panic!("Mul expects 2 inputs")
        };
        vec![(gy * x1).sum_to(&x0.shape()), (gy * x0).sum_to(&x1.shape())]
    }
}

/// 本 ステップ22: 単項マイナス。
pub struct Neg;
impl Forward for Neg {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0] = xs else {
            panic!("Neg expects 1 input")
        };
        -x0
    }
}
impl Function for Neg {
    fn backward(&self, _xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        vec![-gy]
    }
}

/// 本 ステップ22: 非可換演算その1。引かれる側は gy、引く側は −gy。
pub struct Sub;
impl Forward for Sub {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0, x1] = xs else {
            panic!("Sub expects 2 inputs")
        };
        x0 - x1
    }
}
impl Function for Sub {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x0, x1] = xs else {
            panic!("Sub expects 2 inputs")
        };
        vec![gy.sum_to(&x0.shape()), (-gy).sum_to(&x1.shape())]
    }
}

/// 本 ステップ22: 商の微分(gx0 = gy/x1、gx1 = −gy·x0/x1²)。
pub struct Div;
impl Forward for Div {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0, x1] = xs else {
            panic!("Div expects 2 inputs")
        };
        x0 / x1
    }
}
impl Function for Div {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x0, x1] = xs else {
            panic!("Div expects 2 inputs")
        };
        let gx0 = (gy / x1).sum_to(&x0.shape());
        let gx1 = (gy * (-x0 / x1.powf(2.0))).sum_to(&x1.shape());
        vec![gx0, gx1]
    }
}

/// 本 ステップ22: y = x^c。指数 c を持つ、初の状態つき関数
/// (`Node<F>` が関数を値ごと所有するため、フィールドがあっても設計は変わらない)。
pub struct Pow {
    pub c: f32,
}
impl Forward for Pow {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0] = xs else {
            panic!("Pow expects 1 input")
        };
        x0.mapv(|v| v.powf(self.c))
    }
}
impl Function for Pow {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x0] = xs else {
            panic!("Pow expects 1 input")
        };
        let c = self.c;
        let gx0 = gy * c * x0.powf(c - 1.0);
        vec![gx0]
    }
}

/// 本 ステップ27: y = sin x。backward は gy·cos x(テイラー展開の例題で登場)。
pub struct Sin;
impl Forward for Sin {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Sin expects 1 input")
        };
        x.mapv(|v| v.sin())
    }
}
impl Function for Sin {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Sin expects 1 input")
        };
        vec![gy * x.cos()]
    }
}

/// 本 ステップ32: y = cos x。Sin の backward を Variable 演算で書くために必要になった相棒。
pub struct Cos;
impl Forward for Cos {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Cos expects 1 input")
        };
        x.mapv(|v| v.cos())
    }
}
impl Function for Cos {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Cos expects 1 input")
        };
        vec![gy * -x.sin()]
    }
}

/// 本 ステップ35: y = tanh x。backward は gy·(1 − tanh²x)。
/// 本は保存済みの出力 y を使うが、この移植は関数が出力を持たない(=循環参照ゼロの)
/// 設計なので tanh を再計算する — 意図的なトレードオフ。
pub struct Tanh;
impl Forward for Tanh {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Tanh expects 1 input")
        };
        x.mapv(|v| v.tanh())
    }
}
impl Function for Tanh {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Tanh expects 1 input")
        };
        let y = x.tanh();
        vec![gy * (1.0 - y.powf(2.0))]
    }
}

/// 本 ステップ43: シグモイド関数 y = 1/(1+e^(−x))。
/// backward は gy·y(1−y) — Tanh と同様、出力を保存せず入力 x から y を再計算する方式。
pub struct Sigmoid;
impl Forward for Sigmoid {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Sigmoid expects 1 input")
        };
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }
}
impl Function for Sigmoid {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Sigmoid expects 1 input")
        };
        let y = x.sigmoid();
        vec![gy * &y * (1.0 - &y)]
    }
}

/// 本 ステップ38: 形を変える(要素の値と順序はそのまま)。backward は gy を元の形へ
/// reshape するだけ — 元の形は保存せず、backward が受け取る入力 `xs[0]` から読む。
/// forward の `as_standard_layout` は転置直後などの非標準レイアウト対策(vol1 の教訓)。
pub struct Reshape {
    pub shape: Vec<usize>,
}
impl Forward for Reshape {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Reshape expects 1 input")
        };
        x.as_standard_layout()
            .into_owned()
            .into_shape_with_order(self.shape.clone())
            .unwrap()
    }
}
impl Function for Reshape {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Reshape expects 1 input")
        };
        let original_shape = x.shape();
        vec![gy.reshape(&original_shape)]
    }
}

/// 本 ステップ38: 全軸反転の転置。自己逆元なので backward はもう一度 transpose。
pub struct Transpose;
impl Forward for Transpose {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Transpose expects 1 input")
        };
        x.t().as_standard_layout().into_owned()
    }
}
impl Function for Transpose {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [_x] = xs else {
            panic!("Transpose expects 1 input")
        };
        vec![gy.transpose()]
    }
}

/// 本 ステップ40: 形を押し広げる(要素の複製)。backward は sum_to — SumTo と互いが
/// 互いの逆伝播になる双対ペア。
pub struct BroadcastTo {
    pub shape: Vec<usize>,
}
impl Forward for BroadcastTo {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("BroadcastTo expects 1 input")
        };
        x.broadcast(self.shape.clone()).unwrap().into_owned()
    }
}
impl Function for BroadcastTo {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("BroadcastTo expects 1 input")
        };
        vec![gy.sum_to(&x.shape())]
    }
}

/// 本 ステップ40: 指定の形まで和で畳む(BroadcastTo の双対)。
/// Add/Sub/Mul/Div の backward がこれを通ることで、ブロードキャストされた演算の
/// 勾配が正しい形に戻る — ステップ21から抱えていた「スカラー勾配の形」の負債を精算した。
pub struct SumTo {
    pub shape: Vec<usize>,
}
impl Forward for SumTo {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("SumTo expects 1 input")
        };
        crate::utils::sum_to(x, &self.shape)
    }
}
impl Function for SumTo {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("SumTo expects 1 input")
        };
        vec![gy.broadcast_to(&x.shape())]
    }
}

/// 本 ステップ39: 和(None = 全和、Some(ax) = 軸1本)。本の axis タプル+keepdims の
/// フル装備ではなく、線形回帰〜MLP が実際に使う部分集合に絞ってある。
/// backward は「消えた軸を 1 で復元 → broadcast_to」。
pub struct Sum {
    pub axis: Option<usize>,
}
impl Forward for Sum {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Sum expects 1 input")
        };
        match self.axis {
            Some(ax) => x.sum_axis(ndarray::Axis(ax)).into_dyn(),
            None => ndarray::arr0(x.sum()).into_dyn(),
        }
    }
}
impl Function for Sum {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Sum expects 1 input")
        };
        let mut gy_shape = gy.shape();
        if let Some(ax) = self.axis {
            // keepdims=false で計算しているので、gy の shape に軸を復元する
            gy_shape.insert(ax, 1);
        } else if x.ndim() > 0 {
            // scalar result, reshape to all 1s matching original rank
            gy_shape = vec![1; x.ndim()];
        }
        let gy_reshaped = gy.reshape(&gy_shape);
        vec![gy_reshaped.broadcast_to(&x.shape())]
    }
}

/// 本 ステップ41: 行列の積。
/// ndarray の `dot` メソッドは 2 次元配列専用であるため、`into_dimensionality::<Ix2>()`
/// を通して型変換を行う。backward は `gx = gy @ W^T` と `gW = x^T @ gy` になる。
pub struct MatMul;
impl Forward for MatMul {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x, w] = xs else {
            panic!("MatMul expects 2 inputs")
        };
        let x_view = x
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("MatMul requires 2D input for x");
        let w_view = w
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("MatMul requires 2D input for W");
        x_view.dot(&w_view).into_dyn()
    }
}
impl Function for MatMul {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x, w] = xs else {
            panic!("MatMul expects 2 inputs")
        };
        let gx = gy.matmul(&w.transpose());
        let gw = x.transpose().matmul(gy);
        vec![gx, gw]
    }
}

/// 本 ステップ42: 平均二乗誤差 sum((x0−x1)²)/N。
/// フレームワークの演算の合成なので、それ自体が微分可能(二階微分も自動で正しい)。
pub fn mean_squared_error(x0: &Variable, x1: &Variable) -> Variable {
    let diff = x0 - x1;
    let batch_size = x0.shape()[0] as f32;
    diff.powf(2.0).sum() / batch_size
}

/// 本 ステップ43: 線形変換 y = xW (+ b)。本の linear_simple に相当する合成関数。
/// bias の有無は Option で表現(Python のデフォルト引数 None の型付き版)。
/// b の加算は (N,o)+(o,) のブロードキャストで、backward の sum_to が bias 勾配を畳む。
pub fn linear(x: &Variable, w: &Variable, b: Option<&Variable>) -> Variable {
    let t = x.matmul(w);
    match b {
        Some(b_var) => t + b_var,
        None => t,
    }
}
