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
    fn backward(&self, _xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        vec![gy.clone(), gy.clone()]
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
        vec![gy * x1, gy * x0]
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
    fn backward(&self, _xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        vec![gy.clone(), -gy]
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
        let gx0 = gy / x1;
        let gx1 = gy * (-x0 / x1.powf(2.0));
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
