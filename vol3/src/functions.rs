use crate::function::{Forward, Function};
use crate::variable::Variable;
use ndarray::{Array4, ArrayD, Ix4};

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

/// 本 ステップ51: ReLU y = max(x, 0)。
/// backward は「x>0 のマスク定数 × gy」(Clip と同族)。専用の勾配ノードを持たないため、
/// 二階微分も Mul の backward 経由で自動的に正しい(マスクは定数なので ∂²y/∂x² = 0 も自動)。
/// 専用ペア(Gather/GatherGrad 型)は「既存演算で書けない」ときだけ — ReLU は Mul で書ける。
pub struct Relu;
impl Forward for Relu {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Relu expects 1 input")
        };
        x.mapv(|v| v.max(0.0))
    }
}
impl Function for Relu {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Relu expects 1 input")
        };
        let mask = x.data().mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        vec![gy * &Variable::new(mask)]
    }
}
/// ReLU の関数形(`Variable::relu` メソッドと同じ。MLP の活性化に fn ポインタで渡せる)。
pub fn relu(x: &Variable) -> Variable {
    Relu.call(std::slice::from_ref(x))
}

/// 本 ステップ47: 対数関数 y = log(x)
pub struct Log;
impl Forward for Log {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Log expects 1 input")
        };
        x.mapv(|v| v.ln())
    }
}
impl Function for Log {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Log expects 1 input")
        };
        vec![gy / x]
    }
}

/// 本 ステップ47: y = clip(x, min, max)
/// 逆伝播では範囲内の要素だけ勾配を通す (範囲外は 0)
pub struct Clip {
    pub min: f32,
    pub max: f32,
}
impl Forward for Clip {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Clip expects 1 input")
        };
        x.mapv(|v| v.clamp(self.min, self.max))
    }
}
impl Function for Clip {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Clip expects 1 input")
        };
        let mask = x.data().mapv(|v| {
            if v >= self.min && v <= self.max {
                1.0
            } else {
                0.0
            }
        });
        vec![gy * &Variable::new(mask)]
    }
}

/// 本 ステップ47: 多クラス分類用の行ごとの要素抽出 (GetItem の特定ケース)
/// x (N, C) から indices (N,) に従って各行から1要素を抽出して (N,) を返す。
pub struct Gather {
    pub indices: Vec<usize>,
}
impl Forward for Gather {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Gather expects 1 input")
        };
        assert_eq!(x.ndim(), 2, "Gather expects a 2D array");
        let x2d = x.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let mut out = ndarray::Array1::<f32>::zeros(self.indices.len());
        for (i, &col) in self.indices.iter().enumerate() {
            out[i] = x2d[[i, col]];
        }
        out.into_dyn()
    }
}
impl Function for Gather {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Gather expects 1 input")
        };
        // 逆伝播は GatherGrad (ScatterAdd)
        vec![
            GatherGrad {
                indices: self.indices.clone(),
                in_shape: x.shape(),
            }
            .call(std::slice::from_ref(gy)),
        ]
    }
}

/// 本 ステップ47: Gather の逆伝播 — ゼロ配列への scatter-add。
/// GatherGrad の backward は再び Gather(BroadcastTo/SumTo と同じ双対ペア)なので、
/// 二階微分も閉じる(tests/step47.rs で distinct な値で検証済み)。
pub struct GatherGrad {
    pub indices: Vec<usize>,
    pub in_shape: Vec<usize>,
}
impl Forward for GatherGrad {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [gy] = xs else {
            panic!("GatherGrad expects 1 input")
        };
        let mut gx = ndarray::ArrayD::<f32>::zeros(self.in_shape.clone());
        let mut gx_view = gx.view_mut().into_dimensionality::<ndarray::Ix2>().unwrap();
        let gy_view = gy.view().into_dimensionality::<ndarray::Ix1>().unwrap();
        for (i, &col) in self.indices.iter().enumerate() {
            gx_view[[i, col]] += gy_view[i];
        }
        gx
    }
}
impl Function for GatherGrad {
    fn backward(&self, _xs: &[Variable], ggy: &Variable) -> Vec<Variable> {
        // GatherGrad の逆伝播は再び Gather になる (双対関係)
        vec![
            Gather {
                indices: self.indices.clone(),
            }
            .call(std::slice::from_ref(ggy)),
        ]
    }
}

/// 本 ステップ47: softmax(合成版)。新規 Function は不要 — exp / (sum_axis + reshape +
/// ブロードキャスト除算)の合成で、backward はステップ40のブロードキャスト機構が担う。
/// reshape 先は「x.shape() の axis 番目を 1 に置き換えた形」(keepdims 相当)。
/// 注意: 素朴な exp なのでロジットが ~88 を超えると f32 で inf(本家 simple 版と同じ制約。
/// 対策は「行 max を引いてから exp」— 必要になったら)。
pub fn softmax_simple(x: &Variable, axis: usize) -> Variable {
    let x_exp = x.exp();
    let sum_exp = x_exp.sum_axis(axis);

    // sum_axis で潰れた軸を 1 として復元し、ブロードキャストできるようにする (keepdims 相当)
    let mut sum_exp_shape = x.shape();
    sum_exp_shape[axis] = 1;
    let sum_exp_reshaped = sum_exp.reshape(&sum_exp_shape);

    &x_exp / &sum_exp_reshaped
}

/// 本 ステップ47: softmax 交差エントロピー(合成版)−Σ log p[i, t\[i\]] / N。
/// clip(1e-15, 1.0) が log(0) を防ぐ。ラベル t は微分対象でないため Variable ではなく
/// &[usize](Gather の関数状態になる)。勾配の閉形式 (p−t)/N でテスト済み。
pub fn softmax_cross_entropy_simple(x: &Variable, t: &[usize]) -> Variable {
    let n = x.shape()[0];
    let p = softmax_simple(x, 1);
    let p_clipped = p.clip(1e-15, 1.0);
    let log_p = p_clipped.ln();
    let tlog_p = log_p.gather(t);
    let sum_tlog_p = tlog_p.sum();
    sum_tlog_p * (-1.0 / n as f32)
}

/// スレッドローカルのグローバルRNGを使用してDropoutを適用する便利関数
pub fn dropout(x: &Variable, dropout_ratio: f32) -> Variable {
    dropout_with_rng(x, dropout_ratio, &mut rand::rng())
}

/// 乱数生成器(RNG)を明示的に受け取るDropoutの実装。
/// 合成関数として実装されているため、手動でのbackwardは不要（乗算ノードとして計算グラフに記録される）。
pub fn dropout_with_rng(x: &Variable, dropout_ratio: f32, rng: &mut impl rand::Rng) -> Variable {
    if crate::config::Config::train_mode() {
        use rand_distr::{Distribution, Uniform};
        let dist = Uniform::new(0.0, 1.0).unwrap();
        let scale = 1.0 / (1.0 - dropout_ratio);

        let mask_data = x.data().mapv(|_| {
            if dist.sample(rng) > dropout_ratio {
                scale
            } else {
                0.0
            }
        });

        let mask = Variable::new(mask_data);
        x * &mask
    } else {
        x.clone()
    }
}

/// 本 ステップ57: 指定した軸方向の最大値を取る関数。
pub struct Max {
    pub axis: Option<usize>,
    pub keepdims: bool,
}
impl Max {
    pub fn new(axis: Option<usize>, keepdims: bool) -> Self {
        Self { axis, keepdims }
    }
}
impl Forward for Max {
    fn forward(&self, xs: &[ndarray::ArrayD<f32>]) -> ndarray::ArrayD<f32> {
        let [x] = xs else {
            panic!("Max expects 1 input")
        };
        if let Some(ax) = self.axis {
            let mut y = x.fold_axis(ndarray::Axis(ax), f32::MIN, |&a, &b| a.max(b));
            if self.keepdims {
                y.insert_axis_inplace(ndarray::Axis(ax));
            }
            y.into_dyn()
        } else {
            let max_val = x.iter().fold(f32::MIN, |a, &b| a.max(b));
            ndarray::arr0(max_val).into_dyn()
        }
    }
}
impl Function for Max {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x] = xs else {
            panic!("Max expects 1 input")
        };
        let y_array = self.forward(&[x.data()]);

        let mut y_shape = y_array.shape().to_vec();
        let mut gy_shape = gy.shape();

        if !self.keepdims {
            if let Some(ax) = self.axis {
                y_shape.insert(ax, 1);
                gy_shape.insert(ax, 1);
            } else if x.ndim() > 0 {
                y_shape = vec![1; x.ndim()];
                gy_shape = vec![1; x.ndim()];
            }
        }

        let y_reshaped = y_array.into_shape_with_order(y_shape).unwrap();
        let y_bcast = y_reshaped.broadcast(x.shape()).unwrap();

        let cond_data = ndarray::Zip::from(x.data().view())
            .and(y_bcast)
            .map_collect(|&x_val, &y_val| if x_val == y_val { 1.0 } else { 0.0 });

        let cond = Variable::new(cond_data.into_dyn());
        let gy_reshaped = gy.reshape(&gy_shape);
        let gy_bcast = gy_reshaped.broadcast_to(&x.shape());

        vec![&gy_bcast * &cond]
    }
}
pub struct TransposeAxes {
    pub axes: Vec<usize>,
}
impl Forward for TransposeAxes {
    fn forward(&self, xs: &[ndarray::ArrayD<f32>]) -> ndarray::ArrayD<f32> {
        let [x] = xs else {
            panic!("TransposeAxes expects 1 input")
        };
        x.view()
            .permuted_axes(self.axes.clone())
            .as_standard_layout()
            .into_owned()
    }
}
impl Function for TransposeAxes {
    fn backward(&self, _xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let mut inv_axes = vec![0; self.axes.len()];
        for (i, &ax) in self.axes.iter().enumerate() {
            inv_axes[ax] = i;
        }
        vec![gy.transpose_axes(&inv_axes)]
    }
}

/// Vol5 で追加された、軸方向のスライスと結合の関数群。UNet の skip connection で使う。
// ==========================================
// SliceAxis (Concat の逆伝播用・分割オペレータ)
// ==========================================
pub struct SliceAxis {
    pub axis: usize,
    pub start: usize,
    pub end: usize,
}
impl Forward for SliceAxis {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("SliceAxis expects 1 input")
        };
        // slice_axis を使えば、動的な軸に対しても安全にスライス可能です
        x.slice_axis(
            ndarray::Axis(self.axis),
            ndarray::Slice::from(self.start..self.end),
        )
        .to_owned()
    }
}
impl Function for SliceAxis {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        // 元の入力の形状から次元の長さを復元
        let original_shape = xs[0].shape();
        let axis = self.axis;
        let dim = original_shape[axis];
        let mut gx = gy.clone();
        // 前半をゼロ埋め (start > 0 の場合)
        if self.start > 0 {
            let mut left_shape = original_shape.clone();
            left_shape[axis] = self.start;
            let zeros = Variable::new(ArrayD::zeros(left_shape));
            // zeros と gx を Concat (新しい計算グラフが作られる)
            gx = concat(&zeros, &gx, axis);
        }
        // 後半をゼロ埋め (end < dim の場合)
        if self.end < dim {
            let mut right_shape = original_shape.clone();
            right_shape[axis] = dim - self.end;
            let zeros = Variable::new(ArrayD::zeros(right_shape));
            // gx と zeros を Concat (新しい計算グラフが作られる)
            gx = concat(&gx, &zeros, axis);
        }
        vec![gx]
    }
}
pub fn slice_axis(x: &Variable, axis: usize, start: usize, end: usize) -> Variable {
    SliceAxis { axis, start, end }.call(std::slice::from_ref(x))
}
// ==========================================
// Concat
// ==========================================
pub struct Concat {
    pub axis: usize,
}
impl Forward for Concat {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0, x1] = xs else {
            panic!("Concat expects 2 inputs")
        };

        ndarray::concatenate(ndarray::Axis(self.axis), &[x0.view(), x1.view()])
            .expect("Concat failed: shape mismatch along non-concat axes")
    }
}
impl Function for Concat {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let [x0, x1] = xs else {
            panic!("Concat expects 2 inputs")
        };

        // 元の入力の幅を取得
        let w0 = x0.shape()[self.axis];
        let w1 = x1.shape()[self.axis];

        // 勾配 gy を元の幅で切り分ける (SliceAxis ノードがグラフに追加される)
        let gx0 = slice_axis(gy, self.axis, 0, w0);
        let gx1 = slice_axis(gy, self.axis, w0, w0 + w1);

        vec![gx0, gx1]
    }
}
// ユーザー向けフロント API
pub fn concat(x0: &Variable, x1: &Variable, axis: usize) -> Variable {
    Concat { axis }.call(&[x0.clone(), x1.clone()])
}

// ==========================================
// UpsampleBilinear2x
// ==========================================
pub struct UpsampleBilinear2x;
impl Forward for UpsampleBilinear2x {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Upsample expects 1 input")
        };
        assert_eq!(x.ndim(), 4, "Upsample expects 4D input");
        let shape = x.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = (h * 2, w * 2);
        let x_v = x.view().into_dimensionality::<Ix4>().unwrap();
        let mut out = Array4::<f32>::zeros((n, c, out_h, out_w));
        let scale = 2.0;
        for oy in 0..out_h {
            let src_y = (oy as f32 + 0.5) / scale - 0.5;
            let y0 = src_y.floor() as isize;
            let y1 = y0 + 1;
            let dy = src_y - y0 as f32;
            let y0_idx = y0.clamp(0, (h - 1) as isize) as usize;
            let y1_idx = y1.clamp(0, (h - 1) as isize) as usize;
            for ox in 0..out_w {
                let src_x = (ox as f32 + 0.5) / scale - 0.5;
                let x0 = src_x.floor() as isize;
                let x1 = x0 + 1;
                let dx = src_x - x0 as f32;
                let x0_idx = x0.clamp(0, (w - 1) as isize) as usize;
                let x1_idx = x1.clamp(0, (w - 1) as isize) as usize;
                let w00 = (1.0 - dy) * (1.0 - dx);
                let w01 = (1.0 - dy) * dx;
                let w10 = dy * (1.0 - dx);
                let w11 = dy * dx;
                for in_n in 0..n {
                    for in_c in 0..c {
                        let v00 = x_v[[in_n, in_c, y0_idx, x0_idx]];
                        let v01 = x_v[[in_n, in_c, y0_idx, x1_idx]];
                        let v10 = x_v[[in_n, in_c, y1_idx, x0_idx]];
                        let v11 = x_v[[in_n, in_c, y1_idx, x1_idx]];
                        out[[in_n, in_c, oy, ox]] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                    }
                }
            }
        }
        out.into_dyn()
    }
}
impl Function for UpsampleBilinear2x {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        // 黙って打ち切るのをやめ、対となる Grad ノードを計算グラフに接続する
        vec![upsample_bilinear_2x_grad(gy, &xs[0].shape())]
    }
}
pub fn upsample_bilinear_2x(x: &Variable) -> Variable {
    UpsampleBilinear2x.call(std::slice::from_ref(x))
}
// ==========================================
// UpsampleBilinear2xGrad (Upsample の双対)
// ==========================================
pub struct UpsampleBilinear2xGrad {
    pub input_shape: Vec<usize>,
}
impl Forward for UpsampleBilinear2xGrad {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [gy] = xs else {
            panic!("UpsampleGrad expects 1 input")
        };
        let (n, c, h, w) = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
        );
        let (out_h, out_w) = (h * 2, w * 2);
        let gy_v = gy.view().into_dimensionality::<Ix4>().unwrap();
        let mut gx = Array4::<f32>::zeros((n, c, h, w));
        let scale = 2.0;
        for oy in 0..out_h {
            let src_y = (oy as f32 + 0.5) / scale - 0.5;
            let y0 = src_y.floor() as isize;
            let y1 = y0 + 1;
            let dy = src_y - y0 as f32;
            let y0_idx = y0.clamp(0, (h - 1) as isize) as usize;
            let y1_idx = y1.clamp(0, (h - 1) as isize) as usize;
            for ox in 0..out_w {
                let src_x = (ox as f32 + 0.5) / scale - 0.5;
                let x0 = src_x.floor() as isize;
                let x1 = x0 + 1;
                let dx = src_x - x0 as f32;
                let x0_idx = x0.clamp(0, (w - 1) as isize) as usize;
                let x1_idx = x1.clamp(0, (w - 1) as isize) as usize;
                let w00 = (1.0 - dy) * (1.0 - dx);
                let w01 = (1.0 - dy) * dx;
                let w10 = dy * (1.0 - dx);
                let w11 = dy * dx;
                // Scatter 本体
                for in_n in 0..n {
                    for in_c in 0..c {
                        let g = gy_v[[in_n, in_c, oy, ox]];
                        gx[[in_n, in_c, y0_idx, x0_idx]] += w00 * g;
                        gx[[in_n, in_c, y0_idx, x1_idx]] += w01 * g;
                        gx[[in_n, in_c, y1_idx, x0_idx]] += w10 * g;
                        gx[[in_n, in_c, y1_idx, x1_idx]] += w11 * g;
                    }
                }
            }
        }
        gx.into_dyn()
    }
}
impl Function for UpsampleBilinear2xGrad {
    fn backward(&self, _xs: &[Variable], ggy: &Variable) -> Vec<Variable> {
        // 双対の環が完全に閉じる: Grad の backward は Upsample そのもの！
        vec![upsample_bilinear_2x(ggy)]
    }
}
pub fn upsample_bilinear_2x_grad(gy: &Variable, input_shape: &[usize]) -> Variable {
    UpsampleBilinear2xGrad {
        input_shape: input_shape.to_vec(),
    }
    .call(std::slice::from_ref(gy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{EPSILON_FOR_DIFF, approx_equal_arrayd};
    use ndarray::array;

    fn full_numerical_diff<F>(mut f: F, x: &Variable, h: f32) -> ArrayD<f32>
    where
        F: FnMut(&Variable) -> Variable,
    {
        let mut grad = ndarray::Array::zeros(x.shape());
        let mut grad_iter = grad.iter_mut();
        for i in 0..x.data().len() {
            let mut x_plus = x.data().clone();
            let mut x_minus = x.data().clone();

            *x_plus.iter_mut().nth(i).unwrap() += h;
            *x_minus.iter_mut().nth(i).unwrap() -= h;

            let y_plus = f(&Variable::new(x_plus)).data().sum();
            let y_minus = f(&Variable::new(x_minus)).data().sum();

            let diff = (y_plus - y_minus) / (2.0 * h);
            *grad_iter.next().unwrap() = diff;
        }
        grad.into_dyn()
    }

    #[test]
    fn test_concat_forward_and_backward() {
        // C=2 と C=3 の非対称な入力 (axis=1 で結合)
        // x0: shape [2, 2]
        let x0 = Variable::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());

        // x1: shape [2, 3]
        let x1 = Variable::new(array![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].into_dyn());

        // 1. Forward の確認
        let y = concat(&x0, &x1, 1);
        let expected_y = array![[1.0, 2.0, 5.0, 6.0, 7.0], [3.0, 4.0, 8.0, 9.0, 10.0]].into_dyn();
        assert!(approx_equal_arrayd(&y.data(), &expected_y, 1e-6));

        // 2. Backward の手計算値の確認 (出力勾配 gy を流し込む)
        let gy =
            Variable::new(array![[0.1, 0.2, 0.5, 0.6, 0.7], [0.3, 0.4, 0.8, 0.9, 1.0]].into_dyn());

        y.set_grad(gy);
        y.backward(false, false); // create_graph=false

        let expected_gx0 = array![[0.1, 0.2], [0.3, 0.4]].into_dyn();
        let expected_gx1 = array![[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]].into_dyn();

        assert!(approx_equal_arrayd(
            &x0.grad().unwrap(),
            &expected_gx0,
            1e-6
        ));
        assert!(approx_equal_arrayd(
            &x1.grad().unwrap(),
            &expected_gx1,
            1e-6
        ));

        // 3. gradient_check による数値微分との照合

        // 解析的勾配を計算 (gy を指定しないことで、全要素が 1.0 の勾配が流れる)
        let y_check = concat(&x0, &x1, 1);
        x0.cleargrad(); // 手計算テストで汚れている場合はクリア
        x1.cleargrad();
        y_check.backward(false, false);
        let analytical_gx0 = x0.grad().unwrap();
        let analytical_gx1 = x1.grad().unwrap();
        // x0 についての数値微分 (x1 を固定)
        let f_x0 = |x: &Variable| concat(x, &x1, 1);
        let num_gx0 = full_numerical_diff(f_x0, &x0, EPSILON_FOR_DIFF);
        // x1 についての数値微分 (x0 を固定)
        let f_x1 = |x: &Variable| concat(&x0, x, 1);
        let num_gx1 = full_numerical_diff(f_x1, &x1, EPSILON_FOR_DIFF);
        // 解析微分と数値微分を比較 (許容誤差 1e-3)
        assert!(
            approx_equal_arrayd(&analytical_gx0, &num_gx0, 1e-3),
            "gx0 gradient check failed"
        );
        assert!(
            approx_equal_arrayd(&analytical_gx1, &num_gx1, 1e-3),
            "gx1 gradient check failed"
        );
    }

    #[test]
    fn test_concat_double_backprop() {
        let x0 = Variable::new(array![[1.0, 2.0]].into_dyn()); // shape [1, 2]
        let x1 = Variable::new(array![[3.0, 4.0, 5.0]].into_dyn()); // shape [1, 3]
        // 1. 順伝播
        let y = concat(&x0, &x1, 1);
        // 2. 1階微分 (高階微分を有効にするため create_graph=true で逆伝播)
        let gy = Variable::new(array![[1.0, 2.0, 3.0, 4.0, 5.0]].into_dyn());
        y.set_grad(gy.clone());
        y.backward(true, true);
        // 1階微分の結果を取得 (SliceAxis によって分離された状態)
        let gx0 = x0.grad_var().unwrap();
        let gx1 = x1.grad_var().unwrap();
        // 3. 2階微分用の勾配をセット
        // gx0 と gx1 に対してそれぞれ別の勾配を与えて逆伝播させる
        let ggx0 = Variable::new(array![[10.0, 20.0]].into_dyn());
        let ggx1 = Variable::new(array![[30.0, 40.0, 50.0]].into_dyn());

        gx0.set_grad(ggx0);
        gx1.set_grad(ggx1);
        // 各出力勾配からグラフを逆行 (gy を目指して遡行)
        gx0.backward(false, false);
        gx1.backward(false, false);

        // 4. gy に集まった 2階微分の結果を確認
        // SliceAxis::backward 内での Concat(Pad) の結果が gy の grad として集結する
        let ggy = gy.grad().unwrap();
        let expected_ggy = array![[10.0, 20.0, 30.0, 40.0, 50.0]].into_dyn();

        assert!(approx_equal_arrayd(&ggy, &expected_ggy, 1e-6));
    }

    #[test]
    fn test_upsample_bilinear_2x() {
        // N=1, C=1, 2x2 入力
        let x_data = array![[[[1.0, 2.0,], [3.0, 4.0,]]]].into_dyn();
        let x = Variable::new(x_data.clone());
        // 1. Forward の手計算値アサート
        let y = upsample_bilinear_2x(&x);

        // align_corners=False に基づく bilinear 重みで補間された結果
        let expected_y = array![[[
            [1.00, 1.25, 1.75, 2.00,],
            [1.50, 1.75, 2.25, 2.50,],
            [2.50, 2.75, 3.25, 3.50,],
            [3.00, 3.25, 3.75, 4.00,]
        ]]]
        .into_dyn();
        // 誤差を伴う浮動小数演算のため epsilon で比較
        assert!(approx_equal_arrayd(&y.data(), &expected_y, 1e-4));
        // 2. Backward (非対称な勾配 gy を散らす)
        // 1.0 ~ 16.0 のバラバラな値を gy に注入して重み付け Scatter を検証
        let gy_data = array![[[
            [1.0, 2.0, 3.0, 4.0,],
            [5.0, 6.0, 7.0, 8.0,],
            [9.0, 10.0, 11.0, 12.0,],
            [13.0, 14.0, 15.0, 16.0,]
        ]]]
        .into_dyn();

        y.set_grad(Variable::new(gy_data.clone()));
        y.backward(false, false);
        let gx = x.grad().unwrap();
        // 四隅に重みに応じて散らされた勾配の手計算値
        // gx 合計 = 16.5 + 23.5 + 44.5 + 51.5 = 136.0 (gy の総和 1~16 = 136 に一致)
        let expected_gx = array![[[[16.5, 23.5,], [44.5, 51.5,]]]].into_dyn();

        assert!(approx_equal_arrayd(&gx, &expected_gx, 1e-4));

        // 3. Gradient check (非対称な gy を使った数値微分との照合)
        // full_numerical_diff は内部で .sum() を取るため、クロージャ内で gy を掛ければ
        // 任意の出力勾配を用いた内積（dot product）と同等になります。
        let f = |x_var: &Variable| {
            let y_var = upsample_bilinear_2x(x_var);
            let gy_var = Variable::new(gy_data.clone());
            // 出力に gy を掛けることで、sum() 時に目的の勾配がシミュレートされる
            y_var * &gy_var
        };

        // 線形につき打ち切りゼロ、大 h 純利益 (丸め誤差 O(ε·|f| / h) を抑制)
        let num_gx = full_numerical_diff(f, &x, 0.1);

        // 解析微分（先ほど手計算で 136.0 に合致した gx）と数値微分を比較
        assert!(
            approx_equal_arrayd(&gx, &num_gx, 1e-3),
            "UpsampleBilinear2x gradient check failed"
        );
    }

    #[test]
    fn test_upsample_bilinear_2x_double_backprop() {
        // N=1, C=1, 2x2
        let x_data = array![[[[1.0, 2.0], [3.0, 4.0]]]].into_dyn();
        let x = Variable::new(x_data.clone());
        // 1. 順伝播
        let y = upsample_bilinear_2x(&x);
        // 2. 1階微分 (非対称な gy を流してグラフを形成)
        let gy_data = array![[[
            [1.0, 2.0, 3.0, 4.0,],
            [5.0, 6.0, 7.0, 8.0,],
            [9.0, 10.0, 11.0, 12.0,],
            [13.0, 14.0, 15.0, 16.0,]
        ]]]
        .into_dyn();
        let gy = Variable::new(gy_data);

        y.set_grad(gy.clone());
        y.backward(true, true); // 高階微分を有効化
        // 1階微分の結果（UpsampleBilinear2xGrad によって生成された gx）
        let gx = x.grad_var().unwrap();

        // 3. 2階微分用の非対称な勾配 ggx を用意して逆伝播
        let ggx_data = array![[[[10.0, 20.0], [30.0, 40.0]]]].into_dyn();
        let ggx = Variable::new(ggx_data.clone());

        gx.set_grad(ggx.clone());
        gx.backward(false, false); // gy に向かって逆伝播
        // 4. アサート
        let ggy = gy.grad().unwrap();
        // 【証明】
        // gx は UpsampleBilinear2xGrad ノードであるため、その逆伝播は
        // ggx を UpsampleBilinear2x したものと等しくなるはずである。
        let expected_ggy = upsample_bilinear_2x(&ggx).data();

        assert!(
            approx_equal_arrayd(&ggy, &expected_ggy, 1e-4),
            "Double backprop (adjoint of adjoint) did not match forward upsample!"
        );
    }
}
