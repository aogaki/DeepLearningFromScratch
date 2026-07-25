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
pub fn dropout_with_rng(
    x: &Variable,
    dropout_ratio: f32,
    rng: &mut impl rand::RngCore,
) -> Variable {
    if crate::config::Config::train_mode() {
        use ndarray_rand::rand_distr::{Distribution, Uniform};
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
