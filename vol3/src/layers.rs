use crate::utils::random_array;
use crate::variable::Variable;
use ndarray::Array;
use rand_distr::Normal;
use std::cell::RefCell;

/// パラメータであることを明示するためのエイリアス(ドキュメント用)
pub type Parameter = Variable;

/// 本 ステップ44: パラメータをまとめる層(Layer)
///
/// Python 版の `Layer` は `__setattr__` や `isinstance` を使って暗黙的にパラメータを
/// 収集するが、Rust 版では静的型付けを活かして各レイヤが `params()` で明示的に列挙する。
/// これにより実行時の文字列ルックアップによるエラーを防ぎつつ、`cleargrads` 等の一括操作を実現する。
pub trait Layer {
    /// 学習・最適化用(ホットパス)。パラメータの参照(Rcクローン)のみを平坦化して返す。
    fn params(&self) -> Vec<Parameter>;

    /// 保存・デバッグ用(コールドパス)。階層構造をプレフィックス付きの名前として集約する。
    fn named_params(&self) -> Vec<(String, Parameter)>;

    /// 全パラメータの勾配をクリアする(デフォルト実装)。
    fn cleargrads(&self) {
        for p in self.params() {
            p.cleargrad();
        }
    }

    /// Truncated BPTTなどで計算グラフを過去に向かって切り離す(デフォルト実装は何もしない)。
    /// RNNなどの状態を持つレイヤは、これをオーバーライドして内部状態(hやc)の `unchain_backward` を呼ぶ。
    fn unchain_backward(&self) {}

    /// ステップ53: パラメータを .npz 形式でファイルに保存する。
    /// Python (NumPy) と互換性を持たせるため、キー名には `.npy` 拡張子が自動的に付与される
    /// (ndarray-npy ライブラリ側の仕様)。
    fn save_weights(&self, path: &std::path::Path) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut npz = ndarray_npy::NpzWriter::new(file);
        for (name, param) in self.named_params() {
            let arr = param.data();
            npz.add_array(&name, &arr).map_err(std::io::Error::other)?;
        }
        npz.finish().map_err(std::io::Error::other)?;
        Ok(())
    }

    /// ステップ53: .npz 形式のファイルからパラメータを読み込み、既存のパラメータに in-place で代入する。
    /// 遅延初期化を持たないため、モデル構築直後にそのまま読み込むことができる。
    fn load_weights(&self, path: &std::path::Path) -> std::io::Result<()> {
        let file = std::fs::File::open(path)?;
        let mut npz = ndarray_npy::NpzReader::new(file).map_err(std::io::Error::other)?;
        let mut loaded_arrays = Vec::new();
        let named_params = self.named_params();

        // 1パス目: 全パラメータを読み込み、形状の検証を行う
        for (name, param) in &named_params {
            let loaded: ndarray::ArrayD<f32> = npz.by_name(name).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to load '{}': {}", name, e),
                )
            })?;

            let expected_shape = param.shape();
            if loaded.shape() != expected_shape {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Shape mismatch for {}: expected {:?}, got {:?}",
                        name,
                        expected_shape,
                        loaded.shape()
                    ),
                ));
            }
            loaded_arrays.push(loaded);
        }

        // 2パス目: 全ての検証に通過した場合のみ、既存のパラメータに in-place で代入する
        for ((_, param), loaded) in named_params.into_iter().zip(loaded_arrays.into_iter()) {
            param.set_data(loaded);
        }
        Ok(())
    }

    /// 順伝播(単一入出力用)。
    /// 将来 `Vec<Box<dyn Layer>>` などに格納できるよう、動的ディスパッチ(object-safe)を考慮した設計。
    fn forward(&self, x: &Variable) -> Variable;
}

/// 本 ステップ44: 線形変換レイヤ (単純版: in_size 必須)
/// y = x @ W + b
pub struct Linear {
    pub w: Parameter,
    pub b: Option<Parameter>,
}
impl Linear {
    pub fn new(in_size: usize, out_size: usize, nobias: bool, rng: &mut impl rand::Rng) -> Self {
        // Xavierの初期値に近いスケール(1 / sqrt(in_size))
        let std_dev = 1.0 / (in_size as f32).sqrt();
        let w_data = random_array((in_size, out_size), Normal::new(0.0, std_dev).unwrap(), rng);
        let w = Variable::new(w_data.into_dyn());

        let b = if nobias {
            None
        } else {
            let b_data = ndarray::Array::zeros((out_size,));
            Some(Variable::new(b_data.into_dyn()))
        };

        Self { w, b }
    }
}
impl Layer for Linear {
    fn params(&self) -> Vec<Parameter> {
        let mut p = vec![self.w.clone()];
        if let Some(b) = &self.b {
            p.push(b.clone());
        }
        p
    }

    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut p = vec![("W".to_string(), self.w.clone())];
        if let Some(b) = &self.b {
            p.push(("b".to_string(), b.clone()));
        }
        p
    }

    fn forward(&self, x: &Variable) -> Variable {
        crate::functions::linear(x, &self.w, self.b.as_ref())
    }
}

/// 本 ステップ45: TwoLayerNet (固定アーキテクチャのモデル例)
pub struct TwoLayerNet {
    pub l1: Linear,
    pub l2: Linear,
}
impl TwoLayerNet {
    pub fn new(
        in_size: usize,
        hidden_size: usize,
        out_size: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        Self {
            l1: Linear::new(in_size, hidden_size, false, rng),
            l2: Linear::new(hidden_size, out_size, false, rng),
        }
    }
}
impl Layer for TwoLayerNet {
    fn params(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.l1.params());
        p.extend(self.l2.params());
        p
    }

    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut p = Vec::new();
        for (name, param) in self.l1.named_params() {
            p.push((format!("l1/{}", name), param));
        }
        for (name, param) in self.l2.named_params() {
            p.push((format!("l2/{}", name), param));
        }
        p
    }

    fn forward(&self, x: &Variable) -> Variable {
        let h = self.l1.forward(x);
        let h = h.sigmoid();
        self.l2.forward(&h)
    }
}

/// 本 ステップ45: MLP(可変長アーキテクチャ)。sizes は [入力, 隠れ…, 出力] の全サイズ列
/// (単純版 Linear が in_size を要求するため — 本家の遅延 in_size 初期化の代替)。
/// 活性化は fn ポインタ(`Variable::sigmoid` / `Variable::relu` をそのまま渡せる)で、
/// 最終層の後には挟まない(本家の layers[:-1] と同じ意味論)。
pub struct MLP {
    pub layers: Vec<Linear>,
    pub activation: fn(&Variable) -> Variable,
}
impl MLP {
    pub fn new(
        sizes: &[usize],
        activation: fn(&Variable) -> Variable,
        rng: &mut impl rand::Rng,
    ) -> Self {
        assert!(
            sizes.len() >= 2,
            "MLP requires at least input and output size"
        );
        let mut layers = Vec::new();
        for w in sizes.windows(2) {
            layers.push(Linear::new(w[0], w[1], false, rng));
        }
        Self { layers, activation }
    }
}
impl Layer for MLP {
    fn params(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for l in &self.layers {
            p.extend(l.params());
        }
        p
    }

    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut p = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (name, param) in l.named_params() {
                p.push((format!("l{}/{}", i, name), param));
            }
        }
        p
    }

    fn forward(&self, x: &Variable) -> Variable {
        let mut h = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            h = l.forward(&h);
            if i < self.layers.len() - 1 {
                h = (self.activation)(&h);
            }
        }
        h
    }
}

pub struct Conv2d {
    pub w: Parameter,
    pub b: Option<Parameter>,
    pub stride: (usize, usize),
    pub pad: (usize, usize),
}
impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        pad: (usize, usize),
        nobias: bool,
        rng: &mut impl rand::Rng,
    ) -> Self {
        // VGG16/He/Xavier scale. For consistency with DeZero, we use 1/sqrt(in_channels)
        let std_dev = 1.0 / (in_channels as f32).sqrt();
        let w_data = random_array(
            (out_channels, in_channels, kernel_size.0, kernel_size.1),
            Normal::new(0.0, std_dev).unwrap(),
            rng,
        )
        .into_dyn();
        let w = Parameter::new(w_data);

        let b = if nobias {
            None
        } else {
            let b_data = ndarray::Array::zeros((out_channels,)).into_dyn();
            Some(Parameter::new(b_data))
        };

        Self { w, b, stride, pad }
    }
}
impl Layer for Conv2d {
    fn params(&self) -> Vec<Parameter> {
        let mut p = vec![self.w.clone()];
        if let Some(b) = &self.b {
            p.push(b.clone());
        }
        p
    }

    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut p = vec![("W".to_string(), self.w.clone())];
        if let Some(b) = &self.b {
            p.push(("b".to_string(), b.clone()));
        }
        p
    }

    fn forward(&self, x: &Variable) -> Variable {
        x.conv2d_simple(&self.w, self.b.as_ref(), self.stride, self.pad)
    }
}

pub struct RNN {
    pub x2h: Linear,
    pub h2h: Linear,
    pub h: std::cell::RefCell<Option<Variable>>,
}
impl RNN {
    pub fn new(in_size: usize, hidden_size: usize, rng: &mut impl rand::Rng) -> Self {
        Self {
            x2h: Linear::new(in_size, hidden_size, false, rng),
            h2h: Linear::new(hidden_size, hidden_size, true, rng),
            h: std::cell::RefCell::new(None),
        }
    }

    pub fn reset_state(&self) {
        *self.h.borrow_mut() = None;
    }
}
impl Layer for RNN {
    fn params(&self) -> Vec<Parameter> {
        self.named_params().into_iter().map(|(_, p)| p).collect()
    }

    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut p = Vec::new();
        for (name, param) in self.x2h.named_params() {
            p.push((format!("x2h/{}", name), param));
        }
        for (name, param) in self.h2h.named_params() {
            p.push((format!("h2h/{}", name), param));
        }
        p
    }

    fn forward(&self, x: &Variable) -> Variable {
        let h_new = match &*self.h.borrow() {
            Some(h) => (self.x2h.forward(x) + self.h2h.forward(h)).tanh(),
            None => self.x2h.forward(x).tanh(),
        };
        *self.h.borrow_mut() = Some(h_new.clone());
        h_new
    }

    fn unchain_backward(&self) {
        if let Some(h) = self.h.borrow().as_ref() {
            h.unchain();
        }
    }
}

pub struct LSTM {
    pub x2f: Linear,
    pub x2i: Linear,
    pub x2o: Linear,
    pub x2u: Linear,
    pub h2f: Linear,
    pub h2i: Linear,
    pub h2o: Linear,
    pub h2u: Linear,
    pub h: RefCell<Option<Variable>>,
    pub c: RefCell<Option<Variable>>,
}
impl LSTM {
    pub fn new(in_size: usize, hidden_size: usize, rng: &mut impl rand::Rng) -> Self {
        Self {
            x2f: Linear::new(in_size, hidden_size, false, rng),
            x2i: Linear::new(in_size, hidden_size, false, rng),
            x2o: Linear::new(in_size, hidden_size, false, rng),
            x2u: Linear::new(in_size, hidden_size, false, rng),
            h2f: Linear::new(hidden_size, hidden_size, true, rng),
            h2i: Linear::new(hidden_size, hidden_size, true, rng),
            h2o: Linear::new(hidden_size, hidden_size, true, rng),
            h2u: Linear::new(hidden_size, hidden_size, true, rng),
            h: RefCell::new(None),
            c: RefCell::new(None),
        }
    }

    pub fn reset_state(&self) {
        *self.h.borrow_mut() = None;
        *self.c.borrow_mut() = None;
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        let (h_prev, c_prev) = match (&*self.h.borrow(), &*self.c.borrow()) {
            (Some(h), Some(c)) => (Some(h.clone()), Some(c.clone())),
            _ => (None, None),
        };

        let f;
        let i;
        let o;
        let u;

        if let Some(h) = h_prev {
            f = self.x2f.forward(x) + self.h2f.forward(&h);
            i = self.x2i.forward(x) + self.h2i.forward(&h);
            o = self.x2o.forward(x) + self.h2o.forward(&h);
            u = self.x2u.forward(x) + self.h2u.forward(&h);
        } else {
            f = self.x2f.forward(x);
            i = self.x2i.forward(x);
            o = self.x2o.forward(x);
            u = self.x2u.forward(x);
        }

        let f_gate = f.sigmoid();
        let i_gate = i.sigmoid();
        let o_gate = o.sigmoid();
        let u_gate = u.tanh();

        let c_new = if let Some(c) = c_prev {
            &f_gate * &c + &i_gate * &u_gate
        } else {
            &i_gate * &u_gate
        };

        let h_new = &o_gate * c_new.tanh();

        *self.h.borrow_mut() = Some(h_new.clone());
        *self.c.borrow_mut() = Some(c_new);

        h_new
    }
}
impl Layer for LSTM {
    fn forward(&self, x: &Variable) -> Variable {
        self.forward(x)
    }

    fn unchain_backward(&self) {
        if let Some(h) = self.h.borrow().as_ref() {
            h.unchain();
        }
        if let Some(c) = self.c.borrow().as_ref() {
            c.unchain();
        }
    }

    fn params(&self) -> Vec<Parameter> {
        self.named_params().into_iter().map(|(_, p)| p).collect()
    }

    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut p = Vec::new();
        let layers: Vec<(&str, &dyn Layer)> = vec![
            ("x2f", &self.x2f),
            ("x2i", &self.x2i),
            ("x2o", &self.x2o),
            ("x2u", &self.x2u),
            ("h2f", &self.h2f),
            ("h2i", &self.h2i),
            ("h2o", &self.h2o),
            ("h2u", &self.h2u),
        ];

        for (name, l) in layers {
            for (sub_name, param) in l.named_params() {
                p.push((format!("{}/{}", name, sub_name), param));
            }
        }
        p
    }
}

/// Vol5 での利用のため拡張
// ==========================================
// BatchNorm2d
// ==========================================
pub struct BatchNorm2d {
    pub gamma: Parameter,
    pub beta: Parameter,
    pub running_mean: Parameter,
    pub running_var: Parameter,
    pub eps: f32,
    pub momentum: f32,
}
impl BatchNorm2d {
    pub fn new(c: usize) -> Self {
        Self {
            gamma: Variable::new(Array::ones((c,)).into_dyn()),
            beta: Variable::new(Array::zeros((c,)).into_dyn()),
            running_mean: Variable::new(Array::zeros((c,)).into_dyn()),
            running_var: Variable::new(Array::ones((c,)).into_dyn()),
            eps: 1e-5, // Adam の 1e-8 ではなく BatchNorm 標準の 1e-5
            momentum: 0.1,
        }
    }
}
impl Layer for BatchNorm2d {
    fn params(&self) -> Vec<Parameter> {
        // 【要件】running stats はオプティマイザの更新対象 (params) から意図的に除外
        vec![self.gamma.clone(), self.beta.clone()]
    }
    fn named_params(&self) -> Vec<(String, Parameter)> {
        // 【要件】保存・ロード用 (named_params) には全て含める
        vec![
            ("gamma".to_string(), self.gamma.clone()),
            ("beta".to_string(), self.beta.clone()),
            ("running_mean".to_string(), self.running_mean.clone()),
            ("running_var".to_string(), self.running_var.clone()),
        ]
    }
    fn forward(&self, x: &Variable) -> Variable {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "BatchNorm2d expects 4D input (N, C, H, W)");
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        // チャンネルごとの計算を行うため、他次元を潰すブロードキャスト用形状
        let axes_shape = vec![1, c, 1, 1];
        let eps_var = Variable::new(ndarray::arr0(self.eps).into_dyn());
        if crate::config::Config::train_mode() {
            // ========================================================
            // Train モード: 全て Variable の合成演算で書く (backward 自動化)
            // ========================================================
            let m = (n * h * w) as f32;
            let m_var = Variable::new(ndarray::arr0(m).into_dyn());
            // 1. Mean (N, H, W 軸で sum して m で割る)
            let sum = x.sum_to(&axes_shape);
            let batch_mean = &sum / &m_var;
            // 2. Variance (Biased)
            let diff = x - &batch_mean;
            let sq_diff = diff.powf(2.0);
            let batch_var_biased = sq_diff.sum_to(&axes_shape) / &m_var;
            // 3. Normalize: eps はルートの中
            let std = (&batch_var_biased + &eps_var).powf(0.5);
            let x_hat = &diff / &std;
            // 4. Scale and Shift (gamma / beta もブロードキャスト形状に)
            let gamma_b = self.gamma.reshape(&axes_shape);
            let beta_b = self.beta.reshape(&axes_shape);
            let out = &x_hat * &gamma_b + &beta_b;
            // ========================================================
            // 統計量の更新 (副作用・計算グラフには乗せない)
            // ========================================================
            let mom = self.momentum;

            // 【最凶の罠】バッチ分散の蓄積には biased(÷N) ではなく unbiased(÷N-1) を使う
            let m_unbiased_factor = if m > 1.0 { m / (m - 1.0) } else { 1.0 };
            // (1, C, 1, 1) を (C,) に戻す
            let b_mean_arr = batch_mean.data().into_shape_with_order(vec![c]).unwrap();
            let b_var_arr = batch_var_biased
                .data()
                .into_shape_with_order(vec![c])
                .unwrap();
            // 【罠】PyTorch 式の漸化式向き: running = (1 - m) * running + m * batch
            let new_rm = self.running_mean.data() * (1.0 - mom) + &b_mean_arr * mom;
            self.running_mean.set_data(new_rm);
            let new_rv =
                self.running_var.data() * (1.0 - mom) + &b_var_arr * (mom * m_unbiased_factor);
            self.running_var.set_data(new_rv);
            out
        } else {
            // ========================================================
            // Eval モード: バッチ統計を使わず、保存された running stats を使用
            // ========================================================
            let rm = self.running_mean.reshape(&axes_shape);
            let rv = self.running_var.reshape(&axes_shape);
            let std = (&rv + &eps_var).powf(0.5);
            let x_hat = (x - &rm) / &std;
            let gamma_b = self.gamma.reshape(&axes_shape);
            let beta_b = self.beta.reshape(&axes_shape);

            &x_hat * &gamma_b + &beta_b
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_equal_arrayd;
    use ndarray::ArrayD;

    #[test]
    fn test_linear_params() {
        let mut rng = rand::rng();
        // nobias = false の場合 (デフォルト動作)
        let l_bias = Linear::new(2, 3, false, &mut rng);
        let params_bias = l_bias.named_params();
        assert_eq!(params_bias.len(), 2);
        assert_eq!(params_bias[0].0, "W");
        assert_eq!(params_bias[1].0, "b");

        // nobias = true の場合
        let l_nobias = Linear::new(2, 3, true, &mut rng);
        let params_nobias = l_nobias.named_params();
        assert_eq!(params_nobias.len(), 1);
        assert_eq!(params_nobias[0].0, "W");

        // forward 実行 (バイアスなし分岐も通す)
        let x = Variable::new(ndarray::Array::zeros((10, 2)).into_dyn());
        let y = l_nobias.forward(&x);
        assert_eq!(y.shape(), vec![10, 3]);
    }

    #[test]
    fn test_batchnorm2d_params_isolation() {
        // 1. オプティマイザ食われ事故の封じ込めテスト
        let bn = BatchNorm2d::new(3);

        let params = bn.params();
        assert_eq!(
            params.len(),
            2,
            "params() should only contain gamma and beta"
        );

        let named_params = bn.named_params();
        assert_eq!(
            named_params.len(),
            4,
            "named_params() should contain gamma, beta, running_mean, running_var"
        );
    }

    #[test]
    fn test_batchnorm2d_golden() {
        // 2. per-op golden テスト (完全なる容疑者の一掃)
        let file = std::fs::File::open("dataset/batchnorm_golden.npz").unwrap();
        let mut npz = ndarray_npy::NpzReader::new(file).unwrap();
        macro_rules! load {
            ($name:expr) => {
                npz.by_name($name).expect(concat!("Failed to load ", $name))
            };
        }
        let x1_data: ArrayD<f32> = load!("x1.npy");
        let gy1_data: ArrayD<f32> = load!("gy1.npy");
        let gamma_data: ArrayD<f32> = load!("gamma.npy");
        let beta_data: ArrayD<f32> = load!("beta.npy");
        // --- 1周目: Train Forward & Backward ---
        let bn = BatchNorm2d::new(3);
        bn.gamma.set_data(gamma_data.clone());
        bn.beta.set_data(beta_data.clone());
        let x1 = Variable::new(x1_data);
        let y1 = bn.forward(&x1);
        // 容疑者クリア確認: 統計の軸(N·H·W), biased 分散, eps の位置, γ/β
        let expected_y1: ArrayD<f32> = load!("y1.npy");
        assert!(
            approx_equal_arrayd(&y1.data(), &expected_y1, 1e-4),
            "y1 failed"
        );
        // Backward 実行 (合成スタイルにつき完全自動の筈)
        y1.set_grad(Variable::new(gy1_data));
        y1.backward(false, false);
        // 容疑者クリア確認: gx1, dgamma1, dbeta1 がピタリと一致するか
        let expected_gx1: ArrayD<f32> = load!("gx1.npy");
        let expected_dgamma1: ArrayD<f32> = load!("dgamma1.npy");
        let expected_dbeta1: ArrayD<f32> = load!("dbeta1.npy");
        assert!(
            approx_equal_arrayd(&x1.grad().unwrap(), &expected_gx1, 1e-4),
            "gx1 failed"
        );
        assert!(
            approx_equal_arrayd(&bn.gamma.grad().unwrap(), &expected_dgamma1, 1e-4),
            "dgamma1 failed"
        );
        assert!(
            approx_equal_arrayd(&bn.beta.grad().unwrap(), &expected_dbeta1, 1e-4),
            "dbeta1 failed"
        );
        // 容疑者クリア確認: momentum の向き, unbiased 蓄積
        let expected_rm1: ArrayD<f32> = load!("rm1.npy");
        let expected_rv1: ArrayD<f32> = load!("rv1.npy");
        assert!(
            approx_equal_arrayd(&bn.running_mean.data(), &expected_rm1, 1e-4),
            "rm1 failed"
        );
        assert!(
            approx_equal_arrayd(&bn.running_var.data(), &expected_rv1, 1e-4),
            "rv1 failed"
        );
        // --- 2周目: Train Forward (漸化式の 2 周目チェック) ---
        let x2_data: ArrayD<f32> = load!("x2.npy");
        let x2 = Variable::new(x2_data);
        let _y2 = bn.forward(&x2);
        let expected_rm2: ArrayD<f32> = load!("rm2.npy");
        let expected_rv2: ArrayD<f32> = load!("rv2.npy");
        assert!(
            approx_equal_arrayd(&bn.running_mean.data(), &expected_rm2, 1e-4),
            "rm2 failed"
        );
        assert!(
            approx_equal_arrayd(&bn.running_var.data(), &expected_rv2, 1e-4),
            "rv2 failed"
        );
        {
            let _guard = crate::config::test_mode();

            // --- 3周目: Eval 分岐 ---
            let x3_data: ArrayD<f32> = load!("x3.npy");
            let x3 = Variable::new(x3_data);
            let y3_eval = bn.forward(&x3);
            // 容疑者クリア確認: バッチ統計を使わず running stats だけで正規化されているか
            let expected_y3_eval: ArrayD<f32> = load!("y3_eval.npy");
            assert!(
                approx_equal_arrayd(&y3_eval.data(), &expected_y3_eval, 1e-4),
                "y3_eval failed"
            );
        }
    }
}
