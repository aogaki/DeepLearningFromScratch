use crate::variable::Variable;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
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
        let w_data = ndarray::Array::random_using(
            (in_size, out_size),
            Normal::new(0.0, std_dev).unwrap(),
            rng,
        );
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
        let w_data = ndarray::Array::random_using(
            (out_channels, in_channels, kernel_size.0, kernel_size.1),
            ndarray_rand::rand_distr::Normal::new(0.0, std_dev).unwrap(),
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
#[cfg(test)]
mod tests {
    use super::*;

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
}
