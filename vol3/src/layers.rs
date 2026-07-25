use crate::variable::Variable;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

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
