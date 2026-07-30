use ndarray::{Array1, Array2};
use vol3::functions::{concat, relu, upsample_bilinear_2x};
use vol3::layers::{BatchNorm2d, Conv2d, Layer, Linear, Parameter};
use vol3::variable::Variable;

pub fn pos_encoding(t: &Array1<f32>, channels: usize) -> Array2<f32> {
    let mut pe = Array2::<f32>::zeros((t.len(), channels));
    for i in 0..channels {
        // 【要件】標準式のようにペアで分母を揃えず、インデックス i をそのまま適用する
        let div_term = ((i as f32 / channels as f32) * 10000.0f32.ln()).exp();
        for j in 0..t.len() {
            if i % 2 == 0 {
                pe[[j, i]] = (t[j] / div_term).sin();
            } else {
                pe[[j, i]] = (t[j] / div_term).cos();
            }
        }
    }
    pe
}

pub struct ConvBlock {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
}
impl ConvBlock {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        time_embed_dim: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        Self {
            conv1: Conv2d::new(in_ch, out_ch, (3, 3), (1, 1), (1, 1), false, rng),
            bn1: BatchNorm2d::new(out_ch),
            conv2: Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), false, rng),
            bn2: BatchNorm2d::new(out_ch),
            // 時刻埋め込みの出力次元は in_ch
            mlp_fc1: Linear::new(time_embed_dim, in_ch, false, rng),
            mlp_fc2: Linear::new(in_ch, in_ch, false, rng),
        }
    }

    // 固有メソッドとしての forward (Layer トレイトは使わない)
    pub fn forward(&self, x: &Variable, v: &Variable) -> Variable {
        // 1. 時刻埋め込み (mlp)
        let mut emb = self.mlp_fc1.forward(v);
        emb = relu(&emb);
        emb = self.mlp_fc2.forward(&emb);

        // (N, in_ch) -> (N, in_ch, 1, 1) へブロードキャスト用に変形
        let n = x.shape()[0];
        let in_ch = x.shape()[1];
        let emb_reshaped = emb.reshape(&[n, in_ch, 1, 1]);

        // 2. ブロック入力への加算
        // ここで Add の broadcast_to が発動。
        // 逆伝播時には自動的に sum_to 経由で (N,C,H,W) -> (N,C,1,1) に勾配が畳まれます
        let h = x + &emb_reshaped;

        // 3. 畳み込みと正規化の2段重ね
        let mut h = self.conv1.forward(&h);
        h = self.bn1.forward(&h);
        h = relu(&h);

        h = self.conv2.forward(&h);
        h = self.bn2.forward(&h);
        h = relu(&h);

        h
    }
}
impl Layer for ConvBlock {
    fn params(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.params());
        p.extend(self.bn1.params());
        p.extend(self.conv2.params());
        p.extend(self.bn2.params());
        p.extend(self.mlp_fc1.params());
        p.extend(self.mlp_fc2.params());
        p
    }
    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut p = Vec::new();
        // 親（UNet等）が prefix を付けるので、自身をルートとして展開する
        let mut add = |name: &str, params: Vec<(String, Parameter)>| {
            p.extend(
                params
                    .into_iter()
                    .map(|(k, v)| (format!("{}/{}", name, k), v)),
            );
        };
        add("conv1", self.conv1.named_params());
        add("bn1", self.bn1.named_params());
        add("conv2", self.conv2.named_params());
        add("bn2", self.bn2.named_params());
        add("mlp_fc1", self.mlp_fc1.named_params());
        add("mlp_fc2", self.mlp_fc2.named_params());
        p
    }
    fn forward(&self, _x: &Variable) -> Variable {
        unimplemented!("ConvBlock requires (x, v). Use inherent .forward(x, v) instead.");
    }
}

pub struct UNet {
    pub down1: ConvBlock,
    pub down2: ConvBlock,
    pub bot1: ConvBlock,
    pub up2: ConvBlock,
    pub up1: ConvBlock,
    pub out: Conv2d,
    pub time_embed_dim: usize,
}
impl UNet {
    pub fn new(in_ch: usize, time_embed_dim: usize, rng: &mut impl rand::Rng) -> Self {
        Self {
            down1: ConvBlock::new(in_ch, 64, time_embed_dim, rng),
            down2: ConvBlock::new(64, 128, time_embed_dim, rng),
            bot1: ConvBlock::new(128, 256, time_embed_dim, rng),
            // skip connection(128) + bot1(256) = 384
            up2: ConvBlock::new(384, 128, time_embed_dim, rng),
            // skip connection(64) + up2(128) = 192
            up1: ConvBlock::new(192, 64, time_embed_dim, rng),
            // 最終層 1x1 畳み込み
            out: Conv2d::new(64, in_ch, (1, 1), (1, 1), (0, 0), false, rng),

            time_embed_dim,
        }
    }

    pub fn forward(&self, x: &Variable, t: &Array1<f32>) -> Variable {
        // v を 1 度だけ計算
        let pe_data = pos_encoding(t, self.time_embed_dim);
        let v = Variable::new(pe_data.into_dyn());

        // --- Down ---
        let d1 = self.down1.forward(x, &v);
        let p1 = d1.pooling_simple((2, 2), (2, 2), (0, 0));

        let d2 = self.down2.forward(&p1, &v);
        let p2 = d2.pooling_simple((2, 2), (2, 2), (0, 0));

        // --- Bottleneck ---
        let bot = self.bot1.forward(&p2, &v);

        // --- Up ---
        let up_bot = upsample_bilinear_2x(&bot);
        let cat2 = concat(&up_bot, &d2, 1);
        let u2 = self.up2.forward(&cat2, &v);

        let up_u2 = upsample_bilinear_2x(&u2);
        let cat1 = concat(&up_u2, &d1, 1);
        let u1 = self.up1.forward(&cat1, &v);

        // --- Out ---
        self.out.forward(&u1)
    }
}
impl Layer for UNet {
    fn params(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.down1.params());
        p.extend(self.down2.params());
        p.extend(self.bot1.params());
        p.extend(self.up2.params());
        p.extend(self.up1.params());
        p.extend(self.out.params());
        p
    }
    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut p = Vec::new();
        let mut add = |name: &str, params: Vec<(String, Parameter)>| {
            p.extend(
                params
                    .into_iter()
                    .map(|(k, v)| (format!("{}/{}", name, k), v)),
            );
        };
        add("down1", self.down1.named_params());
        add("down2", self.down2.named_params());
        add("bot1", self.bot1.named_params());
        add("up2", self.up2.named_params());
        add("up1", self.up1.named_params());
        add("out", self.out.named_params());
        p
    }
    fn forward(&self, _x: &Variable) -> Variable {
        unimplemented!("UNet requires (x, t). Use inherent .forward(x, t) instead.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, array};
    use rand::SeedableRng;

    #[test]
    fn test_unet_shape_smoke() {
        // 1. 形状スモークテスト (ブロードキャスト加算の実地検証)
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let unet = UNet::new(1, 100, &mut rng);
        // N=2, C=1, H=28, W=28 の入力
        let x_data = Array::zeros((2, 1, 28, 28)).into_dyn();
        let x = Variable::new(x_data);

        // バッチサイズ=2 に対応する2つの時刻
        let t = array![10.0, 20.0];
        // ここを通ることで「(N, C, 1, 1) と (N, C, H, W) の Add」が vol3 で正しく処理されることが証明される
        let y = unet.forward(&x, &t);

        assert_eq!(y.shape(), &[2, 1, 28, 28], "UNet output shape mismatch");
    }

    #[test]
    fn test_unet_gradient_coverage() {
        // 2. 勾配カバレッジテスト (配線忘れの検出とパラメータ数の釘刺し)
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let unet = UNet::new(1, 100, &mut rng);
        let x_data = Array::zeros((2, 1, 28, 28)).into_dyn();
        let x = Variable::new(x_data);
        let t = array![10.0, 20.0];
        // 順伝播
        let y = unet.forward(&x, &t);
        let loss = y.sum();

        // 逆伝播 (ここで全パラメータに勾配が流れるはず)
        loss.backward(false, false);
        let params = unet.params();

        // 釘: ConvBlock(12) × 5 + out(2) = 62
        assert_eq!(
            params.len(),
            62,
            "Total parameter count should be exactly 62"
        );
        // 全パラメータが計算グラフに参加しているか (勾配をもっているか)
        for (i, p) in params.iter().enumerate() {
            assert!(
                p.grad().is_some(),
                "Parameter {} is disconnected from the computation graph! (Check wiring/skips)",
                i
            );
        }
    }
}
