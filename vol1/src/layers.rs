use crate::loss::batch_cross_entropy_error;
use crate::network::{sigmoid, softmax};
use ndarray::{Array, Array2, Dimension, Ix2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// 本 5.4.1「乗算レイヤの実装」forward で入力を保存し backward で入れ替えて掛ける
pub struct MulLayer {
    x: f32,
    y: f32,
}

impl MulLayer {
    pub fn forward(&mut self, x: f32, y: f32) -> f32 {
        self.x = x;
        self.y = y;
        x * y
    }

    pub fn backward(&self, dout: f32) -> (f32, f32) {
        let dx = dout * self.y;
        let dy = dout * self.x;
        (dx, dy)
    }
}

/// 本 5.4.2「加算レイヤの実装」backward は上流の微分をそのまま両側へ配る
pub struct AddLayer;
impl AddLayer {
    pub fn forward(&self, x: f32, y: f32) -> f32 {
        x + y
    }

    pub fn backward(&self, dout: f32) -> (f32, f32) {
        let dx = dout * 1.0;
        let dy = dout * 1.0;
        (dx, dy)
    }
}

/// 本 5.5.1「ReLUレイヤ」x<=0 の位置を mask で覚え、forward/backward で堰き止める
/// 本 7.5 「CNNの実装」でいくつかの次元に対応するためジェネリクス化する
pub struct ReluLayer<D: Dimension = Ix2> {
    mask: Option<Array<bool, D>>,
}
impl<D: Dimension> ReluLayer<D> {
    pub fn new() -> Self {
        Self { mask: None }
    }

    pub fn forward(&mut self, x: Array<f32, D>) -> Array<f32, D> {
        self.mask = Some(x.mapv(|v| v <= 0.0)); // mask は先に保存
        x.mapv(|v| v.max(0.0)) // 0以下は0、あとは素通し
    }

    pub fn backward(&self, dout: Array<f32, D>) -> Array<f32, D> {
        let mut dx = dout;
        let mask = self
            .mask
            .as_ref()
            .expect("ReluLayer: forward must be called before backward");

        dx.iter_mut().zip(mask.iter()).for_each(|(d, &m)| {
            if m {
                *d = 0.0;
            }
        });
        dx
    }
}

/// 本 6.4.3「Dropout」訓練時はニューロンをランダムに消去(mask は true=生存)、
/// テスト時は全ニューロンを使い出力を (1-ratio) 倍する。
/// 訓練/推論の切り替えは明示的な train_flg 引数で行う(渡し忘れがコンパイルエラーになる)
pub struct DropoutLayer {
    dropout_ratio: f32,
    mask: Array2<bool>,
}
impl DropoutLayer {
    pub fn new(dropout_ratio: f32) -> Self {
        Self {
            dropout_ratio,
            mask: Array2::default((0, 0)),
        }
    }

    pub fn forward(&mut self, mut x: Array2<f32>, train_flg: bool) -> Array2<f32> {
        if train_flg {
            let dist = Uniform::new(0.0, 1.0).unwrap();
            let r = Array2::random(x.raw_dim(), dist);
            self.mask = r.mapv(|v| v > self.dropout_ratio);
            x.iter_mut().zip(self.mask.iter()).for_each(|(v, &m)| {
                if !m {
                    *v = 0.0;
                }
            });
            x
        } else {
            x * (1.0 - self.dropout_ratio)
        }
    }

    pub fn backward(&self, dout: Array2<f32>) -> Array2<f32> {
        let mut dx = dout.clone();
        dx.iter_mut().zip(self.mask.iter()).for_each(|(d, &m)| {
            if !m {
                *d = 0.0
            }
        });
        dx
    }
}

/// 本 5.5.2「Sigmoidレイヤ」出力 y を保存し backward は dout*y*(1-y)
pub struct SigmoidLayer {
    out: Array2<f32>,
}
impl SigmoidLayer {
    pub fn new() -> Self {
        Self {
            out: Array2::default((0, 0)),
        }
    }

    pub fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        self.out = x.mapv(|v| sigmoid(v));
        self.out.clone()
    }

    pub fn backward(&self, dout: Array2<f32>) -> Array2<f32> {
        dout * &self.out * &(1.0 - &self.out) // out * (1 - out) はシグモイド関数の微分
    }
}

/// 本 5.6「Affineレイヤ」Y=X·W+B。W,B と勾配 dW,dB を所有し(approach A)、
/// backward で dX=dout·Wᵀ, dW=Xᵀ·dout, dB=dout の列和 を求める
pub struct AffineLayer {
    w: Array2<f32>,
    b: Array2<f32>,
    x: Array2<f32>,
    dw: Array2<f32>,
    db: Array2<f32>,
}
impl AffineLayer {
    pub fn new(w: Array2<f32>, b: Array2<f32>) -> Self {
        let x = Array2::default((0, 0));
        let dw = Array2::zeros(w.raw_dim());
        let db = Array2::zeros(b.raw_dim());
        Self { w, b, x, dw, db }
    }

    pub fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        // self.x = x.clone();
        // x.dot(&self.w) + &self.b
        let out = x.dot(&self.w) + &self.b;
        self.x = x;
        out
    }

    pub fn backward(&mut self, dout: Array2<f32>) -> Array2<f32> {
        self.dw = self.x.t().dot(&dout);
        self.db = dout
            .sum_axis(ndarray::Axis(0))
            .insert_axis(ndarray::Axis(0));
        dout.dot(&self.w.t())
    }

    pub fn dw(&self) -> &Array2<f32> {
        &self.dw
    }

    pub fn db(&self) -> &Array2<f32> {
        &self.db
    }

    pub fn w(&self) -> &Array2<f32> {
        &self.w
    }

    pub fn b(&self) -> &Array2<f32> {
        &self.b
    }

    /// 本 6.1 Optimizer 用の分割借用アクセサ。W(可変)と dW(共有)を同時に貸し出す。
    /// メソッド内でフィールドを直接組にすることで、w_mut() + dw() の借用衝突を回避する
    pub fn w_and_dw(&mut self) -> (&mut Array2<f32>, &Array2<f32>) {
        (&mut self.w, &self.dw)
    }

    pub fn b_and_db(&mut self) -> (&mut Array2<f32>, &Array2<f32>) {
        (&mut self.b, &self.db)
    }

    pub fn w_mut(&mut self) -> &mut Array2<f32> {
        &mut self.w
    }

    pub fn b_mut(&mut self) -> &mut Array2<f32> {
        &mut self.b
    }

    /// 本 5.7.4 保存済みの勾配 dW,dB で自身の W,B を更新する(SGD)
    pub fn update(&mut self, lr: f32) {
        self.w.scaled_add(-lr, &self.dw);
        self.b.scaled_add(-lr, &self.db);
    }

    /// 本 6.4.2「Weight decay」罰則項 (λ/2)ΣW² の微分 λW を dW に加算する。
    /// loss 側の罰則項と必ず対で使う(片方だけだと勾配確認が崩れる)。バイアスには適用しない
    pub fn add_weight_decay(&mut self, weight_decay_lambda: f32) {
        self.dw.scaled_add(weight_decay_lambda, &self.w); // weight decay
    }
}

/// 本 5.6.3「Softmax-with-Lossレイヤ」softmax+交差エントロピー。backward は (y-t)/N
pub struct SoftmaxWithLossLayer {
    y: Array2<f32>,
    t: Array2<f32>,
}
impl SoftmaxWithLossLayer {
    pub fn new() -> Self {
        Self {
            y: Array2::default((0, 0)),
            t: Array2::default((0, 0)),
        }
    }

    pub fn forward(&mut self, x: Array2<f32>, t: Array2<f32>) -> f32 {
        self.t = t;
        self.y = Array2::zeros(x.raw_dim());
        for (i, row) in x.outer_iter().enumerate() {
            let softmax_row = softmax(row.to_owned());
            self.y.row_mut(i).assign(&softmax_row);
        }
        batch_cross_entropy_error(self.y.view(), self.t.view())
    }

    pub fn backward(&self, dout: f32) -> Array2<f32> {
        let batch_size = self.t.shape()[0] as f32;
        (&self.y - &self.t) / batch_size * dout
    }
}

/// 本 6.3「Batch Normalization」Affine の出力をミニバッチ単位で列ごとに正規化し
/// (平均0・分散1)、学習可能な γ(スケール)と β(シフト)で調整する。
/// 初期値のスケールに依存しない学習を可能にする(6.3.2 の実験)。
/// γ, β は現状固定(γ=1, β=0)。dgamma/dbeta は backward で計算済みだが Optimizer 未接続。
/// 推論時にバッチ統計でなく移動平均を使う対応も未実装(訓練と同じ forward を使用)
pub struct BatchNormLayer {
    gamma: Array2<f32>,
    beta: Array2<f32>,
    dgamma: Array2<f32>,
    dbeta: Array2<f32>,

    xc: Array2<f32>,
    xn: Array2<f32>,
    std: Array2<f32>,
    batch_size: usize,
}
impl BatchNormLayer {
    pub fn new(feature_size: usize) -> Self {
        let gamma = Array2::ones((1, feature_size));
        let beta = Array2::zeros((1, feature_size));
        let dgamma = Array2::zeros((1, feature_size));
        let dbeta = Array2::zeros((1, feature_size));
        let xc = Array2::zeros((0, feature_size));
        let xn = Array2::zeros((0, feature_size));
        let std = Array2::zeros((1, feature_size));
        let batch_size = 0;
        Self {
            gamma,
            beta,
            dgamma,
            dbeta,
            xc,
            xn,
            std,
            batch_size,
        }
    }

    pub fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(ndarray::Axis(0)).unwrap();
        let var = x.var_axis(ndarray::Axis(0), 0.0);
        let epsilon = 1e-7;
        let x_hat = (x.clone() - &mean) / (&var + epsilon).mapv(|v| v.sqrt());

        self.batch_size = x.shape()[0];
        self.xc = x - &mean;
        self.xn = x_hat.clone();
        self.std = Array2::from_shape_vec(
            (1, (&var + epsilon).len_of(ndarray::Axis(0))),
            (&var + epsilon).mapv(|v| v.sqrt()).to_vec(),
        )
        .unwrap();
        &self.gamma * x_hat + &self.beta
    }

    /// 本 6.3.1 の計算グラフを逆にたどる。鉄則:順伝播で1つの値が全サンプルに配られた量
    /// (γ, β, std, var, μ は特徴ごとのスカラー)は、逆伝播で必ずバッチ方向の和になる
    /// (sum_axis(0) → (1, features))。正しさは数値微分テストで担保
    pub fn backward(&mut self, dout: Array2<f32>) -> Array2<f32> {
        self.dbeta = dout
            .sum_axis(ndarray::Axis(0))
            .insert_axis(ndarray::Axis(0));
        self.dgamma = (self.xn.clone() * &dout)
            .sum_axis(ndarray::Axis(0))
            .insert_axis(ndarray::Axis(0));

        let dxn = &dout * &self.gamma;
        let dstd = (dxn.clone() * &self.xc)
            .sum_axis(ndarray::Axis(0))
            .insert_axis(ndarray::Axis(0))
            / (&self.std.mapv(|v| v.powi(2)) * -1.0);
        let dvar = 0.5 * dstd / &self.std;
        let dxc = &dxn / &self.std + (2.0 / self.batch_size as f32) * &self.xc * &dvar;
        let dmu = dxc.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        dxc - &dmu / self.batch_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    fn approx_eq_array(a: &Array2<f32>, b: &Array2<f32>, epsilon: f32) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| (x - y).abs() < epsilon)
    }

    #[test]
    fn mul_layer_test() {
        let apple_price = 100.0;
        let apple_num = 2.0;
        let tax = 1.1;
        let mut mul_apple_layer = MulLayer { x: 0.0, y: 0.0 };
        let mut mul_tax_layer = MulLayer { x: 0.0, y: 0.0 };

        let total_price_wo_tax = mul_apple_layer.forward(apple_price, apple_num);
        let total_price = mul_tax_layer.forward(total_price_wo_tax, tax);
        assert!(approx_eq(total_price, 220.0, 1e-6));

        let dprice = 1.0;
        let (dtotal_price_wo_tax, dtax) = mul_tax_layer.backward(dprice);
        let (dapple_price, dapple_num) = mul_apple_layer.backward(dtotal_price_wo_tax);

        let epsilon = 1e-6;
        assert!(approx_eq(dapple_price, 2.2, epsilon));
        assert!(approx_eq(dapple_num, 110.0, epsilon));
        assert!(approx_eq(dtax, 200.0, epsilon));
    }

    #[test]
    fn add_layer_test() {
        let apple_price = 100.0;
        let apple_num = 2.0;
        let mut mul_apple_layer = MulLayer { x: 0.0, y: 0.0 };
        let apple_total_price = mul_apple_layer.forward(apple_price, apple_num);

        let orange_price = 150.0;
        let orange_num = 3.0;
        let mut mul_orange_layer = MulLayer { x: 0.0, y: 0.0 };
        let orange_total_price = mul_orange_layer.forward(orange_price, orange_num);

        let add_apple_orange_layer = AddLayer;
        let total_price_wo_tax =
            add_apple_orange_layer.forward(apple_total_price, orange_total_price);

        let tax = 1.1;
        let mut mul_tax_layer = MulLayer { x: 0.0, y: 0.0 };
        let total_price = mul_tax_layer.forward(total_price_wo_tax, tax);
        assert!(approx_eq(total_price, 715.0, 1e-6));

        let dprice = 1.0;
        let (dtotal_price_wo_tax, dtax) = mul_tax_layer.backward(dprice);
        let (dapple_total_price, dorange_total_price) =
            add_apple_orange_layer.backward(dtotal_price_wo_tax);
        let (dapple_price, dapple_num) = mul_apple_layer.backward(dapple_total_price);
        let (dorange_price, dorange_num) = mul_orange_layer.backward(dorange_total_price);
        let epsilon = 1e-6;
        assert!(approx_eq(dapple_price, 2.2, epsilon));
        assert!(approx_eq(dapple_num, 110.0, epsilon));
        assert!(approx_eq(dorange_price, 3.3, epsilon));
        assert!(approx_eq(dorange_num, 165.0, epsilon));
        assert!(approx_eq(dtax, 650.0, epsilon));
    }

    #[test]
    fn relu_layer_test() {
        let mut relu_layer = ReluLayer::new();
        let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]).unwrap();
        let out = relu_layer.forward(x.clone());
        let expected_out =
            Array2::from_shape_vec((2, 3), vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0]).unwrap();
        assert!(approx_eq_array(&out, &expected_out, 1e-6));

        let dout = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let dx = relu_layer.backward(dout);
        let expected_dx =
            Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        assert!(approx_eq_array(&dx, &expected_dx, 1e-6));
    }

    #[test]
    fn sigmoid_layer_test() {
        let mut sigmoid_layer = SigmoidLayer::new();
        let x = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, 2.0, -3.0, 4.0]).unwrap();
        let out = sigmoid_layer.forward(x.clone());
        let expected_out = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.26894142, 0.5, 0.73105858, 0.88079708, 0.04742587, 0.98201379,
            ],
        )
        .unwrap();
        assert!(approx_eq_array(&out, &expected_out, 1e-6));
    }

    #[test]
    fn affine_layer_test() {
        let w = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let b = Array2::from_shape_vec((1, 3), vec![0.5, 0.5, 0.5]).unwrap();
        let mut affine_layer = AffineLayer::new(w.clone(), b.clone());
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = affine_layer.forward(x.clone());
        let expected_out =
            Array2::from_shape_vec((2, 3), vec![1.4, 1.7, 2.0, 2.4, 3.1, 3.8]).unwrap();

        assert!(approx_eq_array(&out, &expected_out, 1e-6));

        let dout = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let dx = affine_layer.backward(dout.clone());
        assert_eq!(dx.dim(), x.dim());
        assert_eq!(affine_layer.dw.dim(), w.dim());
        assert_eq!(affine_layer.db.dim(), b.dim());
    }

    #[test]
    fn softmax_with_loss_layer_test() {
        // 普遍量の比較でテストを行う, loss は正の有限値。dx の各行の和はほぼ0。dxとxの形状は同じ。
        let mut softmax_with_loss_layer = SoftmaxWithLossLayer::new();
        let x = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.7, 0.3, 0.4, 0.3]).unwrap();
        let t = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
        let loss = softmax_with_loss_layer.forward(x.clone(), t.clone());
        assert!(loss > 0.0 && loss.is_finite());

        let dout = 1.0;
        let dx = softmax_with_loss_layer.backward(dout);
        assert_eq!(dx.dim(), x.dim());

        for i in 0..dx.nrows() {
            let row_sum: f32 = dx.row(i).sum();
            assert!(approx_eq(row_sum, 0.0, 1e-6));
        }
    }

    #[test]
    fn batch_norm_layer_test() {
        let mut batch_norm_layer = BatchNormLayer::new(3);
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = batch_norm_layer.forward(x.clone());
        assert_eq!(out.dim(), x.dim());

        // B = 0, G = 1 の場合、出力は平均0、分散1に正規化される
        let mean = Array2::from_shape_vec(
            (1, out.ncols()),
            out.mean_axis(ndarray::Axis(0)).unwrap().to_vec(),
        )
        .unwrap();
        let var = Array2::from_shape_vec(
            (1, out.ncols()),
            out.var_axis(ndarray::Axis(0), 0.0).to_vec(),
        )
        .unwrap();
        let epsilon = 1e-6;
        println!("mean: {:?}", mean);
        println!("var: {:?}", var);
        assert!(approx_eq_array(
            &mean,
            &Array2::zeros(mean.raw_dim()),
            epsilon
        ));
        assert!(approx_eq_array(&var, &Array2::ones(var.raw_dim()), epsilon));
    }

    #[test]
    fn gradient_check_batch_norm_layer() {
        // 数値微分チェック(非一様な c で sum(out⊙c)、dout=c、max_diff < 1e-2)
        let mut batch_norm_layer = BatchNormLayer::new(3);
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let c = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        batch_norm_layer.forward(x.clone());
        let dout = c.clone();
        let dx = batch_norm_layer.backward(dout);
        assert_eq!(dx.dim(), x.dim());

        // 数値微分チェック
        // dx_num[i,j] = (loss_plus − loss_minus) / (2h)     ← ±h の両側、割る 2h
        // max_diff = max| dx[i,j] − dx_num[i,j] |            ← backward の dx と比較!
        // assert(max_diff < 1e-2)
        let epsilon = 1e-4;
        let mut max_diff = 0.0;
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let mut x_plus = x.clone();
                x_plus[[i, j]] += epsilon;
                let out_plus = batch_norm_layer.forward(x_plus);
                let loss_plus = (out_plus.clone() * c.clone()).sum();

                let mut x_minus = x.clone();
                x_minus[[i, j]] -= epsilon;
                let out_minus = batch_norm_layer.forward(x_minus);
                let loss_minus = (out_minus.clone() * c.clone()).sum();

                let dx_num = (loss_plus - loss_minus) / (2.0 * epsilon);
                let diff = (dx[[i, j]] - dx_num).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        println!("max_diff: {}", max_diff);
        assert!(max_diff < 1e-2);
    }

    #[test]
    fn dropout_layer_test() {
        // 訓練 forward:出力の各要素は「0 か元の値」のどちらか
        // ratio=0.5、要素数多め(例 100×100)で生存率がざっくり半分(0.4〜0.6 くらいの緩さで)
        // テスト forward:全要素 x×(1−ratio) に一致
        // backward:殺された位置の dx = 0、生存位置は dout そのまま(forward の出力から死んだ位置を特定できます)
        let dropout_ratio = 0.5;
        let mut dropout_layer = DropoutLayer::new(dropout_ratio);
        let x =
            Array2::from_shape_vec((100, 100), (1..=10000).map(|v| v as f32).collect()).unwrap();

        // 訓練 forward
        let out = dropout_layer.forward(x.clone(), true);
        assert!(out.iter().zip(x.iter()).all(|(&o, &v)| o == 0.0 || o == v));

        let alive_count = out.iter().filter(|&&o| o != 0.0).count();
        let alive_ratio = alive_count as f32 / (100.0 * 100.0);
        println!("alive_ratio: {}", alive_ratio);
        assert!(alive_ratio > 0.4 && alive_ratio < 0.6);

        // テスト forward
        let out_test = dropout_layer.forward(x.clone(), false);
        assert!(
            out_test
                .iter()
                .zip(x.iter())
                .all(|(&o, &v)| o == v * (1.0 - dropout_ratio))
        );

        // backward
        let dout = Array2::ones((100, 100));
        let forward_out = dropout_layer.forward(x.clone(), true);
        let dx = dropout_layer.backward(dout);
        println!("dx: {:?}", dx);
        println!("forward_out: {:?}", forward_out);
        assert!(
            dx.iter()
                .zip(forward_out.iter())
                .all(|(&d, &o)| (o == 0.0 && d == 0.0) || (o != 0.0 && d == 1.0))
        );
    }
}
