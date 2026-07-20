use crate::conv::{ConvolutionLayer, PoolingLayer};
use crate::layers::{AffineLayer, ReluLayer, SoftmaxWithLossLayer};
use crate::optimizer::Optimizer;
use ndarray::{Array1, Array2, Array4};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

/// 本 7.5「CNNの実装」の conv_param(Python の辞書)に対応する設定構造体。
/// Default は本の既定値(フィルタ 30 本・5×5・stride 1・pad 0)
#[derive(Clone)]
pub struct ConvParams {
    pub filter_num: usize,
    pub filter_size: usize,
    pub stride: usize,
    pub pad: usize,
}
impl Default for ConvParams {
    fn default() -> Self {
        Self {
            filter_num: 30,
            filter_size: 5,
            stride: 1,
            pad: 0,
        } // 本 7.5 の既定値
    }
}

/// 本 7.5「CNNの実装」Conv→ReLU→Pool→Affine→ReLU→Affine→SoftmaxWithLoss の 7 層 CNN。
/// Pool 出力(4次元)は flatten して Affine へ。勾配は各層が所有し、backward 連鎖で埋める
pub struct SimpleConvNet {
    // Layers
    conv: ConvolutionLayer,
    relu1: ReluLayer,
    pool: PoolingLayer,
    affine1: AffineLayer,
    relu2: ReluLayer,
    affine2: AffineLayer,
    last_layer: SoftmaxWithLossLayer,

    // Parameters
    opt_w1: Box<dyn Optimizer>,
    opt_b1: Box<dyn Optimizer>,
    opt_w2: Box<dyn Optimizer>,
    opt_b2: Box<dyn Optimizer>,
    opt_w3: Box<dyn Optimizer>,
    opt_b3: Box<dyn Optimizer>,

    // internal information
    pool_output_shape: Option<(usize, usize, usize, usize)>, // (N, C, H, W)
}
impl SimpleConvNet {
    pub fn new(
        input_dim: (usize, usize, usize), // (チャンネル数、高さ、幅)
        conv_params: ConvParams,
        hidden_size: usize,
        output_size: usize,
        make_opt: impl Fn() -> Box<dyn Optimizer>,
        make_std: impl Fn(usize) -> f32,
    ) -> Self {
        let (input_channels, input_height, input_width) = input_dim;

        let conv_output_height =
            (input_height + 2 * conv_params.pad - conv_params.filter_size) / conv_params.stride + 1;
        let conv_output_width =
            (input_width + 2 * conv_params.pad - conv_params.filter_size) / conv_params.stride + 1;

        let pool_size = 2;
        let pool_output_height = conv_output_height / pool_size;
        let pool_output_width = conv_output_width / pool_size;
        let pool_output_size = conv_params.filter_num * pool_output_height * pool_output_width;

        let fan_in_conv = input_channels * conv_params.filter_size * conv_params.filter_size;
        let w1 = Array4::random(
            (
                conv_params.filter_num,
                input_channels,
                conv_params.filter_size,
                conv_params.filter_size,
            ),
            StandardNormal,
        ) * make_std(fan_in_conv);
        let b1 = Array1::zeros(conv_params.filter_num);

        let w2 = Array2::random((pool_output_size, hidden_size), StandardNormal)
            * make_std(pool_output_size);
        let b2 = Array2::zeros((1, hidden_size));

        let w3 = Array2::random((hidden_size, output_size), StandardNormal) * make_std(hidden_size);
        let b3 = Array2::zeros((1, output_size));

        let conv = ConvolutionLayer::new(w1, b1, conv_params.stride, conv_params.pad);
        let relu1 = ReluLayer::new();
        let pool = PoolingLayer::new(pool_size, pool_size, pool_size, 0);
        let affine1 = AffineLayer::new(w2, b2);
        let relu2 = ReluLayer::new();
        let affine2 = AffineLayer::new(w3, b3);
        let last_layer = SoftmaxWithLossLayer::new();

        Self {
            conv,
            relu1,
            pool,
            affine1,
            relu2,
            affine2,
            last_layer,

            opt_w1: make_opt(),
            opt_b1: make_opt(),
            opt_w2: make_opt(),
            opt_b2: make_opt(),
            opt_w3: make_opt(),
            opt_b3: make_opt(),

            pool_output_shape: None, // will be set during forward pass
        }
    }

    pub fn predict(&mut self, x: &Array4<f32>) -> Array2<f32> {
        let out1 = self.conv.forward(x);
        let out2 = self
            .relu1
            .forward(out1.into_dyn())
            .into_dimensionality()
            .unwrap();
        let out3 = self.pool.forward(&out2);
        let (n, c, h, w) = out3.dim();
        self.pool_output_shape = Some((n, c, h, w)); // store for backward pass
        let out3_reshaped = out3
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((n, c * h * w))
            .unwrap();
        let out4 = self.affine1.forward(out3_reshaped);
        let out5 = self
            .relu2
            .forward(out4.into_dyn())
            .into_dimensionality()
            .unwrap();
        let out6 = self.affine2.forward(out5);

        out6
    }

    pub fn loss(&mut self, x: &Array4<f32>, t: Array2<f32>) -> f32 {
        let y = self.predict(x);
        self.last_layer.forward(y, t)
    }

    pub fn gradient(&mut self, x: &Array4<f32>, t: Array2<f32>) {
        // 順伝播
        self.loss(x, t);

        // 逆伝播
        let dout = self.last_layer.backward(1.0);
        let dout = self.affine2.backward(dout);
        let dout = self
            .relu2
            .backward(dout.into_dyn())
            .into_dimensionality()
            .unwrap();
        let dout = self.affine1.backward(dout);

        let (batch_size, c, pool_h, pool_w) = self.pool_output_shape.unwrap();
        let dout_4d = dout
            .into_shape_with_order((batch_size, c, pool_h, pool_w))
            .unwrap();

        let dout = self.pool.backward(&dout_4d);
        let dout = self
            .relu1
            .backward(dout.into_dyn())
            .into_dimensionality()
            .unwrap();
        let _dout = self.conv.backward(&dout);
    }

    /// 本 7.5 + 6.1: gradient() が各層に保存した dW,dB を、パラメータごとの
    /// Optimizer で W,B に適用する。次元の異なる Array4/Array2/Array1 を
    /// into_dyn() のビューで同じ trait に渡す(Optimizer は ArrayViewMutD を取る)
    pub fn update(&mut self) {
        // Update conv layer parameters
        let (w1, dw1) = self.conv.w_and_dw();
        self.opt_w1
            .update(&mut w1.view_mut().into_dyn(), &dw1.view().into_dyn());
        let (b1, db1) = self.conv.b_and_db();
        self.opt_b1
            .update(&mut b1.view_mut().into_dyn(), &db1.view().into_dyn());

        // Update affine1 layer parameters
        let (w2, dw2) = self.affine1.w_and_dw();
        self.opt_w2
            .update(&mut w2.view_mut().into_dyn(), &dw2.view().into_dyn());
        let (b2, db2) = self.affine1.b_and_db();
        self.opt_b2
            .update(&mut b2.view_mut().into_dyn(), &db2.view().into_dyn());

        // Update affine2 layer parameters
        let (w3, dw3) = self.affine2.w_and_dw();
        self.opt_w3
            .update(&mut w3.view_mut().into_dyn(), &dw3.view().into_dyn());
        let (b3, db3) = self.affine2.b_and_db();
        self.opt_b3
            .update(&mut b3.view_mut().into_dyn(), &db3.view().into_dyn());
    }

    /// 本 7.5: 4.5 の accuracy の CNN 版。行ごとの argmax を正解と突き合わせる
    pub fn accuracy(&mut self, x: &Array4<f32>, t: Array2<f32>) -> f32 {
        let y = self.predict(x);
        let y_max_indices = y.map_axis(ndarray::Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        });
        let t_max_indices = t.map_axis(ndarray::Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        });

        let correct_count = y_max_indices
            .iter()
            .zip(t_max_indices.iter())
            .filter(|(y_idx, t_idx)| y_idx == t_idx)
            .count();
        correct_count as f32 / x.shape()[0] as f32
    }

    /// 本 7.6.1 の可視化用: 1 層目 conv のフィルタ (FN,C,FH,FW) を覗く
    pub fn conv_weights(&self) -> &Array4<f32> {
        self.conv.w()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Dimension;

    // ヘルパー: テンソル間の最大誤差を計算
    fn max_abs_diff<D: Dimension>(a: &ndarray::Array<f32, D>, b: &ndarray::Array<f32, D>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn max_rel_diff<D: Dimension>(a: &ndarray::Array<f32, D>, b: &ndarray::Array<f32, D>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs() / f32::max(x.abs(), y.abs()).max(1.0)) // avoid division by zero
            .fold(0.0, f32::max)
    }

    #[test]
    fn test_predict() {
        let input_dim = (1, 28, 28);
        let conv_params = ConvParams::default();
        let hidden_size = 100;
        let output_size = 10;

        let make_opt = || Box::new(crate::optimizer::SGD::new(0.01)) as Box<dyn Optimizer>;
        let make_std = |fan_in: usize| (2.0 / fan_in as f32).sqrt();

        let mut net = SimpleConvNet::new(
            input_dim,
            conv_params,
            hidden_size,
            output_size,
            make_opt,
            make_std,
        );

        let x = Array4::<f32>::zeros((2, 1, 28, 28));
        let y_pred = net.predict(&x);

        assert_eq!(y_pred.dim(), (2, output_size));
    }

    #[test]
    fn test_loss() {
        let mut net = SimpleConvNet::new(
            (1, 28, 28),
            ConvParams::default(),
            100,
            10,
            || Box::new(crate::optimizer::SGD::new(0.01)) as Box<dyn Optimizer>,
            |fan_in| (2.0 / fan_in as f32).sqrt(),
        );
        let x = Array4::zeros((2, 1, 28, 28));
        let mut t = Array2::zeros((2, 10));
        t[[0, 3]] = 1.0; // one-hot label for first sample
        t[[1, 7]] = 1.0; // one-hot label for second sample

        let loss = net.loss(&x, t);
        assert!(loss.is_finite());
        let expected_loss = (10_f32).ln();
        assert!((loss - expected_loss).abs() < 1e-4);
    }

    #[test]
    fn test_gradient() {
        let mut net = SimpleConvNet::new(
            (1, 28, 28),
            ConvParams::default(),
            100,
            10,
            || Box::new(crate::optimizer::SGD::new(0.01)) as Box<dyn Optimizer>,
            |fan_in| (2.0 / fan_in as f32).sqrt(),
        );
        let x = Array4::zeros((2, 1, 28, 28));
        let mut t = Array2::zeros((2, 10));
        t[[0, 3]] = 1.0; // one-hot label for first sample
        t[[1, 7]] = 1.0; // one-hot label for second sample

        net.gradient(&x, t); // dw, db を計算・保持させる

        // Conv層
        assert_eq!(net.conv.w().dim(), net.conv.dw().dim());
        assert_eq!(net.conv.b().dim(), net.conv.db().dim());
        // Affine1層
        assert_eq!(net.affine1.w().dim(), net.affine1.dw().dim());
        assert_eq!(net.affine1.b().dim(), net.affine1.db().dim());
        // Affine2層
        assert_eq!(net.affine2.w().dim(), net.affine2.dw().dim());
        assert_eq!(net.affine2.b().dim(), net.affine2.db().dim());
    }

    use crate::gradient::numerical_gradient_over;
    use ndarray_rand::rand_distr::Uniform;

    fn run_gradient_check() -> Result<(), String> {
        // --- 1. ミニ・ネットワークの設定 ---
        let input_dim = (1, 4, 4);
        let conv_params = ConvParams {
            filter_num: 2,
            filter_size: 2,
            stride: 1,
            pad: 0,
        };
        let hidden_size = 10;
        let output_size = 2;

        let make_opt = || Box::new(crate::optimizer::SGD::new(0.01)) as Box<dyn Optimizer>;
        // 勾配が消えないように初期値のスケールを意図的に 1.0 に大きくする
        let make_std = |n: usize| (1.0 / n as f32).sqrt();

        let mut net = SimpleConvNet::new(
            input_dim,
            conv_params,
            hidden_size,
            output_size,
            make_opt,
            make_std,
        );

        // 乱数入力とターゲット
        let x = Array4::random((2, 1, 4, 4), Uniform::new(-1.0, 1.0).unwrap());
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 0.0]).unwrap();

        // --- 2. 誤差逆伝播法による勾配計算 ---
        net.gradient(&x, t.clone());
        if !net.affine2.db().iter().any(|&val| val.abs() > 1e-4) {
            return Err("Dead gradient detected: affine2_b (db) is completely zero!".into());
        }
        if !net.conv.dw().iter().any(|&val| val.abs() > 1e-4) {
            return Err("Dead gradient detected: conv_w (dw) is completely zero!".into());
        }

        let epsilon = 1e-2;

        // --- 3. 数値微分と差し替えマクロ ---
        macro_rules! check_grad {
            ($layer:ident, $accessor:ident, $backward_grad:expr, $name:expr) => {
                // net の層のパラメータをクローン
                let mut param_clone = net.$layer.$accessor().clone();

                let num_grad = numerical_gradient_over(&mut param_clone, |modified_param| {
                    // net 内部のパラメータを std::mem::replace で一時的に差し替え
                    let original =
                        std::mem::replace(net.$layer.$accessor(), modified_param.clone());
                    // 差し替えた状態で Loss を計算
                    let l = net.loss(&x, t.clone());
                    // パラメータを元に戻す
                    *net.$layer.$accessor() = original;
                    l
                });

                let abs_diff = max_abs_diff(&num_grad, $backward_grad);
                println!("{}: max_diff = {}", $name, abs_diff);
                let rel_diff = max_rel_diff(&num_grad, $backward_grad);
                println!("{}: max_rel_diff = {}", $name, rel_diff);

                if rel_diff >= epsilon {
                    return Err(format!(
                        "Gradient check failed for {}: max_rel_diff = {}",
                        $name, rel_diff
                    ));
                }
            };
        }

        // --- 4. 全パラメータのチェック実行 ---
        check_grad!(conv, w_mut, net.conv.dw(), "conv_w");
        check_grad!(conv, b_mut, net.conv.db(), "conv_b");

        check_grad!(affine1, w_mut, net.affine1.dw(), "affine1_w");
        check_grad!(affine1, b_mut, net.affine1.db(), "affine1_b");

        check_grad!(affine2, w_mut, net.affine2.dw(), "affine2_w");
        check_grad!(affine2, b_mut, net.affine2.db(), "affine2_b");

        Ok(())
    }

    #[test]
    fn test_gradient_check_all() {
        let max_retries = 3;

        for i in 0..max_retries {
            println!("--- Attempt {}/{} ---", i + 1, max_retries);
            match run_gradient_check() {
                Ok(_) => {
                    println!("Gradient check passed!");
                    return; // 1回でも成功すればテスト合格
                }
                Err(e) => {
                    println!("Attempt failed: {}", e);
                }
            }
        }

        // 全滅した場合のみ panic
        panic!("Gradient check failed after {} attempts.", max_retries);
    }

    #[test]
    fn test_update() {
        let input_dim = (1, 4, 4);
        let conv_params = ConvParams {
            filter_num: 2,
            filter_size: 2,
            stride: 1,
            pad: 0,
        };
        let hidden_size = 10;
        let output_size = 2;

        let lr = 0.1;
        // 学習率 0.1 の SGD を使う
        let make_opt = || Box::new(crate::optimizer::SGD::new(lr)) as Box<dyn Optimizer>;
        let make_std = |n: usize| (1.0 / n as f32).sqrt();

        let mut net = SimpleConvNet::new(
            input_dim,
            conv_params.clone(),
            hidden_size,
            output_size,
            make_opt,
            make_std,
        );

        // 適当な入力データと正解ラベル
        // 乱数入力とターゲット
        let x = Array4::random((2, 1, 4, 4), Uniform::new(-1.0, 1.0).unwrap());
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 0.0]).unwrap();

        // 1. 順伝播・逆伝播を実行し、各層の dw, db を計算・保持させる
        net.gradient(&x, t);
        assert!(
            net.affine2.dw().iter().any(|&val| val.abs() > 1e-4),
            "Dead gradient detected: affine2_w (dw) is completely zero!"
        );
        assert!(
            net.conv.db().iter().any(|&val| val.abs() > 1e-4),
            "Dead gradient detected: conv_b (db) is completely zero!"
        );

        // 2. update() 実行「前」の重みと勾配をクローンして退避しておく
        let conv_w_before = net.conv.w().clone();
        let conv_dw = net.conv.dw().clone();
        let conv_b_before = net.conv.b().clone();
        let conv_db = net.conv.db().clone();
        let affine1_w_before = net.affine1.w().clone();
        let affine1_dw = net.affine1.dw().clone();
        let affine1_b_before = net.affine1.b().clone();
        let affine1_db = net.affine1.db().clone();
        let affine2_w_before = net.affine2.w().clone();
        let affine2_dw = net.affine2.dw().clone();
        let affine2_b_before = net.affine2.b().clone();
        let affine2_db = net.affine2.db().clone();

        // 3. 更新の実行（※これから実装するメソッドです！）
        net.update();

        // 4. SGD の数式 (W_new = W_old - lr * dW) 通りに更新されたか検証
        // w
        let expected_conv_w = &conv_w_before - &(conv_dw * lr);
        assert!(max_abs_diff(net.conv.w(), &expected_conv_w) < 1e-6);
        let expected_affine1_w = &affine1_w_before - &(affine1_dw * lr);
        assert!(max_abs_diff(net.affine1.w(), &expected_affine1_w) < 1e-6);
        let expected_affine2_w = &affine2_w_before - &(affine2_dw * lr);
        assert!(max_abs_diff(net.affine2.w(), &expected_affine2_w) < 1e-6);
        // b
        let expected_conv_b = &conv_b_before - &(conv_db * lr);
        assert!(max_abs_diff(net.conv.b(), &expected_conv_b) < 1e-6);
        let expected_affine1_b = &affine1_b_before - &(affine1_db * lr);
        assert!(max_abs_diff(net.affine1.b(), &expected_affine1_b) < 1e-6);
        let expected_affine2_b = &affine2_b_before - &(affine2_db * lr);
        assert!(max_abs_diff(net.affine2.b(), &expected_affine2_b) < 1e-6);
    }

    use crate::mnist::{load_images, load_labels, to_one_hot};
    use crate::optimizer::Adam;
    use ndarray::Axis;
    #[test]
    #[ignore] // 実行に時間がかかるので CI では無視
    fn train_mnist_backprop_cnn() {
        let images = load_images("dataset/train-images-idx3-ubyte"); // (60000, 784)
        let labels = load_labels("dataset/train-labels-idx1-ubyte");
        let test_images = load_images("dataset/t10k-images-idx3-ubyte");
        let test_labels = load_labels("dataset/t10k-labels-idx1-ubyte");

        let input_dim = (1, 28, 28);
        let hidden_size = 100;
        let output_size = 10;
        let conv_params = ConvParams {
            // From book
            filter_num: 30,
            filter_size: 5,
            stride: 1,
            pad: 0,
        };
        // 学習率 0.1 の SGD を使う
        // let lr = 0.1;
        // let make_opt = || Box::new(crate::optimizer::SGD::new(lr)) as Box<dyn Optimizer>;
        let make_opt = || Box::new(Adam::new(0.001)) as Box<dyn Optimizer>; // Adam で学習率 0.001
        let make_std = |n: usize| (1.0 / n as f32).sqrt();
        let mut net = SimpleConvNet::new(
            input_dim,
            conv_params.clone(),
            hidden_size,
            output_size,
            make_opt,
            make_std,
        );

        let mut rng = rand::rng();
        let batch_size = 100;
        let max_epochs = 20;
        let train_size = images.shape()[0];
        let iter_per_epoch = usize::max(train_size / batch_size, 1);
        let iters_num = max_epochs * iter_per_epoch;

        for i in 0..iters_num {
            let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();

            let x_batch = images.select(Axis(0), &idx);
            let x_batch_reshaped = x_batch
                .into_shape_with_order((batch_size, 1, 28, 28))
                .unwrap();
            let batch_labels: Vec<u8> = idx.iter().map(|&j| labels[j]).collect();
            let t_batch = to_one_hot(&batch_labels, output_size);

            net.gradient(&x_batch_reshaped, t_batch);
            net.update();

            if i % iter_per_epoch == 0 {
                let epoch = i / iter_per_epoch;

                let test_idx =
                    rand::seq::index::sample(&mut rng, test_images.shape()[0], 1000).into_vec();
                let test_x_batch = test_images.select(Axis(0), &test_idx);
                let test_x_batch_reshaped = test_x_batch
                    .into_shape_with_order((1000, 1, 28, 28))
                    .unwrap();
                let test_t_batch = to_one_hot(
                    &test_idx
                        .iter()
                        .map(|&j| test_labels[j])
                        .collect::<Vec<u8>>(),
                    output_size,
                );

                let train_acc =
                    net.accuracy(&x_batch_reshaped, to_one_hot(&batch_labels, output_size));
                let test_acc = net.accuracy(&test_x_batch_reshaped, test_t_batch);

                println!("=== Epoch {} ===", epoch);
                println!(
                    "Train Accuracy: {:.4}, Test Accuracy: {:.4}",
                    train_acc, test_acc
                );
            }
        }

        let mut test_correct = 0;
        let eval_batch_size = 100;

        for i in (0..10000).step_by(eval_batch_size) {
            // 100件ずつ切り出して4次元に変形
            let x_batch = test_images
                .slice(ndarray::s![i..i + eval_batch_size, ..])
                .to_owned();
            let x_4d = x_batch
                .into_shape_with_order((eval_batch_size, 1, 28, 28))
                .unwrap();
            let t_batch = to_one_hot(
                &test_labels
                    .slice(ndarray::s![i..i + eval_batch_size])
                    .to_vec(),
                10,
            );

            // 現在の accuracy は「正解率 (0.0 ~ 1.0)」を返す仕様なので、枚数を掛けて正解「数」に戻して加算する
            let acc = net.accuracy(&x_4d, t_batch);
            test_correct += (acc * eval_batch_size as f32).round() as usize;
        }

        let total_test_acc = test_correct as f32 / 10000.0;
        println!("Total Test Accuracy: {:.4}", total_test_acc);
    }
}
