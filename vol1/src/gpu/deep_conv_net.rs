use super::layers::{GpuAffineLayer, GpuConvLayer, GpuPoolingLayer, GpuReluLayer};
use super::{Gpu, GpuImage, GpuTensor};
use ndarray::{Array1, Array2, Array4};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

pub struct GpuDeepConvNetParams {
    pub w1_1: Array4<f32>,
    pub b1_1: Array1<f32>,
    pub w1_2: Array4<f32>,
    pub b1_2: Array1<f32>,

    pub w2_1: Array4<f32>,
    pub b2_1: Array1<f32>,
    pub w2_2: Array4<f32>,
    pub b2_2: Array1<f32>,

    pub w3_1: Array4<f32>,
    pub b3_1: Array1<f32>,
    pub w3_2: Array4<f32>,
    pub b3_2: Array1<f32>,

    pub wa1: Array2<f32>,
    pub ba1: Array2<f32>,
    pub wa2: Array2<f32>,
    pub ba2: Array2<f32>,
}
impl GpuDeepConvNetParams {
    pub fn random() -> Self {
        let he_conv = |fn_: usize, c: usize, fh: usize, fw: usize| -> Array4<f32> {
            let fan_in = (c * fh * fw) as f32;
            let scale = (2.0 / fan_in).sqrt();
            Array4::random((fn_, c, fh, fw), StandardNormal) * scale
        };
        let he_affine = |fan_in: usize, fan_out: usize| -> Array2<f32> {
            let scale = (2.0 / fan_in as f32).sqrt();
            Array2::random((fan_in, fan_out), StandardNormal) * scale
        };

        Self {
            w1_1: he_conv(16, 1, 3, 3),
            b1_1: Array1::zeros(16),
            w1_2: he_conv(16, 16, 3, 3),
            b1_2: Array1::zeros(16),

            w2_1: he_conv(32, 16, 3, 3),
            b2_1: Array1::zeros(32),
            w2_2: he_conv(32, 32, 3, 3),
            b2_2: Array1::zeros(32),

            w3_1: he_conv(64, 32, 3, 3),
            b3_1: Array1::zeros(64),
            w3_2: he_conv(64, 64, 3, 3),
            b3_2: Array1::zeros(64),

            wa1: he_affine(64 * 4 * 4, 50),
            ba1: Array2::zeros((1, 50)),
            wa2: he_affine(50, 10),
            ba2: Array2::zeros((1, 10)),
        }
    }
}

pub struct GpuDeepConvNet {
    c1_1: GpuConvLayer,
    r1_1: GpuReluLayer,
    c1_2: GpuConvLayer,
    r1_2: GpuReluLayer,
    p1: GpuPoolingLayer,

    c2_1: GpuConvLayer,
    r2_1: GpuReluLayer,
    c2_2: GpuConvLayer,
    r2_2: GpuReluLayer,
    p2: GpuPoolingLayer,

    c3_1: GpuConvLayer,
    r3_1: GpuReluLayer,
    c3_2: GpuConvLayer,
    r3_2: GpuReluLayer,
    p3: GpuPoolingLayer,

    af1: GpuAffineLayer,
    ra1: GpuReluLayer,
    af2: GpuAffineLayer,

    flatten_dims: Option<(usize, usize, usize, usize)>, // 入力画像の形状 (batch, channel, height, width)
}
impl GpuDeepConvNet {
    pub fn new(gpu: &Gpu) -> Self {
        let params = GpuDeepConvNetParams::random();
        Self::new_with_params(gpu, &params)
    }

    pub fn new_with_params(gpu: &Gpu, params: &GpuDeepConvNetParams) -> Self {
        Self {
            c1_1: GpuConvLayer::new(gpu, &params.w1_1, &params.b1_1, 1, 1),
            r1_1: GpuReluLayer::new(),
            c1_2: GpuConvLayer::new(gpu, &params.w1_2, &params.b1_2, 1, 1),
            r1_2: GpuReluLayer::new(),
            p1: GpuPoolingLayer::new(2, 2, 2, 0),

            c2_1: GpuConvLayer::new(gpu, &params.w2_1, &params.b2_1, 1, 1),
            r2_1: GpuReluLayer::new(),
            c2_2: GpuConvLayer::new(gpu, &params.w2_2, &params.b2_2, 1, 2), // pad=2
            r2_2: GpuReluLayer::new(),
            p2: GpuPoolingLayer::new(2, 2, 2, 0),

            c3_1: GpuConvLayer::new(gpu, &params.w3_1, &params.b3_1, 1, 1),
            r3_1: GpuReluLayer::new(),
            c3_2: GpuConvLayer::new(gpu, &params.w3_2, &params.b3_2, 1, 1),
            r3_2: GpuReluLayer::new(),
            p3: GpuPoolingLayer::new(2, 2, 2, 0),

            af1: GpuAffineLayer::new(gpu, &params.wa1, &params.ba1),
            ra1: GpuReluLayer::new(),
            af2: GpuAffineLayer::new(gpu, &params.wa2, &params.ba2),

            flatten_dims: None,
        }
    }

    pub fn forward(&mut self, gpu: &Gpu, x: &GpuImage) -> GpuTensor {
        let mut gout = self.c1_1.forward(gpu, x);
        gout = self.r1_1.forward_image(gpu, gout);
        gout = self.c1_2.forward(gpu, &gout);
        gout = self.r1_2.forward_image(gpu, gout);
        gout = self.p1.forward(gpu, &gout);

        gout = self.c2_1.forward(gpu, &gout);
        gout = self.r2_1.forward_image(gpu, gout);
        gout = self.c2_2.forward(gpu, &gout);
        gout = self.r2_2.forward_image(gpu, gout);
        gout = self.p2.forward(gpu, &gout);

        gout = self.c3_1.forward(gpu, &gout);
        gout = self.r3_1.forward_image(gpu, gout);
        gout = self.c3_2.forward(gpu, &gout);
        gout = self.r3_2.forward_image(gpu, gout);
        gout = self.p3.forward(gpu, &gout);

        // Flatten: ノーコストの型変換
        self.flatten_dims = Some(gout.dims);
        let mut gout_tensor = gout.tensor;

        gout_tensor = self.af1.forward(gpu, &gout_tensor);
        gout_tensor = self.ra1.forward_tensor(gpu, gout_tensor);
        self.af2.forward(gpu, &gout_tensor)
    }

    pub fn backward(&mut self, gpu: &Gpu, dout: &GpuTensor) -> GpuImage {
        let mut gd = self.af2.backward(gpu, dout);
        gd = self.ra1.backward_tensor(gpu, gd);
        gd = self.af1.backward(gpu, &gd);

        // Unflatten: 次元情報を与えてノーコストで Image に戻す

        let mut gd_img = GpuImage {
            tensor: gd,
            dims: self
                .flatten_dims
                .expect("forward must be called before backward"),
        };

        gd_img = self.p3.backward(gpu, &gd_img);
        gd_img = self.r3_2.backward_image(gpu, gd_img);
        gd_img = self.c3_2.backward(gpu, &gd_img);
        gd_img = self.r3_1.backward_image(gpu, gd_img);
        gd_img = self.c3_1.backward(gpu, &gd_img);

        gd_img = self.p2.backward(gpu, &gd_img);
        gd_img = self.r2_2.backward_image(gpu, gd_img);
        gd_img = self.c2_2.backward(gpu, &gd_img);
        gd_img = self.r2_1.backward_image(gpu, gd_img);
        gd_img = self.c2_1.backward(gpu, &gd_img);

        gd_img = self.p1.backward(gpu, &gd_img);
        gd_img = self.r1_2.backward_image(gpu, gd_img);
        gd_img = self.c1_2.backward(gpu, &gd_img);
        gd_img = self.r1_1.backward_image(gpu, gd_img);
        self.c1_1.backward(gpu, &gd_img)
    }

    pub fn update(&mut self, gpu: &Gpu, lr: f32) {
        // 重みを持つ層だけ update を呼び出す
        self.c1_1.update(gpu, lr);
        self.c1_2.update(gpu, lr);
        self.c2_1.update(gpu, lr);
        self.c2_2.update(gpu, lr);
        self.c3_1.update(gpu, lr);
        self.c3_2.update(gpu, lr);
        self.af1.update(gpu, lr);
        self.af2.update(gpu, lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::{ConvolutionLayer, PoolingLayer};
    use crate::layers::{AffineLayer, FlattenLayer, Layer, ReluLayer, SoftmaxWithLossLayer};
    use crate::mnist::{load_images, load_labels, to_one_hot};
    use ndarray::{Axis, Ix4};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal;
    use std::time::Instant;

    #[test]
    fn test_gpu_deep_conv_net() {
        let gpu = Gpu::new();
        let batch = 2;
        let x: Array4<f32> = Array4::random((batch, 1, 28, 28), StandardNormal);

        // GpuDeepConvNetParams に重みをパック
        let params = GpuDeepConvNetParams::random();

        // 2. 両者に注入
        // --- GPU 版 (カプセル化されたネットワーク) ---
        let mut gpu_net = GpuDeepConvNet::new_with_params(&gpu, &params);

        // --- CPU 版 (個別の層として手動組み立て。勾配直接検査のため) ---
        let mut c1_1 = ConvolutionLayer::new(params.w1_1.clone(), params.b1_1.clone(), 1, 1);
        let mut r1_1 = ReluLayer::new();
        let mut c1_2 = ConvolutionLayer::new(params.w1_2.clone(), params.b1_2.clone(), 1, 1);
        let mut r1_2 = ReluLayer::new();
        let mut p1 = PoolingLayer::new(2, 2, 2, 0);

        let mut c2_1 = ConvolutionLayer::new(params.w2_1.clone(), params.b2_1.clone(), 1, 1);
        let mut r2_1 = ReluLayer::new();
        let mut c2_2 = ConvolutionLayer::new(params.w2_2.clone(), params.b2_2.clone(), 1, 2); // pad=2
        let mut r2_2 = ReluLayer::new();
        let mut p2 = PoolingLayer::new(2, 2, 2, 0);

        let mut c3_1 = ConvolutionLayer::new(params.w3_1.clone(), params.b3_1.clone(), 1, 1);
        let mut r3_1 = ReluLayer::new();
        let mut c3_2 = ConvolutionLayer::new(params.w3_2.clone(), params.b3_2.clone(), 1, 1);
        let mut r3_2 = ReluLayer::new();
        let mut p3 = PoolingLayer::new(2, 2, 2, 0);

        let mut flat = FlattenLayer::new();
        let mut af1 = AffineLayer::new(params.wa1.clone(), params.ba1.clone());
        let mut ra1 = ReluLayer::new();
        let mut af2 = AffineLayer::new(params.wa2.clone(), params.ba2.clone());

        // 3. forward logit 比較
        // GPU
        let gx = gpu.upload_image(&x);
        let gout = gpu_net.forward(&gpu, &gx);
        let gpu_logit = gpu.download(&gout);

        // CPU
        let mut out = x.clone().into_dyn();
        out = Layer::forward(&mut c1_1, out, false);
        out = Layer::forward(&mut r1_1, out, false);
        out = Layer::forward(&mut c1_2, out, false);
        out = Layer::forward(&mut r1_2, out, false);
        out = Layer::forward(&mut p1, out, false);

        out = Layer::forward(&mut c2_1, out, false);
        out = Layer::forward(&mut r2_1, out, false);
        out = Layer::forward(&mut c2_2, out, false);
        out = Layer::forward(&mut r2_2, out, false);
        out = Layer::forward(&mut p2, out, false);

        out = Layer::forward(&mut c3_1, out, false);
        out = Layer::forward(&mut r3_1, out, false);
        out = Layer::forward(&mut c3_2, out, false);
        out = Layer::forward(&mut r3_2, out, false);
        out = Layer::forward(&mut p3, out, false);

        out = Layer::forward(&mut flat, out, false);
        out = Layer::forward(&mut af1, out, false);
        out = Layer::forward(&mut ra1, out, false);
        let cpu_logit = Layer::forward(&mut af2, out, false);

        let eps = 1e-2;
        let diff_logit = cpu_logit
            .iter()
            .zip(gpu_logit.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        println!("DeepConvNet logit max_diff: {diff_logit:e}");
        assert!(diff_logit < eps, "logit diff: {diff_logit:e}");

        // 4. 合成 dout → backward dx 比較
        let dout = Array2::random((batch, 10), StandardNormal);
        let gdout = gpu.upload(&dout);

        // GPU
        let gdx_img = gpu_net.backward(&gpu, &gdout);
        let gpu_dx = gpu
            .download(&gdx_img.tensor)
            .into_shape_with_order(gdx_img.dims)
            .unwrap();

        // CPU
        let mut d = dout.clone().into_dyn();
        d = Layer::backward(&mut af2, d);
        d = Layer::backward(&mut ra1, d);
        d = Layer::backward(&mut af1, d);
        d = Layer::backward(&mut flat, d);

        d = Layer::backward(&mut p3, d);
        d = Layer::backward(&mut r3_2, d);
        d = Layer::backward(&mut c3_2, d);
        d = Layer::backward(&mut r3_1, d);
        d = Layer::backward(&mut c3_1, d);

        d = Layer::backward(&mut p2, d);
        d = Layer::backward(&mut r2_2, d);
        d = Layer::backward(&mut c2_2, d);
        d = Layer::backward(&mut r2_1, d);
        d = Layer::backward(&mut c2_1, d);

        d = Layer::backward(&mut p1, d);
        d = Layer::backward(&mut r1_2, d);
        d = Layer::backward(&mut c1_2, d);
        d = Layer::backward(&mut r1_1, d);
        let cpu_dx = Layer::backward(&mut c1_1, d)
            .into_dimensionality::<Ix4>()
            .unwrap();

        let diff_dx = cpu_dx
            .iter()
            .zip(gpu_dx.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        println!("DeepConvNet dx max_diff: {diff_dx:e}");
        assert!(diff_dx < eps, "dx diff: {diff_dx:e}");

        // 5. 端の層の dW/db スポットチェック
        // c1_1
        let cpu_c1_1_dw = c1_1
            .dw()
            .clone()
            .into_shape_with_order((16, 1 * 3 * 3))
            .unwrap()
            .t()
            .to_owned();
        let cpu_c1_1_db = c1_1.db().clone().insert_axis(Axis(0));
        let gpu_c1_1_dw = gpu.download(gpu_net.c1_1.dw_colt());
        let gpu_c1_1_db = gpu.download(gpu_net.c1_1.db());

        let diff_c1_1_dw = cpu_c1_1_dw
            .iter()
            .zip(gpu_c1_1_dw.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        println!("DeepConvNet c1_1 dW max_diff: {diff_c1_1_dw:e}");
        assert!(diff_c1_1_dw < eps, "c1_1 dW diff: {diff_c1_1_dw:e}");

        let diff_c1_1_db = cpu_c1_1_db
            .iter()
            .zip(gpu_c1_1_db.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_c1_1_db < eps, "c1_1 db diff: {diff_c1_1_db:e}");

        // af2
        let cpu_af2_dw = af2.dw().clone();
        let cpu_af2_db = af2.db().clone();
        let gpu_af2_dw = gpu.download(gpu_net.af2.dw());
        let gpu_af2_db = gpu.download(gpu_net.af2.db());

        let diff_af2_dw = cpu_af2_dw
            .iter()
            .zip(gpu_af2_dw.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        println!("DeepConvNet af2 dW max_diff: {diff_af2_dw:e}");
        assert!(diff_af2_dw < eps, "af2 dW diff: {diff_af2_dw:e}");

        let diff_af2_db = cpu_af2_db
            .iter()
            .zip(gpu_af2_db.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_af2_db < eps, "af2 db diff: {diff_af2_db:e}");
    }

    #[test]
    #[ignore] // 実行: cargo test --release train_mnist_deep_gpu_smoke -- --ignored --nocapture
    fn train_mnist_deep_gpu_smoke() {
        println!("Loading MNIST dataset...");
        let x_train = load_images("dataset/train-images-idx3-ubyte")
            .into_shape_with_order((60000, 1, 28, 28))
            .unwrap();
        let t_train = load_labels("dataset/train-labels-idx1-ubyte");

        let gpu = Gpu::new();
        let mut net = GpuDeepConvNet::new(&gpu);
        let mut swl = SoftmaxWithLossLayer::new();

        let train_size = x_train.shape()[0];
        let batch_size = 100;
        let max_iters = 50;
        let lr = 0.01f32; // 今回は素の SGD

        let mut rng = rand::rng();
        let mut first_loss = 0.0;
        let mut last_loss = 0.0;

        println!("--- Training GpuDeepConvNet (Smoke Test / 50 iters) ---");

        for i in 1..=max_iters {
            // 1. バッチ抽出 (CPU)
            let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();
            let x_batch = x_train.select(Axis(0), &idx);
            let batch_labels: Vec<u8> = idx.iter().map(|&j| t_train[j]).collect();
            let t_batch = to_one_hot(&batch_labels, 10);

            // 2. GPU Forward (上り 300KB)
            let gx = gpu.upload_image(&x_batch);
            let gout = net.forward(&gpu, &gx);

            // 3. Logit ダウンロード & CPU Loss計算 (下り 4KB)
            let logit = gpu.download(&gout);
            let loss = swl.forward(logit, t_batch);

            if i == 1 {
                first_loss = loss;
            }
            last_loss = loss;
            if i % 10 == 0 || i == 1 {
                println!("  iter {:3} | loss: {:.4}", i, loss);
            }

            // 4. CPU Softmax Backward
            let dx_cpu = swl.backward(1.0);

            // 5. GPU Backward & Update (上り 4KB)
            let gdout = gpu.upload(&dx_cpu);
            let _ = net.backward(&gpu, &gdout); // dx は捨てる
            net.update(&gpu, lr);
        }

        // Loss が開始時より確実に下がっていることをアサート
        assert!(
            last_loss < first_loss,
            "Loss should decrease. first: {:.4}, last: {:.4}",
            first_loss,
            last_loss
        );
        println!("Smoke test passed! Loss successfully decreased.");
    }

    #[test]
    #[ignore] // 実行: cargo test --release train_mnist_deep_gpu_1epoch -- --ignored --nocapture
    fn train_mnist_deep_gpu_1epoch() {
        println!("Loading MNIST dataset...");
        let x_train = load_images("dataset/train-images-idx3-ubyte")
            .into_shape_with_order((60000, 1, 28, 28))
            .unwrap();
        let t_train = load_labels("dataset/train-labels-idx1-ubyte");

        let x_test = load_images("dataset/t10k-images-idx3-ubyte")
            .into_shape_with_order((10000, 1, 28, 28))
            .unwrap();
        let t_test = load_labels("dataset/t10k-labels-idx1-ubyte");

        let gpu = Gpu::new();
        let mut net = GpuDeepConvNet::new(&gpu);
        let mut swl = SoftmaxWithLossLayer::new();

        let train_size = x_train.shape()[0];
        let batch_size = 100;
        let max_iters = 600; // 1 epoch
        let lr = 0.01f32;

        let mut rng = rand::rng();

        println!("--- Training GpuDeepConvNet (1 Epoch) ---");
        // ロードと初期化を終えたここから計測開始
        let loop_start_time = Instant::now();
        let mut iter_start_time = Instant::now();

        let mut first_losses = 0.0;
        let mut last_losses = 0.0;

        for i in 1..=max_iters {
            // CPU: バッチインデックス抽出 & データ切り出し
            let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();
            let x_batch = x_train.select(Axis(0), &idx);
            let batch_labels: Vec<u8> = idx.iter().map(|&j| t_train[j]).collect();
            let t_batch = to_one_hot(&batch_labels, 10);

            // GPU: 順伝播
            let gx = gpu.upload_image(&x_batch);
            let gout = net.forward(&gpu, &gx);

            // GPU->CPU: logit のみダウンロードし CPU で Loss 算出
            let logit = gpu.download(&gout);
            let loss = swl.forward(logit, t_batch);

            // 偽赤を防ぐ移動平均用の記録
            if i <= 5 {
                first_losses += loss;
            }
            if i > max_iters - 5 {
                last_losses += loss;
            }

            if i % 100 == 0 || i == 1 {
                let elapsed = iter_start_time.elapsed().as_secs_f32();
                println!(
                    "  iter {:3} | loss: {:.4} | Time(100iters): {:.2}s",
                    i, loss, elapsed
                );
                iter_start_time = Instant::now();
            }

            // CPU: 逆伝播 (SoftmaxWithLoss)
            let dx_cpu = swl.backward(1.0);

            // CPU->GPU: dout アップロード, 逆伝播, 更新
            let gdout = gpu.upload(&dx_cpu);
            let _ = net.backward(&gpu, &gdout);
            net.update(&gpu, lr);
        }

        // --- 時間解剖結果の出力 ---
        let total_loop_time = loop_start_time.elapsed().as_secs_f32();
        let s_per_iter = total_loop_time / max_iters as f32;
        println!("\n=== Timing Results ===");
        println!("1 Epoch Total Time: {:.2}s", total_loop_time);
        println!(
            "Seconds per iteration: {:.4}s (Target to beat: CPU ~0.41s)",
            s_per_iter
        );

        // 論点1の修正版 Assert（5回平均で比較）
        let avg_first = first_losses / 5.0;
        let avg_last = last_losses / 5.0;
        assert!(
            avg_last < avg_first,
            "Loss didn't decrease reliably: start_avg {:.4} -> end_avg {:.4}",
            avg_first,
            avg_last
        );

        // --- 精度測定 (先頭 1000 枚) ---
        println!("\nEvaluating accuracy on 1000 test images...");
        let test_subset_size = 1000;
        let eval_batch_size = 100; // メモリ溢れ防止のため分割
        let mut correct = 0;

        for i in (0..test_subset_size).step_by(eval_batch_size) {
            let end = (i + eval_batch_size).min(test_subset_size);
            let x_batch = x_test.slice(ndarray::s![i..end, .., .., ..]).to_owned();

            let gx = gpu.upload_image(&x_batch);
            let gout = net.forward(&gpu, &gx);
            let logit = gpu.download(&gout);

            // CPU argmax 比較
            for j in 0..(end - i) {
                let mut max_val = f32::MIN;
                let mut max_idx = 0;
                for k in 0..10 {
                    if logit[[j, k]] > max_val {
                        max_val = logit[[j, k]];
                        max_idx = k;
                    }
                }
                if max_idx == t_test[i + j] as usize {
                    correct += 1;
                }
            }
        }

        let accuracy = correct as f32 / test_subset_size as f32;
        println!("Test Accuracy: {:.2}%\n", accuracy * 100.0);
    }

    #[test]
    #[ignore] // 実行: cargo test --release test_gpu_timing_diagnostic -- --ignored --nocapture
    fn test_gpu_timing_diagnostic() {
        let x_train = load_images("dataset/train-images-idx3-ubyte")
            .into_shape_with_order((60000, 1, 28, 28))
            .unwrap();
        let t_train = load_labels("dataset/train-labels-idx1-ubyte");

        let gpu = Gpu::new();
        let mut net = GpuDeepConvNet::new(&gpu);
        let mut swl = SoftmaxWithLossLayer::new();

        let batch_size = 100;
        let max_iters = 50; // 50iter で十分傾向が出ます
        let lr = 0.01f32;
        let mut rng = rand::rng();

        let mut total_forward_ms = 0.0;
        let mut total_cpu_ms = 0.0;
        let mut total_backward_ms = 0.0;
        let mut total_update_ms = 0.0;

        println!(
            "--- Running GPU Timing Diagnostic ({} iters) ---",
            max_iters
        );

        for _ in 1..=max_iters {
            // --- 2. CPU 側 (前段: バッチ抽出と upload) ---
            let cpu_start = Instant::now();
            let idx = rand::seq::index::sample(&mut rng, 60000, batch_size).into_vec();
            let x_batch = x_train.select(Axis(0), &idx);
            let t_batch = to_one_hot(&idx.iter().map(|&j| t_train[j]).collect::<Vec<_>>(), 10);
            let gx = gpu.upload_image(&x_batch);
            let mut cpu_elapsed = cpu_start.elapsed().as_secs_f32() * 1000.0;

            // --- 1. Forward (+ logits download による同期) ---
            let fw_start = Instant::now();
            let gout = net.forward(&gpu, &gx);
            let logit = gpu.download(&gout); // ここで Forward 全体の完了待ち
            total_forward_ms += fw_start.elapsed().as_secs_f32() * 1000.0;

            // --- 2. CPU 側 (後段: SWL + dout upload) ---
            let cpu_start2 = Instant::now();
            let _ = swl.forward(logit, t_batch);
            let dx_cpu = swl.backward(1.0);
            let gdout = gpu.upload(&dx_cpu);
            cpu_elapsed += cpu_start2.elapsed().as_secs_f32() * 1000.0;
            total_cpu_ms += cpu_elapsed;

            // --- 3. Backward (+ 強制同期) ---
            let bw_start = Instant::now();
            let _ = net.backward(&gpu, &gdout);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("Poll failed"); // 完了を強制待ち
            total_backward_ms += bw_start.elapsed().as_secs_f32() * 1000.0;

            // --- 4. Update (+ 強制同期) ---
            let up_start = Instant::now();
            net.update(&gpu, lr);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("Poll failed"); // 完了を強制待ち
            total_update_ms += up_start.elapsed().as_secs_f32() * 1000.0;
        }

        println!("=== Cumulative Timing ({} iters) ===", max_iters);
        println!("  1. Forward + DL : {:.2} ms", total_forward_ms);
        println!("  2. CPU (SWL/Up) : {:.2} ms", total_cpu_ms);
        println!("  3. Backward     : {:.2} ms", total_backward_ms);
        println!("  4. Update       : {:.2} ms", total_update_ms);

        let total_ms = total_forward_ms + total_cpu_ms + total_backward_ms + total_update_ms;
        println!("  Total Measured  : {:.2} ms", total_ms);
        println!(
            "  Per Iteration   : {:.2} ms/iter",
            total_ms / max_iters as f32
        );
    }
}
