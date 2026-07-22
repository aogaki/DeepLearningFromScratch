use super::{Gpu, GpuImage, GpuTensor};
use ndarray::{Array1, Array2, Array4, Axis};

pub struct GpuConvLayer {
    w_colt: GpuTensor, // (c·fh·fw, fn) — forward の matmul 用
    b: GpuTensor,      // (1, fn)
    stride: usize,
    pad: usize,
    fh: usize, // w_colt に畳んだ時点で 4D 形状が消えるので別持ち
    fw: usize,
    // forward キャッシュ(CPU 版の col / x_shape と同じ役割)
    col: Option<GpuTensor>,
    input_dims: Option<(usize, usize, usize, usize)>,
    // 勾配(update ステップが後で食う)
    dw_colt: Option<GpuTensor>,
    db: Option<GpuTensor>,
}
impl GpuConvLayer {
    pub fn new(gpu: &Gpu, w: &Array4<f32>, b: &Array1<f32>, stride: usize, pad: usize) -> Self {
        let (fn_, c, fh, fw) = w.dim();

        // w を 2D に変形: (fn, c*fh*fw)
        let w_2d = w.clone().into_shape_with_order((fn_, c * fh * fw)).unwrap();

        // w_colt (c*fh*fw, fn) の作成（順伝播用）
        let mut w_colt_cpu = Array2::<f32>::zeros((c * fh * fw, fn_));
        w_colt_cpu.assign(&w_2d.t());
        let w_colt = gpu.upload(&w_colt_cpu);

        // b (1, fn) の作成
        let b_2d = b.clone().insert_axis(Axis(0));
        let b_gpu = gpu.upload(&b_2d);

        Self {
            w_colt,
            b: b_gpu,
            stride,
            pad,
            fh,
            fw,
            col: None,
            input_dims: None,
            dw_colt: None,
            db: None,
        }
    }

    pub fn forward(&mut self, gpu: &Gpu, x: &GpuImage) -> GpuImage {
        let (n, _c, h, w) = x.dims;
        let fn_ = self.w_colt.shape.1;
        let oh = (h + 2 * self.pad - self.fh) / self.stride + 1;
        let ow = (w + 2 * self.pad - self.fw) / self.stride + 1;

        self.input_dims = Some(x.dims);

        // gpu.conv_forward_gpu の内部実装を直接展開し、col をキャッシュする
        let col = gpu.im2col_gpu(x, self.fh, self.fw, self.stride, self.pad);

        let mut y = gpu.matmul_gpu(&col, &self.w_colt);
        gpu.add_bias_gpu(&mut y, &self.b);
        let out = gpu.nhwc_to_nchw_gpu(&y, (n, fn_, oh, ow));

        self.col = Some(col); // 逆伝播のために保存

        out
    }

    pub fn backward(&mut self, gpu: &Gpu, dout: &GpuImage) -> GpuImage {
        let col = self
            .col
            .as_ref()
            .expect("forward must be called before backward");
        let input_dims = self
            .input_dims
            .expect("forward must be called before backward");

        let (dx, dw_colt, db) = gpu.conv_backward_gpu(
            dout,
            col,
            &self.w_colt,
            input_dims,
            self.fh,
            self.fw,
            self.stride,
            self.pad,
        );

        self.dw_colt = Some(dw_colt);
        self.db = Some(db);

        dx
    }

    pub fn update(&mut self, gpu: &Gpu, lr: f32) {
        let dw_colt = self
            .dw_colt
            .as_ref()
            .expect("backward must be called before update");
        let db = self
            .db
            .as_ref()
            .expect("backward must be called before update");

        gpu.sgd_update_gpu(&mut self.w_colt, dw_colt, lr);
        gpu.sgd_update_gpu(&mut self.b, db, lr);
    }

    pub fn w_colt(&self) -> &GpuTensor {
        &self.w_colt
    }

    pub fn dw_colt(&self) -> &GpuTensor {
        self.dw_colt
            .as_ref()
            .expect("backward must be called before accessing dw")
    }

    pub fn b(&self) -> &GpuTensor {
        &self.b
    }

    pub fn db(&self) -> &GpuTensor {
        self.db
            .as_ref()
            .expect("backward must be called before accessing db")
    }
}

pub struct GpuReluLayer {
    act: Option<GpuTensor>,
}
impl GpuReluLayer {
    pub fn new() -> Self {
        Self { act: None }
    }

    // Affine 後などの 2D テンソル用
    pub fn forward_tensor(&mut self, gpu: &Gpu, mut x: GpuTensor) -> GpuTensor {
        gpu.relu_gpu(&mut x);
        self.act = Some(x.clone()); // ハンドルのクローンを保持
        x
    }

    // Conv 後の 4D 画像テンソル用
    pub fn forward_image(&mut self, gpu: &Gpu, mut x: GpuImage) -> GpuImage {
        gpu.relu_gpu(&mut x.tensor);
        self.act = Some(x.tensor.clone());
        x
    }

    pub fn backward_tensor(&self, gpu: &Gpu, mut dout: GpuTensor) -> GpuTensor {
        gpu.relu_backward_gpu(
            &mut dout,
            self.act
                .as_ref()
                .expect("forward must be called before backward"),
        );
        dout
    }

    pub fn backward_image(&self, gpu: &Gpu, mut dout: GpuImage) -> GpuImage {
        gpu.relu_backward_gpu(
            &mut dout.tensor,
            self.act
                .as_ref()
                .expect("forward must be called before backward"),
        );
        dout
    }
}

pub struct GpuPoolingLayer {
    ph: usize,
    pw: usize,
    stride: usize,
    pad: usize,
    argmax: Option<wgpu::Buffer>,
    input_dims: Option<(usize, usize, usize, usize)>,
}
impl GpuPoolingLayer {
    pub fn new(ph: usize, pw: usize, stride: usize, pad: usize) -> Self {
        Self {
            ph,
            pw,
            stride,
            pad,
            argmax: None,
            input_dims: None,
        }
    }

    pub fn forward(&mut self, gpu: &Gpu, x: &GpuImage) -> GpuImage {
        self.input_dims = Some(x.dims);
        let (y, argmax) = gpu.pool_forward_gpu(x, self.ph, self.pw, self.stride, self.pad);
        self.argmax = Some(argmax); // テンソルではなく u32 バッファを保持
        y
    }

    pub fn backward(&self, gpu: &Gpu, dout: &GpuImage) -> GpuImage {
        gpu.pool_backward_gpu(
            dout,
            self.argmax
                .as_ref()
                .expect("forward must be called before backward"),
            self.input_dims
                .expect("forward must be called before backward"),
            self.ph,
            self.pw,
            self.stride,
            self.pad,
        )
    }
}

pub struct GpuAffineLayer {
    w: GpuTensor,         // (in_size, out_size)
    b: GpuTensor,         // (1, out_size)
    x: Option<GpuTensor>, // forward 時の入力キャッシュ
    dw: Option<GpuTensor>,
    db: Option<GpuTensor>,
}
impl GpuAffineLayer {
    pub fn new(gpu: &Gpu, w: &Array2<f32>, b: &Array2<f32>) -> Self {
        let w_gpu = gpu.upload(w);

        let b_gpu = gpu.upload(b);

        Self {
            w: w_gpu,
            b: b_gpu,
            x: None,
            dw: None,
            db: None,
        }
    }

    pub fn forward(&mut self, gpu: &Gpu, x: &GpuTensor) -> GpuTensor {
        self.x = Some(x.clone()); // ハンドルのクローンを保持
        let mut y = gpu.matmul_gpu(x, &self.w);
        gpu.add_bias_gpu(&mut y, &self.b);
        y
    }

    pub fn backward(&mut self, gpu: &Gpu, dout: &GpuTensor) -> GpuTensor {
        let x = self
            .x
            .as_ref()
            .expect("forward must be called before backward");

        let dx = gpu.matmul_nt_gpu(dout, &self.w);
        let dw = gpu.matmul_tn_gpu(x, dout);
        let db = gpu.column_sum_gpu(dout);

        self.dw = Some(dw);
        self.db = Some(db);

        dx
    }

    pub fn update(&mut self, gpu: &Gpu, lr: f32) {
        let dw = self
            .dw
            .as_ref()
            .expect("backward must be called before update");
        let db = self
            .db
            .as_ref()
            .expect("backward must be called before update");

        gpu.sgd_update_gpu(&mut self.w, dw, lr);
        gpu.sgd_update_gpu(&mut self.b, db, lr);
    }

    pub fn w(&self) -> &GpuTensor {
        &self.w
    }

    pub fn dw(&self) -> &GpuTensor {
        self.dw
            .as_ref()
            .expect("backward must be called before accessing dw")
    }

    pub fn b(&self) -> &GpuTensor {
        &self.b
    }

    pub fn db(&self) -> &GpuTensor {
        self.db
            .as_ref()
            .expect("backward must be called before accessing db")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::{ConvolutionLayer, PoolingLayer};
    use crate::layers::{AffineLayer, FlattenLayer, Layer, ReluLayer};
    use ndarray::{Ix2, Ix4};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal;

    fn setup_conv_layers(
        gpu: &Gpu,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        fn_: usize,
        fh: usize,
        fw: usize,
        stride: usize,
        pad: usize,
    ) -> (Array4<f32>, ConvolutionLayer, GpuConvLayer) {
        let x: Array4<f32> = Array4::random((n, c, h, w), StandardNormal);
        let w_arr: Array4<f32> = Array4::random((fn_, c, fh, fw), StandardNormal);
        let b_arr: Array1<f32> = Array1::random(fn_, StandardNormal);

        let conv_cpu = ConvolutionLayer::new(w_arr.clone(), b_arr.clone(), stride, pad);
        let conv_gpu = GpuConvLayer::new(gpu, &w_arr, &b_arr, stride, pad);

        (x, conv_cpu, conv_gpu)
    }

    #[test]
    fn test_gpu_conv_layer_forward() {
        let gpu = Gpu::new();

        let (x, mut conv_cpu, mut conv_gpu) = setup_conv_layers(&gpu, 2, 3, 5, 5, 4, 3, 3, 1, 1);

        // --- CPU ---
        let cpu_out = Layer::forward(&mut conv_cpu, x.clone().into_dyn(), false)
            .into_dimensionality::<Ix4>()
            .unwrap();

        // --- GPU ---
        let gx = gpu.upload_image(&x);
        let gy = conv_gpu.forward(&gpu, &gx);
        let gpu_out = gpu
            .download(&gy.tensor)
            .into_shape_with_order(gy.dims)
            .unwrap();

        // --- 比較 ---
        let max_diff = cpu_out
            .iter()
            .zip(gpu_out.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 1e-4, "forward diff is too large: {max_diff:e}");
    }

    #[test]
    fn test_gpu_conv_layer_backward() {
        let gpu = Gpu::new();

        let (n, c, h, w, fn_, fh, fw, stride, pad) = (2, 3, 5, 5, 4, 3, 3, 1, 1);
        let (x, mut conv_cpu, mut conv_gpu) =
            setup_conv_layers(&gpu, n, c, h, w, fn_, fh, fw, stride, pad);

        let oh = (h + 2 * pad - fh) / stride + 1;
        let ow = (w + 2 * pad - fw) / stride + 1;
        let dout: Array4<f32> = Array4::random((n, fn_, oh, ow), StandardNormal);

        // --- CPU ---
        let _ = Layer::forward(&mut conv_cpu, x.clone().into_dyn(), false);
        let cpu_dx = Layer::backward(&mut conv_cpu, dout.clone().into_dyn())
            .into_dimensionality::<Ix4>()
            .unwrap();

        let cpu_dw = conv_cpu.dw(); // (fn_, c, fh, fw)
        let cpu_db = conv_cpu.db(); // (fn_,)

        // --- GPU ---
        let gx = gpu.upload_image(&x);
        let _ = conv_gpu.forward(&gpu, &gx); // col のキャッシュ

        let gdout = gpu.upload_image(&dout);
        let gdx = conv_gpu.backward(&gpu, &gdout);

        // --- 取得と変形 ---
        let gpu_dx = gpu
            .download(&gdx.tensor)
            .into_shape_with_order(gdx.dims)
            .unwrap();

        let gpu_dw_colt = gpu.download(conv_gpu.dw_colt.as_ref().unwrap());
        let gpu_db = gpu.download(conv_gpu.db.as_ref().unwrap());

        // --- 比較 ---
        // dx
        let diff_dx = cpu_dx
            .iter()
            .zip(gpu_dx.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_dx < 1e-3, "dx diff: {diff_dx:e}");

        // dW
        let cpu_dw_2d = cpu_dw
            .view()
            .into_shape_with_order((fn_, c * fh * fw))
            .unwrap();
        let cpu_dw_colt = cpu_dw_2d.t();
        let diff_dw = cpu_dw_colt
            .iter()
            .zip(gpu_dw_colt.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_dw < 1e-3, "dw diff: {diff_dw:e}");

        // db
        let cpu_db_2d = cpu_db.view().insert_axis(Axis(0));
        let diff_db = cpu_db_2d
            .iter()
            .zip(gpu_db.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_db < 1e-3, "db diff: {diff_db:e}");
    }

    #[test]
    fn test_gpu_relu_layer() {
        let gpu = Gpu::new();
        let mut cpu_layer = ReluLayer::new();
        let mut gpu_layer = GpuReluLayer::new();

        let x = Array2::random((100, 50), StandardNormal);
        let dout = Array2::random((100, 50), StandardNormal);

        // --- forward ---
        let cpu_y = Layer::forward(&mut cpu_layer, x.clone().into_dyn(), false)
            .into_dimensionality::<Ix2>()
            .unwrap();

        let gx = gpu.upload(&x);
        let gy = gpu_layer.forward_tensor(&gpu, gx);
        let gpu_y = gpu.download(&gy);

        let diff_y = cpu_y
            .iter()
            .zip(gpu_y.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(diff_y, 0.0);

        // --- backward ---
        let cpu_dx = Layer::backward(&mut cpu_layer, dout.clone().into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap();

        let gdout = gpu.upload(&dout);
        let gdx = gpu_layer.backward_tensor(&gpu, gdout);
        let gpu_dx = gpu.download(&gdx);

        let diff_dx = cpu_dx
            .iter()
            .zip(gpu_dx.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(diff_dx, 0.0);
    }

    #[test]
    fn test_gpu_pooling_layer() {
        let gpu = Gpu::new();
        let (n, c, h, w) = (2, 3, 6, 6);
        let (ph, pw, stride, pad) = (2, 2, 2, 0);

        let x = Array4::random((n, c, h, w), StandardNormal);
        let mut cpu_layer = PoolingLayer::new(ph, pw, stride, pad);
        let mut gpu_layer = GpuPoolingLayer::new(ph, pw, stride, pad);

        // --- forward ---
        let cpu_y = cpu_layer.forward(&x);
        let gx = gpu.upload_image(&x);
        let gy = gpu_layer.forward(&gpu, &gx);

        let cpu_y_flat = cpu_y
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((n, c * cpu_y.dim().2 * cpu_y.dim().3))
            .unwrap();
        let gpu_y = gpu.download(&gy.tensor);

        let diff_y = cpu_y_flat
            .iter()
            .zip(gpu_y.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(diff_y, 0.0);

        // --- backward ---
        let (_, _, oh, ow) = cpu_y.dim();
        let dout = Array4::random((n, c, oh, ow), StandardNormal);

        let cpu_dx = cpu_layer.backward(&dout);
        let gdout = gpu.upload_image(&dout);
        let gdx = gpu_layer.backward(&gpu, &gdout);

        let cpu_dx_flat = cpu_dx
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((n, c * h * w))
            .unwrap();
        let gpu_dx = gpu.download(&gdx.tensor);

        let diff_dx = cpu_dx_flat
            .iter()
            .zip(gpu_dx.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(diff_dx, 0.0);
    }

    #[test]
    fn test_gpu_affine_layer() {
        let gpu = Gpu::new();
        let (batch, in_size, out_size) = (100, 50, 10);

        let x = Array2::random((batch, in_size), StandardNormal);
        let w = Array2::random((in_size, out_size), StandardNormal);
        let b = Array2::random((1, out_size), StandardNormal);
        let dout = Array2::random((batch, out_size), StandardNormal);

        let mut cpu_layer = AffineLayer::new(w.clone(), b.clone());
        let mut gpu_layer = GpuAffineLayer::new(&gpu, &w, &b);

        // --- forward ---
        let cpu_y = Layer::forward(&mut cpu_layer, x.clone().into_dyn(), false)
            .into_dimensionality::<Ix2>()
            .unwrap();
        let gx = gpu.upload(&x);
        let gy = gpu_layer.forward(&gpu, &gx);
        let gpu_y = gpu.download(&gy);

        let diff_y = cpu_y
            .iter()
            .zip(gpu_y.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_y < 1e-4);

        // --- backward ---
        let cpu_dx = Layer::backward(&mut cpu_layer, dout.clone().into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap();
        let gdout = gpu.upload(&dout);
        let gdx = gpu_layer.backward(&gpu, &gdout);
        let gpu_dx = gpu.download(&gdx);

        let diff_dx = cpu_dx
            .iter()
            .zip(gpu_dx.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_dx < 1e-3);

        // --- 勾配の比較 ---
        let cpu_dw = cpu_layer.dw();
        let gpu_dw = gpu.download(gpu_layer.dw.as_ref().unwrap());
        let diff_dw = cpu_dw
            .iter()
            .zip(gpu_dw.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_dw < 1e-3);

        let cpu_db = cpu_layer.db();
        let gpu_db = gpu.download(gpu_layer.db.as_ref().unwrap());
        let diff_db = cpu_db
            .iter()
            .zip(gpu_db.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_db < 1e-3);
    }

    #[test]
    fn test_deep_conv_net_backward_gpu() {
        let batch = 2;
        let gpu = Gpu::new();
        let x: Array4<f32> = Array4::random((batch, 1, 28, 28), StandardNormal);

        // 重み初期化ヘルパー
        let he_conv = |fn_: usize, c: usize, fh: usize, fw: usize| -> Array4<f32> {
            let fan_in = (c * fh * fw) as f32;
            let scale = (2.0 / fan_in).sqrt();
            Array4::random((fn_, c, fh, fw), StandardNormal) * scale
        };
        let he_affine = |fan_in: usize, fan_out: usize| -> Array2<f32> {
            let scale = (2.0 / fan_in as f32).sqrt();
            Array2::random((fan_in, fan_out), StandardNormal) * scale
        };

        // 重み・バイアス生成
        let w1_1 = he_conv(16, 1, 3, 3);
        let b1_1 = Array1::<f32>::zeros(16);
        let w1_2 = he_conv(16, 16, 3, 3);
        let b1_2 = Array1::<f32>::zeros(16);
        let w2_1 = he_conv(32, 16, 3, 3);
        let b2_1 = Array1::<f32>::zeros(32);
        let w2_2 = he_conv(32, 32, 3, 3);
        let b2_2 = Array1::<f32>::zeros(32); // pad=2
        let w3_1 = he_conv(64, 32, 3, 3);
        let b3_1 = Array1::<f32>::zeros(64);
        let w3_2 = he_conv(64, 64, 3, 3);
        let b3_2 = Array1::<f32>::zeros(64);
        let wa1 = he_affine(64 * 4 * 4, 50);
        let ba1 = Array2::<f32>::zeros((1, 50));
        let wa2 = he_affine(50, 10);
        let ba2 = Array2::<f32>::zeros((1, 10));

        // --- CPU レイヤ構築 ---
        let mut c1_1 = ConvolutionLayer::new(w1_1.clone(), b1_1.clone(), 1, 1);
        let mut r1_1 = ReluLayer::new();
        let mut c1_2 = ConvolutionLayer::new(w1_2.clone(), b1_2.clone(), 1, 1);
        let mut r1_2 = ReluLayer::new();
        let mut p1 = PoolingLayer::new(2, 2, 2, 0);

        let mut c2_1 = ConvolutionLayer::new(w2_1.clone(), b2_1.clone(), 1, 1);
        let mut r2_1 = ReluLayer::new();
        let mut c2_2 = ConvolutionLayer::new(w2_2.clone(), b2_2.clone(), 1, 2); // pad=2
        let mut r2_2 = ReluLayer::new();
        let mut p2 = PoolingLayer::new(2, 2, 2, 0);

        let mut c3_1 = ConvolutionLayer::new(w3_1.clone(), b3_1.clone(), 1, 1);
        let mut r3_1 = ReluLayer::new();
        let mut c3_2 = ConvolutionLayer::new(w3_2.clone(), b3_2.clone(), 1, 1);
        let mut r3_2 = ReluLayer::new();
        let mut p3 = PoolingLayer::new(2, 2, 2, 0);

        let mut flat = FlattenLayer::new();
        let mut af1 = AffineLayer::new(wa1.clone(), ba1.clone());
        let mut ra1 = ReluLayer::new();
        let mut af2 = AffineLayer::new(wa2.clone(), ba2.clone());

        // --- GPU レイヤ構築 ---
        let mut gc1_1 = GpuConvLayer::new(&gpu, &w1_1, &b1_1, 1, 1);
        let mut gr1_1 = GpuReluLayer::new();
        let mut gc1_2 = GpuConvLayer::new(&gpu, &w1_2, &b1_2, 1, 1);
        let mut gr1_2 = GpuReluLayer::new();
        let mut gp1 = GpuPoolingLayer::new(2, 2, 2, 0);

        let mut gc2_1 = GpuConvLayer::new(&gpu, &w2_1, &b2_1, 1, 1);
        let mut gr2_1 = GpuReluLayer::new();
        let mut gc2_2 = GpuConvLayer::new(&gpu, &w2_2, &b2_2, 1, 2); // pad=2
        let mut gr2_2 = GpuReluLayer::new();
        let mut gp2 = GpuPoolingLayer::new(2, 2, 2, 0);

        let mut gc3_1 = GpuConvLayer::new(&gpu, &w3_1, &b3_1, 1, 1);
        let mut gr3_1 = GpuReluLayer::new();
        let mut gc3_2 = GpuConvLayer::new(&gpu, &w3_2, &b3_2, 1, 1);
        let mut gr3_2 = GpuReluLayer::new();
        let mut gp3 = GpuPoolingLayer::new(2, 2, 2, 0);

        let mut gaf1 = GpuAffineLayer::new(&gpu, &wa1, &ba1);
        let mut gra1 = GpuReluLayer::new();
        let mut gaf2 = GpuAffineLayer::new(&gpu, &wa2, &ba2);

        // ==============================
        // Forward
        // ==============================

        // --- CPU ---
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
        let _cpu_final = Layer::forward(&mut af2, out, false);

        // --- GPU ---
        let mut gout = gpu.upload_image(&x);
        gout = gc1_1.forward(&gpu, &gout);
        gout = gr1_1.forward_image(&gpu, gout);
        gout = gc1_2.forward(&gpu, &gout);
        gout = gr1_2.forward_image(&gpu, gout);
        gout = gp1.forward(&gpu, &gout);

        gout = gc2_1.forward(&gpu, &gout);
        gout = gr2_1.forward_image(&gpu, gout);
        gout = gc2_2.forward(&gpu, &gout);
        gout = gr2_2.forward_image(&gpu, gout);
        gout = gp2.forward(&gpu, &gout);

        gout = gc3_1.forward(&gpu, &gout);
        gout = gr3_1.forward_image(&gpu, gout);
        gout = gc3_2.forward(&gpu, &gout);
        gout = gr3_2.forward_image(&gpu, gout);
        gout = gp3.forward(&gpu, &gout);

        // Flatten: no-op (テンソルを取り出すだけ)
        let mut gout_tensor = gout.tensor;

        gout_tensor = gaf1.forward(&gpu, &gout_tensor);
        gout_tensor = gra1.forward_tensor(&gpu, gout_tensor);
        let _gpu_logits = gaf2.forward(&gpu, &gout_tensor);

        // ==============================
        // Backward
        // ==============================
        let dout = Array2::random((batch, 10), StandardNormal);

        // --- CPU ---
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

        // --- GPU ---
        let mut gd = gpu.upload(&dout);
        gd = gaf2.backward(&gpu, &gd);
        gd = gra1.backward_tensor(&gpu, gd);
        gd = gaf1.backward(&gpu, &gd);

        // Unflatten: no-op (GpuImage に包み直す)
        let mut gd_img = GpuImage {
            tensor: gd,
            dims: (batch, 64, 4, 4),
        };

        gd_img = gp3.backward(&gpu, &gd_img);
        gd_img = gr3_2.backward_image(&gpu, gd_img);
        gd_img = gc3_2.backward(&gpu, &gd_img);
        gd_img = gr3_1.backward_image(&gpu, gd_img);
        gd_img = gc3_1.backward(&gpu, &gd_img);

        gd_img = gp2.backward(&gpu, &gd_img);
        gd_img = gr2_2.backward_image(&gpu, gd_img);
        gd_img = gc2_2.backward(&gpu, &gd_img);
        gd_img = gr2_1.backward_image(&gpu, gd_img);
        gd_img = gc2_1.backward(&gpu, &gd_img);

        gd_img = gp1.backward(&gpu, &gd_img);
        gd_img = gr1_2.backward_image(&gpu, gd_img);
        gd_img = gc1_2.backward(&gpu, &gd_img);
        gd_img = gr1_1.backward_image(&gpu, gd_img);
        let gdx_img = gc1_1.backward(&gpu, &gd_img);

        let gpu_dx = gpu
            .download(&gdx_img.tensor)
            .into_shape_with_order(gdx_img.dims)
            .unwrap();

        // ==============================
        // 検証
        // ==============================
        let eps = 1e-2;

        // ① 最下流の dx (入力勾配)
        let diff_dx = cpu_dx
            .iter()
            .zip(gpu_dx.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        println!("DeepConvNet dx max_diff: {diff_dx:e}");
        assert!(diff_dx < eps, "dx diff: {diff_dx:e}");

        // ② 最初の層 (c1_1) の dW / db
        // GPU は (c*fh*fw, fn_) なので、CPUの (fn_, c, fh, fw) -> (fn_, c*fh*fw) -> 転置
        let cpu_c1_1_dw = c1_1
            .dw()
            .clone()
            .into_shape_with_order((16, 1 * 3 * 3))
            .unwrap()
            .t()
            .to_owned();
        let cpu_c1_1_db = c1_1.db().clone().insert_axis(Axis(0));
        let gpu_c1_1_dw = gpu.download(gc1_1.dw_colt.as_ref().unwrap());
        let gpu_c1_1_db = gpu.download(gc1_1.db.as_ref().unwrap());

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
        println!("DeepConvNet c1_1 db max_diff: {diff_c1_1_db:e}");
        assert!(diff_c1_1_db < eps, "c1_1 db diff: {diff_c1_1_db:e}");

        // ③ 最後の層 (af2) の dW / db
        // CPU の dW も GPU と同様に x^T * dout で (in_size, out_size) のままなので転置不要！
        let cpu_af2_dw = af2.dw().clone();
        let cpu_af2_db = af2.db().clone();
        let gpu_af2_dw = gpu.download(gaf2.dw.as_ref().unwrap());
        let gpu_af2_db = gpu.download(gaf2.db.as_ref().unwrap());

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
        println!("DeepConvNet af2 db max_diff: {diff_af2_db:e}");
        assert!(diff_af2_db < eps, "af2 db diff: {diff_af2_db:e}");
    }

    #[test]
    fn test_gpu_conv_layer_update() {
        use crate::optimizer::SGD; // ★ オプティマイザをインポート

        let gpu = Gpu::new();
        let lr = 0.01f32;

        let (x, mut conv_cpu, mut conv_gpu) = setup_conv_layers(&gpu, 2, 3, 5, 5, 4, 3, 3, 1, 1);

        // ★ CPU 側に本物の SGD オプティマイザをセット
        conv_cpu.set_optimizer(Box::new(SGD::new(lr)), Box::new(SGD::new(lr)));

        let oh = 5;
        let ow = 5;
        let dout = Array4::random((2, 4, oh, ow), StandardNormal);

        // --- CPU ---
        let _ = Layer::forward(&mut conv_cpu, x.clone().into_dyn(), false);
        let _ = Layer::backward(&mut conv_cpu, dout.clone().into_dyn());

        // ★ 正規の学習経路 (Layer::update) でパラメータを更新
        Layer::update(&mut conv_cpu);

        // --- GPU ---
        let gx = gpu.upload_image(&x);
        let gdout = gpu.upload_image(&dout);

        let _ = conv_gpu.forward(&gpu, &gx);
        let _ = conv_gpu.backward(&gpu, &gdout);
        conv_gpu.update(&gpu, lr); // GPU版の更新

        // --- 比較 ---
        let gpu_w_colt = gpu.download(&conv_gpu.w_colt); // (c*fh*fw, fn_)
        let gpu_b = gpu.download(&conv_gpu.b); // (1, fn_)

        // CPU側の更新後パラメータ w を w_colt の形に変換
        let cpu_w_2d = conv_cpu
            .w()
            .view()
            .into_shape_with_order((4, 3 * 3 * 3))
            .unwrap();
        let cpu_w_colt = cpu_w_2d.t();

        let diff_w = cpu_w_colt
            .iter()
            .zip(gpu_w_colt.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let diff_b = conv_cpu
            .b()
            .iter()
            .zip(gpu_b.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        // FMA による 1 ULP 差異が生じるため 1e-6 の許容誤差
        assert!(diff_w < 1e-6, "Conv update w diff: {diff_w:e}");
        assert!(diff_b < 1e-6, "Conv update b diff: {diff_b:e}");
    }

    #[test]
    fn test_gpu_affine_layer_update() {
        use crate::optimizer::SGD;

        let gpu = Gpu::new();
        let lr = 0.01f32;
        let (batch, in_size, out_size) = (10, 20, 5);

        let x = Array2::random((batch, in_size), StandardNormal);
        let w = Array2::random((in_size, out_size), StandardNormal);
        let b = Array2::random((1, out_size), StandardNormal);
        let dout = Array2::random((batch, out_size), StandardNormal);

        let mut affine_cpu = AffineLayer::new(w.clone(), b.clone());
        // ★ CPU 側に本物の SGD オプティマイザをセット
        affine_cpu.set_optimizer(Box::new(SGD::new(lr)), Box::new(SGD::new(lr)));

        let mut affine_gpu = GpuAffineLayer::new(&gpu, &w, &b);

        // --- CPU ---
        let _ = Layer::forward(&mut affine_cpu, x.clone().into_dyn(), false);
        let _ = Layer::backward(&mut affine_cpu, dout.clone().into_dyn());

        // ★ 正規の学習経路
        Layer::update(&mut affine_cpu);

        // --- GPU ---
        let gx = gpu.upload(&x);
        let gdout = gpu.upload(&dout);

        let _ = affine_gpu.forward(&gpu, &gx);
        let _ = affine_gpu.backward(&gpu, &gdout);
        affine_gpu.update(&gpu, lr);

        // --- 比較 ---
        let gpu_w = gpu.download(&affine_gpu.w);
        let gpu_b = gpu.download(&affine_gpu.b);

        let diff_w = affine_cpu
            .w()
            .iter()
            .zip(gpu_w.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let diff_b = affine_cpu
            .b()
            .iter()
            .zip(gpu_b.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        assert!(diff_w < 1e-6, "Affine update w diff: {diff_w:e}");
        assert!(diff_b < 1e-6, "Affine update b diff: {diff_b:e}");
    }
}
