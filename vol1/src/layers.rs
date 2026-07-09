use crate::loss::batch_cross_entropy_error;
use crate::network::{sigmoid, softmax};
use ndarray::Array2;

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
pub struct ReluLayer {
    mask: Array2<bool>,
}

impl ReluLayer {
    pub fn new() -> Self {
        Self {
            mask: Array2::default((0, 0)),
        }
    }

    pub fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        self.mask = x.mapv(|v| v <= 0.0); // mask は先に保存
        x.mapv(|v| v.max(0.0)) // 0以下は0、あとは素通し
    }

    pub fn backward(&self, dout: Array2<f32>) -> Array2<f32> {
        let mut dx = dout.clone();
        dx.iter_mut().zip(self.mask.iter()).for_each(|(d, &m)| {
            if m {
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
}
