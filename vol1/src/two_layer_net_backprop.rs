use crate::layers::{AffineLayer, ReluLayer, SoftmaxWithLossLayer};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

/// 本 5.7.2「誤差逆伝播法に対応したニューラルネットワークの実装」
/// レイヤ(Affine→ReLU→Affine→SoftmaxWithLoss)を保持し、逆伝播で高速に勾配を求める
pub struct TwoLayerNetBackprop {
    affine1: AffineLayer,
    relu: ReluLayer,
    affine2: AffineLayer,
    last_layer: SoftmaxWithLossLayer,
}

impl TwoLayerNetBackprop {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weight_init_std = 0.01;
        let w1 = Array2::random((input_size, hidden_size), StandardNormal) * weight_init_std;
        let b1 = Array2::zeros((1, hidden_size)); // AffineLayer のバイアスは (1, n) 形
        let w2 = Array2::random((hidden_size, output_size), StandardNormal) * weight_init_std;
        let b2 = Array2::zeros((1, output_size));

        let affine1 = AffineLayer::new(w1, b1);
        let affine2 = AffineLayer::new(w2, b2);
        let relu = ReluLayer::new();
        let last_layer = SoftmaxWithLossLayer::new();
        Self {
            affine1,
            relu,
            affine2,
            last_layer,
        }
    }

    pub fn predict(&mut self, x: Array2<f32>) -> Array2<f32> {
        let out1 = self.affine1.forward(x);
        let out2 = self.relu.forward(out1);
        self.affine2.forward(out2)
    }

    pub fn loss(&mut self, x: Array2<f32>, t: Array2<f32>) -> f32 {
        let y = self.predict(x);
        self.last_layer.forward(y, t)
    }

    /// 本 5.7.2 誤差逆伝播で全パラメータの勾配を1パスで求める(順伝播→逆順に backward)
    pub fn gradient(
        &mut self,
        x: Array2<f32>,
        t: Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        // 順伝播
        self.loss(x, t);

        // 逆伝播
        let dout = self.last_layer.backward(1.0);
        let dout = self.affine2.backward(dout);
        let dout = self.relu.backward(dout);
        let _dout = self.affine1.backward(dout);

        (
            self.affine1.dw().clone(),
            self.affine1.db().clone(),
            self.affine2.dw().clone(),
            self.affine2.db().clone(),
        )
    }

    /// 本 5.7.3 勾配確認用。数値微分で全パラメータの勾配を求める(遅いが単純な基準)
    pub fn numerical_gradient(
        &mut self,
        x: Array2<f32>,
        t: Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let h = 1e-4;
        let (rows, cols) = self.affine1.w().dim();
        let mut dw1 = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let original_value = self.affine1.w_mut()[(i, j)];
                self.affine1.w_mut()[(i, j)] = original_value + h;
                let loss_plus_h = self.loss(x.clone(), t.clone());
                self.affine1.w_mut()[(i, j)] = original_value - h;
                let loss_minus_h = self.loss(x.clone(), t.clone());
                dw1[(i, j)] = (loss_plus_h - loss_minus_h) / (2.0 * h);
                self.affine1.w_mut()[(i, j)] = original_value;
            }
        }

        let (rows, cols) = self.affine2.w().dim();
        let mut dw2 = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let original_value = self.affine2.w_mut()[(i, j)];
                self.affine2.w_mut()[(i, j)] = original_value + h;
                let loss_plus_h = self.loss(x.clone(), t.clone());
                self.affine2.w_mut()[(i, j)] = original_value - h;
                let loss_minus_h = self.loss(x.clone(), t.clone());
                dw2[(i, j)] = (loss_plus_h - loss_minus_h) / (2.0 * h);
                self.affine2.w_mut()[(i, j)] = original_value;
            }
        }

        let (rows, cols) = self.affine1.b().dim(); // バイアスは (1, hidden) 形
        let mut db1 = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let original_value = self.affine1.b()[(i, j)];
                self.affine1.b_mut()[(i, j)] = original_value + h;
                let loss_plus_h = self.loss(x.clone(), t.clone());
                self.affine1.b_mut()[(i, j)] = original_value - h;
                let loss_minus_h = self.loss(x.clone(), t.clone());
                self.affine1.b_mut()[(i, j)] = original_value;
                db1[(i, j)] = (loss_plus_h - loss_minus_h) / (2.0 * h);
            }
        }

        let (rows, cols) = self.affine2.b().dim(); // バイアスは (1, output) 形
        let mut db2 = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let original_value = self.affine2.b()[(i, j)];
                self.affine2.b_mut()[(i, j)] = original_value + h;
                let loss_plus_h = self.loss(x.clone(), t.clone());
                self.affine2.b_mut()[(i, j)] = original_value - h;
                let loss_minus_h = self.loss(x.clone(), t.clone());
                self.affine2.b_mut()[(i, j)] = original_value;
                db2[(i, j)] = (loss_plus_h - loss_minus_h) / (2.0 * h);
            }
        }

        (dw1, db1, dw2, db2)
    }

    /// 本 5.7.4 各 Affine に自分の勾配でパラメータを更新させる(SGD 1 ステップ)
    pub fn update(&mut self, lr: f32) {
        self.affine1.update(lr);
        self.affine2.update(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_layer_net_backprop_loss() {
        // 損失が正の有限値であることを確認するテスト
        let mut net = TwoLayerNetBackprop::new(3, 4, 2); // input=3, hidden=4, output=2
        let x = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(); // (batch, input_size)
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap(); // (batch, output_size)
        let loss = net.loss(x, t);
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn test_two_layer_net_backprop_gradient() {
        //gradient を呼ぶと各勾配の形が対応する重みと一致、各要素が有限であることを確認するテスト
        let mut net = TwoLayerNetBackprop::new(3, 4, 2);
        let x = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let (dw1, db1, dw2, db2) = net.gradient(x, t);
        assert_eq!(dw1.dim(), net.affine1.dw().dim());
        assert_eq!(db1.dim(), net.affine1.db().dim());
        assert_eq!(dw2.dim(), net.affine2.dw().dim());
        assert_eq!(db2.dim(), net.affine2.db().dim());
        assert!(dw1.iter().all(|&v| v.is_finite()));
        assert!(db1.iter().all(|&v| v.is_finite()));
        assert!(dw2.iter().all(|&v| v.is_finite()));
        assert!(db2.iter().all(|&v| v.is_finite()));
    }

    fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn test_two_layer_net_backprop_numerical_gradient() {
        // 勾配確認テスト
        // 同じネットで両方を計算して比べます。順序に注意:gradient() は逆伝播で状態を書き換える(dw/db を埋める)だけで重みは変えないので、先に数値微分、後に逆伝播、どちらでもOKですが、混乱を避けるため別々に取ってから比較します。
        let mut net = TwoLayerNetBackprop::new(3, 4, 2);
        *net.affine1.w_mut() *= 100.0; // 勾配を解像可能な大きさに
        *net.affine2.w_mut() *= 100.0; // 勾配確認を f32 で意味あるものにする
        let x = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let (dw1_num, db1_num, dw2_num, db2_num) = net.numerical_gradient(x.clone(), t.clone());
        let (dw1_backprop, db1_backprop, dw2_backprop, db2_backprop) =
            net.gradient(x.clone(), t.clone());

        println!("dw1_num: {:?}", dw1_num);
        println!("dw1_backprop: {:?}", dw1_backprop);
        println!("db1_num: {:?}", db1_num);
        println!("db1_backprop: {:?}", db1_backprop);
        println!("dw2_num: {:?}", dw2_num);
        println!("dw2_backprop: {:?}", dw2_backprop);
        println!("db2_num: {:?}", db2_num);
        println!("db2_backprop: {:?}", db2_backprop);
        let epsilon: f32 = 1e-2;
        assert!(max_abs_diff(&dw1_num, &dw1_backprop) < epsilon);
        assert!(max_abs_diff(&db1_num, &db1_backprop) < epsilon);
        assert!(max_abs_diff(&dw2_num, &dw2_backprop) < epsilon);
        assert!(max_abs_diff(&db2_num, &db2_backprop) < epsilon);
    }

    use crate::mnist::{load_images, load_labels, to_one_hot};
    use ndarray::{Axis, s};
    #[test]
    #[ignore] // 実行に時間がかかるので CI では無視
    fn train_mnist_backprop() {
        let images = load_images("dataset/train-images-idx3-ubyte"); // (60000, 784)
        let labels = load_labels("dataset/train-labels-idx1-ubyte");
        let train_size = images.shape()[0];

        let batch_size = 100;
        let learning_rate = 0.1;
        let iters_num = 1000; // 逆伝播なら現実的に回せる
        let mut net = TwoLayerNetBackprop::new(784, 50, 10);

        // 固定の評価バッチ(トレンドを綺麗に見るため先頭100件)
        let eval_x = images.slice(s![0..100, ..]).to_owned();
        let eval_t = to_one_hot(&labels.slice(s![0..100]).to_vec(), 10);

        let mut rng = rand::rng();
        for i in 0..iters_num {
            let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();
            let x_batch = images.select(Axis(0), &idx);
            let batch_labels: Vec<u8> = idx.iter().map(|&j| labels[j]).collect();
            let t_batch = to_one_hot(&batch_labels, 10);

            net.gradient(x_batch, t_batch); // dw/db を埋める
            net.update(learning_rate);

            if i % 100 == 0 {
                let loss = net.loss(eval_x.clone(), eval_t.clone());
                println!("iter {i}: loss = {loss}");
            }
        }
    }
}
