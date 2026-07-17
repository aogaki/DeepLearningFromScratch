use crate::gradient::numerical_gradient_over;
use crate::loss::batch_cross_entropy_error;
use crate::network::{sigmoid, softmax};
use ndarray::{Array1, Array2, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

/// 本 4.5「学習アルゴリズムの実装」各パラメータに対応する勾配のまとめ
pub struct Gradients {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

/// 本 4.5.1「2層ニューラルネットワークのクラス」(TwoLayerNet)
pub struct TwoLayerNet {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl TwoLayerNet {
    /// 本 4.5.1 重みは微小な正規乱数、バイアスは 0 で初期化
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weight_init_std = 0.01;
        let w1 = Array2::random((input_size, hidden_size), StandardNormal) * weight_init_std;
        let b1 = Array1::zeros(hidden_size);
        let w2 = Array2::random((hidden_size, output_size), StandardNormal) * weight_init_std;
        let b2 = Array1::zeros(output_size);
        TwoLayerNet { w1, b1, w2, b2 }
    }

    /// 本 4.5.1 推論(順伝播)。`forward` を self の重みで呼ぶ薄いラッパ
    pub fn predict(&self, x: ArrayView2<f32>) -> Array2<f32> {
        Self::forward(&self.w1, &self.b1, &self.w2, &self.b2, x)
    }

    /// 本 4.5.1 損失。`forward_loss` を self の重みで呼ぶ薄いラッパ
    pub fn loss(&self, x: ArrayView2<f32>, t: ArrayView2<f32>) -> f32 {
        Self::forward_loss(&self.w1, &self.b1, &self.w2, &self.b2, x, t)
    }

    /// 順伝播の唯一の実装。重みを引数で受けるので数値微分から再利用できる
    fn forward(
        w1: &Array2<f32>,
        b1: &Array1<f32>,
        w2: &Array2<f32>,
        b2: &Array1<f32>,
        x: ArrayView2<f32>,
    ) -> Array2<f32> {
        let a1 = x.dot(w1) + b1;
        let z1 = a1.mapv(sigmoid);
        let a2 = z1.dot(w2) + b2;

        let mut result = Array2::<f32>::zeros(a2.raw_dim());
        for (i, row) in a2.outer_iter().enumerate() {
            let softmax_row = softmax(row.to_owned());
            result.row_mut(i).assign(&softmax_row);
        }
        result
    }

    /// forward + 交差エントロピー誤差。損失をパラメータの純粋関数として表す
    fn forward_loss(
        w1: &Array2<f32>,
        b1: &Array1<f32>,
        w2: &Array2<f32>,
        b2: &Array1<f32>,
        x: ArrayView2<f32>,
        t: ArrayView2<f32>,
    ) -> f32 {
        let y = Self::forward(w1, b1, w2, b2, x);
        batch_cross_entropy_error(y.view(), t)
    }

    /// 本 4.5.1 数値微分で全パラメータ(W1/b1/W2/b2)の勾配を求める
    pub fn numerical_gradient(&self, x: ArrayView2<f32>, t: ArrayView2<f32>) -> Gradients {
        let w1 = numerical_gradient_over(&self.w1, |w| {
            Self::forward_loss(w, &self.b1, &self.w2, &self.b2, x, t)
        });
        let b1 = numerical_gradient_over(&self.b1, |b| {
            Self::forward_loss(&self.w1, b, &self.w2, &self.b2, x, t)
        });
        let w2 = numerical_gradient_over(&self.w2, |w| {
            Self::forward_loss(&self.w1, &self.b1, w, &self.b2, x, t)
        });
        let b2 = numerical_gradient_over(&self.b2, |b| {
            Self::forward_loss(&self.w1, &self.b1, &self.w2, b, x, t)
        });

        Gradients { w1, b1, w2, b2 }
    }

    /// 本 4.5.1 認識精度。予測と正解の argmax が一致した割合
    pub fn accuracy(&self, x: ArrayView2<f32>, t: ArrayView2<f32>) -> f32 {
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

    /// 本 4.5.2 勾配降下法によるパラメータ更新(SGD 1 ステップ)
    pub fn update(&mut self, gradients: Gradients, learning_rate: f32) {
        self.w1.scaled_add(-learning_rate, &gradients.w1);
        self.b1.scaled_add(-learning_rate, &gradients.b1);
        self.w2.scaled_add(-learning_rate, &gradients.w2);
        self.b2.scaled_add(-learning_rate, &gradients.b2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        let input_size = 784;
        let hidden_size = 100;
        let output_size = 10;
        let net = TwoLayerNet::new(input_size, hidden_size, output_size);

        let result = net.predict(ndarray::Array2::<f32>::zeros((3, input_size)).view());
        assert_eq!(result.shape(), &[3, output_size]);
        for row in result.outer_iter() {
            assert!((row.sum() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_loss() {
        let input_size = 784;
        let hidden_size = 100;
        let output_size = 10;
        let net = TwoLayerNet::new(input_size, hidden_size, output_size);

        let x = ndarray::Array2::<f32>::zeros((3, input_size));
        let mut t = ndarray::Array2::<f32>::zeros((3, output_size));
        t[[0, 2]] = 1.0;
        t[[1, 5]] = 1.0;
        t[[2, 0]] = 1.0;
        let loss = net.loss(x.view(), t.view());
        assert!(loss >= 0.0 && loss.is_finite());
        println!("loss test result: {}", loss);
        assert!((loss - 2.302585).abs() < 1e-1);
    }

    #[test]
    fn test_numerical_gradient() {
        // let input_size = 784;
        // let hidden_size = 100;
        // let output_size = 10;
        let input_size = 3;
        let hidden_size = 4;
        let output_size = 2;
        let net = TwoLayerNet::new(input_size, hidden_size, output_size);

        // 各勾配の形が対応パラメータと一致、各要素が is_finite() を確認
        let gradients = net.numerical_gradient(
            ndarray::Array2::<f32>::zeros((3, input_size)).view(),
            ndarray::Array2::<f32>::zeros((3, output_size)).view(),
        );
        assert_eq!(gradients.w1.shape(), &[input_size, hidden_size]);
        assert_eq!(gradients.b1.shape(), &[hidden_size]);
        assert_eq!(gradients.w2.shape(), &[hidden_size, output_size]);
        assert_eq!(gradients.b2.shape(), &[output_size]);
        assert!(gradients.w1.iter().all(|v| v.is_finite()));
        assert!(gradients.b1.iter().all(|v| v.is_finite()));
        assert!(gradients.w2.iter().all(|v| v.is_finite()));
        assert!(gradients.b2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_accuracy() {
        let input_size = 3;
        let hidden_size = 4;
        let output_size = 2;
        let net = TwoLayerNet::new(input_size, hidden_size, output_size);
        let mut t = ndarray::Array2::<f32>::zeros((3, output_size));
        t[[0, 1]] = 1.0;
        t[[1, 0]] = 1.0;
        t[[2, 1]] = 1.0;
        let accuracy = net.accuracy(
            ndarray::Array2::<f32>::zeros((3, input_size)).view(),
            t.view(),
        );
        assert!(accuracy >= 0.0 && accuracy <= 1.0 && accuracy.is_finite());
        println!("accuracy test result: {}", accuracy);
    }

    #[test]
    fn test_update() {
        let input_size = 3;
        let hidden_size = 4;
        let output_size = 2;
        let mut net = TwoLayerNet::new(input_size, hidden_size, output_size);
        let gradients = Gradients {
            w1: ndarray::Array2::<f32>::ones((input_size, hidden_size)),
            b1: ndarray::Array1::<f32>::ones(hidden_size),
            w2: ndarray::Array2::<f32>::ones((hidden_size, output_size)),
            b2: ndarray::Array1::<f32>::ones(output_size),
        };
        let learning_rate = 0.1;
        let w1_before = net.w1.clone();
        let b1_before = net.b1.clone();
        let w2_before = net.w2.clone();
        let b2_before = net.b2.clone();
        net.update(gradients, learning_rate);
        assert_eq!(
            net.w1,
            &(&w1_before - &(0.1 * &ndarray::Array2::<f32>::ones((input_size, hidden_size))))
        );
        assert_eq!(
            net.b1,
            &(&b1_before - &(0.1 * &ndarray::Array1::<f32>::ones(hidden_size)))
        );
        assert_eq!(
            net.w2,
            &(&w2_before - &(0.1 * &ndarray::Array2::<f32>::ones((hidden_size, output_size))))
        );
        assert_eq!(
            net.b2,
            &(&b2_before - &(0.1 * &ndarray::Array1::<f32>::ones(output_size)))
        );
    }

    use crate::mnist::{load_images, load_labels, to_one_hot};
    use ndarray::Axis;
    #[test]
    #[ignore = "遅い: 数値微分での学習。cargo test -- --ignored で実行"]
    fn train_mnist() {
        let images = load_images("dataset/train-images-idx3-ubyte"); // (60000, 784)
        let labels = load_labels("dataset/train-labels-idx1-ubyte"); // (60000,)
        let train_size = images.shape()[0];

        let batch_size = 10;
        let learning_rate = 0.1;
        let mut net = TwoLayerNet::new(784, 50, 10);

        let mut rng = rand::rng();
        for i in 0..5 {
            // まずは数イテレーションだけ
            let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();
            let x_batch = images.select(Axis(0), &idx);
            // ラベル: idx で番号を集めて one-hot に
            let batch_labels: Vec<u8> = idx.iter().map(|&j| labels[j]).collect();
            let t_batch = to_one_hot(&batch_labels, 10);

            let grads = net.numerical_gradient(x_batch.view(), t_batch.view());
            net.update(grads, learning_rate);

            let loss = net.loss(x_batch.view(), t_batch.view());
            println!("iter {i}: loss = {loss}");
        }
    }
}
