pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn identity_function(x: f32) -> f32 {
    x
}

pub fn forward(x: ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    use ndarray::Array2;

    let w1 = Array2::<f32>::from_shape_vec((2, 3), vec![0.1, 0.3, 0.5, 0.2, 0.4, 0.6]).unwrap();
    let b1 = Array2::<f32>::from_shape_vec((1, 3), vec![0.1, 0.2, 0.3]).unwrap();
    let w2 = Array2::<f32>::from_shape_vec((3, 2), vec![0.1, 0.4, 0.2, 0.5, 0.3, 0.6]).unwrap();
    let b2 = Array2::<f32>::from_shape_vec((1, 2), vec![0.1, 0.2]).unwrap();
    let w3 = Array2::<f32>::from_shape_vec((2, 2), vec![0.1, 0.3, 0.2, 0.4]).unwrap();
    let b3 = Array2::<f32>::from_shape_vec((1, 2), vec![0.1, 0.2]).unwrap();

    let a1 = x.dot(&w1) + b1;
    let z1 = a1.mapv(sigmoid);
    let a2 = z1.dot(&w2) + b2;
    let z2 = a2.mapv(sigmoid);
    let a3 = z2.dot(&w3) + b3;
    a3.mapv(identity_function)
}

pub fn softmax(x: ndarray::Array1<f32>) -> ndarray::Array1<f32> {
    let c = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_x = x.mapv(|v| (v - c).exp());
    let sum_exp_x = exp_x.sum();
    exp_x / sum_exp_x
}

pub struct Network {
    w1: ndarray::Array2<f32>,
    b1: ndarray::Array1<f32>,
    w2: ndarray::Array2<f32>,
    b2: ndarray::Array1<f32>,
    w3: ndarray::Array2<f32>,
    b3: ndarray::Array1<f32>,
}

impl Network {
    pub fn load() -> Self {
        let weights_path = "dataset/weights/";
        let w1 = ndarray_npy::read_npy(format!("{}W1.npy", weights_path)).unwrap();
        let b1 = ndarray_npy::read_npy(format!("{}b1.npy", weights_path)).unwrap();
        let w2 = ndarray_npy::read_npy(format!("{}W2.npy", weights_path)).unwrap();
        let b2 = ndarray_npy::read_npy(format!("{}b2.npy", weights_path)).unwrap();
        let w3 = ndarray_npy::read_npy(format!("{}W3.npy", weights_path)).unwrap();
        let b3 = ndarray_npy::read_npy(format!("{}b3.npy", weights_path)).unwrap();

        Network {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        }
    }

    pub fn predict(&self, x: ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        let a1 = x.dot(&self.w1) + &self.b1;
        let z1 = a1.mapv(sigmoid);
        let a2 = z1.dot(&self.w2) + &self.b2;
        let z2 = a2.mapv(sigmoid);
        let a3 = z2.dot(&self.w3) + &self.b3;

        let mut result = ndarray::Array2::<f32>::zeros(a3.raw_dim());
        for (i, row) in a3.outer_iter().enumerate() {
            let softmax_row = softmax(row.to_owned());
            result.row_mut(i).assign(&softmax_row);
        }
        result
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::mnist::{load_images, load_labels};

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn sigmoid_test() {
        let epsilon = 1e-6;
        assert!(approx_eq(sigmoid(0.0), 0.5, epsilon));
        assert!(approx_eq(sigmoid(1.0), 0.73105857863, epsilon));
        assert!(approx_eq(sigmoid(-1.0), 0.26894142137, epsilon));
    }

    #[test]
    fn relu_test() {
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
        assert_eq!(relu(-1.0), 0.0);
    }

    #[test]
    fn matrix_product_test() {
        use ndarray::Array2;

        let a = Array2::<f32>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b =
            Array2::<f32>::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = a.dot(&b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c[[0, 0]], 58.0);
        assert_eq!(c[[0, 1]], 64.0);
        assert_eq!(c[[1, 0]], 139.0);
        assert_eq!(c[[1, 1]], 154.0);
    }

    #[test]
    fn forward_test() {
        use ndarray::Array2;
        let x = Array2::<f32>::from_shape_vec((1, 2), vec![1.0, 0.5]).unwrap();
        let y = forward(x);
        assert_eq!(y.shape(), &[1, 2]);
        let epsilon = 1e-6;
        assert!(approx_eq(y[[0, 0]], 0.3168271, epsilon));
        assert!(approx_eq(y[[0, 1]], 0.6962791, epsilon));
    }

    #[test]
    fn softmax_test() {
        use ndarray::Array1;
        let x = Array1::<f32>::from_vec(vec![0.3, 2.9, 4.0]);
        let y = softmax(x);
        let expected = Array1::<f32>::from_vec(vec![0.01821127, 0.24519183, 0.73659694]);
        let epsilon = 1e-6;
        for (yi, expected_i) in y.iter().zip(expected.iter()) {
            assert!(approx_eq(*yi, *expected_i, epsilon));
        }

        assert!((y.sum() - 1.0).abs() < epsilon);

        let big_numbers = Array1::<f32>::from_vec(vec![1010.0, 1000.0, 990.0]);
        let y_big = softmax(big_numbers);
        assert!(y_big.iter().all(|v| v.is_finite()));
        assert!((y_big.sum() - 1.0).abs() < epsilon);
    }

    #[test]
    fn network_load_test() {
        let network = Network::load();
        assert_eq!(network.w1.shape(), &[784, 50]);
        assert_eq!(network.b1.shape(), &[50]);
    }

    #[test]
    fn network_predict_test() {
        let network = Network::load();
        let image_path = "dataset/t10k-images-idx3-ubyte";
        let label_path = "dataset/t10k-labels-idx1-ubyte";
        let images = load_images(image_path);
        let labels = load_labels(label_path);

        let prediction = network.predict(images);

        // 各行の argmax（予測数字） を取り、ラベルと突き合わせて一致数 ÷ 10000。
        let correct = prediction
            .outer_iter()
            .zip(labels.iter())
            .map(|(pred_row, &label)| {
                let predicted_label = pred_row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(idx, _)| idx)
                    .unwrap();
                predicted_label as u8 == label
            })
            .filter(|&ok| ok)
            .count();
        let accuracy = correct as f64 / labels.len() as f64;
        println!("Accuracy: {:.2}%", accuracy * 100.0);
    }
}
