use ndarray::{ArrayView1, ArrayView2};

/// 本 4.2.1「2乗和誤差」
pub fn sum_of_squared_error(y: ArrayView1<f32>, t: ArrayView1<f32>) -> f32 {
    let sum: f32 = y
        .iter()
        .zip(t.iter())
        .map(|(yi, ti)| (yi - ti).powi(2))
        .sum();
    sum / 2.0
}

/// 本 4.2.2「交差エントロピー誤差」log(0) 対策に微小値 delta を加算
pub fn cross_entropy_error(y: ArrayView1<f32>, t: ArrayView1<f32>) -> f32 {
    let delta = 1e-7;
    let sum: f32 = y
        .iter()
        .zip(t.iter())
        .map(|(yi, ti)| ti * (yi + delta).ln())
        .sum();
    -sum
}

/// 本 4.2.4「【バッチ対応版】交差エントロピー誤差の実装」バッチ平均
pub fn batch_cross_entropy_error(y: ArrayView2<f32>, t: ArrayView2<f32>) -> f32 {
    let batch_size = y.shape()[0] as f32;
    let sum: f32 = y
        .outer_iter()
        .zip(t.outer_iter())
        .map(|(yi, ti)| cross_entropy_error(yi, ti))
        .sum();
    sum / batch_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_of_squared_error() {
        let t = ndarray::array![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let answer_2 = ndarray::array![0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0];
        let result = sum_of_squared_error(answer_2.view(), t.view());
        println!("result (2 is high probability): {}", result);
        assert!((result - 0.0975).abs() < 1e-6);

        let answer_7 = ndarray::array![0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0];
        let result = sum_of_squared_error(answer_7.view(), t.view());
        println!("result (7 is high probability): {}", result);
        assert!((result - 0.5975).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_error() {
        let t = ndarray::array![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let answer_2 = ndarray::array![0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0];
        let result = cross_entropy_error(answer_2.view(), t.view());
        println!("result (2 is high probability): {}", result);
        assert!((result - 0.510825457099338).abs() < 1e-6);

        let answer_7 = ndarray::array![0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0];
        let result = cross_entropy_error(answer_7.view(), t.view());
        println!("result (7 is high probability): {}", result);
        assert!((result - 2.3025840929945458).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cross_entropy_error() {
        let t = ndarray::array![
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // first sample
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // second sample
        ];
        let answer = ndarray::array![
            [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], // first sample
            [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0], // second sample
        ];
        let result = batch_cross_entropy_error(answer.view(), t.view());
        println!("result (batch): {}", result);
        assert!((result - 1.406704775047942).abs() < 1e-6);
    }
}
