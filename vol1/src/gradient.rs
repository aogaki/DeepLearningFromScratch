use ndarray::{Array, Dimension};

/// 本 4.3.2「数値微分の例」中心差分による微分
pub fn numerical_differentiation(f: impl Fn(f32) -> f32, x: f32) -> f32 {
    let h = 1e-4;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// 本 4.4「勾配」各次元の偏微分をまとめた勾配ベクトル
pub fn numerical_gradient(
    f: impl Fn(ndarray::ArrayView1<f32>) -> f32,
    x: ndarray::ArrayView1<f32>,
) -> ndarray::Array1<f32> {
    let h = 1e-4;
    let mut grad = ndarray::Array1::<f32>::zeros(x.len());

    for i in 0..x.len() {
        let mut x_plus_h = x.to_owned();
        x_plus_h[i] += h;
        let mut x_minus_h = x.to_owned();
        x_minus_h[i] -= h;

        grad[i] = (f(x_plus_h.view()) - f(x_minus_h.view())) / (2.0 * h);
    }

    grad
}

/// 配列1つを要素ごとに中心差分する汎用ヘルパー(次元 D に非依存)
pub fn numerical_gradient_over<D: Dimension>(
    param: &Array<f32, D>,
    mut f: impl FnMut(&Array<f32, D>) -> f32,
) -> Array<f32, D> {
    let h = 1e-4;
    let mut grad = Array::<f32, D>::zeros(param.raw_dim());
    let mut p = param.to_owned();

    for i in 0..p.len() {
        let orig = p.as_slice().unwrap()[i];
        p.as_slice_mut().unwrap()[i] = orig + h;
        let loss_plus = f(&p);
        p.as_slice_mut().unwrap()[i] = orig - h;
        let loss_minus = f(&p);
        p.as_slice_mut().unwrap()[i] = orig;
        grad.as_slice_mut().unwrap()[i] = (loss_plus - loss_minus) / (2.0 * h);
    }
    grad
}

/// 本 4.4.1「勾配法」勾配降下法で step_num 回パラメータを更新
pub fn gradient_descent(
    f: impl Fn(ndarray::ArrayView1<f32>) -> f32,
    init_x: ndarray::Array1<f32>,
    lr: f32,
    step_num: usize,
) -> ndarray::Array1<f32> {
    let mut x = init_x;

    for _ in 0..step_num {
        let grad = numerical_gradient(&f, x.view());
        x.scaled_add(-lr, &grad);
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_differentiation() {
        let f = |x: f32| x * x;
        let result = numerical_differentiation(f, 2.0);
        println!("differential test result: {}", result);
        assert!((result - 4.0).abs() < 1e-2);
    }

    #[test]
    fn test_numerical_gradient() {
        let f = |x: ndarray::ArrayView1<f32>| x[0] * x[0] + x[1] * x[1];
        let x = ndarray::array![3.0, 4.0];
        let result = numerical_gradient(f, x.view());
        println!("gradient test result: {:?}", result);
        assert!((result[0] - 6.0).abs() < 2e-2);
        assert!((result[1] - 8.0).abs() < 2e-2);
    }

    #[test]
    fn test_gradient_descent() {
        let f = |x: ndarray::ArrayView1<f32>| x[0] * x[0] + x[1] * x[1];
        let init_x = ndarray::array![-3.0, 4.0];
        let lr = 0.1;
        let step_num = 100;
        let result = gradient_descent(f, init_x, lr, step_num);
        println!("gradient descent test result: {:?}", result);
        assert!((result[0] - 0.0).abs() < 1e-2);
        assert!((result[1] - 0.0).abs() < 1e-2);
    }
}
