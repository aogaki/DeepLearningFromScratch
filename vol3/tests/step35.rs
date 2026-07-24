use ndarray::array;
use vol3::utils::numerical_diff;
use vol3::variable::Variable;

#[test]
fn test_tanh_gradient_check() {
    let x_data = array![[0.5f32]].into_dyn();
    let x = Variable::new(x_data);
    let y = x.tanh();
    y.backward(false, false);

    let num_grad = numerical_diff(
        |x: &[ndarray::ArrayD<f32>]| Variable::new(x[0].clone()).tanh().data(),
        &x,
        1e-4,
    );
    let bp_grad = x.grad().unwrap();

    let diff = (&num_grad.data() - &bp_grad).mapv(|a| a.abs()).sum();
    assert!(
        diff < 1e-4,
        "Gradient check failed for Tanh: diff = {}",
        diff
    );
}
