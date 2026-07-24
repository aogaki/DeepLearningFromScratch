use vol3::utils::{EPSILON_FOR_DIFF, approx_equal_arrayd, numerical_diff};
use vol3::variable::Variable;

#[test]
fn test_cos_gradient_check() {
    let pi = std::f32::consts::PI;
    let x = Variable::from(pi / 4.0);

    let y = x.cos();
    y.backward(false, false);

    let expected_grad = x.grad().unwrap();

    // analytical grad is -sin(pi/4)
    let sin_val = ndarray::arr0(-std::f32::consts::FRAC_PI_4.sin()).into_dyn();
    assert!(approx_equal_arrayd(&expected_grad, &sin_val, 1e-5));

    let num_grad = numerical_diff(
        |x: &[ndarray::ArrayD<f32>]| Variable::new(x[0].clone()).cos().data(),
        &x,
        EPSILON_FOR_DIFF,
    );
    assert!(approx_equal_arrayd(&expected_grad, &num_grad.data(), 1e-3));
}
