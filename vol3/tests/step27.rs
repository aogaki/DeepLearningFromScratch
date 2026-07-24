use vol3::utils::{EPSILON_FOR_DIFF, approx_equal_arrayd, numerical_diff};
use vol3::variable::Variable;

fn my_sin(x: &Variable, threshold: f32) -> Variable {
    let mut y = Variable::from(0.0);
    let mut c = 1.0f32;

    for i in 0..10000 {
        if i > 0 {
            c = -c / ((2 * i) * (2 * i + 1)) as f32;
        }

        let t = x.powf((2 * i + 1) as f32) * &Variable::from(c);
        y = &y + &t;

        let t_val = t.item();
        if t_val.abs() < threshold {
            break;
        }
    }
    y
}

#[test]
fn test_my_sin_gradient_check() {
    let pi = std::f32::consts::PI;
    let x = Variable::from(pi / 4.0);

    let y = my_sin(&x, 1e-4);
    y.backward(false, false);

    let expected_grad = x.grad().unwrap();
    let num_grad = numerical_diff(
        |x: &[ndarray::ArrayD<f32>]| my_sin(&Variable::new(x[0].clone()), 1e-4).data(),
        &x,
        EPSILON_FOR_DIFF,
    );

    assert!(approx_equal_arrayd(&expected_grad, &num_grad.data(), 1e-3));
}

#[test]
fn test_sin_gradient_check() {
    let pi = std::f32::consts::PI;
    let x = Variable::from(pi / 4.0);

    let y = x.sin();
    y.backward(false, false);

    let expected_grad = x.grad().unwrap();

    // analytical grad is cos(pi/4)
    let cos_val = ndarray::arr0(std::f32::consts::FRAC_PI_4.cos()).into_dyn();
    assert!(approx_equal_arrayd(&expected_grad, &cos_val, 1e-5));

    let num_grad = numerical_diff(
        |x: &[ndarray::ArrayD<f32>]| Variable::new(x[0].clone()).sin().data(),
        &x,
        EPSILON_FOR_DIFF,
    );
    assert!(approx_equal_arrayd(&expected_grad, &num_grad.data(), 1e-3));
}
