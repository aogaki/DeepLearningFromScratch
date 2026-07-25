use ndarray::array;
use vol3::functions::{softmax_cross_entropy_simple, softmax_simple};
use vol3::utils::approx_equal_arrayd;
use vol3::variable::Variable;

#[test]
fn test_log_gradient_check() {
    let x = Variable::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
    let y = x.ln();

    // forward
    assert!(approx_equal_arrayd(
        &y.data(),
        &array![[0.0, 2.0f32.ln()], [3.0f32.ln(), 4.0f32.ln()]].into_dyn(),
        1e-4
    ));

    // backward
    y.set_grad(Variable::new(array![[1.0, 1.0], [1.0, 1.0]].into_dyn()));
    y.backward(false, false);

    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[1.0, 0.5], [1.0 / 3.0, 0.25]].into_dyn(),
        1e-4
    ));
}

#[test]
fn test_clip_gradient_check() {
    let x = Variable::new(array![[0.0, 1.0, 2.0, 3.0]].into_dyn());
    let y = x.clip(1.0, 2.0);

    // forward
    assert!(approx_equal_arrayd(
        &y.data(),
        &array![[1.0, 1.0, 2.0, 2.0]].into_dyn(),
        1e-4
    ));

    // backward
    y.set_grad(Variable::new(array![[1.0, 1.0, 1.0, 1.0]].into_dyn()));
    y.backward(false, false);

    // x = 0.0 -> < min, mask 0
    // x = 1.0 -> >= min && <= max, mask 1
    // x = 2.0 -> >= min && <= max, mask 1
    // x = 3.0 -> > max, mask 0
    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[0.0, 1.0, 1.0, 0.0]].into_dyn(),
        1e-4
    ));
}

#[test]
fn test_gather_and_double_backprop() {
    let x = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    let t = vec![1, 2]; // 1st row: 2.0, 2nd row: 6.0

    let y = x.gather(&t);
    assert!(approx_equal_arrayd(
        &y.data(),
        &array![2.0, 6.0].into_dyn(),
        1e-4
    ));

    y.set_grad(Variable::new(array![10.0, 20.0].into_dyn()));
    y.backward(true, true); // retain_grad = true, create_graph = true

    let expected_gx = array![[0.0, 10.0, 0.0], [0.0, 0.0, 20.0]].into_dyn();

    assert!(approx_equal_arrayd(&x.grad().unwrap(), &expected_gx, 1e-4));

    // Test double backprop (ScatterAdd backward = Gather)
    let gx = x.grad_var().unwrap();
    gx.set_grad(Variable::new(
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    ));

    gx.backward(false, false);

    // The gradient of the gradient (ggy -> gx -> gy)
    // gy corresponds to `y`, its gradient is gathered from `ggy`
    let gy = y.grad_var().unwrap();
    // gy gradient should be gather(ggy, t) -> [2.0, 6.0]
    assert!(approx_equal_arrayd(
        &gy.grad().unwrap(),
        &array![2.0, 6.0].into_dyn(),
        1e-4
    ));
}

#[test]
fn test_softmax_simple() {
    let x = Variable::new(array![[0.0, 1.0, 2.0]].into_dyn());
    let y = softmax_simple(&x, 1);

    let sum_exp = 0.0f32.exp() + 1.0f32.exp() + 2.0f32.exp();
    let expected = array![[
        0.0f32.exp() / sum_exp,
        1.0f32.exp() / sum_exp,
        2.0f32.exp() / sum_exp
    ]]
    .into_dyn();

    assert!(approx_equal_arrayd(&y.data(), &expected, 1e-4));

    // Numerical diff check for softmax is complex, but we can do a simple backward
    y.set_grad(Variable::new(array![[1.0, 0.0, 0.0]].into_dyn()));
    y.backward(false, false);

    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[1, 3]);
}

#[test]
fn test_softmax_cross_entropy_simple() {
    let x = Variable::new(array![[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]].into_dyn());
    let t = vec![2, 1]; // target is max for both

    let loss = softmax_cross_entropy_simple(&x, &t);

    loss.backward(false, false);

    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    // For softmax cross entropy, gx is (p - t) / N
    let p = softmax_simple(&x, 1).data();
    let n = 2.0;

    let mut expected_gx = p.clone();
    let mut p_view = expected_gx
        .view_mut()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    p_view[[0, 2]] -= 1.0;
    p_view[[1, 1]] -= 1.0;

    expected_gx.mapv_inplace(|v| v / n);

    assert!(approx_equal_arrayd(&gx, &expected_gx, 1e-4));
}
