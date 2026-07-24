use vol3::variable::Variable;

#[test]
fn test_sin_higher_order_derivatives() {
    let x = Variable::from(1.0);
    let mut y = x.sin();

    // sin(1.0)
    let y_val = y.item();
    assert!((y_val - 1.0f32.sin()).abs() < 1e-4);

    let mut logs = vec![];

    for _ in 0..3 {
        x.cleargrad();
        y.backward(false, true);
        y = x.grad_var().unwrap();
        logs.push(y.item());
    }

    // 1st derivative: cos(1.0)
    assert!((logs[0] - 1.0f32.cos()).abs() < 1e-4);
    // 2nd derivative: -sin(1.0)
    assert!((logs[1] - (-1.0f32.sin())).abs() < 1e-4);
    // 3rd derivative: -cos(1.0)
    assert!((logs[2] - (-1.0f32.cos())).abs() < 1e-4);
}
