use vol3::functions::mean_squared_error;
use vol3::variable::Variable;

#[test]
fn test_linear_regression() {
    // ノイズなしの決定的な固定データ
    // x = [[0.0], [0.1], ..., [0.9]] (10x1)
    // y = 5.0 + 2.0 * x (10x1)
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    for i in 0..10 {
        let val = i as f32 * 0.1;
        x_vec.push(val);
        y_vec.push(5.0 + 2.0 * val);
    }

    let x_data = ndarray::Array::from_shape_vec((10, 1), x_vec).unwrap();
    let y_data = ndarray::Array::from_shape_vec((10, 1), y_vec).unwrap();

    let x = Variable::new(x_data.into_dyn());
    let y = Variable::new(y_data.into_dyn());

    let w = Variable::new(ndarray::Array::zeros((1, 1)).into_dyn());
    let b = Variable::new(ndarray::Array::zeros((1,)).into_dyn());

    let lr = 0.1;
    let iters = 100;

    for _ in 0..iters {
        let y_pred = x.matmul(&w) + &b;
        let loss = mean_squared_error(&y_pred, &y);

        w.cleargrad();
        b.cleargrad();
        loss.backward(false, false);

        let gw = w.grad().unwrap();
        let new_w = w.data() - (gw * lr);
        w.set_data(new_w);

        let gb = b.grad().unwrap();
        let new_b = b.data() - (gb * lr);
        b.set_data(new_b);
    }

    // ノイズがないので W -> 2.0, b -> 5.0 に収束するはず
    let final_w = w.item();
    let final_b = b.item();

    assert!(
        (final_w - 2.0).abs() < 1e-1,
        "W converged to {}, expected 2.0",
        final_w
    );
    assert!(
        (final_b - 5.0).abs() < 1e-1,
        "b converged to {}, expected 5.0",
        final_b
    );
}
