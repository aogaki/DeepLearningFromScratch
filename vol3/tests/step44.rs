use vol3::functions::mean_squared_error;
use vol3::layers::{Layer, Linear};
use vol3::variable::Variable;

#[test]
fn test_layer_mlp() {
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    for i in 0..10 {
        let val = i as f32 * 0.1;
        x_vec.push(val);
        y_vec.push((2.0 * std::f32::consts::PI * val).sin());
    }

    let x_data = ndarray::Array::from_shape_vec((10, 1), x_vec).unwrap();
    let y_data = ndarray::Array::from_shape_vec((10, 1), y_vec).unwrap();

    let x = Variable::new(x_data.into_dyn());
    let y = Variable::new(y_data.into_dyn());
    let mut rng = rand::rng();
    let l1 = Linear::new(1, 5, false, &mut rng);
    let l2 = Linear::new(5, 1, false, &mut rng);

    // Deterministic weights (replace random ones)
    let h = 5;
    let mut w1_vec = Vec::new();
    for j in 0..h {
        w1_vec.push((j as f32 - 2.0) * 0.1);
    }
    let w1_data = ndarray::Array::from_shape_vec((1, h), w1_vec).unwrap();
    l1.w.set_data(w1_data.into_dyn());
    l1.b.as_ref()
        .unwrap()
        .set_data(ndarray::Array::zeros((h,)).into_dyn());

    let mut w2_vec = Vec::new();
    for j in 0..h {
        w2_vec.push((j as f32 - 2.0) * -0.1);
    }
    let w2_data = ndarray::Array::from_shape_vec((h, 1), w2_vec).unwrap();
    l2.w.set_data(w2_data.into_dyn());
    l2.b.as_ref()
        .unwrap()
        .set_data(ndarray::Array::zeros((1,)).into_dyn());

    let predict = |x: &Variable| {
        let h_var = l1.forward(x);
        let h_sig = h_var.sigmoid();
        l2.forward(&h_sig)
    };

    let lr = 0.2;
    let iters = 10000;

    for _ in 0..iters {
        let y_pred = predict(&x);
        let loss = mean_squared_error(&y_pred, &y);

        l1.cleargrads();
        l2.cleargrads();
        loss.backward(false, false);

        for l in [&l1, &l2] {
            for p in l.params() {
                let p_data = p.data();
                let grad = p.grad().unwrap();
                p.set_data(p_data - (grad * lr));
            }
        }
    }

    let y_pred = predict(&x);
    let final_loss = mean_squared_error(&y_pred, &y).item();

    // Past the linear plateau (~0.197) to proper sine learning
    assert!(
        final_loss < 0.01,
        "loss did not converge well, final loss: {}",
        final_loss
    );
}
