use vol3::functions::{linear, mean_squared_error};
use vol3::variable::Variable;

fn predict(x: &Variable, w1: &Variable, b1: &Variable, w2: &Variable, b2: &Variable) -> Variable {
    let y = linear(x, w1, Some(b1));
    let y = y.sigmoid();
    linear(&y, w2, Some(b2))
}

#[test]
fn test_mlp_regression() {
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

    let i = 1;
    let h = 5;
    let o = 1;

    // Use deterministic initial weights
    let mut w1_vec = Vec::new();
    for j in 0..h {
        w1_vec.push((j as f32 - 2.0) * 0.1);
    }
    let w1_data = ndarray::Array::from_shape_vec((i, h), w1_vec).unwrap();
    let b1_data = ndarray::Array::zeros((h,));

    let mut w2_vec = Vec::new();
    for j in 0..h {
        w2_vec.push((j as f32 - 2.0) * -0.1);
    }
    let w2_data = ndarray::Array::from_shape_vec((h, o), w2_vec).unwrap();
    let b2_data = ndarray::Array::zeros((o,));

    let w1 = Variable::new(w1_data.into_dyn());
    let b1 = Variable::new(b1_data.into_dyn());
    let w2 = Variable::new(w2_data.into_dyn());
    let b2 = Variable::new(b2_data.into_dyn());

    let lr = 0.2;
    let iters = 10000;

    for _ in 0..iters {
        let y_pred = predict(&x, &w1, &b1, &w2, &b2);
        let loss = mean_squared_error(&y_pred, &y);

        w1.cleargrad();
        b1.cleargrad();
        w2.cleargrad();
        b2.cleargrad();
        loss.backward(false, false);

        let new_w1 = w1.data() - (w1.grad().unwrap() * lr);
        w1.set_data(new_w1);

        let new_b1 = b1.data() - (b1.grad().unwrap() * lr);
        b1.set_data(new_b1);

        let new_w2 = w2.data() - (w2.grad().unwrap() * lr);
        w2.set_data(new_w2);

        let new_b2 = b2.data() - (b2.grad().unwrap() * lr);
        b2.set_data(new_b2);
    }

    let y_pred = predict(&x, &w1, &b1, &w2, &b2);
    let final_loss = mean_squared_error(&y_pred, &y).item();

    // Since this is a deterministic test, we check if it trained successfully
    // past the linear plateau (loss ~0.197) to properly learn the non-linear sine wave
    assert!(
        final_loss < 0.01,
        "loss did not converge well, final loss: {}",
        final_loss
    );
}
