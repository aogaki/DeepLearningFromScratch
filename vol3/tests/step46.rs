use vol3::functions::mean_squared_error;
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{MomentumSGD, Optimizer, SGD};
use vol3::variable::Variable;

fn train_and_verify_model_with_optimizer(model: &impl Layer, optimizer: &mut impl Optimizer) {
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

    // Deterministic weights
    let h = 5;
    let mut w1_vec = Vec::new();
    for j in 0..h {
        w1_vec.push((j as f32 - 2.0) * 0.1);
    }
    let w1_data = ndarray::Array::from_shape_vec((1, h), w1_vec).unwrap();

    let mut w2_vec = Vec::new();
    for j in 0..h {
        w2_vec.push((j as f32 - 2.0) * -0.1);
    }
    let w2_data = ndarray::Array::from_shape_vec((h, 1), w2_vec).unwrap();

    let params = model.params();
    params[0].set_data(w1_data.into_dyn());
    params[1].set_data(ndarray::Array::zeros((h,)).into_dyn());
    params[2].set_data(w2_data.into_dyn());
    params[3].set_data(ndarray::Array::zeros((1,)).into_dyn());

    let iters = 10000;

    for _ in 0..iters {
        let y_pred = model.forward(&x);
        let loss = mean_squared_error(&y_pred, &y);

        model.cleargrads();
        loss.backward(false, false);

        // Optimizer handles the update loop completely!
        optimizer.update();
    }

    let y_pred = model.forward(&x);
    let final_loss = mean_squared_error(&y_pred, &y).item();

    assert!(
        final_loss < 0.01,
        "loss did not converge well, final loss: {}",
        final_loss
    );
}

#[test]
fn test_sgd_optimizer() {
    let mut rng = rand::rng();
    let mlp = MLP::new(&[1, 5, 1], Variable::sigmoid, &mut rng);
    let mut sgd = SGD::new(0.2);
    sgd.setup(&mlp);
    train_and_verify_model_with_optimizer(&mlp, &mut sgd);
}

#[test]
fn test_momentum_sgd_optimizer() {
    let mut rng = rand::rng();
    let mlp = MLP::new(&[1, 5, 1], Variable::sigmoid, &mut rng);
    let mut momentum = MomentumSGD::new(0.2, 0.9);
    momentum.setup(&mlp);
    train_and_verify_model_with_optimizer(&mlp, &mut momentum);
}
