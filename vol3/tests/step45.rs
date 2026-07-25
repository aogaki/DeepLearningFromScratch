use vol3::functions::mean_squared_error;
use vol3::layers::{Layer, MLP, TwoLayerNet};
use vol3::variable::Variable;

fn train_and_verify_model(model: &impl Layer) {
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

    // Use params() to inject deterministic weights in a model-agnostic way!
    // Since we know both TwoLayerNet(1,5,1) and MLP([1,5,1]) yield params in the order: [W1, b1, W2, b2]
    let params = model.params();
    params[0].set_data(w1_data.into_dyn());
    params[1].set_data(ndarray::Array::zeros((h,)).into_dyn());
    params[2].set_data(w2_data.into_dyn());
    params[3].set_data(ndarray::Array::zeros((1,)).into_dyn());

    let lr = 0.2;
    let iters = 10000;

    for _ in 0..iters {
        let y_pred = model.forward(&x);
        let loss = mean_squared_error(&y_pred, &y);

        model.cleargrads();
        loss.backward(false, false);

        for p in model.params() {
            let p_data = p.data();
            let grad = p.grad().unwrap();
            p.set_data(p_data - (grad * lr));
        }
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
fn test_models_polymorphism() {
    // Both models should train and converge using the exact same generic helper function
    let mut rng = rand::rng();
    let two_layer = TwoLayerNet::new(1, 5, 1, &mut rng);
    train_and_verify_model(&two_layer);

    let mlp = MLP::new(&[1, 5, 1], Variable::sigmoid, &mut rng);
    train_and_verify_model(&mlp);
}

#[test]
fn test_named_params() {
    let mut rng = rand::rng();
    let model = MLP::new(&[1, 5, 1], Variable::sigmoid, &mut rng);
    let params = model.named_params();
    assert_eq!(params.len(), 4);

    let mut names: Vec<String> = params.into_iter().map(|(name, _)| name).collect();
    names.sort();
    assert_eq!(names, vec!["l0/W", "l0/b", "l1/W", "l1/b"]);

    let model2 = TwoLayerNet::new(1, 5, 1, &mut rng);
    let params2 = model2.named_params();
    let mut names2: Vec<String> = params2.into_iter().map(|(name, _)| name).collect();
    names2.sort();
    assert_eq!(names2, vec!["l1/W", "l1/b", "l2/W", "l2/b"]);
}
