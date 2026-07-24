use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use vol3::functions::{linear, mean_squared_error};
use vol3::variable::Variable;

fn predict(x: &Variable, w1: &Variable, b1: &Variable, w2: &Variable, b2: &Variable) -> Variable {
    let y = linear(x, w1, Some(b1));
    let y = y.sigmoid();
    linear(&y, w2, Some(b2))
}

fn main() {
    let x_data = ndarray::Array::random((100, 1), Uniform::new(0.0f32, 1.0f32).unwrap());
    let noise = ndarray::Array::random((100, 1), Uniform::new(0.0f32, 1.0f32).unwrap());
    let mut y_data = x_data.mapv(|v| (2.0 * std::f32::consts::PI * v).sin());
    y_data = y_data + noise;

    let x = Variable::new(x_data.into_dyn());
    let y = Variable::new(y_data.into_dyn());

    let i = 1;
    let h = 10;
    let o = 1;

    let w1_data = ndarray::Array::random((i, h), Uniform::new(-1.0f32, 1.0f32).unwrap()) * 0.01;
    let b1_data = ndarray::Array::zeros((h,));
    let w2_data = ndarray::Array::random((h, o), Uniform::new(-1.0f32, 1.0f32).unwrap()) * 0.01;
    let b2_data = ndarray::Array::zeros((o,));

    let w1 = Variable::new(w1_data.into_dyn());
    let b1 = Variable::new(b1_data.into_dyn());
    let w2 = Variable::new(w2_data.into_dyn());
    let b2 = Variable::new(b2_data.into_dyn());

    let lr = 0.2;
    let iters = 10000;

    for i in 0..iters {
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

        if i % 1000 == 0 {
            println!("iter: {}, loss: {}", i, loss.item());
        }
    }
}
