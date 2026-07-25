use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use vol3::functions::mean_squared_error;
use vol3::layers::{Layer, Linear};
use vol3::variable::Variable;

fn main() {
    let x_data = ndarray::Array::random((100, 1), Uniform::new(0.0f32, 1.0f32).unwrap());
    let noise = ndarray::Array::random((100, 1), Uniform::new(0.0f32, 1.0f32).unwrap());
    let mut y_data = x_data.mapv(|v| (2.0 * std::f32::consts::PI * v).sin());
    y_data = y_data + noise;

    let x = Variable::new(x_data.into_dyn());
    let y = Variable::new(y_data.into_dyn());

    let mut rng = rand::rng();
    let l1 = Linear::new(1, 10, false, &mut rng);
    let l2 = Linear::new(10, 1, false, &mut rng);

    let predict = |x: &Variable| {
        let h = l1.forward(x);
        let h = h.sigmoid();
        l2.forward(&h)
    };

    let lr = 0.2;
    let iters = 10000;

    for i in 0..iters {
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

        if i % 1000 == 0 {
            println!("iter: {}, loss: {}", i, loss.item());
        }
    }
}
