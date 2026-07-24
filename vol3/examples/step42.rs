use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use vol3::functions::mean_squared_error;
use vol3::variable::Variable;

fn main() {
    // Generate toy dataset
    // x = np.random.rand(100, 1)
    // y = 5 + 2 * x + np.random.rand(100, 1)
    let x_data = ndarray::Array::random((100, 1), Uniform::new(0.0f32, 1.0f32).unwrap());
    let noise = ndarray::Array::random((100, 1), Uniform::new(0.0f32, 1.0f32).unwrap());
    let y_data = 5.0 + 2.0 * &x_data + noise;

    let x = Variable::new(x_data.into_dyn());
    let y = Variable::new(y_data.into_dyn());

    let w = Variable::new(ndarray::Array::zeros((1, 1)).into_dyn());
    let b = Variable::new(ndarray::Array::zeros((1,)).into_dyn()); // broadcasting bias

    let lr = 0.1;
    let iters = 100;

    for _ in 0..iters {
        // predict (y_pred = x @ W + b)
        let y_pred = x.matmul(&w) + &b;

        // loss (mean squared error)
        let loss = mean_squared_error(&y_pred, &y);

        // clear gradients and backprop
        w.cleargrad();
        b.cleargrad();
        loss.backward(false, false);

        // update parameters
        // W.data -= lr * W.grad
        let gw = w.grad().unwrap();
        let new_w = w.data() - (gw * lr);
        w.set_data(new_w);

        // b.data -= lr * b.grad
        let gb = b.grad().unwrap();
        let new_b = b.data() - (gb * lr);
        b.set_data(new_b);

        println!("loss: {}", loss.item());
    }

    // print final parameters
    let final_w = w.data();
    let final_b = b.data();
    println!("W = {:?}", final_w);
    println!("b = {:?}", final_b);
}
