use vol3::variable::Variable;

fn f(x: &Variable) -> Variable {
    x.powf(4.0) - 2.0 * x.powf(2.0)
}

fn gx2(x: f32) -> f32 {
    12.0 * x.powi(2) - 4.0
}

fn main() {
    let x = Variable::from(2.0);
    let iters = 10;

    for i in 0..iters {
        println!("iteration {}: x = {:.6}", i, x.item());

        let y = f(&x);
        x.cleargrad();
        y.backward(false);

        let gx = x.grad().unwrap().into_iter().next().unwrap();
        let val_x = x.item();
        let gx2_val = gx2(val_x);

        let new_x = val_x - gx / gx2_val;
        x.set_data(ndarray::arr0(new_x).into_dyn());
    }

    println!("iteration {}: x = {:.6}", iters, x.item());
}
