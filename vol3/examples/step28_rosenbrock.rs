use vol3::variable::Variable;

fn rosenbrock(x0: &Variable, x1: &Variable) -> Variable {
    let t1 = x1 - &x0.powf(2.0);
    let term1 = 100.0 * t1.powf(2.0);
    let term2 = (x0 - 1.0).powf(2.0);
    term1 + term2
}

fn main() {
    let x0 = Variable::from(0.0);
    let x1 = Variable::from(2.0);
    let lr = 0.001;
    let iters = 1000;

    for i in 0..iters {
        let y = rosenbrock(&x0, &x1);

        x0.cleargrad();
        x1.cleargrad();
        y.backward(false, false);

        let v0 = x0.item();
        let v1 = x1.item();
        if i % 100 == 0 || i == iters - 1 {
            println!("iteration {}: x0 = {:.4}, x1 = {:.4}", i, v0, v1);
        }

        let gx0 = x0.grad().unwrap();
        let gx1 = x1.grad().unwrap();

        let new_x0 = x0.data() - (gx0 * lr);
        let new_x1 = x1.data() - (gx1 * lr);

        x0.set_data(new_x0);
        x1.set_data(new_x1);
    }
}
