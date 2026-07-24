use vol3::variable::Variable;

fn f(x: &Variable) -> Variable {
    x.powf(4.0) - 2.0 * x.powf(2.0)
}

fn main() {
    let x = Variable::from(2.0);
    let iters = 10;

    for i in 0..iters {
        println!("iteration {}: x = {:.6}", i, x.item());

        let y = f(&x);
        x.cleargrad();
        y.backward(false, true); // create_graph=true

        let gx = x.grad_var().unwrap();
        x.cleargrad(); // 2階微分のために1階の勾配をクリア
        gx.backward(false, false); // 2階微分
        let gx2 = x.grad_var().unwrap();

        let val_x = x.item();
        let val_gx = gx.item();
        let val_gx2 = gx2.item();

        let new_x = val_x - val_gx / val_gx2;
        x.set_data(ndarray::arr0(new_x).into_dyn());
    }

    println!("iteration {}: x = {:.6}", iters, x.item());
}
