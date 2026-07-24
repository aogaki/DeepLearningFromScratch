use vol3::variable::Variable;

fn f(x: &Variable) -> Variable {
    x.powf(4.0) - 2.0 * x.powf(2.0)
}

fn gx2(x: f32) -> f32 {
    12.0 * x.powi(2) - 4.0
}

#[test]
fn test_newton_manual_optimization() {
    let x = Variable::from(2.0);
    let iters = 10;

    for _ in 0..iters {
        let y = f(&x);
        x.cleargrad();
        y.backward(false, false);

        let gx = x.grad().unwrap().into_iter().next().unwrap();
        let val_x = x.item();
        let gx2_val = gx2(val_x);

        let new_x = val_x - gx / gx2_val;
        x.set_data(ndarray::arr0(new_x).into_dyn());
    }

    // 正解は 1.0 または -1.0。初期値 2.0 からは 1.0 に収束するはず
    assert!((x.item() - 1.0).abs() < 1e-4);
}
