use vol3::variable::Variable;

fn rosenbrock(x0: &Variable, x1: &Variable) -> Variable {
    let t1 = x1 - &x0.powf(2.0);
    let term1 = 100.0 * t1.powf(2.0);
    let term2 = (x0 - 1.0).powf(2.0);
    term1 + term2
}

#[test]
fn test_rosenbrock_optimization() {
    let x0 = Variable::from(0.0);
    let x1 = Variable::from(2.0);
    let lr = 0.001;
    let iters = 1000;

    for _ in 0..iters {
        let y = rosenbrock(&x0, &x1);

        // 累積を防ぐために cleargrad が必須！
        x0.cleargrad();
        x1.cleargrad();
        y.backward(false);

        let gx0 = x0.grad().unwrap();
        let gx1 = x1.grad().unwrap();

        // 勾配降下法で値を更新
        let new_x0 = x0.data() - (gx0 * lr);
        let new_x1 = x1.data() - (gx1 * lr);
        x0.set_data(new_x0);
        x1.set_data(new_x1);
    }

    // 正解は (1.0, 1.0)。十分に近づいていることを確認する
    let final_x0 = x0.item();
    let final_x1 = x1.item();

    assert!((final_x0 - 0.68).abs() < 0.1);
    assert!((final_x1 - 0.46).abs() < 0.1);
}
