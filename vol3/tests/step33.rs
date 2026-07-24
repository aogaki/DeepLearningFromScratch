use vol3::variable::Variable;

fn f(x: &Variable) -> Variable {
    x.powf(4.0) - 2.0 * x.powf(2.0)
}

#[test]
fn test_newton_automated_optimization() {
    let x = Variable::from(2.0);
    let iters = 10;

    for _ in 0..iters {
        let y = f(&x);
        x.cleargrad();
        // 2階微分を計算するため、1階微分の計算グラフを保存
        y.backward(false, true);

        let gx = x.grad_var().unwrap();
        // 1階微分のグラフから2階微分を計算するため、一旦 x の grad をクリア
        x.cleargrad();
        // 2階微分を計算 (グラフはこれ以上不要なので false)
        gx.backward(false, false);

        let gx2 = x.grad_var().unwrap();

        let val_x = x.item();
        let val_gx = gx.item();
        let val_gx2 = gx2.item();

        let new_x = val_x - val_gx / val_gx2;
        x.set_data(ndarray::arr0(new_x).into_dyn());
    }

    // 正解は 1.0 に収束するはず
    assert!((x.item() - 1.0).abs() < 1e-4);
}

#[test]
fn test_second_derivative() {
    let x = Variable::from(2.0);
    let y = f(&x);

    // create_graph=true で1階微分
    y.backward(false, true);
    let gx = x.grad_var().unwrap();

    // y = x^4 - 2x^2
    // y' = 4x^3 - 4x => 4(8) - 4(2) = 24
    assert_eq!(gx.item(), 24.0);

    // 2階微分のためにクリア
    x.cleargrad();

    // 2階微分
    gx.backward(false, false);
    let gx2 = x.grad_var().unwrap();

    // y'' = 12x^2 - 4 => 12(4) - 4 = 44
    assert_eq!(gx2.item(), 44.0);
}

#[test]
fn test_memory_leak_with_create_graph() {
    let weak_x;
    let weak_y;
    {
        let x = Variable::from(2.0);
        let y = x.powf(2.0);
        weak_x = x.downgrade();
        weak_y = y.downgrade();

        // create_graph=false なら循環は発生しない
        y.backward(false, false);
    }
    // スコープを抜ければ確実に解放される
    assert!(!weak_x.is_alive(), "x is leaked without create_graph");
    assert!(!weak_y.is_alive(), "y is leaked without create_graph");

    let weak_x2;
    let weak_y2;
    let weak_gx;
    {
        let x = Variable::from(2.0);
        let y = x.powf(2.0);
        weak_x2 = x.downgrade();
        weak_y2 = y.downgrade();

        // create_graph=true で循環発生
        y.backward(false, true);
        let gx = x.grad_var().unwrap();
        weak_gx = gx.downgrade();

        // cleargrad で意図的に循環を断ち切る
        x.cleargrad();
    }
    // cleargrad したので解放される
    assert!(!weak_x2.is_alive(), "x is leaked after cleargrad");
    assert!(!weak_y2.is_alive(), "y is leaked after cleargrad");
    assert!(!weak_gx.is_alive(), "gx is leaked after cleargrad");
}

#[test]
fn test_memory_leak_without_cleargrad() {
    let weak_x;
    {
        let x = Variable::from(2.0);
        let y = x.powf(2.0);
        weak_x = x.downgrade();

        // create_graph=true で循環発生
        y.backward(false, true);

        // cleargrad せずにスコープを抜ける
    }

    // cleargrad していないので、循環参照によりリークして生き残る！
    assert!(
        weak_x.is_alive(),
        "x should be leaked because of the cycle!"
    );
}
