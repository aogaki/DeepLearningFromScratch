use ndarray::array;
use vol3::utils::approx_equal_arrayd;
use vol3::variable::Variable;

#[test]
fn test_sum() {
    let x = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    let y = x.sum(); // sum all

    assert_eq!(y.shape(), vec![]);
    assert_eq!(y.item(), 21.0);

    y.backward(false, false);
    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    assert!(approx_equal_arrayd(
        &gx,
        &array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn(),
        1e-5
    ));
}

#[test]
fn test_sum_axis() {
    let x = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    let y = x.sum_axis(0);

    assert_eq!(y.shape(), vec![3]);
    assert_eq!(y.data(), array![5.0, 7.0, 9.0].into_dyn());

    let gy = Variable::new(array![10.0, 20.0, 30.0].into_dyn());
    y.set_grad(gy);
    y.backward(false, false);

    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    assert_eq!(
        gx,
        array![[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]].into_dyn()
    );
}

#[test]
fn test_broadcast_to() {
    let x = Variable::new(array![1.0, 2.0, 3.0].into_dyn());
    let y = x.broadcast_to(&[2, 3]);

    assert_eq!(y.shape(), vec![2, 3]);
    assert_eq!(
        y.data(),
        array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]].into_dyn()
    );

    let gy = Variable::new(array![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn());
    y.set_grad(gy);
    y.backward(false, false);

    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[3]);
    assert_eq!(gx, array![50.0, 70.0, 90.0].into_dyn());
}

#[test]
fn test_sum_to() {
    let x = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    let y = x.sum_to(&[3]);

    assert_eq!(y.shape(), vec![3]);
    assert_eq!(y.data(), array![5.0, 7.0, 9.0].into_dyn());

    let gy = Variable::new(array![10.0, 20.0, 30.0].into_dyn());
    y.set_grad(gy);
    y.backward(false, false);

    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    assert_eq!(
        gx,
        array![[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]].into_dyn()
    );
}

#[test]
fn test_add_broadcast() {
    let x0 = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    let x1 = Variable::new(array![10.0, 20.0, 30.0].into_dyn());

    let y = x0.add(&x1);

    assert_eq!(y.shape(), vec![2, 3]);
    assert_eq!(
        y.data(),
        array![[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]].into_dyn()
    );

    // テンソル同士のブロードキャスト足し算の逆伝播
    // gy に各要素異なる値を設定
    let gy = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    y.set_grad(gy);
    y.backward(false, false);

    let gx0 = x0.grad().unwrap();
    let gx1 = x1.grad().unwrap();

    // x0 は (2, 3) なので gy がそのまま伝播
    assert_eq!(gx0.shape(), &[2, 3]);
    assert_eq!(gx0, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());

    // x1 は (3,) なので gy が軸0で和を取られて伝播
    assert_eq!(gx1.shape(), &[3]);
    assert_eq!(gx1, array![5.0, 7.0, 9.0].into_dyn());
}
