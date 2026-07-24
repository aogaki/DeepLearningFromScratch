use ndarray::array;
use vol3::variable::Variable;

#[test]
fn test_matmul() {
    let x = Variable::new(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn());
    let w = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());

    let y = x.matmul(&w);

    assert_eq!(y.shape(), vec![3, 3]);
    assert_eq!(
        y.data(),
        array![[9.0, 12.0, 15.0], [19.0, 26.0, 33.0], [29.0, 40.0, 51.0]].into_dyn()
    );

    // gy のセット (転置しても不変な ones を避け、全要素異なる値を設定)
    let gy = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into_dyn());
    y.set_grad(gy);
    y.backward(false, false);

    let gx = x.grad().unwrap();
    let gw = w.grad().unwrap();

    // gx = gy @ w^T
    assert_eq!(gx.shape(), &[3, 2]);
    assert_eq!(
        gx,
        array![[14.0, 32.0], [32.0, 77.0], [50.0, 122.0]].into_dyn()
    );

    // gw = x^T @ gy
    assert_eq!(gw.shape(), &[2, 3]);
    assert_eq!(
        gw,
        array![[48.0, 57.0, 66.0], [60.0, 72.0, 84.0]].into_dyn()
    );
}
