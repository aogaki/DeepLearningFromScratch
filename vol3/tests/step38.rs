use ndarray::array;
use vol3::utils::approx_equal_arrayd;
use vol3::variable::Variable;

#[test]
fn test_reshape() {
    let x = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    let y = x.reshape(&[6]);

    assert_eq!(y.shape(), vec![6]);
    assert_eq!(y.data(), array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_dyn());

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
fn test_transpose() {
    let x = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    let y = x.transpose();

    assert_eq!(y.shape(), vec![3, 2]);
    assert_eq!(
        y.data(),
        array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]].into_dyn()
    );

    // 逆伝播のテスト。全要素相異なる値を勾配としてセットして経路を確認
    let gy_data = array![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]].into_dyn();
    y.set_grad(Variable::new(gy_data));
    y.backward(false, false);

    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    assert_eq!(
        gx,
        array![[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]].into_dyn()
    );
}

#[test]
fn test_transpose_and_reshape() {
    // 転置後の配列は非標準レイアウトになるため、そのあとの reshape が
    // 正しく標準レイアウトへ変換しているかをテストする
    let x = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    let y = x.transpose();
    let z = y.reshape(&[6]);

    assert_eq!(z.shape(), vec![6]);
    assert_eq!(z.data(), array![1.0, 4.0, 2.0, 5.0, 3.0, 6.0].into_dyn());

    z.backward(false, false);
    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    assert!(approx_equal_arrayd(
        &gx,
        &array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn(),
        1e-5
    ));
}

#[test]
fn test_transpose_layout() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
    let y = x.t();

    // `.to_owned()` preserves the strides, resulting in non-standard layout
    let z1 = y.to_owned();
    assert!(!z1.is_standard_layout());

    // `.as_standard_layout().into_owned()` forces standard layout
    let z2 = y.as_standard_layout().into_owned();
    assert!(z2.is_standard_layout());
}
