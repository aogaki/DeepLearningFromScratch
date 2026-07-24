use ndarray::ArrayD;
use ndarray::array;
use vol3::*;

const EPSILON_FOR_DIFF: f32 = 5e-3;

// ステップ1: Variable がデータを保持する
#[test]
fn test_variable_creation() {
    let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data = ArrayD::from_shape_vec(vec![2, 3], original_data.clone()).unwrap();
    let variable = Variable::new(data);
    assert_eq!(
        variable.data(),
        Variable::new(ArrayD::from_shape_vec(vec![2, 3], original_data).unwrap()).data()
    );
}

// ステップ2: Square の順伝播(呼び方はステップ9のメソッド記法)
#[test]
fn test_square_function() {
    let original_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let variable = Variable::new(original_data.clone().into_dyn());
    let result_variable = variable.square();
    let expected_data = array![[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]];
    assert!(approx_equal_arrayd(
        &result_variable.data(),
        &expected_data.clone().into_dyn(),
        1e-6
    ));
}

// ステップ3: Exp 単体と、Square→Exp→Square の合成
#[test]
fn test_exp_function() {
    let original_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let variable = Variable::new(original_data.clone().into_dyn());
    let result_variable = variable.exp();
    let expected_data = array![
        [1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()],
        [4.0f32.exp(), 5.0f32.exp(), 6.0f32.exp()]
    ];
    assert!(approx_equal_arrayd(
        &result_variable.data(),
        &expected_data.clone().into_dyn(),
        1e-6
    ));

    let final_variable = variable.square().exp().square();
    let expected_final_data = array![
        [
            1.0f32.powi(2).exp().powi(2),
            2.0f32.powi(2).exp().powi(2),
            3.0f32.powi(2).exp().powi(2)
        ],
        [
            4.0f32.powi(2).exp().powi(2),
            5.0f32.powi(2).exp().powi(2),
            6.0f32.powi(2).exp().powi(2)
        ]
    ];
    assert!(approx_equal_arrayd(
        &final_variable.data(),
        &expected_final_data.clone().into_dyn(),
        1e-6
    ));
}

// ステップ4: 数値微分。単一関数(解析解 2x)と、合成関数(4x·e^{2x²}、本と同じ x=0.5)。
// 合成側は値が急伸するので小さな x のみ(f32 の絶対誤差比較が成立する範囲)
#[test]
fn test_numerical_diff() {
    let original_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let variable = Variable::new(original_data.clone().into_dyn());
    let diff_variable = numerical_diff(Square, &variable, EPSILON_FOR_DIFF);
    let expected_diff_data = array![[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]];
    assert!(approx_equal_arrayd(
        &diff_variable.data(),
        &expected_diff_data.clone().into_dyn(),
        1e-3
    ));

    let simple_data = array![[0.5f32]];
    let variable = Variable::new(simple_data.clone().into_dyn());

    let f = |xs: &[ArrayD<f32>]| {
        let y1 = Square.forward(xs);
        let y2 = Exp.forward(&[y1]);
        Square.forward(&[y2])
    };

    let diff_variable = numerical_diff(f, &variable, EPSILON_FOR_DIFF);
    let expected_diff_data = simple_data.mapv(|x| 4.0 * x * (2.0 * x.powi(2)).exp());
    assert!(approx_equal_arrayd(
        &diff_variable.data(),
        &expected_diff_data.clone().into_dyn(),
        1e-3
    ));
}

// ステップ7〜8: y.backward() 一発でグラフを遡る(ループ実装)。
// grad の 1 初期化(ステップ9の先取り)もここで効いている
#[test]
fn test_auto_backward() {
    let x_data = array![[0.5f32]].into_dyn();
    let x = Variable::new(x_data);

    let y = x.square().exp().square();

    // ステップ7の自動逆伝播！
    y.backward(false);

    let expected_grad = array![[3.2974426f32]].into_dyn();
    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &expected_grad,
        1e-5
    ));
}

// ステップ10: 勾配チェック(backprop の解析勾配 vs 数値微分)。
// 以後のステップで backward を書き換えても、この2本が正しさの安全網になる
#[test]
fn test_gradient_check() {
    let x = Variable::new(array![[0.5f32]].into_dyn());

    let y = x.square();
    y.backward(false);
    let analytical_grad = x.grad().unwrap();

    let num_grad_var = numerical_diff(Square, &x, EPSILON_FOR_DIFF);
    let numerical_grad = num_grad_var.data();

    assert!(approx_equal_arrayd(&analytical_grad, &numerical_grad, 1e-3));
}

// ステップ10: 合成関数(Square→Exp→Square)版の勾配チェック
#[test]
fn test_gradient_check_composite() {
    let x = Variable::new(array![[0.5f32]].into_dyn());

    let y = x.square().exp().square();
    y.backward(false);
    let analytical_grad = x.grad().unwrap();

    let f = |xs: &[ArrayD<f32>]| {
        let y1 = Square.forward(xs);
        let y2 = Exp.forward(&[y1]);
        Square.forward(&[y2])
    };
    let num_grad_var = numerical_diff(f, &x, EPSILON_FOR_DIFF);
    let numerical_grad = num_grad_var.data();

    assert!(approx_equal_arrayd(&analytical_grad, &numerical_grad, 1e-3));
}

#[test]
fn test_add_function() {
    let x0 = Variable::new(array![[2.0f32]].into_dyn());
    let x1 = Variable::new(array![[3.0f32]].into_dyn());
    let y = x0.add(&x1);

    assert!(approx_equal_arrayd(
        &y.data(),
        &array![[5.0f32]].into_dyn(),
        1e-5
    ));
}

#[test]
fn test_add_backward() {
    let x = Variable::new(array![[2.0f32]].into_dyn());
    let y = Variable::new(array![[3.0f32]].into_dyn());

    // z = x^2 + y^2
    let z = x.square().add(&y.square());
    z.backward(false);

    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[4.0f32]].into_dyn(),
        1e-5
    ));
    assert!(approx_equal_arrayd(
        &y.grad().unwrap(),
        &array![[6.0f32]].into_dyn(),
        1e-5
    ));
}

#[test]
fn test_add_same_variable() {
    let x = Variable::new(array![[3.0f32]].into_dyn());
    let y = x.add(&x);
    y.backward(false);

    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[2.0f32]].into_dyn(),
        1e-5
    ));

    x.cleargrad();

    let y2 = x.add(&x).add(&x);
    y2.backward(false);

    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[3.0f32]].into_dyn(),
        1e-5
    ));
}

// 本 ステップ16: 複雑な計算グラフ(実装編)
// 世代 (トポロジカルソート)の実装により、同じ変数が複数回 backward されず、
// 正しい値(64.0)になることを確認する。
#[test]
fn test_complex_graph_step16() {
    let x = Variable::new(array![[2.0f32]].into_dyn());
    let a = x.square();
    let y = a.square().add(&a.square());

    y.backward(false);

    // 正解は dy/dx = 8x^3 = 64.0。世代管理により正しく計算される。
    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[64.0f32]].into_dyn(),
        1e-5
    ));
}

#[test]
fn test_retain_grad() {
    let x = Variable::new(array![[2.0f32]].into_dyn());
    let a = x.square();
    let y = a.square();

    y.backward(false);
    assert!(a.grad().is_none(), "中間変数の勾配は破棄されるべき");
    assert!(x.grad().is_some(), "葉変数の勾配は保持されるべき");

    x.cleargrad();

    let a2 = x.square();
    let y2 = a2.square();
    y2.backward(true);
    assert!(
        a2.grad().is_some(),
        "retain_grad=true時は中間変数も保持されるべき"
    );
}

// 本 ステップ19: 変数を使いやすくする(名前・プロパティ委譲・表示)
#[test]
fn test_variable_properties_step19() {
    let x = Variable::new(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    x.set_name("x");

    assert_eq!(x.name().unwrap(), "x");
    assert_eq!(x.shape(), vec![2, 3]);
    assert_eq!(x.ndim(), 2);
    assert_eq!(x.size(), 6); // ndarray の len() は全要素数。Python の size に対応

    let disp = format!("{}", x);
    let expected_disp = r#"variable([[1, 2, 3],
          [4, 5, 6]], name=x)"#;
    assert_eq!(disp, expected_disp);
}

// 本 ステップ20: 演算子のオーバーロード
#[test]
fn test_operator_overloading_step20() {
    let a = Variable::new(array![[3.0f32]].into_dyn());
    let b = Variable::new(array![[2.0f32]].into_dyn());
    let c = Variable::new(array![[1.0f32]].into_dyn());

    // y = a * b + c
    // 葉の変数は参照(&a, &b, &c)として渡し、途中の一時変数はムーブさせる(所有権の活用)
    let y = &a * &b + &c;
    y.backward(false);

    assert!(approx_equal_arrayd(
        &y.data(),
        &array![[7.0f32]].into_dyn(),
        1e-5
    ));
    assert!(approx_equal_arrayd(
        &a.grad().unwrap(),
        &array![[2.0f32]].into_dyn(),
        1e-5
    ));
    assert!(approx_equal_arrayd(
        &b.grad().unwrap(),
        &array![[3.0f32]].into_dyn(),
        1e-5
    ));
}

// 本 ステップ21: 演算子のオーバーロード(2) - スカラーとの混合
#[test]
fn test_operator_overloading_scalar_step21() {
    let x = Variable::new(array![[2.0f32]].into_dyn());

    // 3.0 * x + 1.0
    let y = 3.0 * &x + 1.0;
    y.backward(false);

    assert!(approx_equal_arrayd(
        &y.data(),
        &array![[7.0f32]].into_dyn(),
        1e-5
    ));
    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[3.0f32]].into_dyn(),
        1e-5
    ));
}

// 本 ステップ22: 演算子のオーバーロード(3) - その他の演算子(Neg, Sub, Div, Pow)
#[test]
fn test_operator_overloading_step22() {
    // Neg
    let x = Variable::new(array![[2.0f32]].into_dyn());
    let y = -&x;
    assert!(approx_equal_arrayd(
        &y.data(),
        &array![[-2.0f32]].into_dyn(),
        1e-5
    ));

    // Sub
    let y2 = 3.0 - &x; // スカラーマクロの効果
    assert!(approx_equal_arrayd(
        &y2.data(),
        &array![[1.0f32]].into_dyn(),
        1e-5
    ));

    // Div
    let y3 = &x / 2.0;
    assert!(approx_equal_arrayd(
        &y3.data(),
        &array![[1.0f32]].into_dyn(),
        1e-5
    ));

    // Pow
    let y4 = x.powf(3.0);
    y4.backward(false);
    assert!(approx_equal_arrayd(
        &y4.data(),
        &array![[8.0f32]].into_dyn(),
        1e-5
    )); // 2^3 = 8
    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[12.0f32]].into_dyn(),
        1e-5
    )); // 3 * 2^2 = 12
}

#[test]
fn test_backward_gradients_step22() {
    let eps = EPSILON_FOR_DIFF;
    let tol = 1e-3;

    // Sub の勾配チェック
    let a = Variable::new(array![[3.0f32]].into_dyn());
    let b = Variable::new(array![[2.0f32]].into_dyn());
    let y_sub = &a - &b;
    y_sub.backward(false);

    let b_data = b.data();
    let f_sub_a = |xs: &[ArrayD<f32>]| &xs[0] - &b_data;
    let a_data = a.data();
    let f_sub_b = |xs: &[ArrayD<f32>]| &a_data - &xs[0];

    assert!(approx_equal_arrayd(
        &a.grad().unwrap(),
        &numerical_diff(f_sub_a, &a, eps).data(),
        tol
    ));
    assert!(approx_equal_arrayd(
        &b.grad().unwrap(),
        &numerical_diff(f_sub_b, &b, eps).data(),
        tol
    ));

    // Div の勾配チェック
    let a2 = Variable::new(array![[3.0f32]].into_dyn());
    let b2 = Variable::new(array![[2.0f32]].into_dyn());
    let y_div = &a2 / &b2;
    y_div.backward(false);

    let b2_data = b2.data();
    let f_div_a = |xs: &[ArrayD<f32>]| &xs[0] / &b2_data;
    let a2_data = a2.data();
    let f_div_b = |xs: &[ArrayD<f32>]| &a2_data / &xs[0];

    assert!(approx_equal_arrayd(
        &a2.grad().unwrap(),
        &numerical_diff(f_div_a, &a2, eps).data(),
        tol
    ));
    assert!(approx_equal_arrayd(
        &b2.grad().unwrap(),
        &numerical_diff(f_div_b, &b2, eps).data(),
        tol
    ));
}
