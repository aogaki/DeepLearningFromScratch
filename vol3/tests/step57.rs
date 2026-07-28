use ndarray::ArrayD;
use vol3::utils::random_array;
use rand_distr::Uniform;
use rand::SeedableRng;
use vol3::cnn::Col2Im;
use vol3::function::Function;
use vol3::utils::{EPSILON_FOR_DIFF, approx_equal_arrayd};
use vol3::variable::Variable;

#[test]
fn test_im2col_col2im_duality() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let x_val =
        random_array((1, 1, 4, 4), Uniform::new(0.0, 1.0).unwrap(), &mut rng)
            .into_dyn();
    let x = Variable::new(x_val);

    // (1, 1, 4, 4) kernel=2, stride=1, pad=0 -> OH=3, OW=3
    let y = x.im2col((2, 2), (1, 1), (0, 0), true);
    let gy = Variable::new(ndarray::Array::ones((9, 4)).into_dyn());
    y.set_grad(gy);
    y.backward(false, false);

    let gx = x.grad().unwrap();

    // manual col2im check
    let col2im = Col2Im::new((1, 1, 4, 4), (2, 2), (1, 1), (0, 0), true);
    let expected_gx = col2im.call(&[Variable::new(ndarray::Array::ones((9, 4)).into_dyn())]);

    assert!(approx_equal_arrayd(&gx, &expected_gx.data(), 1e-5));
}

fn full_numerical_diff<F>(mut f: F, x: &Variable, eps: f32) -> ArrayD<f32>
where
    F: FnMut(&Variable) -> Variable,
{
    let mut grad = ndarray::Array::zeros(x.shape());
    let mut grad_iter = grad.iter_mut();
    for i in 0..x.data().len() {
        let mut x_plus = x.data().clone();
        let mut x_minus = x.data().clone();

        *x_plus.iter_mut().nth(i).unwrap() += eps;
        *x_minus.iter_mut().nth(i).unwrap() -= eps;

        let y_plus = f(&Variable::new(x_plus)).data().sum();
        let y_minus = f(&Variable::new(x_minus)).data().sum();

        let diff = (y_plus - y_minus) / (2.0 * eps);
        *grad_iter.next().unwrap() = diff;
    }
    grad.into_dyn()
}

#[test]
fn test_conv2d_simple_gradient_check() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let x_val =
        random_array((1, 2, 4, 4), Uniform::new(0.0, 1.0).unwrap(), &mut rng)
            .into_dyn();
    let w_val =
        random_array((3, 2, 2, 2), Uniform::new(0.0, 1.0).unwrap(), &mut rng)
            .into_dyn();
    let b_val =
        random_array((3,), Uniform::new(0.0, 1.0).unwrap(), &mut rng).into_dyn();

    let x = Variable::new(x_val.clone());
    let w = Variable::new(w_val.clone());
    let b = Variable::new(b_val.clone());

    let y = x.conv2d_simple(&w, Some(&b), (2, 2), (1, 1));
    let gy = Variable::new(ndarray::Array::ones(y.shape()).into_dyn());
    y.set_grad(gy);
    y.backward(false, false);

    let gx_expected = x.grad().unwrap();
    let gx_num = full_numerical_diff(
        |x| x.conv2d_simple(&w, Some(&b), (2, 2), (1, 1)),
        &x,
        EPSILON_FOR_DIFF,
    );
    assert!(approx_equal_arrayd(&gx_expected, &gx_num, 1e-3));

    let gw_expected = w.grad().unwrap();
    let gw_num = full_numerical_diff(
        |w| x.conv2d_simple(w, Some(&b), (2, 2), (1, 1)),
        &w,
        EPSILON_FOR_DIFF,
    );
    assert!(approx_equal_arrayd(&gw_expected, &gw_num, 1e-3));
}

#[test]
fn test_pooling_simple_gradient_check() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let x_val =
        random_array((1, 2, 5, 5), Uniform::new(0.0, 1.0).unwrap(), &mut rng)
            .into_dyn();
    let x = Variable::new(x_val);

    let y = x.pooling_simple((2, 2), (2, 2), (0, 0));
    let gy = Variable::new(ndarray::Array::ones(y.shape()).into_dyn());
    y.set_grad(gy);
    y.backward(false, false);

    let gx_expected = x.grad().unwrap();
    let gx_num = full_numerical_diff(
        |x| x.pooling_simple((2, 2), (2, 2), (0, 0)),
        &x,
        EPSILON_FOR_DIFF,
    );
    assert!(approx_equal_arrayd(&gx_expected, &gx_num, 1e-3));
}

#[test]
fn test_max_gradient_check() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let x_val = random_array((2, 3, 4), Uniform::new(0.0, 1.0).unwrap(), &mut rng)
        .into_dyn();
    let x = Variable::new(x_val);

    let y = x.max_axis(1, false);
    let gy = Variable::new(ndarray::Array::ones(y.shape()).into_dyn());
    y.set_grad(gy);
    y.backward(false, false);

    let gx_expected = x.grad().unwrap();
    let gx_num = full_numerical_diff(|x| x.max_axis(1, false), &x, EPSILON_FOR_DIFF);
    assert!(approx_equal_arrayd(&gx_expected, &gx_num, 1e-3));

    x.cleargrad();
    let y = x.max_axis(2, true);
    let gy = Variable::new(ndarray::Array::ones(y.shape()).into_dyn());
    y.set_grad(gy);
    y.backward(false, false);
    let gx_expected = x.grad().unwrap();
    let gx_num = full_numerical_diff(|x| x.max_axis(2, true), &x, EPSILON_FOR_DIFF);
    assert!(approx_equal_arrayd(&gx_expected, &gx_num, 1e-3));
}

#[test]
fn test_pooling_simple_value() {
    // 0..16 as (1, 1, 4, 4)
    let x_data = ndarray::Array::from_shape_vec((1, 1, 4, 4), (0..16).map(|x| x as f32).collect())
        .unwrap()
        .into_dyn();

    let x = Variable::new(x_data);
    let y = x.pooling_simple((2, 2), (2, 2), (0, 0));

    // kernel 2x2, stride 2, pad 0. OH=2, OW=2.
    // windows:
    // [0, 1]  [2, 3] -> maxes: 5, 7
    // [4, 5]  [6, 7]
    //
    // [8, 9]   [10, 11] -> maxes: 13, 15
    // [12, 13] [14, 15]

    let expected = ndarray::array![[[[5.0, 7.0], [13.0, 15.0]]]].into_dyn();

    assert_eq!(y.data(), expected);
}

#[test]
fn test_conv2d_simple_value() {
    // x: (1, 1, 3, 3) all 1s
    let x_data = ndarray::Array::ones((1, 1, 3, 3)).into_dyn();
    // w: (1, 1, 2, 2) all 2s
    let w_data = (ndarray::Array::ones((1, 1, 2, 2)) * 2.0).into_dyn();

    let x = Variable::new(x_data);
    let w = Variable::new(w_data);
    let y = x.conv2d_simple(&w, None, (1, 1), (0, 0));

    // convolution of 2x2 filter of 2s on 3x3 input of 1s -> moving sum of 4 elements * 2 = 8
    // OH=2, OW=2
    let expected = ndarray::array![[[[8.0, 8.0], [8.0, 8.0]]]].into_dyn();

    assert_eq!(y.data(), expected);
}
