use ndarray::array;
use vol3::*;

pub fn sphere(x: &Variable, y: &Variable) -> Variable {
    x.powf(2.0) + y.powf(2.0)
}

pub fn matyas(x: &Variable, y: &Variable) -> Variable {
    0.26 * (x.powf(2.0) + y.powf(2.0)) - 0.48 * x * y
}

pub fn goldstein_price(x: &Variable, y: &Variable) -> Variable {
    let term1 = 1.0
        + (x + y + 1.0).powf(2.0)
            * (19.0 - 14.0 * x + 3.0 * x.powf(2.0) - 14.0 * y + 6.0 * x * y + 3.0 * y.powf(2.0));
    let term2 = 30.0
        + (2.0 * x - 3.0 * y).powf(2.0)
            * (18.0 - 32.0 * x + 12.0 * x.powf(2.0) + 48.0 * y - 36.0 * x * y + 27.0 * y.powf(2.0));
    term1 * term2
}

#[test]
fn test_sphere() {
    let x = Variable::new(array![[1.0f32]].into_dyn());
    let y = Variable::new(array![[1.0f32]].into_dyn());
    let z = sphere(&x, &y);
    z.backward(false);

    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[2.0f32]].into_dyn(),
        1e-4
    ));
    assert!(approx_equal_arrayd(
        &y.grad().unwrap(),
        &array![[2.0f32]].into_dyn(),
        1e-4
    ));
}

#[test]
fn test_matyas() {
    let x = Variable::new(array![[1.0f32]].into_dyn());
    let y = Variable::new(array![[1.0f32]].into_dyn());
    let z = matyas(&x, &y);
    z.backward(false);

    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[0.04f32]].into_dyn(),
        1e-4
    ));
    assert!(approx_equal_arrayd(
        &y.grad().unwrap(),
        &array![[0.04f32]].into_dyn(),
        1e-4
    ));
}

#[test]
fn test_goldstein_price() {
    let x = Variable::new(array![[1.0f32]].into_dyn());
    let y = Variable::new(array![[1.0f32]].into_dyn());
    let z = goldstein_price(&x, &y);
    z.backward(false);

    assert!(approx_equal_arrayd(
        &x.grad().unwrap(),
        &array![[-5376.0f32]].into_dyn(),
        1e-4
    ));
    assert!(approx_equal_arrayd(
        &y.grad().unwrap(),
        &array![[8064.0f32]].into_dyn(),
        1e-4
    ));
}
