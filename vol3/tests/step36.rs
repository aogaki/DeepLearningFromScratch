use vol3::variable::Variable;

#[test]
fn test_double_backprop() {
    let x = Variable::from(2.0);
    let y = x.powf(2.0);

    y.backward(false, true);
    let gx = x.grad_var().unwrap();

    let z = gx.powf(3.0) + y;
    x.cleargrad();
    z.backward(false, false);

    // y = x^2,  dy/dx = 2x
    // z = (dy/dx)^3 + y = 8x^3 + x^2
    // dz/dx = 24x^2 + 2x
    // at x=2, dz/dx = 24(4) + 4 = 100
    assert_eq!(x.grad_var().unwrap().item(), 100.0);
}
