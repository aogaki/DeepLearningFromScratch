use vol3::config::test_mode;
use vol3::variable::Variable;

#[test]
fn test_dropout_train_mode() {
    let x_data = ndarray::Array::from_elem((10, 10), 1.0).into_dyn();
    let x = Variable::new(x_data);
    let y = x.dropout(0.5);

    let y_data = y.data();
    let mut zeros = 0;
    let mut twos = 0;
    for &val in y_data.iter() {
        if val == 0.0 {
            zeros += 1;
        } else if val == 2.0 {
            twos += 1;
        } else {
            panic!("Unexpected value in dropout output: {}", val);
        }
    }

    assert!(zeros > 20 && zeros < 80, "zeros: {}", zeros);
    assert!(twos > 20 && twos < 80, "twos: {}", twos);

    y.backward(false, false);
    let gx = x.grad().unwrap();
    for (&val, &gy) in y_data.iter().zip(gx.iter()) {
        if val == 0.0 {
            assert_eq!(gy, 0.0);
        } else {
            assert_eq!(gy, 2.0);
        }
    }
}

#[test]
fn test_dropout_test_mode() {
    let x_data = ndarray::Array::from_elem((10, 10), 1.0).into_dyn();
    let x = Variable::new(x_data);

    let _guard = test_mode();
    let y = x.dropout(0.5);

    let y_data = y.data();
    for &val in y_data.iter() {
        assert_eq!(val, 1.0);
    }

    y.backward(false, false);
    let gx = x.grad().unwrap();
    for &gy in gx.iter() {
        assert_eq!(gy, 1.0);
    }
}

#[test]
fn test_dropout_mode_crossing_train_to_test() {
    // Scenario 1: forward in train mode, backward in test mode
    let x_data = ndarray::Array::from_elem((10, 10), 1.0).into_dyn();
    let x = Variable::new(x_data);

    // forward in train mode
    let y = x.dropout(0.5);
    let y_data = y.data();

    // backward in test mode
    {
        let _guard = test_mode();
        y.backward(false, false);
    }

    // The gradient should STILL match the mask used during train mode forward!
    let gx = x.grad().unwrap();
    for (&val, &gy) in y_data.iter().zip(gx.iter()) {
        if val == 0.0 {
            assert_eq!(gy, 0.0);
        } else {
            assert_eq!(gy, 2.0);
        }
    }
}

#[test]
fn test_dropout_mode_crossing_test_to_train() {
    // Scenario 2: forward in test mode, backward in train mode
    let x_data = ndarray::Array::from_elem((10, 10), 1.0).into_dyn();
    let x = Variable::new(x_data);

    let y = {
        let _guard = test_mode();
        x.dropout(0.5)
    };

    // backward in train mode (the guard has dropped)
    // Should NOT panic and should just pass gradient 1.0 everywhere.
    y.backward(false, false);

    let gx = x.grad().unwrap();
    for &gy in gx.iter() {
        assert_eq!(gy, 1.0);
    }
}
