use rand::SeedableRng;
use vol3::layers::{Layer, RNN};
use vol3::variable::Variable;

#[test]
fn test_rnn_unchain_backward() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let rnn = RNN::new(10, 20, &mut rng);

    // Simulate a sequence of 3 inputs
    let x0 = Variable::new(ndarray::ArrayD::zeros(vec![1, 10]));
    let x1 = Variable::new(ndarray::ArrayD::zeros(vec![1, 10]));
    let x2 = Variable::new(ndarray::ArrayD::zeros(vec![1, 10]));

    let _h0 = rnn.forward(&x0);
    let h1 = rnn.forward(&x1);

    // Unchain backward at the RNN layer itself
    // This should disconnect the RNN's internal state (h1) from the first iteration's computations.
    rnn.unchain_backward();

    assert!(!h1.has_creator(), "h1 should be detached from its creator");

    // Since h1 was unchained, backward from h2 will only reach h1, x2, and the parameters used in the second step.
    let h2 = rnn.forward(&x2);

    h2.backward(false, false);

    // Check that x2 has a gradient, but x1 and x0 do not (because h1 was unchained).
    assert!(x2.grad_var().is_some(), "x2 should receive gradients");
    assert!(
        x1.grad_var().is_none(),
        "x1 should NOT receive gradients because model was unchained"
    );
    assert!(
        x0.grad_var().is_none(),
        "x0 should NOT receive gradients because model was unchained"
    );

    // Also test that memory is freed by using Weak pointers if we drop h2.
    let h1_weak = h1.downgrade();
    drop(h1);
    // Note: rnn still holds a strong reference to the latest h (h2), but h1 should be dropped
    // if the graph is properly cleaned up. Wait, h2 holds a reference to h1's data (or h1 itself?
    // Actually h2 is computed from h1. So the Node for h2 owns h1. Therefore h1 is still alive as an input to h2's creator.
    // If we drop h2 (and clear rnn's state), the graph should be fully deallocated.
    drop(h2);
    rnn.reset_state();

    assert!(!h1_weak.is_alive(), "h1 should be dropped and memory freed");
}

#[test]
fn test_simple_rnn_unchain_backward() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let model = vol3::models::SimpleRNN::new(20, 1, &mut rng);

    let x0 = Variable::new(ndarray::ArrayD::zeros(vec![1, 1]));
    let x1 = Variable::new(ndarray::ArrayD::zeros(vec![1, 1]));
    let x2 = Variable::new(ndarray::ArrayD::zeros(vec![1, 1]));

    let _y0 = model.forward(&x0);
    let _y1 = model.forward(&x1);

    // Grab h1 to verify it gets unchained
    let h1 = model.rnn.h.borrow().as_ref().unwrap().clone();

    // Test the unchain_backward delegation in SimpleRNN model
    model.unchain_backward();

    let y2 = model.forward(&x2);
    y2.backward(false, false);

    assert!(x2.grad_var().is_some(), "x2 should receive gradients");
    assert!(
        x1.grad_var().is_none(),
        "x1 should NOT receive gradients because model was unchained"
    );
    assert!(
        x0.grad_var().is_none(),
        "x0 should NOT receive gradients because model was unchained"
    );

    // We can also verify that the internal h (which was h1 at the time of unchain) is detached.
    assert!(
        !h1.has_creator(),
        "internal h1 should be detached from its creator"
    );
}

#[test]
#[ignore] // cargo test test_step59_4_sin_curve_training --release -- --ignored --nocapture
fn test_step59_4_sin_curve_training() {
    let dataset = vol3::datasets::SinCurveDataset::new(true);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let model = vol3::models::SimpleRNN::new(100, 1, &mut rng);

    let lr = 0.001;
    let max_epoch = 100;
    let bptt_length = 30;

    use vol3::optimizers::Optimizer;
    let mut optimizer = vol3::optimizers::Adam::new(lr);
    optimizer.setup(&model);

    let seqlen = dataset.data.nrows();

    for epoch in 0..max_epoch {
        model.reset_state();
        let mut window_loss = Variable::from(0.0);
        let mut total_loss = 0.0;
        let mut count = 0;

        for i in 0..seqlen {
            let x = Variable::new(
                dataset
                    .data
                    .row(i)
                    .to_owned()
                    .into_dyn()
                    .insert_axis(ndarray::Axis(0)),
            );
            let t = Variable::new(
                dataset
                    .label
                    .row(i)
                    .to_owned()
                    .into_dyn()
                    .insert_axis(ndarray::Axis(0)),
            );

            let y = model.forward(&x);
            let loss_t = vol3::functions::mean_squared_error(&y, &t);

            total_loss += loss_t.item();
            window_loss = &window_loss + &loss_t;
            count += 1;

            if count % bptt_length == 0 || count == seqlen {
                model.cleargrads();
                window_loss.backward(false, false);
                model.unchain_backward();
                optimizer.update();

                // グラフを解放するために window_loss をリセット (2本目のピンを切る)
                window_loss = Variable::from(0.0);
            }
        }
        if epoch == max_epoch - 1 {
            let final_train_loss = total_loss / count as f32;
            println!("Epoch {}: loss={}", epoch, final_train_loss);
            assert!(
                final_train_loss < 0.02,
                "Model failed to converge on training data (loss: {})",
                final_train_loss
            );
        } else if epoch % 10 == 0 {
            println!("Epoch {}: loss={}", epoch, total_loss / count as f32);
        }
    }

    // Test set prediction (Autoregressive generation)
    let test_dataset = vol3::datasets::SinCurveDataset::new(false);
    model.reset_state();
    let mut test_loss = 0.0;

    let _guard = vol3::config::no_grad(); // Disable computation graph for memory efficiency

    // Initial input is the very first data point
    let mut x = Variable::new(
        test_dataset
            .data
            .row(0)
            .to_owned()
            .into_dyn()
            .insert_axis(ndarray::Axis(0)),
    );

    for i in 0..seqlen {
        let t = Variable::new(
            test_dataset
                .label
                .row(i)
                .to_owned()
                .into_dyn()
                .insert_axis(ndarray::Axis(0)),
        );
        let y = model.forward(&x);
        let loss_t = vol3::functions::mean_squared_error(&y, &t);
        test_loss += loss_t.item();
        x = y; // Feed prediction as next input
    }

    let final_test_loss = test_loss / seqlen as f32;
    println!("Test loss (Autoregressive MSE): {}", final_test_loss);
}
