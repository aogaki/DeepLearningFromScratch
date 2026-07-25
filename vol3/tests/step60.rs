use rand::SeedableRng;
use vol3::layers::Layer;
use vol3::variable::Variable;

#[test]
#[ignore] // cargo test test_step60_lstm_sin_curve --release -- --ignored --nocapture
fn test_step60_lstm_sin_curve() {
    let train_dataset = vol3::datasets::SinCurveDataset::new(true);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let model = vol3::models::BetterRNN::new(100, 1, &mut rng);

    let max_epoch = 100;
    let batch_size = 30;
    let bptt_length = 30;
    let lr = 0.001;

    use vol3::optimizers::Optimizer;
    let mut optimizer = vol3::optimizers::Adam::new(lr);
    optimizer.setup(&model);

    let mut dataloader = vol3::dataloaders::SeqDataLoader::new(train_dataset, batch_size);

    for epoch in 0..max_epoch {
        model.reset_state();
        let mut window_loss = Variable::from(0.0);
        let mut total_loss = 0.0;
        let mut count = 0;

        for (x, t) in &mut dataloader {
            let y = model.forward(&x);
            let loss_t = vol3::functions::mean_squared_error(&y, &t);

            total_loss += loss_t.item();
            window_loss = &window_loss + &loss_t;
            count += 1;

            if count % bptt_length == 0 {
                model.cleargrads();
                window_loss.backward(false, false);
                model.unchain_backward();
                optimizer.update();

                // グラフを解放するために window_loss をリセット
                window_loss = Variable::from(0.0);
            }
        }

        // Residual update at the end of epoch
        if count % bptt_length != 0 {
            model.cleargrads();
            window_loss.backward(false, false);
            model.unchain_backward();
            optimizer.update();
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

    let seqlen = test_dataset.data.nrows();

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

    // Test loss should be much smaller now with LSTM compared to SimpleRNN (>1.5)
    assert!(
        final_test_loss < 0.5,
        "LSTM failed to autoregress accurately (test loss: {})",
        final_test_loss
    );
}

#[test]
fn test_lstm_unchain_backward_via_trait() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let lstm = vol3::layers::LSTM::new(1, 10, &mut rng);

    // Simulate forward pass to create internal states with creators
    let x = Variable::from(1.0);
    // Dummy creators to simulate graph connections
    let y_h = &x + &x; // y_h has a creator
    let y_c = &x * &x; // y_c has a different creator

    // Manually set internal state to separate variables with creators
    *lstm.h.borrow_mut() = Some(y_h.clone());
    *lstm.c.borrow_mut() = Some(y_c.clone());

    assert!(lstm.h.borrow().as_ref().unwrap().has_creator());

    // Convert to trait object
    let layer: &dyn Layer = &lstm;

    // Call unchain_backward via trait
    layer.unchain_backward();

    // Verify it worked
    assert!(!lstm.h.borrow().as_ref().unwrap().has_creator());
    assert!(!lstm.c.borrow().as_ref().unwrap().has_creator());
}
