use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::env;
use vol3::layers::{Layer, MLP};
use vol3::utils::approx_equal_arrayd;
use vol3::variable::Variable;

#[test]
fn test_save_and_load_weights() {
    let mut rng_a = ChaCha8Rng::seed_from_u64(42);
    let mut rng_b = ChaCha8Rng::seed_from_u64(99);

    let model_a = MLP::new(&[2, 10, 1], Variable::sigmoid, &mut rng_a);
    let model_b = MLP::new(&[2, 10, 1], Variable::sigmoid, &mut rng_b);

    let x_data = ndarray::array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let x = Variable::new(x_data.clone());

    // Before loading, models should produce different outputs
    let y_a = model_a.forward(&x);
    let y_b_before = model_b.forward(&x);
    assert!(!approx_equal_arrayd(&y_a.data(), &y_b_before.data(), 1e-4));

    // Save model A
    let mut temp_path = env::temp_dir();
    temp_path.push("my_mlp.npz");
    model_a
        .save_weights(&temp_path)
        .expect("Failed to save weights");

    // Load into model B
    model_b
        .load_weights(&temp_path)
        .expect("Failed to load weights");

    // After loading, model B should produce same output as model A
    let y_b_after = model_b.forward(&x);
    assert!(approx_equal_arrayd(&y_a.data(), &y_b_after.data(), 1e-4));

    // Clean up
    let _ = std::fs::remove_file(temp_path);
}

#[test]
fn test_load_weights_shape_mismatch() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let model_a = MLP::new(&[2, 10, 1], Variable::sigmoid, &mut rng);
    let mut temp_path = env::temp_dir();
    temp_path.push("my_mlp_mismatch.npz");
    model_a
        .save_weights(&temp_path)
        .expect("Failed to save weights");

    // model_c has different hidden size (20 instead of 10)
    let model_c = MLP::new(&[2, 20, 1], Variable::sigmoid, &mut rng);

    // Loading should fail due to shape mismatch
    let res = model_c.load_weights(&temp_path);
    assert!(res.is_err());
    let err_msg = res.unwrap_err().to_string();
    assert!(err_msg.contains("Shape mismatch"));

    // Clean up
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_load_weights_partial_failure() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    // Original model: [2, 10, 1]
    let model_a = MLP::new(&[2, 10, 1], Variable::sigmoid, &mut rng);
    let mut temp_path = env::temp_dir();
    temp_path.push("my_mlp_partial.npz");
    model_a
        .save_weights(&temp_path)
        .expect("Failed to save weights");

    // model_b: [2, 10, 3] -> First layer matches [2, 10], but second layer [10, 3] mismatches [10, 1]
    let model_b = MLP::new(&[2, 10, 3], Variable::sigmoid, &mut rng);

    // Save state before load attempt
    let l0_w_before = model_b
        .named_params()
        .into_iter()
        .find(|(name, _)| name == "l0/W")
        .unwrap()
        .1
        .data();

    let res = model_b.load_weights(&temp_path);
    assert!(res.is_err());

    // Check that l0/W was NOT mutated by the aborted load
    let l0_w_after = model_b
        .named_params()
        .into_iter()
        .find(|(name, _)| name == "l0/W")
        .unwrap()
        .1
        .data();
    assert!(approx_equal_arrayd(&l0_w_before, &l0_w_after, 1e-6));

    let _ = std::fs::remove_file(temp_path);
}
