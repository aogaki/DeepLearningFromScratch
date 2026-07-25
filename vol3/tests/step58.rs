use rand::SeedableRng;
use vol3::layers::Layer;
use vol3::models::VGG16;

#[test]
#[ignore]
fn test_vgg16_load_weights() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let vgg = VGG16::new(&mut rng);

    // We expect the weights to be loaded. It takes a few seconds.
    let res = vgg.load_weights(std::path::Path::new("dataset/weights/vgg16.npz"));

    // Check if loading fails because of '.npy' mismatch.
    if let Err(e) = res {
        panic!("Failed to load VGG16 weights: {}", e);
    }
}

use vol3::utils::approx_equal_arrayd;
use vol3::variable::Variable;

#[test]
#[ignore]
fn test_vgg16_inference_parity() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let vgg = VGG16::new(&mut rng);

    vgg.load_weights(std::path::Path::new("dataset/weights/vgg16.npz"))
        .unwrap();

    // Read input and expected output from Python's dezero
    let file = std::fs::File::open("dataset/weights/vgg16_test.npz").unwrap();
    let mut npz = ndarray_npy::NpzReader::new(file).unwrap();

    let x_data: ndarray::ArrayD<f32> = npz.by_name("x").unwrap();
    let expected_y: ndarray::ArrayD<f32> = npz.by_name("y").unwrap();

    let _guard = vol3::config::test_mode(); // Use test_mode to disable dropout

    let x = Variable::new(x_data);
    let y = vgg.forward(&x);

    assert!(approx_equal_arrayd(&y.data(), &expected_y, 1e-4));
}
