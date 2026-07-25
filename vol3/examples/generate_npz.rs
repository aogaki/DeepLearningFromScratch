use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use vol3::layers::{Layer, MLP};

fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let model = MLP::new(&[2, 10, 1], vol3::variable::Variable::sigmoid, &mut rng);
    model
        .save_weights(std::path::Path::new("test_model.npz"))
        .unwrap();
}
