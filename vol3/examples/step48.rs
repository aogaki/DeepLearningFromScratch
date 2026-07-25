use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use vol3::functions::softmax_cross_entropy_simple;
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Optimizer, SGD};
use vol3::utils::{accuracy, get_spiral};
use vol3::variable::Variable;

fn main() {
    let max_epoch = 300;
    let batch_size = 30;
    let hidden_size = 10;
    let lr = 1.0;

    let (x, t) = get_spiral(true);
    let num_data = x.shape()[0];
    // Using a seeded RNG for consistent example output
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let model = MLP::new(&[2, hidden_size, 3], Variable::sigmoid, &mut rng);
    let mut optimizer = SGD::new(lr);

    optimizer.setup(&model);

    for epoch in 0..max_epoch {
        let mut indices: Vec<usize> = (0..num_data).collect();
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0.0;

        for i in (0..num_data).step_by(batch_size) {
            let end = (i + batch_size).min(num_data);
            let mut batch_x = ndarray::Array2::<f32>::zeros((end - i, 2));
            let mut batch_t = Vec::with_capacity(end - i);

            let x_view = x.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            for j in 0..(end - i) {
                let idx = indices[i + j];
                batch_x[[j, 0]] = x_view[[idx, 0]];
                batch_x[[j, 1]] = x_view[[idx, 1]];
                batch_t.push(t[idx]);
            }

            let bx = Variable::new(batch_x.into_dyn());
            let y = model.forward(&bx);
            let loss = softmax_cross_entropy_simple(&y, &batch_t);

            epoch_correct += accuracy(&y, &batch_t) * (end - i) as f32;
            epoch_loss += loss.item() * (end - i) as f32;

            model.cleargrads();
            loss.backward(false, false);
            optimizer.update();
        }

        let avg_loss = epoch_loss / num_data as f32;
        let avg_acc = epoch_correct / num_data as f32;

        println!(
            "epoch {} | loss {:.4} | accuracy {:.4}",
            epoch + 1,
            avg_loss,
            avg_acc
        );
    }
}
