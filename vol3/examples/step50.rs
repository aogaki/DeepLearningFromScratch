use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use vol3::dataloaders::DataLoader;
use vol3::datasets::{Dataset, SpiralDataset};
use vol3::functions::softmax_cross_entropy_simple;
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Optimizer, SGD};
use vol3::utils::accuracy;
use vol3::variable::Variable;

fn main() {
    let max_epoch = 300;
    let batch_size = 30;
    let hidden_size = 10;
    let lr = 1.0;

    let dataset = SpiralDataset::new(true);
    let num_data = dataset.len();

    let test_dataset = SpiralDataset::new(false);
    let num_test_data = test_dataset.len();

    let mut model_rng = ChaCha8Rng::seed_from_u64(42);
    let model = MLP::new(&[2, hidden_size, 3], Variable::sigmoid, &mut model_rng);
    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);

    let loader_rng = Box::new(ChaCha8Rng::seed_from_u64(42));
    let mut data_loader = DataLoader::new(dataset, batch_size, true, loader_rng);

    let test_loader_rng = Box::new(ChaCha8Rng::seed_from_u64(43));
    let mut test_data_loader = DataLoader::new(test_dataset, batch_size, false, test_loader_rng);

    for epoch in 0..max_epoch {
        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0.0;

        for (batch_x, batch_t) in &mut data_loader {
            let b_size = batch_t.len() as f32;
            let y = model.forward(&batch_x);
            let loss = softmax_cross_entropy_simple(&y, &batch_t);

            epoch_correct += accuracy(&y, &batch_t) * b_size;
            epoch_loss += loss.item() * b_size;

            model.cleargrads();
            loss.backward(false, false);
            optimizer.update();
        }

        let avg_loss = epoch_loss / num_data as f32;
        let avg_acc = epoch_correct / num_data as f32;

        let mut test_epoch_loss = 0.0;
        let mut test_epoch_correct = 0.0;

        {
            let _guard = vol3::config::no_grad();
            for (batch_x, batch_t) in &mut test_data_loader {
                let b_size = batch_t.len() as f32;
                let y = model.forward(&batch_x);
                let loss = softmax_cross_entropy_simple(&y, &batch_t);

                test_epoch_correct += accuracy(&y, &batch_t) * b_size;
                test_epoch_loss += loss.item() * b_size;
            }
        }

        let test_avg_loss = test_epoch_loss / num_test_data as f32;
        let test_avg_acc = test_epoch_correct / num_test_data as f32;

        println!(
            "epoch {} | loss {:.4} | accuracy {:.4} | test_loss {:.4} | test_accuracy {:.4}",
            epoch + 1,
            avg_loss,
            avg_acc,
            test_avg_loss,
            test_avg_acc
        );
    }
}
