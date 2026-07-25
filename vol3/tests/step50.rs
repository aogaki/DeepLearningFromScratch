use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use vol3::dataloaders::DataLoader;
use vol3::datasets::{Dataset, SpiralDataset};
use vol3::functions::softmax_cross_entropy_simple;
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Optimizer, SGD};
use vol3::utils::accuracy;
use vol3::variable::Variable;

#[test]
fn test_dataloader_iteration() {
    let dataset = SpiralDataset::new(true);
    let num_data = dataset.len();
    let batch_size = 30;

    let rng = Box::new(ChaCha8Rng::seed_from_u64(42));
    let mut loader = DataLoader::new(dataset, batch_size, true, rng);

    // 2エポック回して、どちらも全データが出力されるかを検証 (reset動作の確認)
    for _ in 0..2 {
        let mut count = 0;
        for (batch_x, batch_t) in &mut loader {
            assert_eq!(batch_x.shape()[0], batch_t.len());
            count += batch_t.len();
        }
        assert_eq!(
            count, num_data,
            "DataLoader should yield exactly `num_data` items per epoch"
        );
    }
}

#[test]
fn test_dataloader_no_shuffle() {
    let dataset = SpiralDataset::new(false);
    let num_data = dataset.len();
    let batch_size = 30;

    let rng = Box::new(ChaCha8Rng::seed_from_u64(42));
    let mut loader = DataLoader::new(dataset, batch_size, false, rng);

    // If shuffle=false, the data should be yielded in exact original order
    let mut current_idx = 0;
    let dataset = SpiralDataset::new(false);
    for (_batch_x, batch_t) in &mut loader {
        let b_size = batch_t.len();

        // shuffle=false では元データセットと同順で出ることを添字照合で確認する
        for (i, item) in batch_t.iter().enumerate() {
            let (_, t) = dataset.get_item(current_idx + i);
            assert_eq!(*item, t);
        }
        current_idx += b_size;
    }
    assert_eq!(current_idx, num_data);
}

#[test]
fn test_spiral_training_with_dataloader() {
    let max_epoch = 300;
    let batch_size = 30;
    let hidden_size = 10;
    let lr = 1.0;

    let dataset = SpiralDataset::new(true);
    let num_data = dataset.len();

    let mut model_rng = ChaCha8Rng::seed_from_u64(1984);
    let model = MLP::new(&[2, hidden_size, 3], Variable::sigmoid, &mut model_rng);
    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);

    let loader_rng = Box::new(ChaCha8Rng::seed_from_u64(1984));
    let mut data_loader = DataLoader::new(dataset, batch_size, true, loader_rng);

    let mut start_loss = 0.0;
    let mut final_loss = 0.0;
    let mut start_acc = 0.0;
    let mut final_acc = 0.0;

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

        if epoch == 0 {
            start_loss = avg_loss;
            start_acc = avg_acc;
        }
        if epoch == max_epoch - 1 {
            final_loss = avg_loss;
            final_acc = avg_acc;
        }
    }

    assert!(final_loss < start_loss, "Loss should decrease");
    assert!(final_acc > start_acc, "Accuracy should increase");
    assert!(final_acc > 0.9, "Accuracy should be reasonably high (>90%)");
}
