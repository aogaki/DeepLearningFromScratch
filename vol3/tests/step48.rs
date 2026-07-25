use vol3::functions::softmax_cross_entropy_simple;
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Optimizer, SGD};
use vol3::utils::{accuracy, get_spiral};
use vol3::variable::Variable;

#[test]
fn test_spiral_training() {
    let max_epoch = 300;
    let batch_size = 30;
    let hidden_size = 10;
    let lr = 1.0;

    let (x, t) = get_spiral(true);
    let num_data = x.shape()[0];
    // Use a fixed RNG seed for deterministic model initialization and data shuffling
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(1984);
    let model = MLP::new(&[2, hidden_size, 3], Variable::sigmoid, &mut rng);
    let mut optimizer = SGD::new(lr);

    optimizer.setup(&model);

    let mut start_loss = 0.0;
    let mut final_loss = 0.0;
    let mut start_acc = 0.0;
    let mut final_acc = 0.0;

    for epoch in 0..max_epoch {
        use rand::seq::SliceRandom;
        // spiral はクラス順に並ぶためシャッフル必須

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
