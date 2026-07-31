// caffeinate -i cargo run --release --example step10_train

use ndarray::{ArrayViewD, Axis};
use rand::RngExt;
use rand::SeedableRng;
use std::fs;
use std::path::Path;
use std::time::Instant;

use vol3::dataloaders::DataLoader;
use vol3::datasets::MnistDataset;
use vol3::functions::mean_squared_error;
use vol3::layers::Layer;
use vol3::optimizers::{Adam, Optimizer};
use vol3::variable::Variable;

use vol5::diffusion::Diffusion;
use vol5::unet::UNet;

fn transform_diffusion(mut x: ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
    x.mapv_inplace(|v| v / 255.0);
    x.into_shape_with_order((1, 28, 28)).unwrap().into_dyn()
}

fn main() {
    let output_dir = Path::new("output");
    fs::create_dir_all(output_dir).unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(123);

    println!("Initializing UNet and Diffusion...");
    let unet = UNet::new(1, 100, &mut rng);
    let mut optimizer = Adam::new(1e-3);
    optimizer.setup(&unet);

    let diffusion = Diffusion::new(1000, 1e-4, 0.02);

    println!("Loading MNIST dataset (60k)...");
    let dataset = MnistDataset::new(true, Some(transform_diffusion));
    let mut loader = DataLoader::new(
        dataset,
        128,
        true,
        Box::new(rand::rngs::StdRng::seed_from_u64(42)),
    );

    let epochs = 10;
    println!("Start training for {} epochs...", epochs);

    for epoch in 0..epochs {
        let mut sum_loss = 0.0;
        let mut batches = 0;
        let start = Instant::now();

        for (x, labels) in &mut loader {
            let x_data = x.data();
            let b_size = x_data.shape()[0];

            let y_batch = ndarray::Array1::from_vec(labels);

            let keep_label = rng.random_bool(0.9);
            let labels_opt = if keep_label { Some(&y_batch) } else { None };

            let mut t_array = ndarray::Array1::<f32>::zeros(b_size);
            let mut x_t_singles = Vec::new();
            let mut noise_singles = Vec::new();

            for b in 0..b_size {
                let t_val = rng.random_range(1..=1000) as usize;
                t_array[b] = t_val as f32;

                let x_0_single = x_data.index_axis(Axis(0), b).into_owned().into_dyn();
                let noise_single = vol3::utils::random_array(
                    x_0_single.dim(),
                    rand_distr::StandardNormal,
                    &mut rng,
                );

                let x_t_single = diffusion.add_noise_with_noise(&x_0_single, t_val, &noise_single);

                x_t_singles.push(x_t_single);
                noise_singles.push(noise_single);
            }

            let views_x_t: Vec<ArrayViewD<f32>> = x_t_singles.iter().map(|a| a.view()).collect();
            let views_noise: Vec<ArrayViewD<f32>> =
                noise_singles.iter().map(|a| a.view()).collect();

            let x_t_data = ndarray::stack(Axis(0), &views_x_t).unwrap();
            let noise_data = ndarray::stack(Axis(0), &views_noise).unwrap();

            let x_t = Variable::new(x_t_data.into_dyn());
            let noise = Variable::new(noise_data.into_dyn());

            let noise_pred = unet.forward(&x_t, &t_array, labels_opt);

            let loss = mean_squared_error(&noise_pred, &noise) / 784.0;

            sum_loss += loss.data().sum();
            batches += 1;

            unet.cleargrads();
            loss.backward(false, false);
            optimizer.update();

            // 定期的な進捗報告 (生存確認)
            if batches % 50 == 0 {
                let current_avg = sum_loss / (batches as f32);
                println!(
                    "  [Epoch {} - Batch {:>3}] Avg Loss: {:.4}, Elapsed: {:?}",
                    epoch + 1,
                    batches,
                    current_avg,
                    start.elapsed()
                );
            }
        }

        let avg_loss = sum_loss / (batches as f32);
        println!(
            "Epoch {:2} Completed: Avg Loss = {:.4}, Total Time = {:?}",
            epoch + 1,
            avg_loss,
            start.elapsed()
        );

        let save_path = output_dir.join(format!("step10_unet_epoch_{}.npz", epoch + 1));
        unet.save_weights(&save_path).unwrap();
        println!("Saved weights to {:?}", save_path);
    }
    println!("Training completed!");
}
