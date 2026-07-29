//! 本 7.4「VAE」
//! MNISTデータセットを用いて VAE をゼロから学習し、学習完了後に
//! 標準正規分布のノイズ z から新しい画像 (8x8 = 64枚) を生成して PGM 保存する。
//! 実行: vol5/ で `cargo run --example step07_generate --release`

use ndarray::Array4;
use rand::SeedableRng;
use rand_distr::Normal;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use vol3::dataloaders::DataLoader;
use vol3::datasets::MnistDataset;
use vol3::functions::mean_squared_error;
use vol3::layers::Layer;
use vol3::optimizers::{Adam, Optimizer};
use vol3::utils::random_array;
use vol3::variable::Variable;

use vol5::vae::{VAE, kl_divergence};

/// 画像を [0.0, 1.0] にスケールし、VAE 用に 784次元の1次元ベクトルに平坦化する
fn transform_vae(mut x: ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
    x.mapv_inplace(|v| v / 255.0);
    x.into_shape_with_order((784,)).unwrap().into_dyn()
}

/// 複数の画像 (N, 1, H, W) をタイル状に並べて1枚のPGM画像として保存する
fn save_images_grid_as_pgm(
    images: &Array4<f32>,
    filepath: &Path,
    num_cols: usize,
    scale: usize,
) -> io::Result<()> {
    let (num_images, _c, h, w) = images.dim();
    let num_rows = num_images.div_ceil(num_cols);
    let margin = 2; // 画像間の余白

    let tile_h = h * scale;
    let tile_w = w * scale;

    let canvas_h = num_rows * tile_h + (num_rows + 1) * margin;
    let canvas_w = num_cols * tile_w + (num_cols + 1) * margin;

    // キャンバスを作成し、背景色(128: 灰色)で塗りつぶす
    let mut canvas = vec![vec![128u8; canvas_w]; canvas_h];

    for i in 0..num_images {
        let row = i / num_cols;
        let col = i % num_cols;
        let img = images.slice(ndarray::s![i, 0, .., ..]);

        // 各画像で min-max 正規化 (コントラストの最大化)
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in img.iter() {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        if (max_val - min_val).abs() < 1e-7 {
            max_val = min_val + 1.0;
        }

        let start_y = margin + row * (tile_h + margin);
        let start_x = margin + col * (tile_w + margin);

        for y in 0..tile_h {
            for x in 0..tile_w {
                let orig_y = y / scale;
                let orig_x = x / scale;
                let val = img[[orig_y, orig_x]];

                let normalized = (val - min_val) / (max_val - min_val);
                let pixel_val = (normalized * 255.0).clamp(0.0, 255.0).round() as u8;

                canvas[start_y + y][start_x + x] = pixel_val;
            }
        }
    }

    let mut file = File::create(filepath)?;
    writeln!(file, "P2")?;
    writeln!(file, "{} {}", canvas_w, canvas_h)?;
    writeln!(file, "255")?;
    for row in canvas {
        for pixel in row {
            write!(file, "{} ", pixel)?;
        }
        writeln!(file)?;
    }
    Ok(())
}

fn main() {
    let output_dir = Path::new("output");
    fs::create_dir_all(output_dir).unwrap();
    let filepath = output_dir.join("vae_generated.pgm");

    let mut rng = rand::rngs::StdRng::seed_from_u64(123);

    // 1. VAEとオプティマイザの初期化
    println!("Initializing VAE...");
    let vae = VAE::new(784, 200, 20, &mut rng);
    let mut optimizer = Adam::new(3e-4);
    optimizer.setup(&vae);

    // 2. MNIST データローダーの準備
    println!("Loading MNIST dataset...");
    let dataset = MnistDataset::new(true, Some(transform_vae));
    let mut loader = DataLoader::new(
        dataset,
        100,
        true,
        Box::new(rand::rngs::StdRng::seed_from_u64(42)),
    );

    // 3. 学習ループ
    let epochs = 30;
    println!("Start training for {} epochs...", epochs);

    for epoch in 0..epochs {
        let mut sum_loss = 0.0;
        let mut batches = 0;

        for (x, _labels) in &mut loader {
            let (y_pred, mu, ln_var) = vae.forward_vae(&x, &mut rng);

            let rec_loss = mean_squared_error(&y_pred, &x);
            let kl_loss = kl_divergence(&mu, &ln_var);
            let loss = rec_loss + kl_loss;

            sum_loss += loss
                .data()
                .into_dimensionality::<ndarray::Ix0>()
                .unwrap()
                .into_scalar();
            batches += 1;

            vae.cleargrads();
            loss.backward(false, false);
            optimizer.update();
        }

        let avg_loss = sum_loss / (batches as f32);
        println!("Epoch {:2}: Loss = {:.4}", epoch + 1, avg_loss);
    }
    println!("Training completed!");

    // 4. 画像の生成
    {
        let _guard = vol3::config::no_grad();

        let num_images = 64; // 8x8 = 64枚
        let latent_dim = 20;
        println!("Generating images from z ~ N(0, I)...");

        let z_data = random_array(
            (num_images, latent_dim),
            Normal::new(0.0, 1.0).unwrap(),
            &mut rng,
        );
        let z = Variable::new(z_data.into_dyn());

        let generated = vae.dec.forward(&z);

        let gen_data = generated
            .data()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let images_4d = gen_data
            .into_shape_with_order((num_images, 1, 28, 28))
            .unwrap();

        // 5. 画像保存 (スケール=2 で 448x448 の見やすいサイズに)
        println!("Saving generated images to {:?}", filepath);
        save_images_grid_as_pgm(&images_4d, &filepath, 8, 2).unwrap();

        println!("Done! Check output/vae_generated.pgm");
    }
}
