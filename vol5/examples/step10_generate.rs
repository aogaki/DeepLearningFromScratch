// cargo run --release --example step10_generate [重みのパス] [gamma]
// cargo run --release --example step10_generate output/step10_unet_epoch_5.npz 5.0

use ndarray::{Array1, Array4};
use rand::SeedableRng;
use std::fs;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use vol3::layers::Layer;
use vol3::variable::Variable;

use vol5::diffusion::Diffusion;
use vol5::unet::UNet;

/// 複数の画像 (N, 1, H, W) をタイル状に並べて1枚のPGM画像として保存する
fn save_images_grid_as_pgm(
    images: &Array4<f32>,
    filepath: &Path,
    num_cols: usize,
    scale: usize,
) -> io::Result<()> {
    let (num_images, _c, h, w) = images.dim();
    let num_rows = num_images.div_ceil(num_cols);
    let margin = 2;

    let tile_h = h * scale;
    let tile_w = w * scale;

    let canvas_h = num_rows * tile_h + (num_rows + 1) * margin;
    let canvas_w = num_cols * tile_w + (num_cols + 1) * margin;

    let mut canvas = vec![vec![128u8; canvas_w]; canvas_h];

    for i in 0..num_images {
        let row = i / num_cols;
        let col = i % num_cols;
        let img = images.slice(ndarray::s![i, 0, .., ..]);

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
    let args: Vec<String> = std::env::args().collect();

    // 引数1: NPZ パス (省略時 output/step10_unet_epoch_10.npz)
    let weights_path_str = if args.len() > 1 {
        args[1].clone()
    } else {
        "output/step10_unet_epoch_10.npz".to_string()
    };

    // 引数2: gamma (省略時 3.0)
    let gamma: f32 = if args.len() > 2 {
        args[2].parse().expect("Failed to parse gamma as f32")
    } else {
        3.0
    };

    let weights_path = Path::new(&weights_path_str);
    if !weights_path.exists() {
        panic!("Weights file not found: {:?}", weights_path);
    }

    // パスからファイル名を抽出 (例: step10_unet_epoch_10)
    let stem = weights_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown_epoch");

    let output_dir = Path::new("output");
    fs::create_dir_all(output_dir).unwrap();

    // 出力ファイル名に epoch 名と gamma を織り込む
    let out_filename = format!("step10_generated_{}_gamma_{:.1}.pgm", stem, gamma);
    let filepath = output_dir.join(&out_filename);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    println!("Initializing UNet and Diffusion...");
    let unet = UNet::new(1, 100, &mut rng);
    let diffusion = Diffusion::new(1000, 1e-4, 0.02);

    println!("Loading weights from {:?}...", weights_path);
    unet.load_weights(weights_path).unwrap();

    let _guard = vol3::config::test_mode();
    let _no_grad = vol3::config::no_grad();

    let num_cols = 10;
    let num_rows = 4;
    let n_samples = num_cols * num_rows;

    let mut labels_vec = Vec::new();
    for _ in 0..num_rows {
        for l in 0..10 {
            labels_vec.push(l as usize);
        }
    }
    let labels = Array1::from_vec(labels_vec);

    println!(
        "Generating {} conditional images with gamma={:.1}...",
        n_samples, gamma
    );

    let mut x_sample = vol3::utils::random_array(
        ndarray::IxDyn(&[n_samples, 1, 28, 28]),
        rand_distr::StandardNormal,
        &mut rng,
    );

    for t_step in (1..=1000).rev() {
        if t_step % 100 == 0 || t_step == 1000 || t_step == 1 {
            println!("Sampling step {}/1000", t_step);
        }

        let t_array = Array1::from_elem(n_samples, t_step as f32);
        let x_var = Variable::new(x_sample.clone());

        // 1. 条件付き (Some(labels))
        let eps_cond_var = unet.forward(&x_var, &t_array, Some(&labels));
        let eps_cond = eps_cond_var.data();

        // 2. 無条件 (None)
        let eps_uncond_var = unet.forward(&x_var, &t_array, None);
        let eps_uncond = eps_uncond_var.data();

        // 3. ブレンド (CFG)
        let eps = &eps_uncond + gamma * (&eps_cond - &eps_uncond);

        // 4. デノイズ
        let z_data =
            vol3::utils::random_array(x_sample.dim(), rand_distr::StandardNormal, &mut rng);
        let z_opt = if t_step == 1 { None } else { Some(&z_data) };
        x_sample = diffusion.denoise_with_noise(&x_sample, &eps, t_step, z_opt);
    }

    let images_4d = x_sample.into_dimensionality::<ndarray::Ix4>().unwrap();

    println!("Saving generated images to {:?}", filepath);
    save_images_grid_as_pgm(&images_4d, &filepath, num_cols, 2).unwrap();
    println!("Done!");
}
