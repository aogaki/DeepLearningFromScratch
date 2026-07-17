//! 本 7.6.1「1層目の重みの可視化」学習前後の conv フィルタ(30, 1, 5, 5)を
//! タイル状の PGM 画像に保存し、エッジ/ブロブ検出器が育つ様子を目視する。
//! 実行: vol1/ で `cargo run -p vol1 --example visualize_filters --release`
//! (出力は実行時のカレントディレクトリ基準で output/filters/ に書かれる)。
//! 学習設定は本の train_convnet.py 相当(Adam lr=0.001・init std=0.01・20 エポック)。
//! 経験則: フィルタの見た目は「学習した構造 ÷ 初期値ノイズ」の比で決まる —
//! lr の桁違い(Adam に 0.1)や大きすぎる init(Xavier=0.2)では構造がノイズに埋もれる。

use ndarray::{Array4, Axis};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use vol1::mnist::{load_images, load_labels, to_one_hot};
use vol1::optimizer::{Adam, Optimizer};
use vol1::simple_conv_net::{ConvParams, SimpleConvNet};

/// 全てのフィルタをタイル状に並べて1枚のPGM画像として保存する
fn save_filters_grid_as_pgm(w: &Array4<f32>, filepath: &Path, scale: usize) -> io::Result<()> {
    let (filter_num, _c, fh, fw) = w.dim(); // (30, 1, 5, 5)

    // タイル配置の設定 (30枚なら 5行 x 6列)
    let num_cols = 6;
    let num_rows = (filter_num + num_cols - 1) / num_cols;
    let margin = 2; // 各フィルタの間の隙間(ピクセル数)

    let tile_h = fh * scale;
    let tile_w = fw * scale;

    let canvas_h = num_rows * tile_h + (num_rows + 1) * margin;
    let canvas_w = num_cols * tile_w + (num_cols + 1) * margin;

    // キャンバスを作成し、背景色(128: 灰色)で塗りつぶす
    let mut canvas = vec![vec![128u8; canvas_w]; canvas_h];

    // 各フィルタを描画していく
    for i in 0..filter_num {
        let row = i / num_cols;
        let col = i % num_cols;

        let filter = w.slice(ndarray::s![i, 0, .., ..]);

        // フィルタごとの min-max を求める (コントラストの最大化)
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in filter.iter() {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        if (max_val - min_val).abs() < 1e-7 {
            max_val = min_val + 1.0; // ゼロ割防止
        }

        // キャンバス上の描画開始位置
        let start_y = margin + row * (tile_h + margin);
        let start_x = margin + col * (tile_w + margin);

        for y in 0..tile_h {
            for x in 0..tile_w {
                // scale倍に最近傍補間
                let orig_y = y / scale;
                let orig_x = x / scale;
                let val = filter[[orig_y, orig_x]];

                let normalized = (val - min_val) / (max_val - min_val);
                let pixel_val = (normalized * 255.0).clamp(0.0, 255.0).round() as u8;

                canvas[start_y + y][start_x + x] = pixel_val;
            }
        }
    }

    // PGM として書き出し
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
    let output_dir = Path::new("output/filters");
    fs::create_dir_all(output_dir).unwrap();

    let input_dim = (1, 28, 28);
    let conv_params = ConvParams {
        filter_num: 30,
        filter_size: 5,
        stride: 1,
        pad: 0,
    };

    let lr = 0.001;
    // let make_opt = || Box::new(SGD::new(lr)) as Box<dyn Optimizer>;
    let make_opt = || Box::new(Adam::new(lr)) as Box<dyn Optimizer>;
    // let make_std = |n: usize| (1.0 / n as f32).sqrt();
    let make_std = |_| 0.01;

    let mut net = SimpleConvNet::new(input_dim, conv_params, 100, 10, make_opt, make_std);

    // --- 1. 学習前のフィルタを「1枚の画像」として保存 ---
    println!("Saving pre-training filters grid...");
    let w_before = net.conv_weights().clone();
    let path_before = output_dir.join("filter_before.pgm");
    save_filters_grid_as_pgm(&w_before, &path_before, 10).unwrap();

    // --- 2. MNIST で軽く学習 ---
    println!("Loading MNIST dataset...");
    let images = load_images("dataset/train-images-idx3-ubyte");
    let labels = load_labels("dataset/train-labels-idx1-ubyte");
    let train_size = images.shape()[0];

    let batch_size = 100;
    let iters_num = 12000;

    println!("Training for {} iterations...", iters_num);
    let mut rng = rand::rng();

    for _ in 0..iters_num {
        let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();

        let x_batch = images.select(Axis(0), &idx);
        let x_batch_4d = x_batch
            .into_shape_with_order((batch_size, 1, 28, 28))
            .unwrap();
        let t_batch = to_one_hot(&idx.iter().map(|&j| labels[j]).collect::<Vec<_>>(), 10);

        net.gradient(&x_batch_4d, t_batch);
        net.update();
    }

    // --- 3. 学習後のフィルタを「1枚の画像」として保存 ---
    println!("Saving post-training filters grid...");
    let w_after = net.conv_weights();
    let path_after = output_dir.join("filter_after.pgm");
    save_filters_grid_as_pgm(w_after, &path_after, 10).unwrap();

    println!("Done! Check 'filter_before.pgm' and 'filter_after.pgm' in output/filters.");
}
