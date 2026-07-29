use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// vol3 と同様の、乱数生成器 (RNG) を注入して配列を埋めるヘルパー関数
pub fn random_normal_array<R: Rng>(size: usize, mu: f32, sigma: f32, rng: &mut R) -> Array1<f32> {
    let dist = Normal::new(mu, sigma).unwrap();
    Array1::from_shape_simple_fn(size, || dist.sample(rng))
}

/// 実数の比較用関数
pub fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// スペース区切りのプレーンテキストを読み込み、N x D の Array2<f32> を返す
pub fn load_txt(path: &str) -> Array2<f32> {
    let file = File::open(path).expect("Failed to open dataset file");
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    let mut cols = 0;

    for line in reader.lines() {
        let line = line.unwrap();
        let row: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if !row.is_empty() {
            if cols == 0 {
                cols = row.len();
            }
            data.extend(row);
        }
    }

    let rows = data.len() / cols;
    Array2::from_shape_vec((rows, cols), data).expect("Failed to create Array2")
}
