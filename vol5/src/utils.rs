use ndarray::Array1;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// vol3 と同様の、乱数生成器 (RNG) を注入して配列を埋めるヘルパー関数
pub fn random_normal_array<R: Rng>(size: usize, mu: f32, sigma: f32, rng: &mut R) -> Array1<f32> {
    let dist = Normal::new(mu, sigma).unwrap();
    Array1::from_shape_simple_fn(size, || dist.sample(rng))
}

/// 実数の比較用関数
pub fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}
