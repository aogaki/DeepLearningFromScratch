use ndarray::{Array1, Array2};
use std::f32::consts::PI;

/// Step 1: 1次元の正規分布の確率密度関数 (スカラー版)
pub fn normal(x: f32, mu: f32, sigma: f32) -> f32 {
    let z = (x - mu) / sigma;
    (1.0 / ((2.0 * PI).sqrt() * sigma)) * (-0.5 * z * z).exp()
}

/// Step 1: 1次元の正規分布の確率密度関数 (配列版)
pub fn normal_array(x: &Array1<f32>, mu: f32, sigma: f32) -> Array1<f32> {
    x.mapv(|v| normal(v, mu, sigma))
}

/// Step 2: 1次元正規分布の最尤推定
pub fn fit_normal(x: &Array1<f32>) -> (f32, f32) {
    let mu = x.mean().unwrap_or(0.0);
    let variance = x.mapv(|v| (v - mu).powi(2)).mean().unwrap_or(0.0);
    let sigma = variance.sqrt();
    (mu, sigma)
}

/// Step 3: 多次元の正規分布の確率密度関数
pub fn multivariate_normal(x: &Array1<f32>, mu: &Array1<f32>, cov: &Array2<f32>) -> f32 {
    let d = x.len() as f32;

    // スクラッチ実装した行列計算を使用
    let det = determinant(cov);
    let inv_cov = invert_matrix(cov).expect("Covariance matrix must be invertible");

    let diff = x - mu;
    let exponent = -0.5 * diff.dot(&inv_cov).dot(&diff);

    let denom = ((2.0 * PI).powf(d) * det).sqrt();

    exponent.exp() / denom
}

// --- 以下、ゼロから作る行列計算 (Pure Rust) ---

/// 行列式を求める (ガウスの消去法)
pub fn determinant(matrix: &Array2<f32>) -> f32 {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return 0.0;
    }

    let mut m = matrix.clone();
    let mut det = 1.0;

    for i in 0..n {
        let mut pivot_row = i;
        let mut max_val = m[[i, i]].abs();
        // ピボットの探索
        for k in (i + 1)..n {
            let val = m[[k, i]].abs();
            if val > max_val {
                max_val = val;
                pivot_row = k;
            }
        }
        if max_val < 1e-6 {
            return 0.0;
        } // ゼロ割りを防ぐ

        // 行の入れ替え
        if pivot_row != i {
            for j in i..n {
                let temp = m[[i, j]];
                m[[i, j]] = m[[pivot_row, j]];
                m[[pivot_row, j]] = temp;
            }
            det *= -1.0; // 行を入れ替えると行列式の符号が反転する
        }

        let pivot = m[[i, i]];
        det *= pivot;

        // 掃き出し
        for k in (i + 1)..n {
            let factor = m[[k, i]] / pivot;
            for j in i..n {
                m[[k, j]] -= factor * m[[i, j]];
            }
        }
    }
    det
}

/// 逆行列を求める (ガウス・ジョルダン法)
pub fn invert_matrix(matrix: &Array2<f32>) -> Option<Array2<f32>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return None;
    }

    // 拡大行列 [A | I] の作成
    let mut aug = Array2::<f32>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, i + n]] = 1.0;
    }

    for i in 0..n {
        let mut pivot_row = i;
        let mut max_val = aug[[i, i]].abs();
        for k in (i + 1)..n {
            let val = aug[[k, i]].abs();
            if val > max_val {
                max_val = val;
                pivot_row = k;
            }
        }

        if max_val < 1e-6 {
            return None;
        } // 逆行列が存在しない

        if pivot_row != i {
            for j in i..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = temp;
            }
        }

        // ピボット行の正規化
        let pivot = aug[[i, i]];
        for j in i..(2 * n) {
            aug[[i, j]] /= pivot;
        }

        // 他の行の掃き出し
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in i..(2 * n) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    // 右半分から逆行列を抽出
    let mut inv = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, j + n]];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use crate::utils::approx_eq;

    use super::*;
    use ndarray::array;

    #[test]
    fn test_normal() {
        let y = normal(0.0, 0.0, 1.0);
        assert!(approx_eq(y, 0.3989423, 1e-5));
    }

    #[test]
    fn test_fit_normal() {
        let x = array![0.5, 1.5, 2.5, 3.5];
        let (mu, sigma) = fit_normal(&x);
        assert!(approx_eq(mu, 2.0, 1e-6), "mu = {}", mu);
        assert!(approx_eq(sigma, 1.25_f32.sqrt(), 1e-6), "sigma = {}", sigma);
    }

    #[test]
    fn test_multivariate_normal() {
        let x = array![0.0, 0.0];
        let mu = array![0.0, 0.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        let y = multivariate_normal(&x, &mu, &cov);
        assert!(approx_eq(y, 0.15915494, 1e-5));
    }

    #[test]
    fn test_linalg_3x3_swap() {
        // a[0][0] = 0.0 の非対称行列で、ピボット交換 (det *= -1.0) の分岐を強制する
        let m = array![[0.0, 2.0, 1.0], [1.0, 0.0, -1.0], [2.0, 1.0, 1.0]];

        // 1. 行列式のテスト (手計算: det = -5.0)
        let det = determinant(&m);
        assert!(approx_eq(det, -5.0, 1e-5), "det was {}", det);

        // 2. 逆行列のテスト (手計算の adj(A) / det(A) と比較)
        let inv = invert_matrix(&m).unwrap();
        let expected_inv = array![[-0.2, 0.2, 0.4], [0.6, 0.4, -0.2], [-0.2, -0.8, 0.4]];

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (inv[[i, j]] - expected_inv[[i, j]]).abs() < 1e-5,
                    "Mismatch at [{}, {}]",
                    i,
                    j
                );
            }
        }

        // 3. A * A^-1 ≈ I の全要素の整合テスト
        let i_matrix = m.dot(&inv);
        let expected_i = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(i_matrix[[i, j]], expected_i[[i, j]], 1e-5));
            }
        }
    }

    #[test]
    fn test_linalg_singular() {
        // 特異行列 (行列式が 0 のケース)
        let m = array![[1.0, 2.0], [2.0, 4.0]];

        // 0.0 であり、逆行列は None となることをテスト
        assert!(approx_eq(determinant(&m), 0.0, 1e-5));
        assert!(invert_matrix(&m).is_none());
    }
}
