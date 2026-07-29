use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;
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

/// Step 4: 混合ガウスモデル (GMM) の確率密度関数
/// x: 評価する点 (D次元)
/// pis: 各ガウス分布の重み (K次元)
/// mus: 各ガウス分布の平均ベクトル (K個のD次元ベクトル)
/// covs: 各ガウス分布の共分散行列 (K個のD x D行列)
pub fn gmm(x: &Array1<f32>, pis: &[f32], mus: &[Array1<f32>], covs: &[Array2<f32>]) -> f32 {
    // インデックスループを排除し、イテレータチェーンで重み付き和を計算
    pis.iter()
        .zip(mus)
        .zip(covs)
        .map(|((&pi, mu), cov)| pi * multivariate_normal(x, mu, cov))
        .sum()
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

// --- ゼロから作るサンプリング処理 ---
/// コレスキー分解 (A = L * L^T を満たす下三角行列 L を求める)
pub fn cholesky(matrix: &Array2<f32>) -> Option<Array2<f32>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return None;
    }

    let mut l = Array2::<f32>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }

            if i == j {
                let val = matrix[[i, i]] - sum;
                if val <= 0.0 {
                    return None;
                } // 正定値でない場合
                l[[i, j]] = val.sqrt();
            } else {
                l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
            }
        }
    }
    Some(l)
}

/// 多次元正規分布からのサンプリング (コレスキー分解を利用: x = μ + L * z)
pub fn sample_multivariate_normal<R: Rng>(
    mu: &Array1<f32>,
    cov: &Array2<f32>,
    rng: &mut R,
) -> Option<Array1<f32>> {
    let n = mu.len();
    let l = cholesky(cov)?;

    let z = crate::utils::random_normal_array(n, 0.0, 1.0, rng);

    Some(mu + l.dot(&z))
}

/// 混合ガウスモデル (GMM) からのトイデータのサンプリング
pub fn sample_gmm<R: Rng>(
    pis: &[f32],
    mus: &[Array1<f32>],
    covs: &[Array2<f32>],
    rng: &mut R,
) -> Option<Array1<f32>> {
    let dist = WeightedIndex::new(pis).ok()?;
    let k = dist.sample(rng);
    sample_multivariate_normal(&mus[k], &covs[k], rng)
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

    #[test]
    fn test_cholesky() {
        // 1. 手計算できる 2x2 行列
        // A = [[4, 2], [2, 3]]
        // L_00 = sqrt(4) = 2
        // L_10 = 2 / 2 = 1
        // L_11 = sqrt(3 - 1^2) = sqrt(2) ≈ 1.4142135
        let a = array![[4.0f32, 2.0f32], [2.0f32, 3.0f32]];
        let l = cholesky(&a).unwrap();

        // 要素が手計算と一致するか
        assert!(approx_eq(l[[0, 0]], 2.0, 1e-5));
        assert!(approx_eq(l[[0, 1]], 0.0, 1e-5));
        assert!(approx_eq(l[[1, 0]], 1.0, 1e-5));
        assert!(approx_eq(l[[1, 1]], 2.0f32.sqrt(), 1e-5));
        // L * L^T ≈ A の往復テスト
        let a_reconstructed = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx_eq(a[[i, j]], a_reconstructed[[i, j]], 1e-5));
            }
        }
        // 2. 非正定値行列 (行列式 det = 1*1 - 2*2 = -3 < 0) は None が返るべき
        let not_pd = array![[1.0f32, 2.0f32], [2.0f32, 1.0f32]];
        assert!(cholesky(&not_pd).is_none());
    }
}
