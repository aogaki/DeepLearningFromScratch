use ndarray::{Array1, Array2, array};
use rand::SeedableRng;
use rand::rngs::StdRng;
use vol5::gaussian::{gmm, sample_gmm, sample_multivariate_normal};
use vol5::utils::approx_eq;

#[test]
fn experiment_step4() {
    println!("--- Step 4: 混合ガウスモデル ---");

    // 【検証0】多次元正規分布のサンプリング「形状」検証
    // サンプルが本当に指定した相関を持つか、標本平均と標本共分散でチェックする
    let mut rng = StdRng::seed_from_u64(42);
    let n_samples_cov = 10000;

    let mu_true = array![1.0f32, 2.0f32];
    let cov_true = array![[2.0f32, 0.5f32], [0.5f32, 1.0f32]];

    let mut sum_x = Array1::<f32>::zeros(2);
    let mut sum_xx = Array2::<f32>::zeros((2, 2));

    for _ in 0..n_samples_cov {
        let x = sample_multivariate_normal(&mu_true, &cov_true, &mut rng).unwrap();
        sum_x += &x;
        // x * x^T の計算 (外積)
        sum_xx[[0, 0]] += x[0] * x[0];
        sum_xx[[0, 1]] += x[0] * x[1];
        sum_xx[[1, 0]] += x[1] * x[0];
        sum_xx[[1, 1]] += x[1] * x[1];
    }

    let mu_est = sum_x / (n_samples_cov as f32);
    // 標本共分散 = E[xx^T] - E[x]E[x]^T
    let mut cov_est = sum_xx / (n_samples_cov as f32);
    cov_est[[0, 0]] -= mu_est[0] * mu_est[0];
    cov_est[[0, 1]] -= mu_est[0] * mu_est[1];
    cov_est[[1, 0]] -= mu_est[1] * mu_est[0];
    cov_est[[1, 1]] -= mu_est[1] * mu_est[1];

    println!("標本平均: {:?}", mu_est);
    println!("標本共分散: \n{:?}", cov_est);

    // 1万サンプルなので、0.1 未満の誤差に収束する
    assert!(approx_eq(mu_est[0], mu_true[0], 0.1));
    assert!(approx_eq(mu_est[1], mu_true[1], 0.1));
    assert!(approx_eq(cov_est[[0, 0]], cov_true[[0, 0]], 0.1));
    assert!(approx_eq(cov_est[[1, 1]], cov_true[[1, 1]], 0.1));
    assert!(approx_eq(cov_est[[0, 1]], cov_true[[0, 1]], 0.1));
    assert!(approx_eq(cov_est[[1, 0]], cov_true[[1, 0]], 0.1)); // 最重要: 非対角（相関）成分の検証
    // --- 以下、GMM の検証 ---
    let mu1 = array![0.0f32, 0.0f32];
    let cov1 = array![[1.0f32, 0.0f32], [0.0f32, 1.0f32]];
    let mu2 = array![3.0f32, 3.0f32];
    let cov2 = array![[1.0f32, 0.5f32], [0.5f32, 1.0f32]];
    let mus = vec![mu1, mu2];
    let covs = vec![cov1, cov2];
    let pis = vec![0.4f32, 0.6f32];

    // 【検証1】特定座標での確率密度を手計算と照合
    let x1 = array![0.0f32, 0.0f32];
    let p_x1 = gmm(&x1, &pis, &mus, &covs);
    assert!(approx_eq(p_x1, 0.063935, 1e-4));

    // 【検証2】2重リーマン和で全確率=1を検証 (積分範囲 -5.0 〜 8.0)
    let dx = 0.2f32;
    let mut integral = 0.0f32;
    let mut x_val = -5.0f32;
    while x_val <= 8.0 {
        let mut y_val = -5.0f32;
        while y_val <= 8.0 {
            let point = array![x_val, y_val];
            integral += gmm(&point, &pis, &mus, &covs) * (dx * dx);
            y_val += dx;
        }
        x_val += dx;
    }
    println!("2Dリーマン和による積分値: {:.6}", integral);
    assert!(approx_eq(integral, 1.0, 1e-2));

    // 【検証3】トイデータの生成 (Step5 の EMアルゴリズム用)
    let mut sample_count_k1_region = 0;
    let n_samples_gmm = 10000;

    for _ in 0..n_samples_gmm {
        let sample = sample_gmm(&pis, &mus, &covs, &mut rng).unwrap();
        // x座標が 1.5 未満の数をカウント
        if sample[0] < 1.5 {
            sample_count_k1_region += 1;
        }
    }

    // 理論値の計算:
    // P(x < 1.5) = pi_1 * Φ((1.5 - mu1_x) / sigma1_x) + pi_2 * Φ((1.5 - mu2_x) / sigma2_x)
    // 累積分布関数 Φ について:
    // Φ(1.5) ≈ 0.9332
    // Φ(-1.5) ≈ 0.0668
    // 真値 ≈ 0.4 * 0.9332 + 0.6 * 0.0668 = 0.37328 + 0.04008 = 0.41336
    let expected_ratio = 0.41336f32;
    let ratio = sample_count_k1_region as f32 / n_samples_gmm as f32;
    println!(
        "x < 1.5 に生成された割合: {:.4} (真値: {:.4})",
        ratio, expected_ratio
    );

    // 手計算値との照合に格上げされたため、tol は大数の法則に従いより厳しく設定できる (2.3σ でなく)
    assert!(approx_eq(ratio, expected_ratio, 0.02));
}
