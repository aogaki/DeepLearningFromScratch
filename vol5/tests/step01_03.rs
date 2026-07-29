use ndarray::{Array1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

use vol5::gaussian::{fit_normal, multivariate_normal, normal_array};
use vol5::utils::{approx_eq, random_normal_array};

#[test]
fn experiment_step1() {
    println!("--- Step 1: 1次元の正規分布 ---");
    let x_vals: Array1<f32> = Array1::range(-5.0, 5.0, 0.1);
    let y_vals = normal_array(&x_vals, 0.0, 1.0);

    println!(
        "x[0]  (x={0:+.1}) = {0:.6}, y[0]  = {1:.6}",
        x_vals[0], y_vals[0]
    );
    println!(
        "x[50] (x={0:+.1}) = {0:.6}, y[50] = {1:.6}",
        x_vals[50], y_vals[50]
    );
    println!(
        "x[99] (x={0:+.1}) = {0:.6}, y[99] = {1:.6}",
        x_vals[99], y_vals[99]
    );

    // 【検証】「図」の代わりとして、確率密度関数の規格化条件をリーマン和で確認
    // ∫p(x)dx ≈ Σ p(x) * dx = 1 になることをテストする
    let dx = 0.1f32;
    let integral = y_vals.sum() * dx;
    println!("リーマン和による積分値: {:.6}", integral);
    assert!(approx_eq(integral, 1.0, 1e-5), "integral = {}", integral);
}

#[test]
fn experiment_step2() {
    println!("--- Step 2: 最尤推定 ---");
    // シード固定の StdRng を使用して「宝くじ」を排除
    let mut rng = StdRng::seed_from_u64(42);

    // 生成関数へ乱数生成器を注入
    let xs = random_normal_array(1000, 2.0, 1.0, &mut rng);

    let (mu_est, sigma_est) = fit_normal(&xs);

    println!("真のパラメータ: mu = 2.0, sigma = 1.0");
    println!(
        "推定パラメータ: mu = {:.4}, sigma = {:.4}",
        mu_est, sigma_est
    );

    // シードが固定されているため、何度走らせても一貫した結果で確実に pass します
    assert!(approx_eq(mu_est, 2.0, 0.1), "mu_est = {}", mu_est);
    assert!(approx_eq(sigma_est, 1.0, 0.1), "sigma_est = {}", sigma_est);
}

#[test]
fn experiment_step3() {
    println!("--- Step 3: 多次元の正規分布 ---");
    let x = array![0.5f32, -0.2f32];
    let mu = array![0.0f32, 0.0f32];
    let cov = array![[2.0f32, 0.3f32], [0.3f32, 0.5f32]];

    let y = multivariate_normal(&x, &mu, &cov);

    println!("評価点 x = {:?}", x);
    println!("平均 mu = {:?}", mu);
    println!("共分散行列 cov = \n{:?}", cov);
    println!("確率密度 p(x) = {:.6}", y);

    // 【検証】「実験」も本の結論を手計算して Assert する (家風の徹底)
    // 1. det = 2.0 * 0.5 - 0.3 * 0.3 = 0.91
    // 2. inv = [[0.5/0.91, -0.3/0.91], [-0.3/0.91, 2.0/0.91]]
    // 3. diff = [0.5, -0.2]^T
    // 4. diff^T * inv * diff = (1 / 0.91) * (0.5*(0.25+0.06) - 0.2*(-0.15-0.4))
    //                        = (1 / 0.91) * 0.265 ≈ 0.29120879
    // 5. exp_part = exp(-0.5 * 0.29120879) ≈ exp(-0.145604395) ≈ 0.864500
    // 6. denom = 2π * sqrt(0.91) ≈ 6.2831853 * 0.9539392 ≈ 5.993952
    // 7. p(x) = 0.864500 / 5.993952 ≈ 0.144228

    let expected_y = 0.144228;
    assert!(
        approx_eq(y, expected_y, 1e-5),
        "y = {}, expected_y = {}",
        y,
        expected_y
    );
}
