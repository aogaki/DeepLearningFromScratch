use crate::gaussian::multivariate_normal;
use ndarray::{Array1, Array2};

pub struct GaussianMixtureModel {
    pub k: usize,
    pub d: usize,
    pub pis: Vec<f32>,
    pub mus: Vec<Array1<f32>>,
    pub covs: Vec<Array2<f32>>,
}

impl GaussianMixtureModel {
    pub fn new(
        k: usize,
        d: usize,
        pis: Vec<f32>,
        mus: Vec<Array1<f32>>,
        covs: Vec<Array2<f32>>,
    ) -> Self {
        Self {
            k,
            d,
            pis,
            mus,
            covs,
        }
    }

    /// 対数尤度の計算 (単調増加の Assert などに用いる)
    pub fn log_likelihood(&self, x: &Array2<f32>) -> f32 {
        let n_samples = x.nrows();
        let mut ll = 0.0;
        for n in 0..n_samples {
            let x_n = x.row(n).to_owned();
            let mut prob = 0.0;
            for k in 0..self.k {
                prob += self.pis[k] * multivariate_normal(&x_n, &self.mus[k], &self.covs[k]);
            }
            // prob が 0 になるのを防ぐための微小値（アンダーフロー対策）
            ll += (prob + 1e-8).ln();
        }
        ll
    }

    /// Eステップ: 各データが各クラスタに属する負担率(gamma)を計算
    pub fn e_step(&self, x: &Array2<f32>) -> Array2<f32> {
        let n_samples = x.nrows();
        let mut gammas = Array2::<f32>::zeros((n_samples, self.k));

        for n in 0..n_samples {
            let x_n = x.row(n).to_owned();
            let mut denominator = 0.0;

            // 分子（負担率の元）の計算
            for k in 0..self.k {
                let numerator =
                    self.pis[k] * multivariate_normal(&x_n, &self.mus[k], &self.covs[k]);
                gammas[[n, k]] = numerator;
                denominator += numerator;
            }

            // 全クラスタの合計で割って正規化 (足して 1 にする)
            for k in 0..self.k {
                gammas[[n, k]] /= denominator + 1e-8;
            }
        }
        gammas
    }

    /// Mステップ: 負担率(gamma)を用いてパラメータを更新
    pub fn m_step(&mut self, x: &Array2<f32>, gammas: &Array2<f32>) {
        let n_samples = x.nrows();

        for k in 0..self.k {
            // N_k = sum_{n} gamma_{nk}
            let mut nk = 0.0;
            for n in 0..n_samples {
                nk += gammas[[n, k]];
            }

            // ゼロ割りを防ぐ
            nk = nk.max(1e-8);

            // 1. 混合係数の更新
            self.pis[k] = nk / (n_samples as f32);

            // 2. 平均ベクトルの更新
            let mut new_mu = Array1::<f32>::zeros(self.d);
            for n in 0..n_samples {
                let x_n = x.row(n);
                for i in 0..self.d {
                    new_mu[i] += gammas[[n, k]] * x_n[i];
                }
            }
            new_mu /= nk;
            self.mus[k] = new_mu.clone();

            // 3. 共分散行列の更新 (外積和)
            let mut new_cov = Array2::<f32>::zeros((self.d, self.d));
            for n in 0..n_samples {
                let x_n = x.row(n);
                let diff = &x_n - &new_mu;

                for i in 0..self.d {
                    for j in 0..self.d {
                        new_cov[[i, j]] += gammas[[n, k]] * diff[i] * diff[j];
                    }
                }
            }
            new_cov /= nk;
            self.covs[k] = new_cov;
        }
    }

    /// EMアルゴリズムの学習ループ
    /// 戻り値: 各イテレーションでの対数尤度の履歴 (Vec<f32>)
    pub fn fit(&mut self, x: &Array2<f32>, max_iter: usize, tol: f32) -> Vec<f32> {
        let mut ll_history = Vec::new();
        let mut prev_ll = self.log_likelihood(x);
        ll_history.push(prev_ll);
        for _ in 0..max_iter {
            let gammas = self.e_step(x);
            self.m_step(x, &gammas);

            let current_ll = self.log_likelihood(x);
            ll_history.push(current_ll);

            // 対数尤度の変化が十分小さくなれば収束 (絶対値で判定)
            if (current_ll - prev_ll).abs() < tol {
                break; // ライブラリは黙ってループを抜ける
            }
            prev_ll = current_ll;
        }

        ll_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_eq;
    use ndarray::array;

    #[test]
    fn test_em_algorithm_steps() {
        // 【手計算できる 1次元のトイデータ】
        // x = [1.0, 2.0, 8.0, 9.0]^T
        // 明らかに [1, 2] のグループと [8, 9] のグループに分かれる
        let x = array![[1.0f32], [2.0f32], [8.0f32], [9.0f32]];

        let k = 2;
        let d = 1;
        let pis = vec![0.5, 0.5];
        // 初期値として、グループ1は0付近、グループ2は10付近とする
        let mus = vec![array![0.0], array![10.0]];
        let covs = vec![array![[1.0]], array![[1.0]]];

        let mut gmm = GaussianMixtureModel::new(k, d, pis, mus, covs);

        // --- 1. Eステップの検証 ---
        let gammas = gmm.e_step(&x);

        // データ [1.0] と [2.0] は圧倒的にクラスタ0（mu=0）に近い
        assert!(gammas[[0, 0]] > 0.99); // x=1.0 は クラスタ0 の負担率ほぼ1
        assert!(gammas[[0, 1]] < 0.01);
        assert!(gammas[[1, 0]] > 0.99); // x=2.0 は クラスタ0 の負担率ほぼ1
        assert!(gammas[[1, 1]] < 0.01);

        // データ [8.0] と [9.0] は圧倒的にクラスタ1（mu=10）に近い
        assert!(gammas[[2, 0]] < 0.01);
        assert!(gammas[[2, 1]] > 0.99); // x=8.0 は クラスタ1 の負担率ほぼ1
        assert!(gammas[[3, 0]] < 0.01);
        assert!(gammas[[3, 1]] > 0.99); // x=9.0 は クラスタ1 の負担率ほぼ1

        // --- 2. Mステップの検証 ---
        // gammas が理想的に [1,0], [1,0], [0,1], [0,1] に分かれたとして手計算:
        // N_0 ≈ 2, N_1 ≈ 2
        // mu_0 ≈ (1.0 + 2.0) / 2 = 1.5
        // mu_1 ≈ (8.0 + 9.0) / 2 = 8.5
        // cov_0 ≈ ((1.0-1.5)^2 + (2.0-1.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
        // cov_1 ≈ ((8.0-8.5)^2 + (9.0-8.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25

        gmm.m_step(&x, &gammas);

        // 平均の検証
        assert!(
            approx_eq(gmm.mus[0][0], 1.5, 0.1),
            "mu0 was {}",
            gmm.mus[0][0]
        );
        assert!(
            approx_eq(gmm.mus[1][0], 8.5, 0.1),
            "mu1 was {}",
            gmm.mus[1][0]
        );

        // 共分散（分散）の検証
        assert!(
            approx_eq(gmm.covs[0][[0, 0]], 0.25, 0.1),
            "cov0 was {}",
            gmm.covs[0][[0, 0]]
        );
        assert!(
            approx_eq(gmm.covs[1][[0, 0]], 0.25, 0.1),
            "cov1 was {}",
            gmm.covs[1][[0, 0]]
        );

        // 混合係数の検証 (N=4 に対しそれぞれ2個ずつなので 0.5 になるはず)
        assert!(approx_eq(gmm.pis[0], 0.5, 0.05));
        assert!(approx_eq(gmm.pis[1], 0.5, 0.05));
    }
}
