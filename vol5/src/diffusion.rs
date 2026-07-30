use ndarray::{Array1, ArrayD};
use rand_distr::StandardNormal;
use vol3::utils::random_array;

pub struct Diffusion {
    pub num_timesteps: usize,
    pub betas: Array1<f32>,
    pub alphas: Array1<f32>,
    pub alpha_bars: Array1<f32>,
}
impl Diffusion {
    pub fn new(num_timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let betas = Array1::linspace(beta_start, beta_end, num_timesteps);
        let alphas = 1.0 - &betas;

        let mut alpha_bars = Array1::zeros(num_timesteps);
        let mut prod = 1.0;
        for i in 0..num_timesteps {
            prod *= alphas[i];
            alpha_bars[i] = prod;
        }

        Self {
            num_timesteps,
            betas,
            alphas,
            alpha_bars,
        }
    }

    // ==========================================
    // 順過程 (q(x_t | x_0))
    // ==========================================

    /// パリティ検証用：外部からノイズ ε を直接注入する
    pub fn add_noise_with_noise(
        &self,
        x_0: &ArrayD<f32>,
        t: usize,
        noise: &ArrayD<f32>,
    ) -> ArrayD<f32> {
        assert!(t > 0 && t <= self.num_timesteps, "t must be in 1..=T");
        let t_idx = t - 1;
        let alpha_bar = self.alpha_bars[t_idx];

        x_0.mapv(|v| v * alpha_bar.sqrt()) + noise.mapv(|v| v * (1.0 - alpha_bar).sqrt())
    }

    /// 通常用：内部で乱数を生成して注入する（学習ターゲットとしてノイズも返す）
    pub fn add_noise(
        &self,
        x_0: &ArrayD<f32>,
        t: usize,
        rng: &mut impl rand::Rng,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        let noise = random_array(x_0.dim(), StandardNormal, rng);
        let x_t = self.add_noise_with_noise(x_0, t, &noise);
        (x_t, noise)
    }

    // ==========================================
    // 逆過程 (p(x_{t-1} | x_t))
    // ==========================================

    /// パリティ検証用：外部からサンプリングノイズ z を直接注入する
    pub fn denoise_with_noise(
        &self,
        x_t: &ArrayD<f32>,
        eps_hat: &ArrayD<f32>,
        t: usize,
        z: Option<&ArrayD<f32>>,
    ) -> ArrayD<f32> {
        assert!(t > 0 && t <= self.num_timesteps, "t must be in 1..=T");
        let t_idx = t - 1;
        let alpha = self.alphas[t_idx];
        let alpha_bar = self.alpha_bars[t_idx];

        let eps_coeff = (1.0 - alpha) / (1.0 - alpha_bar).sqrt();
        let mean = (x_t - eps_hat.mapv(|v| v * eps_coeff)).mapv(|v| v / alpha.sqrt());

        if t == 1 {
            // [地雷の解決] t=1 (ᾱ_0 = 1.0 に相当する地点) では分散が 0 になるため、ノイズは足さない
            mean
        } else {
            // ここでのみ alpha_bar_prev を参照し、死荷重を排除
            let alpha_bar_prev = self.alpha_bars[t_idx - 1];
            let std = ((1.0 - alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)).sqrt();

            let z_array = z.expect("t > 1 requires a noise array (z)");
            mean + z_array.mapv(|v| v * std)
        }
    }

    /// 通常用：内部で乱数を生成して注入する
    pub fn denoise(
        &self,
        x_t: &ArrayD<f32>,
        eps_hat: &ArrayD<f32>,
        t: usize,
        rng: &mut impl rand::Rng,
    ) -> ArrayD<f32> {
        let z = if t == 1 {
            None
        } else {
            Some(random_array(x_t.dim(), StandardNormal, rng))
        };
        self.denoise_with_noise(x_t, eps_hat, t, z.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_eq;
    use rand::SeedableRng; // 固定シードを使う場合は StdRng などを使用
    #[test]
    fn test_step8_diffusion_theory() {
        let diff = Diffusion::new(1000, 1e-4, 0.02);
        let t = 100;
        let t_idx = t - 1;
        let n_samples = 100_000;
        let x_0 = ArrayD::from_elem(vec![n_samples], 5.0);

        // 再現性のため固定シードを推奨 (プロジェクト規約)
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // 1. 閉形式
        let (x_t_closed, _) = diff.add_noise(&x_0, t, &mut rng);
        let mean_closed = x_t_closed.mean().unwrap();
        let var_closed = x_t_closed.var(0.0);
        // 2. 逐次 1 ステップ
        let mut x_t_seq = x_0.clone();
        for step in 1..=t {
            let step_idx = step - 1;
            let alpha = diff.alphas[step_idx];
            let noise = random_array(x_t_seq.dim(), StandardNormal, &mut rng);

            x_t_seq = x_t_seq.mapv(|v| v * alpha.sqrt()) + noise.mapv(|v| v * (1.0 - alpha).sqrt());
        }
        let mean_seq = x_t_seq.mean().unwrap();
        let var_seq = x_t_seq.var(0.0);

        // 理論値
        let alpha_bar_t = diff.alpha_bars[t_idx];
        let expected_mean = 5.0 * alpha_bar_t.sqrt();
        let expected_var = 1.0 - alpha_bar_t;

        // 比較
        let tol = 1e-2;
        assert!(
            approx_eq(mean_closed, expected_mean, tol),
            "閉形式 平均不一致"
        );
        assert!(
            approx_eq(var_closed, expected_var, tol),
            "閉形式 分散不一致"
        );
        assert!(
            approx_eq(mean_seq, expected_mean, tol),
            "逐次更新 平均不一致"
        );
        assert!(approx_eq(var_seq, expected_var, tol), "逐次更新 分散不一致");
    }
}
