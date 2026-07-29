use rand_distr::Normal;
use vol3::layers::{Layer, Linear};
use vol3::utils::random_array;
use vol3::variable::Variable;

/// --- Reparameterization Trick ---
/// テスト用/パリティ検証用の注入口 (勾配は mu と sigma に流れる)
pub fn reparameterize_with_eps(mu: &Variable, sigma: &Variable, eps: &Variable) -> Variable {
    mu + &(sigma * eps)
}

/// 通常利用のラッパー
pub fn reparameterize(mu: &Variable, sigma: &Variable, rng: &mut impl rand::Rng) -> Variable {
    let eps_data = random_array(mu.shape(), Normal::new(0.0, 1.0).unwrap(), rng);
    let eps = Variable::new(eps_data.into_dyn());
    reparameterize_with_eps(mu, sigma, &eps)
}

/// --- KL Divergence ---
/// 注意: 標準 KL の 2 倍を返す(本の 2×ELBO 流儀)
/// vol3 の mean_squared_error と整合させるため、バッチサイズで除算する
pub fn kl_divergence(mu: &Variable, ln_var: &Variable) -> Variable {
    let batch_size = mu.shape()[0] as f32;
    // 2×ELBO流儀: 0.5 を掛けない
    let term = ln_var + 1.0 - &(mu * mu) - ln_var.exp();
    term.sum() * (-1.0 / batch_size)
}

/// --- Layers ---
pub struct Encoder {
    pub l1: Linear,
    pub l_mu: Linear,
    pub l_ln_var: Linear,
}
impl Encoder {
    pub fn new(
        in_size: usize,
        hidden_size: usize,
        latent_size: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        Self {
            l1: Linear::new(in_size, hidden_size, false, rng),
            l_mu: Linear::new(hidden_size, latent_size, false, rng),
            l_ln_var: Linear::new(hidden_size, latent_size, false, rng),
        }
    }

    pub fn encode(&self, x: &Variable) -> (Variable, Variable) {
        let h = self.l1.forward(x).relu();
        let mu = self.l_mu.forward(&h);
        let ln_var = self.l_ln_var.forward(&h);
        (mu, ln_var)
    }
}

// 委譲パターン (forward は使えないため unimplemented にし、npz保存等の機能を活かす)
impl Layer for Encoder {
    fn params(&self) -> Vec<Variable> {
        let mut p = self.l1.params();
        p.extend(self.l_mu.params());
        p.extend(self.l_ln_var.params());
        p
    }
    fn named_params(&self) -> Vec<(String, Variable)> {
        let mut p = Vec::new();
        for (name, param) in self.l1.named_params() {
            p.push((format!("l1/{}", name), param));
        }
        for (name, param) in self.l_mu.named_params() {
            p.push((format!("l_mu/{}", name), param));
        }
        for (name, param) in self.l_ln_var.named_params() {
            p.push((format!("l_ln_var/{}", name), param));
        }
        p
    }
    fn forward(&self, _x: &Variable) -> Variable {
        unimplemented!("Use encode()")
    }
}

pub struct Decoder {
    pub l1: Linear,
    pub l2: Linear,
}
impl Decoder {
    pub fn new(
        latent_size: usize,
        hidden_size: usize,
        out_size: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        Self {
            l1: Linear::new(latent_size, hidden_size, false, rng),
            l2: Linear::new(hidden_size, out_size, false, rng),
        }
    }
}

impl Layer for Decoder {
    fn params(&self) -> Vec<Variable> {
        let mut p = self.l1.params();
        p.extend(self.l2.params());
        p
    }
    fn named_params(&self) -> Vec<(String, Variable)> {
        let mut p = Vec::new();
        for (name, param) in self.l1.named_params() {
            p.push((format!("l1/{}", name), param));
        }
        for (name, param) in self.l2.named_params() {
            p.push((format!("l2/{}", name), param));
        }
        p
    }
    fn forward(&self, z: &Variable) -> Variable {
        let h = self.l1.forward(z).relu();
        self.l2.forward(&h).sigmoid()
    }
}

pub struct VAE {
    pub enc: Encoder,
    pub dec: Decoder,
}
impl VAE {
    pub fn new(
        in_size: usize,
        hidden_size: usize,
        latent_size: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        Self {
            enc: Encoder::new(in_size, hidden_size, latent_size, rng),
            dec: Decoder::new(latent_size, hidden_size, in_size, rng),
        }
    }

    /// Loss 計算用に y, mu, ln_var を返す
    pub fn forward_vae(
        &self,
        x: &Variable,
        rng: &mut impl rand::Rng,
    ) -> (Variable, Variable, Variable) {
        let (mu, ln_var) = self.enc.encode(x);
        let sigma = (&ln_var * 0.5).exp();
        let z = reparameterize(&mu, &sigma, rng);
        let y = self.dec.forward(&z);
        (y, mu, ln_var)
    }
}

impl Layer for VAE {
    fn params(&self) -> Vec<Variable> {
        let mut p = self.enc.params();
        p.extend(self.dec.params());
        p
    }
    fn named_params(&self) -> Vec<(String, Variable)> {
        let mut p = Vec::new();
        for (name, param) in self.enc.named_params() {
            p.push((format!("enc/{}", name), param));
        }
        for (name, param) in self.dec.named_params() {
            p.push((format!("dec/{}", name), param));
        }
        p
    }
    fn forward(&self, _x: &Variable) -> Variable {
        unimplemented!("Use forward_vae()")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_eq;
    use rand::SeedableRng;

    #[test]
    fn test_kl_divergence_manual() {
        // 1. KL 手計算テスト
        // バッチサイズ 2, 潜在次元 2 の非対称テンソル
        let mu_data = ndarray::array![[0.0, 1.0], [-1.0, 0.0]].into_dyn();
        let ln_var_data = ndarray::array![[0.0, 0.0], [1.0, -1.0]].into_dyn();

        let mu = Variable::new(mu_data);
        let ln_var = Variable::new(ln_var_data);

        let kl = kl_divergence(&mu, &ln_var);
        let kl_val = kl
            .data()
            .into_dimensionality::<ndarray::Ix0>()
            .unwrap()
            .into_scalar();

        // KLの手計算: -0.5 / N * sum(1 + ln_var - mu^2 - exp(ln_var))   ※ N = 2
        // Element (0,0): mu= 0, ln_var= 0 -> 1 + 0 - 0 - e^0 = 0
        // Element (0,1): mu= 1, ln_var= 0 -> 1 + 0 - 1 - e^0 = -1
        // Element (1,0): mu=-1, ln_var= 1 -> 1 + 1 - 1 - e^1 = 1 - e
        // Element (1,1): mu= 0, ln_var=-1 -> 1 - 1 - 0 - e^-1 = -e^-1
        // Sum = -1 + (1 - e) - e^-1 = -e - e^-1
        // Expected = -0.5 / 2.0 * (-e - e^-1) = 0.25 * (e + e^-1)
        let e = std::f32::consts::E;
        let expected = 0.5 * (e + 1.0 / e);

        assert!(
            approx_eq(kl_val, expected, 1e-5),
            "KL manual check failed. Expected: {}, Got: {}",
            expected,
            kl_val
        );
    }

    #[test]
    fn test_reparameterize_numerical_grad() {
        // 2. 勾配の数値微分テスト
        let mu_data = ndarray::array![[1.0, 2.0]].into_dyn();
        let sigma_data = ndarray::array![[0.5, 1.5]].into_dyn();
        let eps_data = ndarray::array![[0.1, -0.2]].into_dyn();

        let mu = Variable::new(mu_data.clone());
        let sigma = Variable::new(sigma_data.clone());
        let eps = Variable::new(eps_data.clone());

        // backprop用
        let z = reparameterize_with_eps(&mu, &sigma, &eps);
        let z_sum = z.sum();
        z_sum.backward(false, false);

        let mu_grad = mu
            .grad()
            .unwrap()
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let sigma_grad = sigma
            .grad()
            .unwrap()
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        let h = 5e-3; // 数値微分のステップサイズ f32 では分子の桁落ちが支配的、∛ε が最適刻み
        // mu[0,0] の数値微分
        let mut mu_plus = mu_data.clone();
        mu_plus[[0, 0]] += h;
        let z_plus = reparameterize_with_eps(&Variable::new(mu_plus), &sigma, &eps).sum();
        let z_plus_val = z_plus
            .data()
            .into_dimensionality::<ndarray::Ix0>()
            .unwrap()
            .into_scalar();

        let mut mu_minus = mu_data.clone();
        mu_minus[[0, 0]] -= h;
        let z_minus = reparameterize_with_eps(&Variable::new(mu_minus), &sigma, &eps).sum();
        let z_minus_val = z_minus
            .data()
            .into_dimensionality::<ndarray::Ix0>()
            .unwrap()
            .into_scalar();

        let num_grad_mu00 = (z_plus_val - z_minus_val) / (2.0 * h);
        assert!(
            approx_eq(mu_grad[[0, 0]], num_grad_mu00, 1e-3),
            "mu[0,0] grad mismatch: autograd={}, numerical={}",
            mu_grad[[0, 0]],
            num_grad_mu00
        );

        // sigma[0,1] の数値微分
        let mut sigma_plus = sigma_data.clone();
        sigma_plus[[0, 1]] += h;
        let z_plus_s = reparameterize_with_eps(&mu, &Variable::new(sigma_plus), &eps).sum();
        let z_plus_s_val = z_plus_s
            .data()
            .into_dimensionality::<ndarray::Ix0>()
            .unwrap()
            .into_scalar();

        let mut sigma_minus = sigma_data.clone();
        sigma_minus[[0, 1]] -= h;
        let z_minus_s = reparameterize_with_eps(&mu, &Variable::new(sigma_minus), &eps).sum();
        let z_minus_s_val = z_minus_s
            .data()
            .into_dimensionality::<ndarray::Ix0>()
            .unwrap()
            .into_scalar();

        let num_grad_sigma01 = (z_plus_s_val - z_minus_s_val) / (2.0 * h);
        assert!(
            approx_eq(sigma_grad[[0, 1]], num_grad_sigma01, 1e-3),
            "sigma[0,1] grad mismatch: autograd={}, numerical={}",
            sigma_grad[[0, 1]],
            num_grad_sigma01
        );
    }

    #[test]
    fn test_reparameterize_statistics() {
        // 3. 統計テスト (step04の再演)
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let n = 10000;
        let mu_val = 2.0;
        let sigma_val = 3.0;

        let mu = Variable::new(ndarray::array![[mu_val]].into_dyn());
        let sigma = Variable::new(ndarray::array![[sigma_val]].into_dyn());

        let mut sum_z = 0.0;
        let mut sum_z_sq = 0.0;

        for _ in 0..n {
            let z = reparameterize(&mu, &sigma, &mut rng);
            let z_val = z.data().into_dimensionality::<ndarray::Ix2>().unwrap()[[0, 0]];
            sum_z += z_val;
            sum_z_sq += z_val * z_val;
        }

        let mean = sum_z / (n as f32);
        // E[Z^2] - (E[Z])^2
        let var = (sum_z_sq / (n as f32)) - mean * mean;

        // 統計的な揺らぎを考慮して eps は 1e-1 級の緩めに設定
        assert!(
            approx_eq(mean, mu_val, 1e-1),
            "Mean mismatch: expected {}, got {}",
            mu_val,
            mean
        );
        assert!(
            approx_eq(var, sigma_val * sigma_val, 3e-1),
            "Var mismatch: expected {}, got {}",
            sigma_val * sigma_val,
            var
        );
    }
}
