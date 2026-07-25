use crate::layers::Layer;
use crate::variable::Variable;
use ndarray::ArrayD;

/// 本 ステップ46: Optimizer (最適化手法)
///
/// レイヤ(モデル)のパラメータ `Variable` の参照(クローン)を保持し、
/// 学習ループ内で各パラメータを一括して更新する役割を持つ。
pub trait Optimizer {
    fn setup(&mut self, layer: &impl Layer);
    fn update(&mut self);
}

/// 本 ステップ46: 確率的勾配降下法 W ← W − lr·g。
/// setup で `params()`(Rc ハンドルの clone)を保持し、update はハンドル越しに書き換える —
/// vol1 で必要だった split-borrow 機構(w_and_dw)はこの設計では最初から不要。
pub struct SGD {
    pub lr: f32,
    params: Vec<Variable>,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            params: Vec::new(),
        }
    }
}

impl Optimizer for SGD {
    fn setup(&mut self, layer: &impl Layer) {
        self.params = layer.params();
    }

    fn update(&mut self) {
        for p in &self.params {
            if let Some(grad) = p.grad() {
                let p_data = p.data();
                p.set_data(p_data - (grad * self.lr));
            }
        }
    }
}

/// 本 ステップ46: モメンタム SGD v ← momentum·v − lr·g、W ← W + v。
/// velocities は setup 時に zeros で実体化(params のスナップショットがあり形が確定して
/// いるため — vol1 の Option 遅延初期化は「形が最初の update まで不明」だった時代の作法)。
/// 慣性の実効はモジュールテストで検証: 1ステップ目は SGD と一致、2ステップ目で分岐する。
pub struct MomentumSGD {
    pub lr: f32,
    pub momentum: f32,
    params: Vec<Variable>,
    velocities: Vec<ArrayD<f32>>,
}

impl MomentumSGD {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            params: Vec::new(),
            velocities: Vec::new(),
        }
    }
}

impl Optimizer for MomentumSGD {
    fn setup(&mut self, layer: &impl Layer) {
        self.params = layer.params();
        // パラメータの形は全て確定しているので setup 時に実体化する
        self.velocities = self
            .params
            .iter()
            .map(|p| ndarray::Array::zeros(p.shape()).into_dyn())
            .collect();
    }

    fn update(&mut self) {
        for (i, p) in self.params.iter().enumerate() {
            if let Some(grad) = p.grad() {
                let v = &mut self.velocities[i];
                *v = &*v * self.momentum - &grad * self.lr;

                let p_data = p.data();
                p.set_data(p_data + &*v);
            }
        }
    }
}

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub t: f32,
    pub params: Vec<crate::variable::Variable>,
    pub ms: Vec<ndarray::ArrayD<f32>>,
    pub vs: Vec<ndarray::ArrayD<f32>>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0.0,
            params: vec![],
            ms: vec![],
            vs: vec![],
        }
    }
}

impl Optimizer for Adam {
    fn setup(&mut self, layer: &impl crate::layers::Layer) {
        self.params = layer.params();
        for p in &self.params {
            self.ms.push(ndarray::ArrayD::zeros(p.data().dim()));
            self.vs.push(ndarray::ArrayD::zeros(p.data().dim()));
        }
    }

    fn update(&mut self) {
        self.t += 1.0;
        let lr_t =
            self.lr * (1.0 - self.beta2.powf(self.t)).sqrt() / (1.0 - self.beta1.powf(self.t));
        for i in 0..self.params.len() {
            if let Some(grad) = self.params[i].grad() {
                self.ms[i] = &self.ms[i] * self.beta1 + &grad * (1.0 - self.beta1);
                self.vs[i] = &self.vs[i] * self.beta2 + (&grad * &grad) * (1.0 - self.beta2);
                let p_data = self.params[i].data();

                let mut v_hat = self.vs[i].clone();
                v_hat.mapv_inplace(|v| v.sqrt() + self.eps);

                let step = &self.ms[i] / v_hat;
                self.params[i].set_data(p_data - (step * lr_t));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Linear;

    #[test]
    fn test_sgd_vs_momentum() {
        let mut rng = rand::rng();
        let layer1 = Linear::new(1, 1, false, &mut rng);
        layer1.w.set_data(
            ndarray::Array::from_shape_vec((1, 1), vec![1.0])
                .unwrap()
                .into_dyn(),
        );
        layer1.b.as_ref().unwrap().set_data(
            ndarray::Array::from_shape_vec((1,), vec![1.0])
                .unwrap()
                .into_dyn(),
        );

        let layer2 = Linear::new(1, 1, false, &mut rng);
        layer2.w.set_data(
            ndarray::Array::from_shape_vec((1, 1), vec![1.0])
                .unwrap()
                .into_dyn(),
        );
        layer2.b.as_ref().unwrap().set_data(
            ndarray::Array::from_shape_vec((1,), vec![1.0])
                .unwrap()
                .into_dyn(),
        );

        let mut sgd = SGD::new(0.1);
        sgd.setup(&layer1);

        let mut momentum = MomentumSGD::new(0.1, 0.9);
        momentum.setup(&layer2);

        // Step 1: Same gradient (W_grad=1.0, b_grad=0.0)
        layer1.w.set_grad(Variable::new(
            ndarray::Array::from_shape_vec((1, 1), vec![1.0])
                .unwrap()
                .into_dyn(),
        ));
        layer2.w.set_grad(Variable::new(
            ndarray::Array::from_shape_vec((1, 1), vec![1.0])
                .unwrap()
                .into_dyn(),
        ));

        sgd.update();
        momentum.update();

        // 1st step: Both SGD and Momentum should move by -lr * grad = -0.1
        let w1_1 = layer1
            .w
            .data()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()[[0, 0]];
        let w2_1 = layer2
            .w
            .data()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()[[0, 0]];
        assert!((w1_1 - 0.9).abs() < 1e-5);
        assert!((w2_1 - 0.9).abs() < 1e-5);

        // Step 2: Same gradient again (W_grad=1.0)
        layer1.w.set_grad(Variable::new(
            ndarray::Array::from_shape_vec((1, 1), vec![1.0])
                .unwrap()
                .into_dyn(),
        ));
        layer2.w.set_grad(Variable::new(
            ndarray::Array::from_shape_vec((1, 1), vec![1.0])
                .unwrap()
                .into_dyn(),
        ));

        sgd.update();
        momentum.update();

        let w1_2 = layer1
            .w
            .data()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()[[0, 0]];
        let w2_2 = layer2
            .w
            .data()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()[[0, 0]];

        // SGD moves by -0.1 again -> 0.8
        assert!((w1_2 - 0.8).abs() < 1e-5, "Expected 0.8, got {}", w1_2);

        // Momentum velocity = 0.9 * (-0.1) - 0.1 * 1.0 = -0.09 - 0.1 = -0.19
        // W2 = 0.9 - 0.19 = 0.71
        assert!((w2_2 - 0.71).abs() < 1e-5, "Expected 0.71, got {}", w2_2);
    }
}
