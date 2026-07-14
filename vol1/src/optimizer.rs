use ndarray::Array2;

/// 本 6.1「パラメータの更新」最適化手法の共通インターフェース。
/// パラメータは所有せず、毎回 &mut で借りて更新する(W の本体は各レイヤが持つ)。
/// &mut self なのは Momentum の v など「パラメータごとの履歴」を自身が抱えるため。
/// 状態は 1 パラメータ専用 —— W ごとに別インスタンスを作ること(v/h/m の混線防止)
pub trait Optimizer {
    fn update(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>);
}

/// 本 6.1.2「SGD」W ← W − η·∂L/∂W。状態を持たない最も単純な更新則
pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>) {
        param.scaled_add(-self.lr, grad);
    }
}

/// 本 6.1.4「Momentum」v ← αv − η·∂L/∂W、W ← W + v。速度 v の慣性で谷を滑らかに下る。
/// v の形は初回 update まで不明なので Option で遅延確保(get_or_insert_with)
pub struct Momentum {
    lr: f32,
    momentum: f32,
    velocity: Option<Array2<f32>>,
}

impl Momentum {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity: None,
        }
    }
}

impl Optimizer for Momentum {
    fn update(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>) {
        let v = self
            .velocity
            .get_or_insert_with(|| Array2::zeros(param.raw_dim()));
        *v = v.mapv(|x| x * self.momentum) - grad.mapv(|x| x * self.lr);
        *param += &*v;
    }
}

/// 本 6.1.5「AdaGrad」h ← h + (∂L/∂W)²、W ← W − η·(1/√h)·∂L/∂W。
/// 勾配2乗の累積 h で要素ごとに実効学習率を減衰(よく動いたパラメータほど小刻みに)
pub struct AdaGrad {
    lr: f32,
    h: Option<Array2<f32>>,
}

impl AdaGrad {
    pub fn new(lr: f32) -> Self {
        Self { lr, h: None }
    }
}

impl Optimizer for AdaGrad {
    fn update(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>) {
        let h = self.h.get_or_insert_with(|| Array2::zeros(param.raw_dim()));
        *h += &grad.mapv(|x| x * x);
        *param -= &(grad / (h.mapv(|x| x.sqrt() + 1e-7)) * self.lr);
    }
}

/// 本 6.1.6「Adam」Momentum の 1次モーメント m と AdaGrad 系の 2次モーメント v を併用し、
/// バイアス補正 m̂=m/(1−β1^t), v̂=v/(1−β2^t) を掛ける(原論文の標準形。本の lr_t 形と等価)。
/// 勾配のスケールを自動正規化するため、勾配消失気味の悪条件にも頑健(6.3.2 の対照実験で確認)
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    iter: i32,
    m: Option<Array2<f32>>,
    v: Option<Array2<f32>>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            iter: 0,
            m: None,
            v: None,
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>) {
        self.iter += 1;
        let m = self.m.get_or_insert_with(|| Array2::zeros(param.raw_dim()));
        let v = self.v.get_or_insert_with(|| Array2::zeros(param.raw_dim()));

        *m = m.mapv(|x| x * self.beta1) + grad.mapv(|x| x * (1.0 - self.beta1));
        *v = v.mapv(|x| x * self.beta2) + grad.mapv(|x| x * x * (1.0 - self.beta2));

        let m_hat = m.mapv(|x| x / (1.0 - self.beta1.powi(self.iter)));
        let v_hat = v.mapv(|x| x / (1.0 - self.beta2.powi(self.iter)));

        *param -= &(m_hat / (v_hat.mapv(|x| x.sqrt() + 1e-7)) * self.lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_array(a: &Array2<f32>, b: &Array2<f32>, epsilon: f32) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| (x - y).abs() < epsilon)
    }

    #[test]
    fn test_sgd_update() {
        let mut sgd = SGD::new(0.1);
        let mut param = Array2::from_elem((2, 2), 1.0);
        let grad = Array2::from_elem((2, 2), 0.5);
        sgd.update(&mut param, &grad);
        assert!(approx_eq_array(
            &param,
            &Array2::from_elem((2, 2), 0.95),
            1e-6
        ));
    }

    #[test]
    fn test_momentum_update() {
        let mut momentum = Momentum::new(0.1, 0.9);
        let mut param = Array2::from_elem((2, 2), 1.0);
        let grad = Array2::from_elem((2, 2), 0.5);

        momentum.update(&mut param, &grad);
        assert!(approx_eq_array(
            &param,
            &Array2::from_elem((2, 2), 0.95),
            1e-6
        ));

        momentum.update(&mut param, &grad);
        assert!(approx_eq_array(
            &param,
            &Array2::from_elem((2, 2), 0.855),
            1e-6
        ));
    }

    #[test]
    fn test_adagrad_update() {
        let mut adagrad = AdaGrad::new(0.1);
        let mut param = Array2::from_elem((2, 2), 1.0);
        let grad = Array2::from_elem((2, 2), 0.5);

        adagrad.update(&mut param, &grad);
        assert!(approx_eq_array(
            &param,
            &Array2::from_elem((2, 2), 0.9),
            1e-6
        ));

        adagrad.update(&mut param, &grad);
        assert!(approx_eq_array(
            &param,
            &Array2::from_elem((2, 2), 0.829289),
            1e-6
        ));
    }

    #[test]
    fn test_adam_update() {
        let mut adam = Adam::new(0.1);
        let mut param = Array2::from_elem((2, 2), 1.0);
        let grad = Array2::from_elem((2, 2), 0.5);

        adam.update(&mut param, &grad);
        assert!(approx_eq_array(
            &param,
            &Array2::from_elem((2, 2), 0.9), // Expected value after one update
            1e-6
        ));

        adam.update(&mut param, &grad);
        assert!(approx_eq_array(
            &param,
            &Array2::from_elem((2, 2), 0.8), // Expected value after two updates
            1e-6
        ));
    }
}
