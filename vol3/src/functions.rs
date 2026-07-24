use crate::function::{Forward, Function};
use ndarray::ArrayD;

pub struct Square;
impl Forward for Square {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Square expects 1 input")
        };
        x.mapv(|v| v * v)
    }
}
impl Function for Square {
    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let [x] = xs else {
            panic!("Square expects 1 input")
        };
        vec![gy * (2.0 * x)]
    }
}

pub struct Exp;
impl Forward for Exp {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Exp expects 1 input")
        };
        x.mapv(|v| v.exp())
    }
}
impl Function for Exp {
    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let [x] = xs else {
            panic!("Exp expects 1 input")
        };
        vec![gy * x.mapv(|v| v.exp())]
    }
}

pub struct Add;
impl Forward for Add {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0, x1] = xs else {
            panic!("Add expects 2 inputs")
        };
        x0 + x1
    }
}
impl Function for Add {
    fn backward(&self, _xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        vec![gy.clone(), gy.clone()]
    }
}

pub struct Mul;
impl Forward for Mul {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0, x1] = xs else {
            panic!("Mul expects 2 inputs")
        };
        x0 * x1
    }
}
impl Function for Mul {
    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let [x0, x1] = xs else {
            panic!("Mul expects 2 inputs")
        };
        vec![gy * x1, gy * x0]
    }
}

pub struct Neg;
impl Forward for Neg {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0] = xs else {
            panic!("Neg expects 1 input")
        };
        -x0
    }
}
impl Function for Neg {
    fn backward(&self, _xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        vec![-gy]
    }
}

pub struct Sub;
impl Forward for Sub {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0, x1] = xs else {
            panic!("Sub expects 2 inputs")
        };
        x0 - x1
    }
}
impl Function for Sub {
    fn backward(&self, _xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        vec![gy.clone(), -gy]
    }
}

pub struct Div;
impl Forward for Div {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0, x1] = xs else {
            panic!("Div expects 2 inputs")
        };
        x0 / x1
    }
}
impl Function for Div {
    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let [x0, x1] = xs else {
            panic!("Div expects 2 inputs")
        };
        let gx0 = gy / x1;
        let gx1 = gy * (-x0 / (x1 * x1));
        vec![gx0, gx1]
    }
}

pub struct Pow {
    pub c: f32,
}
impl Forward for Pow {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x0] = xs else {
            panic!("Pow expects 1 input")
        };
        x0.mapv(|v| v.powf(self.c))
    }
}
impl Function for Pow {
    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let [x0] = xs else {
            panic!("Pow expects 1 input")
        };
        let c = self.c;
        let gx0 = gy * c * x0.mapv(|v| v.powf(c - 1.0));
        vec![gx0]
    }
}

pub struct Sin;
impl Forward for Sin {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        let [x] = xs else {
            panic!("Sin expects 1 input")
        };
        x.mapv(|v| v.sin())
    }
}
impl Function for Sin {
    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let [x] = xs else {
            panic!("Sin expects 1 input")
        };
        vec![gy * x.mapv(|v| v.cos())]
    }
}
