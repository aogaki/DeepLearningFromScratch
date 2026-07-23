use ndarray::ArrayD;

pub struct Variable {
    data: ArrayD<f32>,
}
impl Variable {
    pub fn new(data: ArrayD<f32>) -> Self {
        Variable { data }
    }
}

pub fn numerical_diff<F>(f: F, x: &Variable, eps: f32) -> Variable
where
    F: Function,
{
    let mut x0 = x.data.clone();
    let mut x1 = x.data.clone();

    x0 += eps;
    x1 -= eps;

    let y0 = f.forward(&x0);
    let y1 = f.forward(&x1);

    let diff_data = (y0 - y1) / (2.0 * eps);
    Variable::new(diff_data)
}

pub trait Function {
    fn call(&self, x: &Variable) -> Variable {
        let result_data = self.forward(&x.data);
        Variable::new(result_data)
    }

    fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32>;
}

impl<T> Function for T
where
    T: Fn(&ArrayD<f32>) -> ArrayD<f32>,
{
    fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        self(x)
    }
}

pub struct Square;
impl Function for Square {
    fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        x.mapv(|v| v * v)
    }
}

pub struct Exp;
impl Function for Exp {
    fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        x.mapv(|v| v.exp())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    fn approx_equal_arrayd(a: &ArrayD<f32>, b: &ArrayD<f32>, tol: f32) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (*x - *y).abs() < tol)
    }

    #[test]
    fn test_variable_creation() {
        let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data = ArrayD::from_shape_vec(vec![2, 3], original_data.clone()).unwrap();
        let variable = Variable::new(data);
        assert_eq!(
            variable.data,
            Variable::new(ArrayD::from_shape_vec(vec![2, 3], original_data).unwrap()).data
        );
    }

    #[test]
    fn test_square_function() {
        let original_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let variable = Variable::new(original_data.clone().into_dyn());
        let square_function = Square;
        let result_variable = square_function.call(&variable);
        let expected_data = array![[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]];
        assert!(
            approx_equal_arrayd(
                &result_variable.data,
                &expected_data.clone().into_dyn(),
                1e-6
            ),
            "Expected: {:?}, Got: {:?}",
            expected_data,
            result_variable.data
        );
    }

    #[test]
    fn test_exp_function() {
        // simple test
        let original_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let variable = Variable::new(original_data.clone().into_dyn());
        let exp_function = Exp;
        let result_variable = exp_function.call(&variable);
        let expected_data = array![
            [1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()],
            [4.0f32.exp(), 5.0f32.exp(), 6.0f32.exp()]
        ];
        assert!(
            approx_equal_arrayd(
                &result_variable.data,
                &expected_data.clone().into_dyn(),
                1e-6
            ),
            "Expected: {:?}, Got: {:?}",
            expected_data,
            result_variable.data
        );

        // same as the book
        // Square -> Exp -> Square
        let square_function = Square;
        let intermediate_variable = square_function.call(&variable);
        let exp_variable = exp_function.call(&intermediate_variable);
        let final_variable = square_function.call(&exp_variable);
        let expected_final_data = array![
            [
                1.0f32.powi(2).exp().powi(2),
                2.0f32.powi(2).exp().powi(2),
                3.0f32.powi(2).exp().powi(2)
            ],
            [
                4.0f32.powi(2).exp().powi(2),
                5.0f32.powi(2).exp().powi(2),
                6.0f32.powi(2).exp().powi(2)
            ]
        ];
        assert!(
            approx_equal_arrayd(
                &final_variable.data,
                &expected_final_data.clone().into_dyn(),
                1e-6
            ),
            "Expected: {:?}, Got: {:?}",
            expected_final_data,
            final_variable.data
        );
    }

    #[test]
    fn test_numerical_diff() {
        // 単一関数
        let original_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let variable = Variable::new(original_data.clone().into_dyn());
        let square_function = Square;
        let eps = 5e-3;
        let diff_variable = numerical_diff(square_function, &variable, eps);
        let expected_diff_data = array![[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]];
        assert!(
            approx_equal_arrayd(
                &diff_variable.data,
                &expected_diff_data.clone().into_dyn(),
                1e-3
            ),
            "Expected: {:?}, Got: {:?}",
            expected_diff_data,
            diff_variable.data
        );

        // 合成関数、値がとんでもないことになるので 0.5 だけ
        let simple_data = array![[0.5f32]];
        let variable = Variable::new(simple_data.clone().into_dyn());

        let f = |x: &ArrayD<f32>| {
            let square_function = Square;
            let exp_function = Exp;
            let y1 = square_function.forward(x);
            let y2 = exp_function.forward(&y1);
            square_function.forward(&y2)
        };

        let diff_variable = numerical_diff(f, &variable, eps);
        let expected_diff_data = simple_data.mapv(|x| 4.0 * x * (2.0 * x.powi(2)).exp());
        assert!(
            approx_equal_arrayd(
                &diff_variable.data,
                &expected_diff_data.clone().into_dyn(),
                1e-3
            ),
            "Expected: {:?}, Got: {:?}",
            expected_diff_data,
            diff_variable.data
        );
    }
}
