use crate::function::Creator;
use crate::function::Function;
use crate::functions::{Add, Div, Exp, Mul, Neg, Pow, Sin, Square, Sub};
use ndarray::ArrayD;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

struct VariableInner {
    data: ArrayD<f32>,
    grad: Option<ArrayD<f32>>,
    creator: Option<Box<dyn Creator>>,
    generation: usize,
    name: Option<String>,
}

#[derive(Clone)]
pub struct Variable(Rc<RefCell<VariableInner>>);

impl Variable {
    pub fn new(data: ArrayD<f32>) -> Self {
        Variable(Rc::new(RefCell::new(VariableInner {
            data,
            grad: None,
            creator: None,
            generation: 0,
            name: None,
        })))
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.clone()
    }

    pub fn set_data(&self, data: ArrayD<f32>) {
        self.0.borrow_mut().data = data;
    }

    pub fn generation(&self) -> usize {
        self.0.borrow().generation
    }

    pub fn set_generation(&self, generation: usize) {
        self.0.borrow_mut().generation = generation;
    }

    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    pub fn name(&self) -> Option<String> {
        self.0.borrow().name.clone()
    }

    pub fn set_name(&self, name: &str) {
        self.0.borrow_mut().name = Some(name.to_string());
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.0.borrow().data.ndim()
    }

    pub fn size(&self) -> usize {
        self.0.borrow().data.len()
    }

    pub fn item(&self) -> f32 {
        *self
            .0
            .borrow()
            .data
            .iter()
            .next()
            .expect("Variable has no data")
    }

    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0.borrow().grad.clone()
    }

    pub fn set_grad(&self, grad: ArrayD<f32>) {
        self.0.borrow_mut().grad = Some(grad);
    }

    pub fn add_grad(&self, gx: ArrayD<f32>) {
        let mut borrow = self.0.borrow_mut();
        if let Some(grad) = &mut borrow.grad {
            *grad += &gx;
        } else {
            borrow.grad = Some(gx);
        }
    }

    pub fn cleargrad(&self) {
        self.0.borrow_mut().grad = None;
    }

    pub fn set_creator(&self, func: Box<dyn Creator>) {
        self.0.borrow_mut().creator = Some(func);
    }

    pub fn has_creator(&self) -> bool {
        self.0.borrow().creator.is_some()
    }

    pub fn creator_info(&self) -> Option<(usize, String, Vec<Variable>)> {
        let borrow = self.0.borrow();
        borrow.creator.as_ref().map(|c| {
            let id = c.as_ref() as *const dyn Creator as *const () as usize;
            (id, c.label(), c.get_inputs())
        })
    }

    pub fn backward(&self, retain_grad: bool) {
        if self.grad().is_none() {
            self.set_grad(ArrayD::from_elem(self.data().shape(), 1.0f32));
        }

        let mut queue = vec![];
        let mut seen_set = std::collections::HashSet::new();

        let ptr = self.id();
        seen_set.insert(ptr);
        queue.push(self.clone());

        while !queue.is_empty() {
            queue.sort_by_key(|v| v.generation());
            let var = queue.pop().unwrap();

            let computed_gradients = {
                let borrow = var.0.borrow();
                if let Some(creator) = &borrow.creator {
                    let grad = borrow.grad.as_ref().unwrap();
                    let gxs = creator.backward(grad);
                    let inputs = creator.get_inputs();
                    Some((gxs, inputs))
                } else {
                    None
                }
            };

            if let Some((gxs, inputs)) = computed_gradients {
                for (gx, input) in gxs.into_iter().zip(inputs.into_iter()) {
                    input.add_grad(gx);
                    let ptr = input.id();
                    if !seen_set.contains(&ptr) {
                        seen_set.insert(ptr);
                        queue.push(input);
                    }
                }
            }

            if !retain_grad && var.0.borrow().creator.is_some() {
                var.cleargrad();
            }
        }
    }

    pub fn square(&self) -> Variable {
        Square.call(std::slice::from_ref(self))
    }

    pub fn exp(&self) -> Variable {
        Exp.call(std::slice::from_ref(self))
    }

    pub fn sin(&self) -> Variable {
        Sin.call(std::slice::from_ref(self))
    }

    pub fn add(&self, other: &Variable) -> Variable {
        Add.call(&[self.clone(), other.clone()])
    }

    pub fn mul(&self, other: &Variable) -> Variable {
        Mul.call(&[self.clone(), other.clone()])
    }

    pub fn neg(&self) -> Variable {
        Neg.call(std::slice::from_ref(self))
    }

    pub fn sub(&self, other: &Variable) -> Variable {
        Sub.call(&[self.clone(), other.clone()])
    }

    pub fn div(&self, other: &Variable) -> Variable {
        Div.call(&[self.clone(), other.clone()])
    }

    pub fn powf(&self, c: f32) -> Variable {
        Pow { c }.call(std::slice::from_ref(self))
    }
}

impl std::ops::Add for &Variable {
    type Output = Variable;
    fn add(self, rhs: &Variable) -> Variable {
        Variable::add(self, rhs)
    }
}

impl std::ops::Mul for &Variable {
    type Output = Variable;
    fn mul(self, rhs: &Variable) -> Variable {
        Variable::mul(self, rhs)
    }
}

impl std::ops::Neg for &Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        Variable::neg(self)
    }
}

impl std::ops::Neg for Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        Variable::neg(&self)
    }
}

impl std::ops::Sub for &Variable {
    type Output = Variable;
    fn sub(self, rhs: &Variable) -> Variable {
        Variable::sub(self, rhs)
    }
}

impl std::ops::Div for &Variable {
    type Output = Variable;
    fn div(self, rhs: &Variable) -> Variable {
        Variable::div(self, rhs)
    }
}

crate::impl_op_combinations!(Add, add);
crate::impl_op_combinations!(Mul, mul);
crate::impl_op_combinations!(Sub, sub);
crate::impl_op_combinations!(Div, div);

impl From<f32> for Variable {
    fn from(val: f32) -> Self {
        Variable::new(ndarray::arr0(val).into_dyn())
    }
}

crate::impl_op_scalar!(Add, add);
crate::impl_op_scalar!(Mul, mul);
crate::impl_op_scalar!(Sub, sub);
crate::impl_op_scalar!(Div, div);

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let borrow = self.0.borrow();
        let data_str = format!("{}", borrow.data);
        let indented_str = data_str.replace('\n', "\n         ");

        if let Some(name) = &borrow.name {
            write!(f, "variable({}, name={})", indented_str, name)
        } else {
            write!(f, "variable({})", indented_str)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::no_grad;
    use crate::utils::approx_equal_arrayd;
    use ndarray::array;

    // ステップ6: 手作業の逆伝播(creator を1段ずつ辿る)。
    // 末尾で backward の冪等性(同じ creator で2回計算しても同じ答え — 決定性の
    // 検証なのでここだけビット一致の assert_eq が正しい)も確認する
    #[test]
    fn test_manual_backward_chain() {
        let x_data = array![[0.5f32]].into_dyn();
        let x = Variable::new(x_data);

        let a = x.square();
        let b = a.exp();
        let y = b.square();

        // y.grad = 1.0 (shapeは揃える)
        y.set_grad(array![[1.0f32]].into_dyn());

        fn get_gx(var: &Variable, gy: &ArrayD<f32>) -> ArrayD<f32> {
            var.0
                .borrow()
                .creator
                .as_ref()
                .unwrap()
                .backward(gy)
                .into_iter()
                .next()
                .unwrap()
        }

        let gy = y.grad().unwrap();
        b.set_grad(get_gx(&y, &gy));

        let gb = b.grad().unwrap();
        a.set_grad(get_gx(&b, &gb));

        let ga = a.grad().unwrap();
        x.set_grad(get_gx(&a, &ga));

        let expected_grad = array![[3.2974426f32]].into_dyn();
        assert!(approx_equal_arrayd(
            &x.grad().unwrap(),
            &expected_grad,
            1e-5 // 誤差の許容範囲
        ));

        let ga2 = a.grad().unwrap();
        let x_grad_2 = get_gx(&a, &ga2);
        assert_eq!(x.grad().unwrap(), x_grad_2);
    }

    #[test]
    fn test_no_grad() {
        let x = Variable::new(array![[2.0f32]].into_dyn());
        let y;
        {
            let _guard = no_grad();
            y = x.square();
        }
        assert!(!y.has_creator(), "no_grad時はcreatorがセットされない");

        let z = x.square();
        assert!(z.has_creator(), "ガードが外れればcreatorがセットされる");
    }
}
