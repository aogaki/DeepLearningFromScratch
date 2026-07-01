
pub fn perceptron(x_vec: &[f32], w_vec: &[f32], b: f32) -> f32 {
    x_vec.iter().zip(w_vec.iter()).map(|(x, w)| x * w).sum::<f32>() + b
}

pub fn step_function(x: f32) -> f32 {
    if x <= 0.0 {
         0.0
    } else {
         1.0
    }
}

pub fn and_gate(x1: f32, x2: f32) -> f32 {
    let tmp = perceptron(&[x1, x2], &[0.5, 0.5], -0.7);
    step_function(tmp)
}

pub fn nand_gate(x1: f32, x2: f32) -> f32 {
    let tmp = perceptron(&[x1, x2], &[-0.5, -0.5], 0.7);
    step_function(tmp)
}

pub fn or_gate(x1: f32, x2: f32) -> f32 {
    let tmp = perceptron(&[x1, x2], &[0.5, 0.5], -0.2);
    step_function(tmp)
}

pub fn xor_gate(x1: f32, x2: f32) -> f32 {
    let s1 = nand_gate(x1, x2);
    let s2 = or_gate(x1, x2);
    and_gate(s1, s2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn and_gate_test() {
        assert_eq!(and_gate(0.0, 0.0), 0.0);
        assert_eq!(and_gate(0.0, 1.0), 0.0);
        assert_eq!(and_gate(1.0, 0.0), 0.0);
        assert_eq!(and_gate(1.0, 1.0), 1.0);
    }

    #[test]
    fn nand_gate_test() {
        assert_eq!(nand_gate(0.0, 0.0), 1.0);
        assert_eq!(nand_gate(0.0, 1.0), 1.0);
        assert_eq!(nand_gate(1.0, 0.0), 1.0);
        assert_eq!(nand_gate(1.0, 1.0), 0.0);
    }

    #[test]
    fn or_gate_test() {
        assert_eq!(or_gate(0.0, 0.0), 0.0);
        assert_eq!(or_gate(0.0, 1.0), 1.0);
        assert_eq!(or_gate(1.0, 0.0), 1.0);
        assert_eq!(or_gate(1.0, 1.0), 1.0);
    }

    #[test]
    fn xor_gate_test() {
        assert_eq!(xor_gate(0.0, 0.0), 0.0);
        assert_eq!(xor_gate(0.0, 1.0), 1.0);
        assert_eq!(xor_gate(1.0, 0.0), 1.0);
        assert_eq!(xor_gate(1.0, 1.0), 0.0);
    }
}
