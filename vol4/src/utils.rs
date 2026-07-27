use ndarray::Array1;

pub fn argmax(arr: &Array1<f32>) -> Option<usize> {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}

pub fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}
