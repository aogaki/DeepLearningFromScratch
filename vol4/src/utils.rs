use ndarray::Array1;

pub fn argmax(arr: &Array1<f32>) -> Option<usize> {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}
