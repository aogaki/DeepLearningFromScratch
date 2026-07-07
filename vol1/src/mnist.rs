use ndarray::{Array1, Array2};

/// 本 3.6.1「MNISTデータセット」ラベルIDXファイルのパース
pub fn parse_idx_labels(bytes: &[u8]) -> Array1<u8> {
    let magic_number = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic_number, 2049, "Invalid IDX label file header");
    let num_items = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let labels = &bytes[8..8 + num_items];
    Array1::from_vec(labels.to_vec())
}

/// 本 3.6.1「MNISTデータセット」画像IDXファイルのパース(0..1 に正規化)
pub fn parse_idx_images(bytes: &[u8]) -> Array2<f32> {
    let magic_number = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic_number, 2051, "Invalid IDX image file header");
    let num_images = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let num_rows = u32::from_be_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let num_cols = u32::from_be_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let image_size = num_rows * num_cols;
    let images: Vec<f32> = bytes[16..16 + num_images * image_size]
        .iter()
        .map(|&b| b as f32 / 255.0)
        .collect();

    Array2::from_shape_vec((num_images, image_size), images).unwrap()
}

/// 本 3.6.1「MNISTデータセット」画像ファイルを読み込む
pub fn load_images(path: &str) -> Array2<f32> {
    let bytes = std::fs::read(path).expect("Failed to read image file");
    parse_idx_images(&bytes)
}

/// 本 3.6.1「MNISTデータセット」ラベルファイルを読み込む
pub fn load_labels(path: &str) -> Array1<u8> {
    let bytes = std::fs::read(path).expect("Failed to read label file");
    parse_idx_labels(&bytes)
}

/// 本 4.5.2 ミニバッチ学習用: ラベル番号を one-hot 表現 (rows, num_classes) に変換
pub fn to_one_hot(labels: &[u8], num_classes: usize) -> Array2<f32> {
    let mut one_hot = Array2::<f32>::zeros((labels.len(), num_classes));
    for (i, &label) in labels.iter().enumerate() {
        one_hot[[i, label as usize]] = 1.0;
    }
    one_hot
}

#[cfg(test)]
mod tests {

    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn parse_idx_labels_test() {
        let bytes: Vec<u8> = vec![
            0x00, 0x00, 0x08, 0x01, // magic number
            0x00, 0x00, 0x00, 0x03, // number of items
            0x07, 0x02, 0x09, // labels
        ];
        let labels = parse_idx_labels(&bytes);
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0], 7);
        assert_eq!(labels[1], 2);
        assert_eq!(labels[2], 9);
    }

    #[test]
    fn parse_idx_images_test() {
        let bytes: Vec<u8> = vec![
            0x00, 0x00, 0x08, 0x03, // magic number
            0x00, 0x00, 0x00, 0x02, // number of images
            0x00, 0x00, 0x00, 0x02, // number of rows
            0x00, 0x00, 0x00, 0x02, // number of columns
            0x00, 0xFF, 0x00, 0xFF, // first image
            0xFF, 0x00, 0xFF, 0x00, // second image
        ];
        let images = parse_idx_images(&bytes);
        assert_eq!(images.shape(), &[2, 4]);
        let epsilon = 1e-6;
        assert!(approx_eq(images[[0, 0]], 0.0, epsilon));
        assert!(approx_eq(images[[0, 1]], 1.0, epsilon));
        assert!(approx_eq(images[[0, 2]], 0.0, epsilon));
        assert!(approx_eq(images[[0, 3]], 1.0, epsilon));
        assert!(approx_eq(images[[1, 0]], 1.0, epsilon));
        assert!(approx_eq(images[[1, 1]], 0.0, epsilon));
        assert!(approx_eq(images[[1, 2]], 1.0, epsilon));
        assert!(approx_eq(images[[1, 3]], 0.0, epsilon));
    }

    #[test]
    fn load_images_and_labels_test() {
        let image_path = "dataset/train-images-idx3-ubyte";
        let label_path = "dataset/train-labels-idx1-ubyte";
        let images = load_images(image_path);
        let labels = load_labels(label_path);
        assert_eq!(images.shape(), &[60000, 784]);
        assert_eq!(labels.len(), 60000);
        assert_eq!(labels[0], 5);
    }

    #[test]
    fn to_one_hot_test() {
        let labels = vec![0, 1, 2, 1];
        let num_classes = 3;
        let one_hot = to_one_hot(&labels, num_classes);
        assert_eq!(one_hot.shape(), &[4, 3]);
        assert_eq!(one_hot[[0, 0]], 1.0);
        assert_eq!(one_hot[[0, 1]], 0.0);
        assert_eq!(one_hot[[0, 2]], 0.0);
        assert_eq!(one_hot[[1, 0]], 0.0);
        assert_eq!(one_hot[[1, 1]], 1.0);
        assert_eq!(one_hot[[1, 2]], 0.0);
        assert_eq!(one_hot[[2, 0]], 0.0);
        assert_eq!(one_hot[[2, 1]], 0.0);
        assert_eq!(one_hot[[2, 2]], 1.0);
        assert_eq!(one_hot[[3, 0]], 0.0);
        assert_eq!(one_hot[[3, 1]], 1.0);
        assert_eq!(one_hot[[3, 2]], 0.0);
    }
}
