use ndarray::{Array1, Array4};

/// 本 3.6.1 / 51: MNISTデータセットのラベル IDX ファイルのパース
pub fn parse_idx_labels(bytes: &[u8]) -> Array1<u8> {
    let magic_number = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic_number, 2049, "Invalid IDX label file header");
    let num_items = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let labels = &bytes[8..8 + num_items];
    Array1::from_vec(labels.to_vec())
}

/// 本 3.6.1 / 51: MNISTデータセットの画像 IDX ファイルのパース
/// `u8` のまま `(N, 1, 28, 28)` の形状で返す。
pub fn parse_idx_images(bytes: &[u8]) -> Array4<u8> {
    let magic_number = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic_number, 2051, "Invalid IDX image file header");
    let num_images = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let num_rows = u32::from_be_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let num_cols = u32::from_be_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let image_size = num_rows * num_cols;

    let images_bytes = &bytes[16..16 + num_images * image_size];

    // (N, 1, 28, 28) に整形して返す
    Array4::from_shape_vec((num_images, 1, num_rows, num_cols), images_bytes.to_vec()).unwrap()
}

/// IDX 画像ファイルを読み込んでパースする(std::fs::read + parse_idx_images)。
pub fn load_images(path: &str) -> Array4<u8> {
    let bytes = std::fs::read(path).expect("Failed to read image file");
    parse_idx_images(&bytes)
}

/// IDX ラベルファイルを読み込んでパースする(std::fs::read + parse_idx_labels)。
pub fn load_labels(path: &str) -> Array1<u8> {
    let bytes = std::fs::read(path).expect("Failed to read label file");
    parse_idx_labels(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_idx_labels() {
        let mut bytes = vec![];
        // Magic number: 2049 (0x00000801)
        bytes.extend_from_slice(&[0, 0, 8, 1]);
        // Number of items: 2 (0x00000002)
        bytes.extend_from_slice(&[0, 0, 0, 2]);
        // Labels: 5, 0
        bytes.extend_from_slice(&[5, 0]);

        let labels = parse_idx_labels(&bytes);
        assert_eq!(labels.shape(), &[2]);
        assert_eq!(labels[0], 5);
        assert_eq!(labels[1], 0);
    }

    #[test]
    fn test_parse_idx_images() {
        let mut bytes = vec![];
        // Magic number: 2051 (0x00000803)
        bytes.extend_from_slice(&[0, 0, 8, 3]);
        // Number of images: 2 (0x00000002)
        bytes.extend_from_slice(&[0, 0, 0, 2]);
        // Rows: 2 (0x00000002)
        bytes.extend_from_slice(&[0, 0, 0, 2]);
        // Cols: 2 (0x00000002)
        bytes.extend_from_slice(&[0, 0, 0, 2]);

        // Pixels for 2 images of 2x2 (8 bytes total)
        // Image 1: [1, 2, 3, 4]
        // Image 2: [5, 6, 7, 8]
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let images = parse_idx_images(&bytes);
        assert_eq!(images.shape(), &[2, 1, 2, 2]);
        assert_eq!(images[[0, 0, 0, 0]], 1);
        assert_eq!(images[[0, 0, 1, 1]], 4);
        assert_eq!(images[[1, 0, 0, 0]], 5);
        assert_eq!(images[[1, 0, 1, 1]], 8);
    }
}
