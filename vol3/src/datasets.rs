use ndarray::ArrayD;

/// 本 ステップ49: データセットの基底トレイト
pub trait Dataset {
    type Target;

    /// データの総数を返す
    fn len(&self) -> usize;

    /// len_without_is_empty 対策
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// インデックスを指定してデータ1件を返す
    /// 戻り値は入力特徴量と正解ラベル
    fn get_item(&self, index: usize) -> (ArrayD<f32>, Self::Target);
}

/// 本 ステップ48〜49: スパイラルデータセット(3クラス渦巻き、クラス順に並んでいるため
/// 学習にはシャッフル必須)。データ生成は utils::get_spiral(シード 1984/2020 は本家準拠)。
pub struct SpiralDataset {
    pub x: ndarray::Array2<f32>,
    pub t: Vec<usize>,
}

impl SpiralDataset {
    pub fn new(train: bool) -> Self {
        let (x, t) = crate::utils::get_spiral(train);
        let x2d = x.into_dimensionality::<ndarray::Ix2>().unwrap();
        Self { x: x2d, t }
    }
}

impl Dataset for SpiralDataset {
    type Target = usize;

    fn len(&self) -> usize {
        self.x.shape()[0]
    }

    fn get_item(&self, index: usize) -> (ArrayD<f32>, Self::Target) {
        let feature = self.x.row(index).to_owned().into_dyn();
        let label = self.t[index];
        (feature, label)
    }
}

/// 本 ステップ51: MNIST データセット。
/// 生データは u8 (N,1,28,28) のまま保持(f32 の 1/4 のメモリ、CNN 対応の形)し、
/// get_item で f32 化 → transform 適用 —「パースはバイト解読、前処理は Dataset の責務」。
/// ファイルパスは CARGO_MANIFEST_DIR 起点で CWD 非依存(cargo run は cwd を変えない)。
pub struct MnistDataset {
    pub data: ndarray::Array4<u8>,
    pub labels: ndarray::Array1<u8>,
    pub transform: Option<fn(ArrayD<f32>) -> ArrayD<f32>>,
}

impl MnistDataset {
    pub fn new(train: bool, transform: Option<fn(ArrayD<f32>) -> ArrayD<f32>>) -> Self {
        let suffix = if train { "train" } else { "t10k" };
        let image_path = format!(
            "{}/dataset/{}-images-idx3-ubyte",
            env!("CARGO_MANIFEST_DIR"),
            suffix
        );
        let label_path = format!(
            "{}/dataset/{}-labels-idx1-ubyte",
            env!("CARGO_MANIFEST_DIR"),
            suffix
        );

        let data = crate::mnist::load_images(&image_path);
        let labels = crate::mnist::load_labels(&label_path);

        Self {
            data,
            labels,
            transform,
        }
    }
}

impl Dataset for MnistDataset {
    type Target = usize;

    fn len(&self) -> usize {
        self.data.shape()[0]
    }

    fn get_item(&self, index: usize) -> (ArrayD<f32>, Self::Target) {
        let x_u8 = self.data.index_axis(ndarray::Axis(0), index);
        let mut x_f32 = x_u8.mapv(|v| v as f32).into_dyn();

        if let Some(t) = self.transform {
            x_f32 = t(x_f32);
        }

        let t = self.labels[index] as usize;
        (x_f32, t)
    }
}

/// MLP 用の標準前処理: (1,28,28) → (784,) 平坦化+ /255 正規化。
/// `MnistDataset::new` の transform に関数ポインタで渡す(example と test の共通の家)。
pub fn mnist_flatten_normalize(x: ArrayD<f32>) -> ArrayD<f32> {
    // (1, 28, 28) を (784,) に平坦化して / 255.0
    let flat_x = x.into_shape_with_order(vec![784]).unwrap();
    flat_x / 255.0
}

pub struct SinCurveDataset {
    pub data: ndarray::Array2<f32>,
    pub label: ndarray::Array2<f32>,
}

impl SinCurveDataset {
    pub fn new(train: bool) -> Self {
        use rand::{Rng, SeedableRng};
        let num_data = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut y = Vec::with_capacity(num_data);
        for i in 0..num_data {
            let x = (i as f32) / ((num_data - 1) as f32) * 2.0 * std::f32::consts::PI;
            let val = if train {
                let noise: f32 = rng.random_range(-0.05..0.05);
                x.sin() + noise
            } else {
                x.cos()
            };
            y.push(val);
        }

        let mut data = Vec::with_capacity(num_data - 1);
        let mut label = Vec::with_capacity(num_data - 1);
        for i in 0..num_data - 1 {
            data.push(y[i]);
            label.push(y[i + 1]);
        }

        Self {
            data: ndarray::Array2::from_shape_vec((num_data - 1, 1), data).unwrap(),
            label: ndarray::Array2::from_shape_vec((num_data - 1, 1), label).unwrap(),
        }
    }
}

impl Dataset for SinCurveDataset {
    type Target = ArrayD<f32>;

    fn len(&self) -> usize {
        self.data.shape()[0]
    }

    fn get_item(&self, index: usize) -> (ArrayD<f32>, Self::Target) {
        let x = self.data.row(index).to_owned().into_dyn();
        let t = self.label.row(index).to_owned().into_dyn();
        (x, t)
    }
}
