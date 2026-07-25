use crate::datasets::Dataset;
use crate::variable::Variable;
use rand::seq::SliceRandom;

/// 本 ステップ50: DataLoader — Dataset からミニバッチ (Variable, Vec<usize>) を組み立てる。
/// シャッフル用 RNG は注入(テストの決定性)。バッチ形状は先頭要素の shape から導出する
/// ため特徴量の形に非依存(spiral の (2,) も MNIST の (784,) も同じコードで扱える)。
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    rng: Box<dyn rand::RngCore>,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool, rng: Box<dyn rand::RngCore>) -> Self {
        let len = dataset.len();
        let indices = (0..len).collect();
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            rng,
        }
    }
}

/// 1エポックぶんのイテレータ。`&mut DataLoader` の IntoIterator が「for ループ1つ=
/// 1エポック」を型で表現する: シャッフルは into_iter()(=エポック開始点)で発火し、
/// エポック末端で必ず None を返す。カーソルはこちら側にあるので、DeZero の
/// 「StopIteration 後に reset」のような None 後の再開問題は構造的に存在しない。
pub struct DataLoaderIter<'a, D: Dataset> {
    loader: &'a mut DataLoader<D>,
    cursor: usize,
}

impl<'a, D: Dataset> IntoIterator for &'a mut DataLoader<D> {
    type Item = (Variable, Vec<usize>);
    type IntoIter = DataLoaderIter<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        if self.shuffle {
            self.indices.shuffle(&mut self.rng);
        }
        DataLoaderIter {
            loader: self,
            cursor: 0,
        }
    }
}

impl<'a, D: Dataset> Iterator for DataLoaderIter<'a, D> {
    type Item = (Variable, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let dataset_len = self.loader.dataset.len();
        if self.cursor >= dataset_len {
            return None; // エポック末端で必ず終了する
        }

        let end = (self.cursor + self.loader.batch_size).min(dataset_len);
        let batch_size = end - self.cursor;

        // 最初のデータを見てバッチの形を決める
        let (first_x, first_t) = self
            .loader
            .dataset
            .get_item(self.loader.indices[self.cursor]);
        let mut shape = vec![batch_size];
        shape.extend(first_x.shape());

        let mut batch_x = ndarray::Array::zeros(shape);
        let mut batch_t = Vec::with_capacity(batch_size);

        // 最初の要素を代入
        batch_x.index_axis_mut(ndarray::Axis(0), 0).assign(&first_x);
        batch_t.push(first_t);

        // 残りのデータを代入
        for (i, idx) in self.loader.indices[self.cursor + 1..end].iter().enumerate() {
            let (x, t) = self.loader.dataset.get_item(*idx);
            batch_x.index_axis_mut(ndarray::Axis(0), i + 1).assign(&x);
            batch_t.push(t);
        }

        self.cursor = end;
        Some((Variable::new(batch_x.into_dyn()), batch_t))
    }
}
