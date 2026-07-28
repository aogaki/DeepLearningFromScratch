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
    rng: Box<dyn rand::Rng>,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool, rng: Box<dyn rand::Rng>) -> Self {
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
    type Item = (Variable, Vec<D::Target>);
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
    type Item = (Variable, Vec<D::Target>);

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

/// 本 ステップ60: SeqDataLoader — シャッフルしないミニバッチ (時系列並行ストリーム)
pub struct SeqDataLoader<D: Dataset<Target = ndarray::ArrayD<f32>>> {
    dataset: D,
    batch_size: usize,
}

impl<D: Dataset<Target = ndarray::ArrayD<f32>>> SeqDataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
        }
    }
}

pub struct SeqDataLoaderIter<'a, D: Dataset<Target = ndarray::ArrayD<f32>>> {
    loader: &'a mut SeqDataLoader<D>,
    iteration: usize,
    max_iter: usize,
    jump: usize,
}

impl<'a, D: Dataset<Target = ndarray::ArrayD<f32>>> IntoIterator for &'a mut SeqDataLoader<D> {
    type Item = (Variable, Variable);
    type IntoIter = SeqDataLoaderIter<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        let data_size = self.dataset.len();
        let jump = data_size / self.batch_size;
        let max_iter = jump; // After `jump` iterations, it loops back, so 1 epoch = jump steps
        SeqDataLoaderIter {
            loader: self,
            iteration: 0,
            max_iter,
            jump,
        }
    }
}

impl<'a, D: Dataset<Target = ndarray::ArrayD<f32>>> Iterator for SeqDataLoaderIter<'a, D> {
    type Item = (Variable, Variable);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iteration >= self.max_iter {
            return None;
        }

        let data_size = self.loader.dataset.len();
        let batch_size = self.loader.batch_size;

        let (first_x, first_t) = self.loader.dataset.get_item(self.iteration % data_size);

        let mut shape_x = vec![batch_size];
        shape_x.extend(first_x.shape());
        let mut shape_t = vec![batch_size];
        shape_t.extend(first_t.shape());

        let mut batch_x = ndarray::Array::zeros(shape_x);
        let mut batch_t = ndarray::Array::zeros(shape_t);

        batch_x.index_axis_mut(ndarray::Axis(0), 0).assign(&first_x);
        batch_t.index_axis_mut(ndarray::Axis(0), 0).assign(&first_t);

        for i in 1..batch_size {
            let idx = (i * self.jump + self.iteration) % data_size;
            let (x, t) = self.loader.dataset.get_item(idx);
            batch_x.index_axis_mut(ndarray::Axis(0), i).assign(&x);
            batch_t.index_axis_mut(ndarray::Axis(0), i).assign(&t);
        }

        self.iteration += 1;
        Some((
            Variable::new(batch_x.into_dyn()),
            Variable::new(batch_t.into_dyn()),
        ))
    }
}
