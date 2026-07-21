use crate::conv::{ConvolutionLayer, PoolingLayer};
use crate::layers::{
    AffineLayer, DropoutLayer, FlattenLayer, Layer, ReluLayer, SoftmaxWithLossLayer,
};
use crate::optimizer::{Adam, Optimizer};
use ndarray::{Array1, Array2, Array4, Ix2, Ix4, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

/// 本 8.1「ネットワークをより深く」3×3 conv を 6 層重ねたディープ CNN。
/// He 初期化 + Adam(lr=0.001)を層ごとに内蔵(案B: 層が自分の optimizer を所有)し、
/// forward/backward は Vec を fold で流すだけ。MNIST full 10k で 99.32%
/// (本の ~99.4% と σ≈±0.08% の 1σ 以内で整合)。conv4 だけ pad=2 なのは本の仕込みで、
/// 14→16 に膨らませて 3 回の pooling をすべて偶数で割り切らせるため
pub struct DeepConvNet {
    //   1. [Conv(16) -> ReLU -> Conv(16) -> ReLU -> Pooling]
    //   2. [Conv(32) -> ReLU -> Conv(32) -> ReLU -> Pooling]
    //   3. [Conv(64) -> ReLU -> Conv(64) -> ReLU -> Pooling]
    //   4. Flatten
    //   5. [Affine(50) -> ReLU -> Dropout]
    //   6. [Affine(10) -> Dropout]
    //   7. SoftmaxWithLoss (last_layer)
    pub layers: Vec<Box<dyn Layer>>,
    pub last_layer: SoftmaxWithLossLayer,
}
impl DeepConvNet {
    pub fn new() -> Self {
        Self::new_with_dropout(0.5)
    }
    pub fn new_with_dropout(dropout_ratio: f32) -> Self {
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        //このネットワークでは、重みの初期値として「Heの初期値」を使用し、重みパラメータの更新にAdamを用います
        let he_conv = |fn_: usize, c: usize, fh: usize, fw: usize| -> Array4<f32> {
            let fan_in = (c * fh * fw) as f32;
            let scale = (2.0 / fan_in).sqrt();
            let w = Array4::random((fn_, c, fh, fw), StandardNormal);
            w * scale
        };
        let he_affine = |fan_in: usize, fan_out: usize| -> Array2<f32> {
            let scale = (2.0 / fan_in as f32).sqrt();
            let w = Array2::random((fan_in, fan_out), StandardNormal);
            w * scale
        };
        let lr = 0.001;
        let make_opt = |lr: f32| -> Box<dyn Optimizer> { Box::new(Adam::new(lr)) };

        //   1. [Conv(16) -> ReLU -> Conv(16) -> ReLU -> Pooling]
        let mut conv1_1 = ConvolutionLayer::new(he_conv(16, 1, 3, 3), Array1::zeros(16), 1, 1);
        conv1_1.set_optimizer(make_opt(lr), make_opt(lr));
        layers.push(Box::new(conv1_1));
        layers.push(Box::new(ReluLayer::new()));
        let mut conv1_2 = ConvolutionLayer::new(he_conv(16, 16, 3, 3), Array1::zeros(16), 1, 1);
        conv1_2.set_optimizer(make_opt(lr), make_opt(lr));
        layers.push(Box::new(conv1_2));
        layers.push(Box::new(ReluLayer::new()));
        layers.push(Box::new(PoolingLayer::new(2, 2, 2, 0)));

        //   2. [Conv(32) -> ReLU -> Conv(32) -> ReLU -> Pooling]
        let mut conv2_1 = ConvolutionLayer::new(he_conv(32, 16, 3, 3), Array1::zeros(32), 1, 1);
        conv2_1.set_optimizer(make_opt(lr), make_opt(lr));
        layers.push(Box::new(conv2_1));
        layers.push(Box::new(ReluLayer::new()));
        let mut conv2_2 = ConvolutionLayer::new(he_conv(32, 32, 3, 3), Array1::zeros(32), 1, 2);
        conv2_2.set_optimizer(make_opt(lr), make_opt(lr));
        layers.push(Box::new(conv2_2));
        layers.push(Box::new(ReluLayer::new()));
        layers.push(Box::new(PoolingLayer::new(2, 2, 2, 0)));

        //   3. [Conv(64) -> ReLU -> Conv(64) -> ReLU -> Pooling]
        let mut conv3_1 = ConvolutionLayer::new(he_conv(64, 32, 3, 3), Array1::zeros(64), 1, 1);
        conv3_1.set_optimizer(make_opt(lr), make_opt(lr));
        layers.push(Box::new(conv3_1));
        layers.push(Box::new(ReluLayer::new()));
        let mut conv3_2 = ConvolutionLayer::new(he_conv(64, 64, 3, 3), Array1::zeros(64), 1, 1);
        conv3_2.set_optimizer(make_opt(lr), make_opt(lr));
        layers.push(Box::new(conv3_2));
        layers.push(Box::new(ReluLayer::new()));
        layers.push(Box::new(PoolingLayer::new(2, 2, 2, 0)));

        //   4. Flatten
        layers.push(Box::new(FlattenLayer::new()));

        //   5. [Affine(50) -> ReLU -> Dropout]
        let mut affine1 = AffineLayer::new(he_affine(64 * 4 * 4, 50), Array2::zeros((1, 50)));
        affine1.set_optimizer(make_opt(lr), make_opt(lr));
        layers.push(Box::new(affine1));
        layers.push(Box::new(ReluLayer::new()));
        layers.push(Box::new(DropoutLayer::new(dropout_ratio)));

        //   6. [Affine(10) -> Dropout]
        let mut affine2 = AffineLayer::new(he_affine(50, 10), Array2::zeros((1, 10)));
        affine2.set_optimizer(make_opt(lr), make_opt(lr));
        layers.push(Box::new(affine2));
        layers.push(Box::new(DropoutLayer::new(dropout_ratio)));

        //   7. SoftmaxWithLoss (last_layer)
        let last_layer = SoftmaxWithLossLayer::new();

        Self { layers, last_layer }
    }

    pub fn predict(&mut self, x: Array4<f32>, train_flg: bool) -> Array2<f32> {
        self.layers
            .iter_mut()
            .fold(x.into_dyn(), |x, layer| layer.forward(x, train_flg))
            .into_dimensionality::<Ix2>()
            .unwrap()
    }

    pub fn loss(&mut self, x: Array4<f32>, t: Array2<f32>) -> f32 {
        let y = self.predict(x, true);
        self.last_layer.forward(y, t)
    }

    pub fn gradient(&mut self, x: Array4<f32>, t: Array2<f32>) -> (f32, Array4<f32>) {
        // forward
        let loss = self.loss(x, t);

        // backward
        let dout = self.last_layer.backward(1.0);
        let dx_dyn = self
            .layers
            .iter_mut()
            .rev()
            .fold(dout.into_dyn(), |acc, layer| layer.backward(acc));

        (loss, dx_dyn.into_dimensionality::<Ix4>().unwrap())
    }

    pub fn update(&mut self) {
        self.layers.iter_mut().for_each(|layer| {
            layer.update();
        });
    }

    pub fn accuracy(&mut self, x: &Array4<f32>, t: &Array2<f32>, batch_size: usize) -> f32 {
        let n_samples = x.shape()[0];
        let mut correct_count = 0;

        // 先に正解ラベル (t) を One-hot 表現から正解のインデックス（0〜9）に変換しておく
        let t_max_indices = t.map_axis(ndarray::Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        });

        // 0 から n_samples まで batch_size 刻みでループ
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples); // 最後の端数バッチにも対応

            // 画像データのスライスを切り出して所有権を持つ (to_owned)
            let tx = x.slice(s![i..end, .., .., ..]).to_owned();

            // 推論時は Dropout を無効にするので train_flg = false
            let y = self.predict(tx, false);

            // 推論結果の確率が一番高いインデックスを取得
            let y_max_indices = y.map_axis(ndarray::Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            });

            // 対応する正解ラベルのスライス
            let tt = t_max_indices.slice(s![i..end]);

            // 一致した数をカウントアップ
            correct_count += y_max_indices
                .iter()
                .zip(tt.iter())
                .filter(|(y_idx, t_idx)| y_idx == t_idx)
                .count();
        }

        // 全体の正解率を計算
        correct_count as f32 / n_samples as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        let mut net = DeepConvNet::new();
        let x = Array4::random((1, 1, 28, 28), StandardNormal);
        let y = net.predict(x.clone(), false);
        assert_eq!(y.shape(), &[1, 10]);
        assert!(y.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_loss() {
        let mut net = DeepConvNet::new();
        let x = Array4::zeros((1, 1, 28, 28));
        let t = Array2::from_shape_vec(
            (1, 10),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let loss = net.loss(x, t);

        let expected_initial_loss = 10.0f32.ln();
        assert!(
            (loss - expected_initial_loss).abs() < 1e-6,
            "Initial loss {} is too far from expected {}",
            loss,
            expected_initial_loss
        );
    }

    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal;
    fn run_gradcheck_once() -> Result<(), String> {
        let mut net = DeepConvNet::new_with_dropout(0.0);

        // リトライごとに新しい乱数で引く
        let x = Array4::random((1, 1, 28, 28), StandardNormal);
        let t = Array2::from_shape_vec(
            (1, 10),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let (_, dx_backprop) = net.gradient(x.clone(), t.clone());

        // 【非空虚ガード】
        if !dx_backprop.iter().any(|&v| v.abs() > 1e-6) {
            return Err("dx is perfectly zero!".to_string());
        }

        // 1. |dx| の大きい順（信号の強い場所）にインデックスをソートしてトップ5を抽出
        let mut dx_indexed: Vec<((usize, usize, usize, usize), f32)> = dx_backprop
            .indexed_iter()
            .map(|(idx, &v)| (idx, v))
            .collect();

        // 降順ソート
        dx_indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        // なぜ 1e-3 か: ulp 換算と kink 確率の挟み撃ち
        let h = 1e-3;
        // 上位5点の「S/Nが良い場所」だけをピンポイントで数値微分
        for &(idx, dx_bp) in dx_indexed.iter().take(5) {
            let mut x_plus = x.clone();
            x_plus[idx] += h;
            let loss_plus = net.loss(x_plus, t.clone());

            let mut x_minus = x.clone();
            x_minus[idx] -= h;
            let loss_minus = net.loss(x_minus, t.clone());

            let dx_num = (loss_plus - loss_minus) / (2.0 * h);

            // 信号が強い場所なので、純粋な相対誤差が機能する
            let diff = (dx_bp - dx_num).abs() / (dx_bp.abs() + dx_num.abs() + 1e-8);

            // S/Nが良いので 1e-2 の閾値でも十分安全圏
            if diff >= 1e-2 {
                return Err(format!(
                    "Mismatch at {:?}: bp={}, num={}, rel_diff={}",
                    idx, dx_bp, dx_num, diff
                ));
            }
        }

        Ok(())
    }

    #[test]
    fn test_grad() {
        let max_retries = 3;

        // 第7章の遺産: Kink踏みを許容する3回リトライ機構
        for attempt in 1..=max_retries {
            match run_gradcheck_once() {
                Ok(_) => {
                    println!("Gradcheck passed on attempt {}", attempt);
                    return; // テスト成功
                }
                Err(e) => {
                    println!("Attempt {} failed: {}", attempt, e);
                    if attempt == max_retries {
                        // 3回連続で落ちたなら、それは不運（Kink）ではなくバグ（配線ミス）
                        panic!(
                            "Gradcheck failed after {} attempts. Last error: {}",
                            max_retries, e
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_update() {
        // Dropout の乱数ノイズを排除し、純粋な学習能力（単調減少）をテストする
        let mut net = DeepConvNet::new_with_dropout(0.0);

        // 1. 小バッチのダミーデータを準備 (2枚, 1ch, 28x28)
        let x = Array4::random((2, 1, 28, 28), StandardNormal);
        let t = Array2::from_shape_vec(
            (2, 10),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();

        // 最初の損失を記録
        let initial_loss = net.loss(x.clone(), t.clone());
        println!("iter 0: loss = {}", initial_loss);

        // 2. 20 iter ほど学習を回す
        let iters = 10;
        let mut current_loss = initial_loss;

        for i in 1..=iters {
            // 順伝播・逆伝播で各レイヤの dw, db を計算・保持
            net.gradient(x.clone(), t.clone());

            // 内部の Optimizer (Adam) を使って w, b を更新
            net.update();

            let new_loss = net.loss(x.clone(), t.clone());
            println!("iter {}: loss = {}", i, new_loss);

            current_loss = new_loss;
        }

        // 3. 最終的に loss は初期状態より確実に下がり、ほぼ 0 に収束しているはず
        assert!(
            current_loss < initial_loss,
            "Loss did not drop after training"
        );
        assert!(current_loss < 0.1, "Loss did not converge well");
    }

    #[test]
    fn test_accuracy() {
        let mut net = DeepConvNet::new_with_dropout(0.0);
        // 5件のダミーデータ
        let x = Array4::random((5, 1, 28, 28), StandardNormal);
        // 1. まず predict を呼んでネットワークの「素の出力」の argmax を知る
        let y = net.predict(x.clone(), false);
        let y_max_indices: Vec<usize> = y
            .map_axis(ndarray::Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .to_vec();

        // 2. その出力結果に合わせて、意図的に正解率が 0.6 (3/5) になるラベル t を逆算して作る
        let mut t = Array2::zeros((5, 10));
        for i in 0..5 {
            let pred_idx = y_max_indices[i];
            if i < 3 {
                // 先頭の3件は predict と「一致」させる (正解)
                t[[i, pred_idx]] = 1.0;
            } else {
                // 残りの2件はわざと「外す」 (不正解)
                // 10クラスなので +1 して剰余をとれば確実にズレる
                let wrong_idx = (pred_idx + 1) % 10;
                t[[i, wrong_idx]] = 1.0;
            }
        }

        // 3. batch_size=2 という半端なサイズで呼び、端数(2,2,1)でカウント漏れがないかテスト
        let acc = net.accuracy(&x, &t, 2);

        // カウントロジックとバッチ分割が完璧なら、精度は厳密に 3/5 = 0.6 になるはず
        assert!(
            (acc - 0.6).abs() < 1e-6,
            "Expected accuracy 0.6, but got {}",
            acc
        );
    }

    use crate::mnist::{load_images, load_labels, to_one_hot};
    use ndarray::Axis;
    use std::time::Instant;
    #[test]
    #[ignore] // 実行に時間がかかるため通常は無視
    fn train_mnist_deep() {
        println!("Loading MNIST dataset...");
        // データの読み込みと Array4 (N, 1, 28, 28) への変換
        let x_train = load_images("dataset/train-images-idx3-ubyte")
            .into_shape_with_order((60000, 1, 28, 28))
            .unwrap();
        let t_train = load_labels("dataset/train-labels-idx1-ubyte");

        let x_test = load_images("dataset/t10k-images-idx3-ubyte")
            .into_shape_with_order((10000, 1, 28, 28))
            .unwrap();
        let t_test = load_labels("dataset/t10k-labels-idx1-ubyte");
        let t_test_onehot = to_one_hot(&t_test.to_vec(), 10);

        // Optimizer(Adam lr=0.001) と Dropout(0.5) を内包した本番用ネットワーク
        let mut net = DeepConvNet::new();

        let train_size = x_train.shape()[0];
        let batch_size = 100;
        let max_epochs = 20;
        let iter_per_epoch = (train_size / batch_size).max(1); // 600
        let iters_num = iter_per_epoch * max_epochs; // 12000

        let mut rng = rand::rng();

        println!("--- Training DeepConvNet ---");
        println!(
            "Max Epochs: {}, Batch Size: {}, Total Iters: {}",
            max_epochs, batch_size, iters_num
        );
        println!("(Run with: cargo test --release train_mnist_deep -- --ignored --nocapture)");

        let total_start_time = Instant::now();
        let mut epoch_start_time = Instant::now();
        let mut iter_start_time = Instant::now();

        for i in 1..=iters_num {
            // ミニバッチのランダム抽出
            let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();
            let x_batch = x_train.select(Axis(0), &idx);
            let batch_labels: Vec<u8> = idx.iter().map(|&j| t_train[j]).collect();
            let t_batch = to_one_hot(&batch_labels, 10);

            // 勾配計算と更新
            let (loss, _) = net.gradient(x_batch, t_batch);
            net.update();

            if i % 100 == 0 {
                let elapsed = iter_start_time.elapsed().as_secs_f32();
                println!(
                    "  iter {:5} | loss: {:.4} | Time(100iters): {:.2}s",
                    i, loss, elapsed
                );
                iter_start_time = Instant::now();
            }

            // 1エポック完了ごとに簡易評価と所要時間を表示
            if i % iter_per_epoch == 0 {
                let current_epoch = i / iter_per_epoch;

                // エポック中の評価は全量ではなく 1000 サンプル抽出で高速化 (誤差 σ ≈ ±0.3%)
                let test_sample_idx =
                    rand::seq::index::sample(&mut rng, x_test.shape()[0], 1000).into_vec();
                let x_test_sample = x_test.select(Axis(0), &test_sample_idx);
                let t_test_sample_labels: Vec<u8> =
                    test_sample_idx.iter().map(|&j| t_test[j]).collect();
                let t_test_sample = to_one_hot(&t_test_sample_labels, 10);

                // batch_size=100 で区切りながら 1000 件を評価
                let acc = net.accuracy(&x_test_sample, &t_test_sample, 100);
                let elapsed = epoch_start_time.elapsed().as_secs_f32();

                println!(
                    "Epoch {:2} | Acc(1000): {:.4} | Time: {:.2}s",
                    current_epoch, acc, elapsed
                );

                // 次のエポックの計測開始
                epoch_start_time = Instant::now();
            }
        }

        // 全エポック終了後、10,000件全データで厳密な最終評価
        println!("--- Training Completed ---");
        println!(
            "Total Time: {:.2}s",
            total_start_time.elapsed().as_secs_f32()
        );

        println!("Evaluating final accuracy on full 10,000 test set...");
        let final_acc = net.accuracy(&x_test, &t_test_onehot, 100);
        println!("Final Accuracy: {:.4} (Expected ~0.994)", final_acc);

        // 最終的に 99% の大台に乗っていれば大成功
        assert!(
            final_acc > 0.99,
            "Final accuracy {} is below 99%",
            final_acc
        );
    }
}
