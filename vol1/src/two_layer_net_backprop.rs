use crate::layers::{AffineLayer, BatchNormLayer, DropoutLayer, ReluLayer, SoftmaxWithLossLayer};
use crate::optimizer::Optimizer;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

/// 本 5.7.2「誤差逆伝播法に対応したニューラルネットワークの実装」+ 6章のテクニック統合版。
/// レイヤ構成:Affine→BatchNorm(6.3)→ReLU→Dropout(6.4.3)→Affine→SoftmaxWithLoss。
/// パラメータ(w1/b1/w2/b2)ごとに独立した Optimizer(6.1)を持ち、
/// weight decay(6.4.2)は loss の罰則項と gradient の λW 加算を対で適用する
pub struct TwoLayerNetBackprop {
    affine1: AffineLayer,
    bn: BatchNormLayer,
    relu: ReluLayer,
    affine2: AffineLayer,
    dropout: DropoutLayer,
    last_layer: SoftmaxWithLossLayer,
    opt_w1: Box<dyn Optimizer>,
    opt_b1: Box<dyn Optimizer>,
    opt_w2: Box<dyn Optimizer>,
    opt_b2: Box<dyn Optimizer>,
    weight_decay_lambda: f32,
}
impl TwoLayerNetBackprop {
    /// make_opt: Optimizer のファクトリ(6.1)。中で4回呼び、パラメータごとに独立した
    /// インスタンス(=独立した v/h/m の履歴)を作る。Box は Clone できないためこの形。
    /// make_std: fan-in から重み初期値の標準偏差を決める(6.2)。
    /// He なら |n| (2.0/n as f32).sqrt()、Xavier なら 1.0/n、固定なら |_| 0.01
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_decay_lambda: f32,
        dropout_ratio: f32,
        make_opt: impl Fn() -> Box<dyn Optimizer>,
        make_std: impl Fn(usize) -> f32,
    ) -> Self {
        let w1 = Array2::random((input_size, hidden_size), StandardNormal) * make_std(input_size);
        let b1 = Array2::zeros((1, hidden_size)); // AffineLayer のバイアスは (1, n) 形
        let w2 = Array2::random((hidden_size, output_size), StandardNormal) * make_std(hidden_size);
        let b2 = Array2::zeros((1, output_size));

        let affine1 = AffineLayer::new(w1, b1);
        let bn = BatchNormLayer::new(hidden_size);
        let affine2 = AffineLayer::new(w2, b2);
        let relu = ReluLayer::new();
        let last_layer = SoftmaxWithLossLayer::new();
        Self {
            affine1,
            bn,
            relu,
            affine2,
            dropout: DropoutLayer::new(dropout_ratio),
            last_layer,
            opt_w1: make_opt(),
            opt_b1: make_opt(),
            opt_w2: make_opt(),
            opt_b2: make_opt(),
            weight_decay_lambda,
        }
    }

    /// train_flag: Dropout の訓練/推論モード切り替え(6.4.3)。学習時 true、評価時 false。
    /// 明示引数方式なので渡し忘れはコンパイルエラーになる
    pub fn predict(&mut self, x: Array2<f32>, train_flag: bool) -> Array2<f32> {
        let out1 = self.affine1.forward(x);
        let out2 = self.bn.forward(out1);
        let out3 = self.relu.forward(out2.into_dyn()).into_dimensionality().unwrap();
        let out4 = self.dropout.forward(out3, train_flag);
        self.affine2.forward(out4)
    }

    /// 交差エントロピー + weight decay 罰則項 (λ/2)(ΣW1²+ΣW2²)(6.4.2)。
    /// 罰則項は gradient() 側の add_weight_decay と対(数値微分との整合が保たれる)
    pub fn loss(&mut self, x: Array2<f32>, t: Array2<f32>, train_flag: bool) -> f32 {
        let y = self.predict(x, train_flag);
        let weight_decay = 0.5
            * self.weight_decay_lambda
            * (self.affine1.w().iter().map(|v| v * v).sum::<f32>()
                + self.affine2.w().iter().map(|v| v * v).sum::<f32>());
        self.last_layer.forward(y, t) + weight_decay
    }

    /// 本 5.7.2 誤差逆伝播で全パラメータの勾配を1パスで求める(順伝播→逆順に backward)
    pub fn gradient(
        &mut self,
        x: Array2<f32>,
        t: Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        // 順伝播
        self.loss(x, t, true);

        // 逆伝播
        let dout = self.last_layer.backward(1.0);
        let dout = self.affine2.backward(dout);
        let dout = self.dropout.backward(dout);
        let dout = self.relu.backward(dout.into_dyn()).into_dimensionality().unwrap();
        let dout = self.bn.backward(dout);
        let _dout = self.affine1.backward(dout);

        self.affine1.add_weight_decay(self.weight_decay_lambda);
        self.affine2.add_weight_decay(self.weight_decay_lambda);

        (
            self.affine1.dw().clone(),
            self.affine1.db().clone(),
            self.affine2.dw().clone(),
            self.affine2.db().clone(),
        )
    }

    /// 本 5.7.3 勾配確認用。数値微分で全パラメータの勾配を求める(遅いが単純な基準)
    pub fn numerical_gradient(
        &mut self,
        x: Array2<f32>,
        t: Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let h = 1e-4;
        let (rows, cols) = self.affine1.w().dim();
        let mut dw1 = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let original_value = self.affine1.w_mut()[(i, j)];
                self.affine1.w_mut()[(i, j)] = original_value + h;
                let loss_plus_h = self.loss(x.clone(), t.clone(), true);
                self.affine1.w_mut()[(i, j)] = original_value - h;
                let loss_minus_h = self.loss(x.clone(), t.clone(), true);
                dw1[(i, j)] = (loss_plus_h - loss_minus_h) / (2.0 * h);
                self.affine1.w_mut()[(i, j)] = original_value;
            }
        }

        let (rows, cols) = self.affine2.w().dim();
        let mut dw2 = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let original_value = self.affine2.w_mut()[(i, j)];
                self.affine2.w_mut()[(i, j)] = original_value + h;
                let loss_plus_h = self.loss(x.clone(), t.clone(), true);
                self.affine2.w_mut()[(i, j)] = original_value - h;
                let loss_minus_h = self.loss(x.clone(), t.clone(), true);
                dw2[(i, j)] = (loss_plus_h - loss_minus_h) / (2.0 * h);
                self.affine2.w_mut()[(i, j)] = original_value;
            }
        }

        let (rows, cols) = self.affine1.b().dim(); // バイアスは (1, hidden) 形
        let mut db1 = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let original_value = self.affine1.b()[(i, j)];
                self.affine1.b_mut()[(i, j)] = original_value + h;
                let loss_plus_h = self.loss(x.clone(), t.clone(), true);
                self.affine1.b_mut()[(i, j)] = original_value - h;
                let loss_minus_h = self.loss(x.clone(), t.clone(), true);
                self.affine1.b_mut()[(i, j)] = original_value;
                db1[(i, j)] = (loss_plus_h - loss_minus_h) / (2.0 * h);
            }
        }

        let (rows, cols) = self.affine2.b().dim(); // バイアスは (1, output) 形
        let mut db2 = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let original_value = self.affine2.b()[(i, j)];
                self.affine2.b_mut()[(i, j)] = original_value + h;
                let loss_plus_h = self.loss(x.clone(), t.clone(), true);
                self.affine2.b_mut()[(i, j)] = original_value - h;
                let loss_minus_h = self.loss(x.clone(), t.clone(), true);
                self.affine2.b_mut()[(i, j)] = original_value;
                db2[(i, j)] = (loss_plus_h - loss_minus_h) / (2.0 * h);
            }
        }

        (dw1, db1, dw2, db2)
    }

    /// 本 5.7.4 の学習ステップを 6.1 の Optimizer 経由に置き換えたもの。
    /// 各パラメータを専属 Optimizer に渡して1ステップ更新する。
    /// lr 等の更新則の詳細は Optimizer の内部事情になったため引数から消えた
    pub fn update(&mut self) {
        let (w, dw) = self.affine1.w_and_dw();
        self.opt_w1.update(&mut w.view_mut().into_dyn(), &dw.view().into_dyn());
        let (b, db) = self.affine1.b_and_db();
        self.opt_b1.update(&mut b.view_mut().into_dyn(), &db.view().into_dyn());
        let (w, dw) = self.affine2.w_and_dw();
        self.opt_w2.update(&mut w.view_mut().into_dyn(), &dw.view().into_dyn());
        let (b, db) = self.affine2.b_and_db();
        self.opt_b2.update(&mut b.view_mut().into_dyn(), &db.view().into_dyn());
    }

    /// 本 6.4.1 の過学習実験用に ch4 TwoLayerNet から移植。行ごとの argmax を突き合わせる。
    /// 評価なので predict は train_flag=false(Dropout 無効)で呼ぶ
    pub fn accuracy(&mut self, x: Array2<f32>, t: Array2<f32>) -> f32 {
        let x_n_dim = x.shape()[0];
        let y = self.predict(x, false);
        let y_max_indices = y.map_axis(ndarray::Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        });
        let t_max_indices = t.map_axis(ndarray::Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        });

        let correct_count = y_max_indices
            .iter()
            .zip(t_max_indices.iter())
            .filter(|(y_idx, t_idx)| y_idx == t_idx)
            .count();
        correct_count as f32 / x_n_dim as f32
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::optimizer::{Adam, Momentum, SGD};

    #[test]
    fn test_two_layer_net_backprop_loss() {
        // 損失が正の有限値であることを確認するテスト
        let mut net = TwoLayerNetBackprop::new(
            3,
            4,
            2,
            0.0,
            0.0,
            || Box::new(SGD::new(0.1)),
            |x| (2.0 / x as f32).sqrt(),
        ); // input=3, hidden=4, output=2
        let x = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(); // (batch, input_size)
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap(); // (batch, output_size)
        let loss = net.loss(x, t, true);
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn test_two_layer_net_backprop_gradient() {
        //gradient を呼ぶと各勾配の形が対応する重みと一致、各要素が有限であることを確認するテスト
        let mut net = TwoLayerNetBackprop::new(
            3,
            4,
            2,
            0.0,
            0.0,
            || Box::new(SGD::new(0.1)),
            |x| (2.0 / x as f32).sqrt(),
        );
        let x = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let (dw1, db1, dw2, db2) = net.gradient(x, t);
        assert_eq!(dw1.dim(), net.affine1.dw().dim());
        assert_eq!(db1.dim(), net.affine1.db().dim());
        assert_eq!(dw2.dim(), net.affine2.dw().dim());
        assert_eq!(db2.dim(), net.affine2.db().dim());
        assert!(dw1.iter().all(|&v| v.is_finite()));
        assert!(db1.iter().all(|&v| v.is_finite()));
        assert!(dw2.iter().all(|&v| v.is_finite()));
        assert!(db2.iter().all(|&v| v.is_finite()));
    }

    fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn test_two_layer_net_backprop_numerical_gradient() {
        // 勾配確認テスト
        // 同じネットで両方を計算して比べます。順序に注意:gradient() は逆伝播で状態を書き換える(dw/db を埋める)だけで重みは変えないので、先に数値微分、後に逆伝播、どちらでもOKですが、混乱を避けるため別々に取ってから比較します。
        let mut net =
            TwoLayerNetBackprop::new(3, 4, 2, 0.0, 0.0, || Box::new(SGD::new(0.1)), |_| 0.01);
        *net.affine1.w_mut() *= 100.0; // 勾配を解像可能な大きさに
        *net.affine2.w_mut() *= 100.0; // 勾配確認を f32 で意味あるものにする
        let x = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let (dw1_num, db1_num, dw2_num, db2_num) = net.numerical_gradient(x.clone(), t.clone());
        let (dw1_backprop, db1_backprop, dw2_backprop, db2_backprop) =
            net.gradient(x.clone(), t.clone());

        println!("dw1_num: {:?}", dw1_num);
        println!("dw1_backprop: {:?}", dw1_backprop);
        println!("db1_num: {:?}", db1_num);
        println!("db1_backprop: {:?}", db1_backprop);
        println!("dw2_num: {:?}", dw2_num);
        println!("dw2_backprop: {:?}", dw2_backprop);
        println!("db2_num: {:?}", db2_num);
        println!("db2_backprop: {:?}", db2_backprop);
        let epsilon: f32 = 1e-2;
        assert!(max_abs_diff(&dw1_num, &dw1_backprop) < epsilon);
        assert!(max_abs_diff(&db1_num, &db1_backprop) < epsilon);
        assert!(max_abs_diff(&dw2_num, &dw2_backprop) < epsilon);
        assert!(max_abs_diff(&db2_num, &db2_backprop) < epsilon);
    }

    use crate::mnist::{load_images, load_labels, to_one_hot};
    use ndarray::{Axis, s};
    #[test]
    #[ignore] // 実行に時間がかかるので CI では無視
    fn train_mnist_backprop() {
        let images = load_images("dataset/train-images-idx3-ubyte"); // (60000, 784)
        let labels = load_labels("dataset/train-labels-idx1-ubyte");
        let train_size = images.shape()[0];

        let batch_size = 100;
        let iters_num = 1000; // 逆伝播なら現実的に回せる
        let mut net_sgd = TwoLayerNetBackprop::new(
            784,
            50,
            10,
            0.0,
            0.0,
            || Box::new(SGD::new(0.1)),
            // |x| (2.0 / x as f32).sqrt(),
            |_| 0.0001,
        );

        // 固定の評価バッチ(トレンドを綺麗に見るため先頭100件)
        let eval_x = images.slice(s![0..100, ..]).to_owned();
        let eval_t = to_one_hot(&labels.slice(s![0..100]).to_vec(), 10);

        println!("Training TwoLayerNetBackprop with SGD...");
        let mut rng = rand::rng();
        for i in 0..iters_num {
            let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();
            let x_batch = images.select(Axis(0), &idx);
            let batch_labels: Vec<u8> = idx.iter().map(|&j| labels[j]).collect();
            let t_batch = to_one_hot(&batch_labels, 10);

            net_sgd.gradient(x_batch, t_batch); // dw/db を埋める
            net_sgd.update();

            if i % 100 == 0 {
                let loss = net_sgd.loss(eval_x.clone(), eval_t.clone(), true);
                println!("iter {i}: loss = {loss}");
            }
        }
        let loss = net_sgd.loss(eval_x.clone(), eval_t.clone(), true);
        println!("Final loss with SGD: {loss}");

        let mut net_adam = TwoLayerNetBackprop::new(
            784,
            50,
            10,
            0.0,
            0.0,
            || Box::new(Adam::new(0.001)),
            // |x| (2.0 / x as f32).sqrt(),
            |_| 0.0001,
        );
        println!("Training TwoLayerNetBackprop with Adam...");
        for i in 0..iters_num {
            let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();
            let x_batch = images.select(Axis(0), &idx);
            let batch_labels: Vec<u8> = idx.iter().map(|&j| labels[j]).collect();
            let t_batch = to_one_hot(&batch_labels, 10);
            net_adam.gradient(x_batch, t_batch); // dw/db を埋める
            net_adam.update();

            if i % 100 == 0 {
                let loss = net_adam.loss(eval_x.clone(), eval_t.clone(), true);
                println!("iter {i}: loss = {loss}");
            }
        }
        let loss = net_adam.loss(eval_x.clone(), eval_t.clone(), true);
        println!("Final loss with Adam: {loss}");

        let mut net_momentum = TwoLayerNetBackprop::new(
            784,
            50,
            10,
            0.0,
            0.0,
            || Box::new(Momentum::new(0.01, 0.9)),
            // |x| (2.0 / x as f32).sqrt(),
            |_| 0.0001,
        );
        println!("Training TwoLayerNetBackprop with Momentum...");
        for i in 0..iters_num {
            let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();
            let x_batch = images.select(Axis(0), &idx);
            let batch_labels: Vec<u8> = idx.iter().map(|&j| labels[j]).collect();
            let t_batch = to_one_hot(&batch_labels, 10);
            net_momentum.gradient(x_batch, t_batch); // dw/db を埋める
            net_momentum.update();
            if i % 100 == 0 {
                let loss = net_momentum.loss(eval_x.clone(), eval_t.clone(), true);
                println!("iter {i}: loss = {loss}");
            }
        }
        let loss = net_momentum.loss(eval_x.clone(), eval_t.clone(), true);
        println!("Final loss with Momentum: {loss}");
    }

    #[test]
    #[ignore] // 実行に時間がかかるので CI では無視
    fn test_overfitting() {
        // 1. 訓練データ先頭 300 サンプル(x_train, t_train)
        // 2. 評価用に t10k データ(ch3 で使った dataset/t10k-* ファイル)
        // 3. SGD(lr=0.01)などで 2000〜3000 iters、バッチは 300 の中から 100 サンプル
        // 4. 100 iter ごとに train_acc(300 全部で)と test_acc を println
        let x_train = load_images("dataset/train-images-idx3-ubyte")
            .slice(s![0..300, ..])
            .to_owned();
        let t_train = load_labels("dataset/train-labels-idx1-ubyte")
            .slice(s![0..300])
            .to_owned();
        let x_test = load_images("dataset/t10k-images-idx3-ubyte");
        let t_test = load_labels("dataset/t10k-labels-idx1-ubyte");

        let mut net = TwoLayerNetBackprop::new(
            784,
            50,
            10,
            0.0,
            0.2,
            || Box::new(SGD::new(0.01)),
            |x| (2.0 / x as f32).sqrt(),
        );

        let batch_size = 100;
        let iters_num = 2000;
        let mut rng = rand::rng();
        for i in 0..iters_num {
            let idx = rand::seq::index::sample(&mut rng, 300, batch_size).into_vec();
            let x_batch = x_train.select(Axis(0), &idx);
            let batch_labels: Vec<u8> = idx.iter().map(|&j| t_train[j]).collect();
            let t_batch = to_one_hot(&batch_labels, 10);
            net.gradient(x_batch, t_batch);
            net.update();
            if i % 100 == 0 {
                let train_acc = net.accuracy(x_train.clone(), to_one_hot(&t_train.to_vec(), 10));
                let test_acc = net.accuracy(x_test.clone(), to_one_hot(&t_test.to_vec(), 10));
                println!("iter {i}: train_acc = {train_acc}, test_acc = {test_acc}");
            }
        }
    }

    use rand::Rng;
    use rayon::prelude::*;
    #[test]
    #[ignore] // 実行に時間がかかるので CI では無視
    fn test_hyperparameter_tuning() {
        // 1. 訓練 500 サンプル、検証 200 サンプル程度に分割(速さ優先の小規模で)
        let x_train = load_images("dataset/train-images-idx3-ubyte")
            .slice(s![0..500, ..])
            .to_owned();
        let t_train = load_labels("dataset/train-labels-idx1-ubyte")
            .slice(s![0..500])
            .to_owned();
        let x_val = load_images("dataset/train-images-idx3-ubyte")
            .slice(s![500..700, ..])
            .to_owned();
        let t_val = load_labels("dataset/train-labels-idx1-ubyte")
            .slice(s![500..700])
            .to_owned();

        // 2. 30〜50 回試行:lr と λ(お好みで dropout_ratio も)を対数一様サンプル
        let mut parameter_results: Vec<(f32, f32, f32, f32)> = (0..25)
            .into_par_iter() // ← ここが並列化の全て
            .map(|_| {
                let mut rng = rand::rng();
            // let lr = 10f32.powf(rng.random_range(-6.0..-2.0)); // First trial
            // let lr = 10f32.powf(rng.random_range(-3.0..0.0)); // Second trial
            let lr = 0.1;
            let lambda = 10f32.powf(rng.random_range(-8.0..-4.0));
            // let lambda = 0.0;
            let dropout_ratio = rng.random_range(0.0..0.4);
            // let dropout_ratio = 0.0;
            let mut net = TwoLayerNetBackprop::new(
                784,
                50,
                10,
                lambda,
                dropout_ratio,
                || Box::new(SGD::new(lr)),
                |x| (2.0 / x as f32).sqrt(),
            );

            // 3. 各試行:ネットを新規作成 → 短く学習(200 iters 程度)→ val_acc を記録
            let batch_size = 100;
            // let iters_num = 200; // For lr search
            let iters_num = 2000;
            for _ in 0..iters_num {
                let idx = rand::seq::index::sample(&mut rng, 500, batch_size).into_vec();
                let x_batch = x_train.select(Axis(0), &idx);
                let batch_labels: Vec<u8> = idx.iter().map(|&j| t_train[j]).collect();
                let t_batch = to_one_hot(&batch_labels, 10);
                net.gradient(x_batch, t_batch);
                net.update();
            }
            let val_acc = net.accuracy(x_val.clone(), to_one_hot(&t_val.to_vec(), 10));
            println!(
                "lr={lr:.6}, λ={lambda:.8}, dropout_ratio={dropout_ratio:.2}, final val_acc = {val_acc:.4}"
            );
        
            (lr, lambda, dropout_ratio, val_acc) // タプルで返して collect
        }).collect();

        // 4. 最後に val_acc の降順で「lr=..., λ=... → val_acc=...」を一覧表示
        parameter_results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        for (lr, lambda, dropout_ratio, val_acc) in parameter_results {
            println!(
                "lr={lr:.6}, λ={lambda:.8}, dropout_ratio={dropout_ratio:.2}, val_acc={val_acc:.4}"
            );
        }
        // 5. 上位に共通する lr の桁を観察(→ 範囲を狭めて再走、が本の流儀)
        // その後、λ や dropout_ratio も同様に範囲を狭めて再走するのが本の流儀
        // First lr search
        // lr=0.009355, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7300
        // lr=0.009888, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7250
        // lr=0.008723, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6950
        // lr=0.008148, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6900
        // lr=0.009316, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6800
        // lr=0.007341, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6800
        // lr=0.006192, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6750
        // lr=0.007387, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6350
        // lr=0.002951, λ=0.00000000, dropout_ratio=0.00, val_acc=0.5750
        // lr=0.003336, λ=0.00000000, dropout_ratio=0.00, val_acc=0.5550
        // lr=0.002010, λ=0.00000000, dropout_ratio=0.00, val_acc=0.4900
        // lr=0.001840, λ=0.00000000, dropout_ratio=0.00, val_acc=0.4350
        // lr=0.001049, λ=0.00000000, dropout_ratio=0.00, val_acc=0.3800
        // lr=0.001008, λ=0.00000000, dropout_ratio=0.00, val_acc=0.3650
        // lr=0.000745, λ=0.00000000, dropout_ratio=0.00, val_acc=0.3300
        // lr=0.000889, λ=0.00000000, dropout_ratio=0.00, val_acc=0.3100
        // lr=0.001220, λ=0.00000000, dropout_ratio=0.00, val_acc=0.3000
        // lr=0.000514, λ=0.00000000, dropout_ratio=0.00, val_acc=0.2850
        // lr=0.000719, λ=0.00000000, dropout_ratio=0.00, val_acc=0.2750
        // lr=0.000858, λ=0.00000000, dropout_ratio=0.00, val_acc=0.2600
        // lr=0.000499, λ=0.00000000, dropout_ratio=0.00, val_acc=0.2000
        // lr=0.000225, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1800
        // lr=0.000656, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1700
        // lr=0.000553, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1500
        // lr=0.000002, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1400
        // lr=0.000073, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1350
        // lr=0.000013, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1250
        // lr=0.000401, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1200
        // lr=0.000248, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1200
        // lr=0.000079, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1200
        // lr=0.000003, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1150
        // lr=0.000056, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1100
        // lr=0.000057, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1100
        // lr=0.000098, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1100
        // lr=0.000127, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1050
        // lr=0.000174, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1050
        // lr=0.000003, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1050
        // lr=0.000020, λ=0.00000000, dropout_ratio=0.00, val_acc=0.1000
        // lr=0.000009, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0950
        // lr=0.000012, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0950
        // lr=0.000026, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0900
        // lr=0.000007, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0850
        // lr=0.000003, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0800
        // lr=0.000017, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0800
        // lr=0.000002, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0700
        // lr=0.000007, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0700
        // lr=0.000005, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0700
        // lr=0.000002, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0700
        // lr=0.000069, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0600
        // lr=0.000013, λ=0.00000000, dropout_ratio=0.00, val_acc=0.0600
        //
        // Second lr search
        // lr=0.098026, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7900
        // lr=0.950267, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7900
        // lr=0.298855, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7900
        // lr=0.756477, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7850
        // lr=0.038396, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7850
        // lr=0.499032, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7850
        // lr=0.898662, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7800
        // lr=0.284083, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7750
        // lr=0.140571, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7700
        // lr=0.950340, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7700
        // lr=0.090118, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7700
        // lr=0.371965, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7650
        // lr=0.709765, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7600
        // lr=0.123031, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7600
        // lr=0.526537, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7600
        // lr=0.035591, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7600
        // lr=0.038384, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7550
        // lr=0.018126, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7500
        // lr=0.120593, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7500
        // lr=0.019503, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7500
        // lr=0.144299, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7450
        // lr=0.036782, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7400
        // lr=0.010920, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7400
        // lr=0.022138, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7350
        // lr=0.028041, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7350
        // lr=0.012664, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7300
        // lr=0.010442, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7150
        // lr=0.015054, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7150
        // lr=0.014571, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7100
        // lr=0.016792, λ=0.00000000, dropout_ratio=0.00, val_acc=0.7050
        // lr=0.009559, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6950
        // lr=0.008547, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6950
        // lr=0.008463, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6900
        // lr=0.005617, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6900
        // lr=0.005375, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6800
        // lr=0.005106, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6700
        // lr=0.007750, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6700
        // lr=0.004553, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6450
        // lr=0.003808, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6400
        // lr=0.003829, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6400
        // lr=0.002690, λ=0.00000000, dropout_ratio=0.00, val_acc=0.6000
        // lr=0.003150, λ=0.00000000, dropout_ratio=0.00, val_acc=0.5700
        // lr=0.002765, λ=0.00000000, dropout_ratio=0.00, val_acc=0.5650
        // lr=0.002563, λ=0.00000000, dropout_ratio=0.00, val_acc=0.5500
        // lr=0.002017, λ=0.00000000, dropout_ratio=0.00, val_acc=0.5100
        // lr=0.001566, λ=0.00000000, dropout_ratio=0.00, val_acc=0.4600
        // lr=0.001404, λ=0.00000000, dropout_ratio=0.00, val_acc=0.4550
        // lr=0.001422, λ=0.00000000, dropout_ratio=0.00, val_acc=0.4450
        // lr=0.001267, λ=0.00000000, dropout_ratio=0.00, val_acc=0.4200
        // lr=0.001359, λ=0.00000000, dropout_ratio=0.00, val_acc=0.3850
    }
}
