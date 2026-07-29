use ndarray_npy::NpzReader;
use rand::SeedableRng;
use std::fs::File;
use std::path::Path;
use vol3::dataloaders::DataLoader;
use vol3::datasets::MnistDataset;
use vol3::functions::mean_squared_error;
use vol3::layers::Layer;
use vol3::optimizers::{Adam, Optimizer};
use vol3::variable::Variable;
use vol5::utils::approx_eq;
use vol5::vae::{VAE, kl_divergence, reparameterize_with_eps};

#[test]
fn experiment_step07_parity_10iters() {
    println!("--- Step 7: VAE 10イテレーション厳密パリティ検証 ---");

    let file = File::open("dataset/step07_golden.npz").unwrap();
    let mut npz = NpzReader::new(file).unwrap();

    let x_data: ndarray::ArrayD<f32> = npz.by_name("x_batch").unwrap();
    let eps_seq: ndarray::ArrayD<f32> = npz.by_name("eps_seq").unwrap(); // (10, 32, 20) など
    let loss_seq_pt: ndarray::ArrayD<f32> = npz.by_name("loss_seq").unwrap(); // (10,)

    let x = Variable::new(x_data);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let vae = VAE::new(784, 200, 20, &mut rng);
    vae.load_weights(Path::new("dataset/step07_golden.npz"))
        .unwrap();

    let mut optimizer = Adam::new(3e-4);
    optimizer.setup(&vae);

    let iters = 10;
    let mut my_loss_history = Vec::new();

    // eps_seq をイテレーションごとに切り出す
    let eps_view = eps_seq.into_dimensionality::<ndarray::Ix3>().unwrap();

    for i in 0..iters {
        // 1. 各イテレーションの固定化された eps (乱数) を取得 (shape: 32x20)
        let eps_i = eps_view.slice(ndarray::s![i, .., ..]).to_owned().into_dyn();
        let eps = Variable::new(eps_i);

        // 2. 順伝播
        let (mu, ln_var) = vae.enc.encode(&x);
        let sigma = (&ln_var * 0.5).exp();

        // 乱数生成器ではなく、PyTorchが使ったのと同じ eps を注入する！
        let z = reparameterize_with_eps(&mu, &sigma, &eps);
        let y_pred = vae.dec.forward(&z);

        // 3. 損失関数の計算
        let rec_loss = mean_squared_error(&y_pred, &x);
        let kl_loss = kl_divergence(&mu, &ln_var);
        let loss = rec_loss + kl_loss;

        my_loss_history.push(
            loss.data()
                .into_dimensionality::<ndarray::Ix0>()
                .unwrap()
                .into_scalar(),
        );

        // 4. 逆伝播と更新
        vae.cleargrads();
        loss.backward(false, false);
        optimizer.update();
    }

    // パリティ検証 (Loss)
    for i in 0..iters {
        let diff = (my_loss_history[i] - loss_seq_pt[[i]]).abs();
        // Loss は ~184。f32 の相対精度限界を考慮し、絶対 tol を 1e-3 (相対 ~5e-6) に設定
        assert!(
            approx_eq(my_loss_history[i], loss_seq_pt[[i]], 1e-3),
            "Loss mismatch at iter {}: my={}, pt={}, diff={}",
            i,
            my_loss_history[i],
            loss_seq_pt[[i]],
            diff
        );
    }
    println!(
        "10 iters Loss parity passed. Final Loss: {:.6}",
        my_loss_history[9]
    );

    // パリティ検証 (最終重みの Max |Δ|)
    let mut max_weight_delta = 0.0f32;
    for (name, param) in vae.named_params() {
        let golden_name = format!("final10/{}", name);
        let pt_weight: ndarray::ArrayD<f32> = npz.by_name(&golden_name).unwrap();
        let my_weight = param.data();

        let local_max = (&my_weight - &pt_weight)
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, |a, b| a.max(b));
        if local_max > max_weight_delta {
            max_weight_delta = local_max;
        }
    }

    println!(
        "Max Weight Delta (max |Δ|) after 10 iters: {:.2e}",
        max_weight_delta
    );
    // 5.35e-4 実測
    assert!(
        max_weight_delta < 5e-3,
        "Weight parity failed! max |Δ| = {}",
        max_weight_delta
    );
}

// --- 追加: transform 関数 ---
// [1, 28, 28] などの f32 テンソルを受け取り、255で割って [784] に平坦化する
fn transform_vae(mut x: ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
    x.mapv_inplace(|v| v / 255.0);
    x.into_shape_with_order((784,)).unwrap().into_dyn()
}

#[test]
#[ignore] // 実データ学習は時間がかかるため、通常のテストではスキップ
fn experiment_step07_full_training() {
    println!("--- Step 7: VAE 30 Epoch 実データ学習 (帯域 Assert) ---");

    // ゴールデンカーブの読み込み
    let file = File::open("dataset/step07_epoch_curve.npz").unwrap();
    let _npz = NpzReader::new(file).unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let vae = VAE::new(784, 200, 20, &mut rng);
    let mut optimizer = Adam::new(3e-4);
    optimizer.setup(&vae);

    // train = true, transform 関数を指定
    let dataset = MnistDataset::new(true, Some(transform_vae));

    // batch_size = 32, shuffle = true
    let mut loader = DataLoader::new(
        dataset,
        32,
        true,
        Box::new(rand::rngs::StdRng::seed_from_u64(1)),
    );
    let epochs = 30;
    let mut my_epoch_losses = Vec::new();
    for epoch in 0..epochs {
        let mut sum_loss = 0.0;
        let mut batches = 0;
        // loader は (Variable(バッチ化された x), Vec<Label>) を返す
        for (x, _labels) in &mut loader {
            // 通常の reparameterize (乱数注入) を使用
            let (y_pred, mu, ln_var) = vae.forward_vae(&x, &mut rng);

            // 損失計算 (L1 + L2)
            let rec_loss = mean_squared_error(&y_pred, &x);
            let kl_loss = kl_divergence(&mu, &ln_var);
            let loss = rec_loss + kl_loss;
            // スカラー値を取り出して加算
            sum_loss += loss
                .data()
                .into_dimensionality::<ndarray::Ix0>()
                .unwrap()
                .into_scalar();
            batches += 1;
            vae.cleargrads();
            loss.backward(false, false);
            optimizer.update();
        }

        let avg_loss = sum_loss / (batches as f32);
        my_epoch_losses.push(avg_loss);
        println!("Epoch {}: Loss = {:.4}", epoch + 1, avg_loss);
    }
    // 帯域 Assert (終盤 30 Epoch の収束値が 35.0 〜 43.0 の範囲に収まっているか)
    let final_loss = my_epoch_losses[epochs - 1];
    println!("Final Epoch Loss: {:.4}", final_loss);

    assert!(
        final_loss > 35.0 && final_loss < 43.0,
        "Convergence failed! Expected final loss in [35.0, 43.0], but got {:.4}",
        final_loss
    );
}
