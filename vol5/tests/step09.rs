use ndarray::{Array1, Array2, ArrayD, Axis};
use ndarray_npy::NpzReader;
use rand::RngExt;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::Read;
use std::println;
use std::time::Instant;
use vol3::functions::mean_squared_error;
use vol3::layers::Layer;
use vol3::optimizers::{Adam, Optimizer};
use vol3::utils::approx_eq;
use vol3::variable::Variable;
use vol5::diffusion::Diffusion;
use vol5::unet::UNet;

#[test]
fn test_step09_parity() {
    let mut npz = NpzReader::new(File::open("dataset/step09_golden.npz").unwrap()).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let unet = UNet::new(1, 100, &mut rng);

    // Diffusion スケジューラの初期化 (ステップ数1000など、ゴールデンと同条件を想定)
    let diffusion = Diffusion::new(1000, 1e-4, 0.02);
    // 1. 初期重みのロード (KEYMAP方式)
    let mut loaded_count = 0;
    for (name, param) in unet.named_params() {
        let key = format!("{}.npy", name);

        let golden_res: Result<ndarray::ArrayD<f32>, _> = npz.by_name(&key);
        match golden_res {
            Ok(golden_data) => {
                param.set_data(golden_data);
                loaded_count += 1;
            }
            Err(_) => {
                // Step 09 ゴールデンには label_emb は存在しない。それ以外の欠損はパニック
                assert_eq!(
                    name, "label_emb/W",
                    "Unexpected missing key in golden npz: {}",
                    key
                );
            }
        }
    }
    // 釘: Step 09 は総パラメータ83本中、label_emb を除く 82 本をカバーする
    assert_eq!(
        loaded_count, 82,
        "Expected exactly 82 keys loaded from step09_golden.npz"
    );
    let x_batch: ArrayD<f32> = npz.by_name("x_batch.npy").unwrap(); // (16, 1, 28, 28)
    let t_seq: Array2<f32> = npz.by_name("t_seq.npy").unwrap(); // (5, 16)
    let noise_seq: ArrayD<f32> = npz.by_name("noise_seq.npy").unwrap(); // (5, 16, 1, 28, 28)
    let loss_seq: Array1<f32> = npz.by_name("loss_seq.npy").unwrap(); // (5,)
    let mut optimizer = Adam::new(1e-3);
    optimizer.setup(&unet);
    // =========================================================================
    // 2. 5 iter の学習ループ (順伝播・逆伝播・パラメータ更新)
    // =========================================================================
    for i in 0..5 {
        let t_array = t_seq.row(i).to_owned();
        let noise_i = noise_seq.index_axis(Axis(0), i).to_owned();
        // 順過程 (q_sample): Diffusion実装が t: usize なので、各画像ごとに計算して結合する
        let mut x_t_singles = Vec::new();
        for b in 0..16 {
            let x_0_single = x_batch.index_axis(Axis(0), b).into_owned().into_dyn();
            let noise_single = noise_i.index_axis(Axis(0), b).into_owned().into_dyn();
            let t_val = t_array[b] as usize;

            // 1件ずつ add_noise_with_noise
            let x_t_single = diffusion.add_noise_with_noise(&x_0_single, t_val, &noise_single);
            x_t_singles.push(x_t_single);
        }

        // 分割した配列を Axis(0) で結合してバッチ (16, 1, 28, 28) に戻す
        let views: Vec<_> = x_t_singles.iter().map(|a| a.view()).collect();
        let x_t_data = ndarray::stack(Axis(0), &views).unwrap().into_dyn();
        let x_t = Variable::new(x_t_data);
        let noise = Variable::new(noise_i.into_dyn());
        let start = Instant::now();
        // 順伝播
        let noise_pred = unet.forward(&x_t, &t_array, None);
        // 【罠回避】 損失の 784 倍罠
        let loss = mean_squared_error(&noise_pred, &noise) / 784.0;
        // 逆伝播
        unet.cleargrads();
        loss.backward(false, false);
        optimizer.update();
        // 1 iter 目の時間計測 (重量級 im2col のベンチマーク)
        if i == 0 {
            let elapsed = start.elapsed();
            println!("1 iter time (batch=16, im2col): {:?}", elapsed);
        }
        let loss_val = loss.data().sum();
        println!(
            "iter {}: loss = {:.6} (golden = {:.6})",
            i, loss_val, loss_seq[i]
        );

        assert!(
            approx_eq(loss_val, loss_seq[i], 1e-2),
            "Loss mismatch at iter {}",
            i
        );
    }
    // =========================================================================
    // 3. Final 5 重みの検証 (BN の統計量が意図通り育っているか込み)
    // =========================================================================
    let mut verified_count = 0;
    for (name, param) in unet.named_params() {
        let key = format!("final5/{}.npy", name);

        let golden_res: Result<ndarray::ArrayD<f32>, _> = npz.by_name(&key);
        match golden_res {
            Ok(golden_data) => {
                let max_diff = (&param.data() - &golden_data)
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_diff < 1e-1,
                    "Weight mismatch for '{}' after 5 iters: max_diff = {}",
                    name,
                    max_diff
                );
                param.set_data(golden_data);
                verified_count += 1;
            }
            Err(_) => {
                // ここでも明示的に例外を許可
                assert_eq!(
                    name, "label_emb/W",
                    "Unexpected missing key in final5 npz: {}",
                    key
                );
            }
        }
    }
    // 釘: final5 も同様に 82 本の重み検証と再ロードが行われたか
    assert_eq!(
        verified_count, 82,
        "Expected exactly 82 keys verified and reloaded in final5"
    );
    // =========================================================================
    // 4. サンプリング部分軌道の検証 (eval 分岐のテスト)
    // =========================================================================

    {
        let _guard = vol3::config::test_mode();
        let mut x_sample: ArrayD<f32> = npz.by_name("x_start.npy").unwrap();
        let z_seq: ArrayD<f32> = npz.by_name("z_seq.npy").unwrap();
        // ★ バッチサイズを x_start から動的に取得
        let n_samples = x_sample.shape()[0];
        // 10 -> 1 のデノイズ
        for t_step in (1..=10).rev() {
            let z_data = z_seq.index_axis(Axis(0), 10 - t_step).to_owned();

            // ★ ハードコードの 16 を n_samples に変更
            let t_array = Array1::from_elem(n_samples, t_step as f32);

            let x_var = Variable::new(x_sample.clone());
            let eps_hat_var = unet.forward(&x_var, &t_array, None);
            let eps_hat = eps_hat_var.data();
            let z_opt = if t_step == 1 { None } else { Some(&z_data) };
            x_sample = diffusion.denoise_with_noise(&x_sample, &eps_hat, t_step, z_opt);
        }
        let x_after10: ArrayD<f32> = npz.by_name("x_after10.npy").unwrap();
        let max_diff_x = (&x_sample - &x_after10)
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff_x < 1e-4,
            "Sampling trajectory mismatch: max_diff = {}",
            max_diff_x
        );
    }
}

#[test]
#[ignore] // cargo test --release test_step09_tier3_statistical -- --ignored --nocapture
fn test_step09_tier3_statistical() {
    // 1. ゴールデンからの初期重みロード
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let unet = UNet::new(1, 100, &mut rng);
    let mut npz = NpzReader::new(File::open("dataset/step09_golden.npz").unwrap()).unwrap();

    let mut loaded_count = 0;
    for (name, param) in unet.named_params() {
        let key = format!("{}.npy", name);

        let golden_res: Result<ndarray::ArrayD<f32>, _> = npz.by_name(&key);
        match golden_res {
            Ok(golden_data) => {
                param.set_data(golden_data);
                loaded_count += 1;
            }
            Err(_) => {
                // Step 09 ゴールデンには label_emb は存在しない。それ以外の欠損はパニック
                assert_eq!(
                    name, "label_emb/W",
                    "Unexpected missing key in golden npz: {}",
                    key
                );
            }
        }
    }
    // 釘: Step 09 は総パラメータ83本中、label_emb を除く 82 本をカバーする
    assert_eq!(
        loaded_count, 82,
        "Expected exactly 82 keys loaded from step09_golden.npz"
    );

    // 2. MNIST 先頭 1024 枚をロード (正規化込み)
    let mut bytes = Vec::new();
    File::open("../vol3/dataset/train-images-idx3-ubyte")
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();

    let images_all = vol3::mnist::parse_idx_images(&bytes);
    // [0, 1] へスケーリング
    let x_data = images_all
        .slice(ndarray::s![0..1024, .., .., ..])
        .mapv(|v| v as f32 / 255.0);

    let batch_size = 64;
    let max_epoch = 5;

    let mut optimizer = Adam::new(1e-3);
    optimizer.setup(&unet);
    let diffusion = Diffusion::new(1000, 1e-4, 0.02);

    let mut epoch_losses = Vec::new();

    // 3. 学習ループ (Epoch 単位)
    for epoch in 0..max_epoch {
        let mut indices: Vec<usize> = (0..1024).collect();
        indices.shuffle(&mut rng);

        let mut sum_loss = 0.0;
        let mut iters = 0;

        for i in (0..1024).step_by(batch_size) {
            let end = std::cmp::min(i + batch_size, 1024);
            let b_size = end - i;

            // ランダムシャッフルされたバッチ画像を抽出
            let mut batch_views = Vec::new();
            for &idx in &indices[i..end] {
                batch_views.push(x_data.index_axis(Axis(0), idx));
            }
            let x_batch = ndarray::stack(Axis(0), &batch_views).unwrap();

            let mut t_array = Array1::<f32>::zeros(b_size);
            let mut x_t_singles = Vec::new();
            let mut noise_singles = Vec::new();

            for b in 0..b_size {
                let t_val = rng.random_range(1..=1000) as usize;
                t_array[b] = t_val as f32;

                let x_0_single = x_batch.index_axis(Axis(0), b).into_owned().into_dyn();
                let noise_single = vol3::utils::random_array(
                    x_0_single.dim(),
                    rand_distr::StandardNormal,
                    &mut rng,
                );

                let x_t_single = diffusion.add_noise_with_noise(&x_0_single, t_val, &noise_single);

                x_t_singles.push(x_t_single);
                noise_singles.push(noise_single);
            }

            // バッチへの結合
            let views_x_t: Vec<_> = x_t_singles.iter().map(|a| a.view()).collect();
            let views_noise: Vec<_> = noise_singles.iter().map(|a| a.view()).collect();

            let x_t_data = ndarray::stack(Axis(0), &views_x_t).unwrap();
            let noise_data = ndarray::stack(Axis(0), &views_noise).unwrap();

            let x_t = Variable::new(x_t_data.into_dyn());
            let noise = Variable::new(noise_data.into_dyn());

            // 順伝播・損失計算
            let noise_pred = unet.forward(&x_t, &t_array, None);
            let loss = mean_squared_error(&noise_pred, &noise) / 784.0;

            unet.cleargrads();
            loss.backward(false, false);
            optimizer.update();

            sum_loss += loss.data().sum();
            iters += 1;
        }

        let avg_loss = sum_loss / iters as f32;
        println!("epoch {}: avg_loss = {:.4}", epoch + 1, avg_loss);
        epoch_losses.push(avg_loss);
    }

    // 4. アサーション (Golden カーブに基づく統計的パリティ)
    let mut curve_npz =
        NpzReader::new(File::open("dataset/step09_tier3_curve.npz").unwrap()).unwrap();
    let golden_curve: Array1<f32> = curve_npz.by_name("epoch_losses.npy").unwrap();
    let final_loss = epoch_losses[4];
    let expected_final = golden_curve[4]; // (例: 0.0596)

    // Golden に対しておよそ ±30% の帯域幅を設定してアサーション
    let lower_bound = expected_final * 0.7;
    let upper_bound = expected_final * 1.3;

    assert!(
        final_loss >= lower_bound && final_loss <= upper_bound,
        "Final epoch loss {:.4} is strictly out of the expected band [{:.4}, {:.4}] (Golden was {:.4})",
        final_loss,
        lower_bound,
        upper_bound,
        expected_final
    );

    // Loss が確実に下降していること (epoch 5 < epoch 2)
    assert!(
        epoch_losses[4] < epoch_losses[1],
        "Loss is not decreasing: Epoch 2 loss ({:.4}) vs Epoch 5 loss ({:.4})",
        epoch_losses[1],
        epoch_losses[4]
    );
}
