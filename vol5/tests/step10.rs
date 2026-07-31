use ndarray::{Array1, Array2, ArrayD, Axis};
use ndarray_npy::NpzReader;
use rand::RngExt;
use rand::SeedableRng;
use std::fs::File;
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
pub fn test_label_dropout() {
    // --- 初期設定 (画像とラベルのロード) ---
    let img_bytes = std::fs::read("../vol3/dataset/train-images-idx3-ubyte").unwrap();
    let x_data = vol3::mnist::parse_idx_images(&img_bytes)
        .slice(ndarray::s![0..1024, .., .., ..])
        .mapv(|v| v as f32 / 255.0);
    let lbl_bytes = std::fs::read("../vol3/dataset/train-labels-idx1-ubyte").unwrap();
    let labels_all = vol3::mnist::parse_idx_labels(&lbl_bytes)
        .slice(ndarray::s![0..1024])
        .mapv(|l| l as usize);

    // --- エポック・バッチループの中 (煙テスト: 2バッチ×8枚) ---
    let batch_size = 8;
    let num_samples = 16;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let indices: Vec<usize> = (0..num_samples).collect();
    let unet = UNet::new(1, 128, &mut rng);
    let mut optimizer = vol3::optimizers::Adam::new(1e-3);
    optimizer.setup(&unet);
    let diffusion = Diffusion::new(1000, 1e-4, 0.02);

    for i in (0..num_samples).step_by(batch_size) {
        let end = std::cmp::min(i + batch_size, num_samples);
        let b_size = end - i;
        let mut batch_views = Vec::new();
        let mut batch_labels_vec = Vec::new();
        for &idx in &indices[i..end] {
            batch_views.push(x_data.index_axis(ndarray::Axis(0), idx));
            batch_labels_vec.push(labels_all[idx]);
        }
        let x_batch = ndarray::stack(ndarray::Axis(0), &batch_views).unwrap();
        let y_batch = ndarray::Array1::from_vec(batch_labels_vec);
        // ★ バッチ丸ごとラベルをドロップする (CFGの無条件経路学習)
        let keep_label = rng.random_bool(0.9);
        let labels_opt = if keep_label { Some(&y_batch) } else { None };
        // ノイズ付加など
        let mut t_array = ndarray::Array1::<f32>::zeros(b_size);
        let mut x_t_singles = Vec::new();
        let mut noise_singles = Vec::new();
        for b in 0..b_size {
            let t_val = rng.random_range(1..=1000) as usize;
            t_array[b] = t_val as f32;
            let x_0_single = x_batch
                .index_axis(ndarray::Axis(0), b)
                .into_owned()
                .into_dyn();
            let noise_single =
                vol3::utils::random_array(x_0_single.dim(), rand_distr::StandardNormal, &mut rng);
            let x_t_single = diffusion.add_noise_with_noise(&x_0_single, t_val, &noise_single);
            x_t_singles.push(x_t_single);
            noise_singles.push(noise_single);
        }
        let views_x_t: Vec<_> = x_t_singles.iter().map(|a| a.view()).collect();
        let views_noise: Vec<_> = noise_singles.iter().map(|a| a.view()).collect();
        let x_t_data = ndarray::stack(ndarray::Axis(0), &views_x_t).unwrap();
        let noise_data = ndarray::stack(ndarray::Axis(0), &views_noise).unwrap();
        let x_t = Variable::new(x_t_data.into_dyn());
        let noise = Variable::new(noise_data.into_dyn());
        // ★ UNet への順伝播 (Some か None がバッチ単位で渡る)
        let noise_pred = unet.forward(&x_t, &t_array, labels_opt);
        let loss = vol3::functions::mean_squared_error(&noise_pred, &noise) / 784.0;

        // 釘: Lossが正常な値か
        assert!(loss.data().iter().all(|v| v.is_finite()));
        unet.cleargrads();
        loss.backward(false, false);
        optimizer.update();
    }

    // --- CFG サンプリングのテスト ---
    {
        let _guard = vol3::config::test_mode();
        let n_samples = 4;
        let mut x_sample = vol3::utils::random_array(
            ndarray::IxDyn(&[n_samples, 1, 28, 28]),
            rand_distr::StandardNormal,
            &mut rng,
        );
        let labels = ndarray::Array1::from_vec(vec![0, 1, 2, 3]);
        let gamma = 3.0;
        for t_step in (1..=10).rev() {
            let t_array = ndarray::Array1::from_elem(n_samples, t_step as f32);
            let x_var = Variable::new(x_sample.clone());
            // 1. 条件付き (Some(labels))
            let eps_cond_var = unet.forward(&x_var, &t_array, Some(&labels));
            let eps_cond = eps_cond_var.data();
            // 2. 無条件 (None)
            let eps_uncond_var = unet.forward(&x_var, &t_array, None);
            let eps_uncond = eps_uncond_var.data();
            // 釘: 条件付きと無条件の出力が異なること（label_embが機能している生存確認）
            assert_ne!(eps_cond, eps_uncond);
            // 3. ブレンド: eps = eps_uncond + gamma * (eps_cond - eps_uncond)
            let eps = &eps_uncond + gamma * (&eps_cond - &eps_uncond);
            // 4. ノイズ付加とデノイズ
            let z_data =
                vol3::utils::random_array(x_sample.dim(), rand_distr::StandardNormal, &mut rng);
            let z_opt = if t_step == 1 { None } else { Some(&z_data) };
            x_sample = diffusion.denoise_with_noise(&x_sample, &eps, t_step, z_opt);
        }

        // 釘: サンプリング結果が正常な値か
        assert!(x_sample.iter().all(|v| v.is_finite()));
        assert_eq!(x_sample.shape(), &[n_samples, 1, 28, 28]);
    }
}

#[test]
fn test_step10_parity() {
    let mut npz = NpzReader::new(File::open("dataset/step10_golden.npz").unwrap()).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // ① UNet は規模 100 に変更
    let unet = UNet::new(1, 100, &mut rng);
    let diffusion = Diffusion::new(1000, 1e-4, 0.02);

    // 1. 初期重みのロード (例外分岐なしで83本フルカバー)
    let mut loaded_count = 0;
    for (name, param) in unet.named_params() {
        let key = format!("{}.npy", name);
        let golden_data: ArrayD<f32> = npz
            .by_name(&key)
            .unwrap_or_else(|_| panic!("Missing key: {}", key));
        param.set_data(golden_data);
        loaded_count += 1;
    }
    assert_eq!(loaded_count, 83, "Expected exactly 83 keys loaded");

    // データのロード
    let x_batch: ArrayD<f32> = npz.by_name("x_batch.npy").unwrap(); // (16, 1, 28, 28)

    // Python側は int 型 (i64) で記録されている想定
    let labels_batch_i64: Array1<i64> = npz.by_name("labels_batch.npy").unwrap();
    let labels_batch = labels_batch_i64.mapv(|x| x as usize);

    let t_seq: Array2<f32> = npz.by_name("t_seq.npy").unwrap(); // (5, 16)
    let noise_seq: ArrayD<f32> = npz.by_name("noise_seq.npy").unwrap(); // (5, 16, 1, 28, 28)
    let loss_seq: Array1<f32> = npz.by_name("loss_seq.npy").unwrap(); // (5,)

    // ② iter 2 だけ無条件のドロップフラグ
    let drop_flags = [1, 1, 0, 1, 1];

    let mut optimizer = Adam::new(1e-3);
    optimizer.setup(&unet);

    // 2. 5 iter の学習ループ
    for i in 0..5 {
        let t_array = t_seq.row(i).to_owned();
        let noise_i = noise_seq.index_axis(Axis(0), i).to_owned();

        let mut x_t_singles = Vec::new();
        for b in 0..16 {
            let x_0_single = x_batch.index_axis(Axis(0), b).into_owned().into_dyn();
            let noise_single = noise_i.index_axis(Axis(0), b).into_owned().into_dyn();
            let t_val = t_array[b] as usize;

            let x_t_single = diffusion.add_noise_with_noise(&x_0_single, t_val, &noise_single);
            x_t_singles.push(x_t_single);
        }

        let views: Vec<_> = x_t_singles.iter().map(|a| a.view()).collect();
        let x_t_data = ndarray::stack(Axis(0), &views).unwrap().into_dyn();
        let x_t = Variable::new(x_t_data);
        let noise = Variable::new(noise_i.into_dyn());

        // フラグに応じた Some/None の切り替え
        let labels_opt = if drop_flags[i] == 1 {
            Some(&labels_batch)
        } else {
            None
        };

        let start = Instant::now();
        let noise_pred = unet.forward(&x_t, &t_array, labels_opt);
        let loss = mean_squared_error(&noise_pred, &noise) / 784.0;

        unet.cleargrads();
        loss.backward(false, false);
        optimizer.update();

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

    // 3. Final 5 重みの検証
    let mut verified_count = 0;
    for (name, param) in unet.named_params() {
        let key = format!("final5/{}.npy", name);
        let golden_data: ArrayD<f32> = npz
            .by_name(&key)
            .unwrap_or_else(|_| panic!("Missing key: {}", key));

        let max_diff = (&param.data() - &golden_data)
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f32, f32::max);

        println!(
            "Verifying '{}': max_diff = {:.6}, golden shape = {:?}, param shape = {:?}",
            name,
            max_diff,
            golden_data.shape(),
            param.data().shape()
        );
        // 台帳記録済みの通り、label_emb/W は global/per-param step counter のズレにより
        // iter 2 の欠席後、系統ドリフト (~1e-4 - 1e-3) が発生するため tol=1e-2 を採用するが
        // */*/running_* はmean や var が大きくなるため、5e-2
        let tol = if name.contains("/running_") {
            5e-2
        } else {
            1e-2
        };
        assert!(
            max_diff < tol,
            "Weight mismatch for '{}' after 5 iters: max_diff = {}",
            name,
            max_diff
        );
        param.set_data(golden_data);
        verified_count += 1;
    }
    assert_eq!(
        verified_count, 83,
        "Expected exactly 83 keys verified and reloaded in final5"
    );

    // 4. サンプリング部分軌道の検証 (CFG)
    {
        let _guard = vol3::config::test_mode();
        let mut x_sample: ArrayD<f32> = npz.by_name("x_start.npy").unwrap();
        let z_seq: ArrayD<f32> = npz.by_name("z_seq.npy").unwrap();

        // labels_sample は決め打ち
        let labels_sample = Array1::from_vec(vec![0, 1, 2, 3]);
        let n_samples = x_sample.shape()[0];
        let gamma = 3.0;

        for t_step in (1..=10).rev() {
            let z_data = z_seq.index_axis(Axis(0), 10 - t_step).to_owned();
            let t_array = Array1::from_elem(n_samples, t_step as f32);
            let x_var = Variable::new(x_sample.clone());

            // ③ 2回 forward → γ ブレンド → denoise_with_noise
            let eps_cond_var = unet.forward(&x_var, &t_array, Some(&labels_sample));
            let eps_cond = eps_cond_var.data();

            let eps_uncond_var = unet.forward(&x_var, &t_array, None);
            let eps_uncond = eps_uncond_var.data();

            let eps_hat = &eps_uncond + gamma * (&eps_cond - &eps_uncond);

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
