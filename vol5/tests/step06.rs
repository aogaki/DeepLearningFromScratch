use ndarray_npy::NpzReader;
use rand::SeedableRng;
use std::fs::File;
use std::path::Path;
use vol3::functions::mean_squared_error;
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Optimizer, SGD};
use vol3::variable::Variable;
use vol5::utils::approx_eq;

#[test]
fn experiment_step06_parity() {
    println!("--- Step 6: PyTorch パリティ検証 ---");

    // 1. ゴールデンデータ (.npz) の読み込み
    // ndarray-npy は内部で `.npy` 拡張子を透過的に処理するため、そのままのキー名で読み出せます
    let file = File::open("dataset/step06_golden.npz").expect("Failed to open step06_golden.npz");
    let mut npz = NpzReader::new(file).unwrap();

    let x_data: ndarray::ArrayD<f32> = npz.by_name("x").unwrap();
    let y_data: ndarray::ArrayD<f32> = npz.by_name("y").unwrap();
    let loss_history_pt: ndarray::ArrayD<f32> = npz.by_name("loss_history").unwrap();

    let x = Variable::new(x_data);
    let y = Variable::new(y_data);

    // 2. モデルの構築と初期重みのロード
    // rng は初期化用ですが、直後に load_weights で上書きするため実質的に無効化されます
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // [1, 10, 1] 構成、活性化関数は Variable::sigmoid の関数ポインタを直接渡す
    let mlp = MLP::new(&[1, 10, 1], Variable::sigmoid, &mut rng);

    // PyTorchから抽出した初期重み (l0/W, l0/b, l1/W, l1/b) で上書き
    // load_weights が良きに計らって余分なキーを無視しつつロードしてくれます
    let path = Path::new("dataset/step06_golden.npz");
    mlp.load_weights(path)
        .expect("Failed to load golden weights");

    // 3. オプティマイザの準備 (lr = 0.2)
    let mut optimizer = SGD::new(0.2);
    optimizer.setup(&mlp);

    // 4. 学習ループ (10,000イテレーション)
    let iters = 10000;
    let mut my_loss_history = Vec::with_capacity(iters);

    for _ in 0..iters {
        let y_pred = mlp.forward(&x);
        let loss = mean_squared_error(&y_pred, &y);

        // 0次元テンソル(スカラー)を取り出して Vec に記録
        my_loss_history.push(
            loss.data()
                .into_dimensionality::<ndarray::Ix0>()
                .unwrap()
                .into_scalar(),
        );

        mlp.cleargrads();
        loss.backward(false, false); // retain_grad=false, create_graph=false
        optimizer.update();
    }

    // 5. パリティ検証

    // 【第1段】Iter 0: 純粋な forward + MSE のパリティ検証
    // 初期重みが完全一致しているため、1e-6 級でタイトに一致するはず
    let pt_loss_0 = loss_history_pt[[0]];
    let my_loss_0 = my_loss_history[0];
    println!(
        "Iter 0: My Loss = {:.7}, PyTorch Loss = {:.7}",
        my_loss_0, pt_loss_0
    );
    assert!(
        approx_eq(my_loss_0, pt_loss_0, 1e-6),
        "Iter 0 Parity Failed! drift: {}",
        (my_loss_0 - pt_loss_0).abs()
    );

    // 【第2段】Iter 9999: f32 の丸め誤差蓄積 (Drift) の検証
    // 10000反復の backward と update を繰り返した結果、PyTorch 側と
    // 自作フレームワーク間でどの程度のズレ(Drift)が生じるかを観測します。
    let final_iter = iters - 1;
    let pt_final_loss = loss_history_pt[[final_iter]];
    let my_final_loss = my_loss_history[final_iter];
    let drift = (my_final_loss - pt_final_loss).abs();

    println!(
        "Iter 9999: My Loss = {:.7}, PyTorch Loss = {:.7}",
        my_final_loss, pt_final_loss
    );
    println!("Total Drift over 10000 iterations: {:.2e}", drift);

    // この Assert は、実際に上の println! で出力される drift 値を観測してから
    // 台帳の初エントリとして正確な限界値（tol）を定める想定です。
    // 今回は安全マージンとしてひとまず 1e-3 を置いています。
    assert!(drift < 1e-3, "Final Loss Parity Failed! drift: {}", drift);

    // 6. 重み空間のパリティ検証 (Max |Δ|)
    // 10000イテレーション後の全ての重みとバイアスについて、PyTorchとの最大誤差を計算する
    let mut max_weight_delta = 0.0f32;

    for (name, param) in mlp.named_params() {
        // ゴールデンデータのキー名を作成 (例: "l0/W" -> "final/l0/W")
        let golden_name = format!("final/{}", name);
        let pt_weight: ndarray::ArrayD<f32> = npz
            .by_name(&golden_name)
            .unwrap_or_else(|_| panic!("Failed to load {}", golden_name));

        let my_weight = param.data();

        // 要素ごとの差分の絶対値を計算し、最大値を求める (L_infinity norm)
        let diff = &my_weight - &pt_weight;
        let local_max = diff.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));

        if local_max > max_weight_delta {
            max_weight_delta = local_max;
        }
    }

    println!("Max Weight Delta (max |Δ|): {:.2e}", max_weight_delta);

    assert!(
        max_weight_delta < 5e-3,
        "Weight Parity Failed! max |Δ|: {}",
        max_weight_delta
    );
}
