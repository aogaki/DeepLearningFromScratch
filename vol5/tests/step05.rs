use ndarray::{Array1, array};
use vol5::em::GaussianMixtureModel;
use vol5::utils::{approx_eq, load_txt};

#[test]
fn experiment_step5_old_faithful() {
    println!("--- Step 5: EMアルゴリズム (Old Faithful) ---");

    // 1. データセットの読み込みと形状検証
    let data = load_txt("dataset/old_faithful.txt");
    assert_eq!(data.nrows(), 272);
    assert_eq!(data.ncols(), 2);

    // 平均値の Assert
    let mut mean = Array1::<f32>::zeros(2);
    for i in 0..272 {
        mean = mean + data.row(i);
    }
    mean /= 272.0;

    assert!(approx_eq(mean[0], 3.488, 0.05), "mean[0] was {}", mean[0]);
    assert!(approx_eq(mean[1], 70.90, 0.5), "mean[1] was {}", mean[1]);

    // 2. GMMの初期化 (本に準拠した初期値)
    let k = 2;
    let d = 2;
    let pis = vec![0.5, 0.5];

    // 平均は (0.0, 50.0) と (0.0, 100.0) を初期値に
    let mus = vec![array![0.0, 50.0], array![0.0, 100.0]];

    // 共分散行列は単位行列を初期値に
    // 修正前: array![[1.0, 0.0], [0.0, 1.0]]
    // これだとf32によるアンダーフローが起きて計算がうまくいかない。
    // let covs = vec![
    //     array![[1.0, 0.0], [0.0, 1.0]],
    //     array![[1.0, 0.0], [0.0, 1.0]],
    // ];
    // 修正後: Y軸のスケールに合わせて分散を 100.0 にする
    let covs = vec![
        array![[1.0, 0.0], [0.0, 100.0]],
        array![[1.0, 0.0], [0.0, 100.0]],
    ];

    let mut gmm = GaussianMixtureModel::new(k, d, pis, mus, covs);

    // 3. EMアルゴリズムによるフィッティング (最大30回ループ)
    let ll_history = gmm.fit(&data, 30, 1e-4);

    // 4. 定理のテスト化: 対数尤度は単調増加する (log p(x;θ_new) >= log p(x;θ_old))
    // ただし、f32の精度限界によりプラトー付近で 1e-4 程度の数値ノイズが生じるため、
    // eps を設けて「定理と f32 の距離」を Assert する。
    let eps = 1e-2;
    for i in 1..ll_history.len() {
        let prev = ll_history[i - 1];
        let curr = ll_history[i];
        println!("Iter {}: log likelihood = {:.4}", i, curr); // 報告は呼び出し側の責務
        assert!(
            curr >= prev - eps,
            "Log-likelihood decreased beyond numerical noise! prev={}, curr={}",
            prev,
            curr
        );
    }

    // 5. 学習後のクラスタの妥当性検証 (目視からの脱却、外部パリティテスト)
    // ラベル順は初期値 (y=50 と y=100) で決定的に定まるため、添字直書きでOK

    // Cluster 0: 待ち時間短(50台)・噴出時間短(2弱) のグループ
    assert!(approx_eq(gmm.pis[0], 0.356, 0.05));
    assert!(approx_eq(gmm.mus[0][0], 2.04, 0.1));
    assert!(approx_eq(gmm.mus[0][1], 54.48, 0.5));
    // Cluster 1: 待ち時間長(80台)・噴出時間長(4強) のグループ
    assert!(approx_eq(gmm.pis[1], 0.644, 0.05));
    assert!(approx_eq(gmm.mus[1][0], 4.29, 0.1));
    assert!(approx_eq(gmm.mus[1][1], 79.97, 0.5));
}
