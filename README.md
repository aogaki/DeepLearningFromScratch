# ゼロから作る Deep Learning — Rust 移植(学習用)

斎藤康毅『ゼロから作る Deep Learning』(オライリー・ジャパン)全6巻を、Python + NumPy から
**Rust** へ自分の手で移植しながら学ぶリポジトリ。成果物よりも「自分で書いて理解すること」が目的で、
Claude Code をガイド兼レビュアーとして一歩ずつ進めている。

## 方針

- **Rust のイディオムを優先**して写経・移植する(`ndarray` を NumPy の代わりに使用)。
- 浮動小数は **`f32` で統一**。将来 [wgpu](https://wgpu.rs/) の compute シェーダに載せることを見据えている(WGSL に f64 が無いため)。
- 各巻を独立した Cargo クレート(`vol1`〜`vol6`)として実装し、共有クレートは作らない(巻ごとに再実装して理解を深める)。
- 常に `cargo test` を green に保ち、浮動小数の比較は誤差付きで行う。
- 本は各節のスクリプトをコマンドラインで実行していくスタイルだが、この移植では**実験も含めて基本 `#[test]` に書く**(学習ループなど遅いものは `#[ignore]` を付け、名前指定で明示的に実行する)。ファイル出力を伴う可視化などに限り `examples/` を使う。

## 進捗

| 巻   | 状態      | 内容                                                                                   |
| ---- | --------- | -------------------------------------------------------------------------------------- |
| vol1 | 8.1 完了 | 2章 パーセプトロン / 3章 順伝播・softmax・MNIST 推論(93.5%)/ 4章 損失関数・数値微分・ミニバッチ学習 / 5章 誤差逆伝播(レイヤ実装・勾配確認・高速学習: 損失 2.3→0.26 を約30秒/1000回)/ 6章 学習テクニック(Optimizer 4種・He 初期化・BatchNorm・Weight decay・Dropout・ハイパーパラメータ探索を rayon 並列化)/ 7章 CNN(im2col・Conv/Pooling レイヤ・SimpleConvNet で MNIST テスト精度 **98.75%**・フィルタ可視化)/ 8.1 ディープ CNN(Layer トレイトで全層を `Vec<Box<dyn Layer>>` に、conv 6 層の DeepConvNet で MNIST テスト精度 **99.32%**、本の ~99.4% と 1σ 以内)。次は wgpu 導入(8.3.2 が読み物の伴走)。 |
| vol2 | 未着手    | —                                                                                      |
| vol3 | 未着手    | —                                                                                      |
| vol4 | 未着手    | —                                                                                      |
| vol5 | 未着手    | —                                                                                      |
| vol6 | 未着手    | —                                                                                      |

## 構成

ルートは Cargo ワークスペース(`resolver = "3"`)。`Cargo.lock` と `target/` を全巻で共有する。

```
.
├── vol1/            # 1巻目のクレート
│   └── src/
│       ├── lib.rs
│       ├── perceptron.rs   # 2章 パーセプトロン
│       ├── network.rs      # 3章 順伝播・活性化関数・softmax
│       ├── mnist.rs        # 3.6 MNIST データ読み込み・推論
│       ├── loss.rs         # 4.2 損失関数
│       ├── gradient.rs     # 4.3-4.4 数値微分・勾配降下法
│       ├── two_layer_net.rs # 4.5 2層ネットのクラス・ミニバッチ学習
│       ├── layers.rs       # 5.4-5.6 レイヤ(Relu/Sigmoid/Affine/SoftmaxWithLoss)+ 6.3 BatchNorm / 6.4.3 Dropout / 8.1 Layer トレイト・Flatten
│       ├── optimizer.rs    # 6.1 Optimizer トレイト(SGD/Momentum/AdaGrad/Adam、7.5 で ArrayD 対応)
│       ├── two_layer_net_backprop.rs # 5.7 逆伝播対応の2層ネット + 6章統合(初期化・正則化・実験群)
│       ├── conv.rs         # 7.4 im2col/col2im・Convolution/Pooling レイヤ
│       ├── simple_conv_net.rs # 7.5 SimpleConvNet(CNN の学習)
│       ├── deep_conv_net.rs # 8.1 DeepConvNet(Layer トレイトで層をリスト化、99.32%)
│       └── ../examples/
│           └── visualize_filters.rs # 7.6.1 フィルタ可視化(PGM 出力)
├── books/           # 本の PDF(gitignore 済み)
└── Cargo.toml       # ワークスペース定義
```

各巻の `dataset/` にデータセットや変換後の重みを置く(gitignore 済み・再取得可能)。

## 本との対応

本のどの章がどのファイルに対応するかの鳥瞰。細かい対応(節・見出し)は各関数の
doc コメント `/// 本 X.Y「見出し」` に書いてあり、`cargo doc --open` で閲覧できる。
「4.5 のコードはどこ?」となったら `rg "本 4.5"` で該当箇所へ飛べる。

各巻は独立クレート(`vol1`〜`vol5`)。巻を進めたらこの節に対応表を追記していく。

### 第1巻 ― Python で学ぶディープラーニングの理論と実装(`vol1`)

| 本の章                         | ファイル                          |
| ------------------------------ | --------------------------------- |
| 2章 パーセプトロン             | `vol1/src/perceptron.rs`          |
| 3章 ニューラルネットワーク     | `vol1/src/network.rs`             |
| 3.6 手書き数字認識(MNIST)    | `vol1/src/mnist.rs`, `network.rs` |
| 4.2 損失関数                   | `vol1/src/loss.rs`                |
| 4.3-4.4 数値微分・勾配         | `vol1/src/gradient.rs`            |
| 4.5 2層ネットの学習            | `vol1/src/two_layer_net.rs`       |
| 5.4-5.6 レイヤ実装             | `vol1/src/layers.rs`              |
| 5.7 誤差逆伝播法の実装         | `vol1/src/two_layer_net_backprop.rs` |
| 6.1 パラメータの更新(SGD/Momentum/AdaGrad/Adam) | `vol1/src/optimizer.rs`        |
| 6.2 重みの初期値(He/Xavier)  | `vol1/src/two_layer_net_backprop.rs`(`make_std` 注入) |
| 6.3 Batch Normalization        | `vol1/src/layers.rs`(`BatchNormLayer`) |
| 6.4 正則化(Weight decay・Dropout) | `vol1/src/layers.rs`, `two_layer_net_backprop.rs` |
| 6.5 ハイパーパラメータの検証   | `vol1/src/two_layer_net_backprop.rs`(`test_hyperparameter_tuning`, rayon 並列) |
| 7.4 Convolution/Pooling レイヤ(im2col) | `vol1/src/conv.rs`                |
| 7.5 CNN の実装(SimpleConvNet) | `vol1/src/simple_conv_net.rs`(MNIST 学習は `train_mnist_backprop_cnn`) |
| 7.6.1 1層目の重みの可視化      | `vol1/examples/visualize_filters.rs` |
| 8.1 ネットワークをより深く(DeepConvNet) | `vol1/src/deep_conv_net.rs`(Layer トレイト・Flatten は `layers.rs`、MNIST 学習は `train_mnist_deep`) |

### 第2巻 ― 自然言語処理編(`vol2`)

未着手。

### 第3巻 ― フレームワーク編(`vol3`)

未着手。

### 第4巻 ― 強化学習編(`vol4`)

未着手。

### 第5巻 ― 生成モデル編(`vol5`)

未着手。

### 第6巻 ― LLM編(`vol6`)

未着手。

## 実行環境

本 README 中の実行時間(「約30秒」「約80分」「0.4 s/iter」など)はすべて以下のマシンでの実測値。
環境が違えば相応にスケールする。

- Apple **M4 Pro**(14 コア)/ RAM 48 GB / macOS
- 学習ループは単一スレッド(rayon 並列はハイパーパラメータ探索のみ)
- 学習系はすべて `--release` ビルドでの計測(デバッグビルドは約100倍遅い)

## コマンド

```sh
cargo test            # 全巻テスト
cargo test -p vol1    # 単一巻テスト
cargo check           # 型チェックのみ(速い)
cargo test -- --nocapture   # println! を表示
cargo doc --open      # 本との対応を含む API ドキュメントを生成・閲覧
```

数値微分による学習など遅いテストは `#[ignore]` を付けてあり、通常の `cargo test` では走らない。
明示的に回すときは `cargo test -- --ignored --nocapture`。

6章の実験も `#[ignore]` 付きのテストとして残してある(名前で個別に実行できる):

```sh
cargo test train_mnist_backprop -- --ignored --nocapture      # Optimizer 3種の学習曲線比較
cargo test test_overfitting -- --ignored --nocapture          # 過学習の再現と正則化の効果
cargo test test_hyperparameter_tuning -- --ignored --nocapture # ランダムサーチ(rayon 並列)
```

7章の CNN は計算が重いので **`--release` 必須**(デバッグビルドとの差は約100倍):

```sh
cargo test --release train_mnist_backprop_cnn -- --ignored --nocapture  # CNN の MNIST 学習(テスト精度 98.75%)
cd vol1 && cargo run -p vol1 --example visualize_filters --release      # 7.6.1 フィルタ可視化(output/filters/ に PGM)
```

8章の DeepConvNet は 20 エポックで約80分かかる(0.4 s/iter、release 実測):

```sh
cargo test --release train_mnist_deep -- --ignored --nocapture  # 8.1 DeepConvNet の MNIST 学習(テスト精度 99.32%)
```
