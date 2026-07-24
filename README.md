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
| vol1 | **完了** | 2章 パーセプトロン / 3章 順伝播・softmax・MNIST 推論(93.5%)/ 4章 損失関数・数値微分・ミニバッチ学習 / 5章 誤差逆伝播(レイヤ実装・勾配確認・高速学習: 損失 2.3→0.26 を約30秒/1000回)/ 6章 学習テクニック(Optimizer 4種・He 初期化・BatchNorm・Weight decay・Dropout・ハイパーパラメータ探索を rayon 並列化)/ 7章 CNN(im2col・Conv/Pooling レイヤ・SimpleConvNet で MNIST テスト精度 **98.75%**・フィルタ可視化)/ 8.1 ディープ CNN(Layer トレイトで全層を `Vec<Box<dyn Layer>>` に、conv 6 層の DeepConvNet で MNIST テスト精度 **99.32%**、本の ~99.4% と 1σ 以内)/ **wgpu GPU 化(8.3 の先の独自拡張)**: DeepConvNet の forward・backward・Optimizer を WGSL シェーダ 17 本で GPU 常駐化。カーネル最適化 3 段(vec4 レジスタタイル → workgroup リダクション → タイル×リダクション)で **1 iter 0.41 s → 21.8 ms(×18.8)**、20 エポック 82 分 → **4.5 分**、Adam で **テスト精度 99.41%(peak)** — CPU 版と同水準を達成。記録: [`vol1/docs/wgpu-journey.md`](vol1/docs/wgpu-journey.md) |
| vol2 | 未着手    | (個人的興味の巻として後回し)                                                        |
| vol3 | **進行中** | **第2ステージ(ステップ1〜24)完了**: `Variable`(`Rc<RefCell>` の薄いハンドル — Python の共有参照を Rust で明示)/ Function を `Forward`・`Function`・`Creator` の 3 trait に分割し、`call(self)` が関数を `Node<F>` としてグラフへ移す(「入力未設定」状態が型で排除される設計)/ 数値微分(f32 では eps ≈ ∛ε ≈ 5e-3 と導出)/ 自動逆伝播(ループ+世代管理のトポロジカル順+勾配累積)/ `no_grad`(thread_local + RAII ガード)/ 演算子オーバーロード(`std::ops` 4通り+スカラー混合をマクロ量産、`3.0 * &x` が `__rmul__` なしで書ける)/ モジュール分割。Goldstein-Price の勾配が本と厳密一致 |
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
│       ├── gpu.rs          # wgpu: デバイス初期化・GpuTensor/GpuImage・カーネル部品(*.wgsl シェーダ 17 本と対)
│       ├── gpu/
│       │   ├── layers.rs   # GPU 版レイヤ(Conv/ReLU/Pooling/Affine、SGD/Adam 状態持ち)
│       │   └── deep_conv_net.rs # GPU 版 DeepConvNet と学習ループ(20 epoch 4.5 分・99.4%)
│       └── ../examples/
│           └── visualize_filters.rs # 7.6.1 フィルタ可視化(PGM 出力)
│   └── docs/
│       └── wgpu-journey.md # GPU(wgpu)導入のステップバイステップ記録(実測値・ハマりどころ付き)
├── vol3/            # 3巻目(フレームワーク編 / DeZero)のクレート
│   ├── src/
│   │   ├── lib.rs        # ファサード再エクスポート(dezero/__init__.py 相当)
│   │   ├── variable.rs   # Variable(Rc<RefCell> ハンドル・backward・演算子)
│   │   ├── function.rs   # Forward/Function/Creator トレイトと Node
│   │   ├── functions.rs  # Square/Exp/Add/Mul/Neg/Sub/Div/Pow
│   │   ├── config.rs     # enable_backprop(thread_local)と no_grad ガード
│   │   ├── macros.rs     # 演算子 impl 量産マクロ($crate 絶対パス)
│   │   └── utils.rs      # 数値微分・近似比較
│   └── tests/            # ステップ番号付き統合テスト(本の各ステップの実例集)
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
| 8.3 高速化(GPU)— 本を超えて実装 | `vol1/src/gpu.rs`, `vol1/src/gpu/`, `vol1/src/*.wgsl`(記録: `vol1/docs/wgpu-journey.md`) |

### 第2巻 ― 自然言語処理編(`vol2`)

未着手。

### 第3巻 ― フレームワーク編(`vol3`)

DeZero(小さな自動微分フレームワーク)を 60 ステップで作る巻。doc コメントは
`/// 本 ステップX「見出し」` の形式(`rg "ステップ7"` などで該当箇所へ)。

| 本のステップ                   | ファイル                          |
| ------------------------------ | --------------------------------- |
| 第1ステージ(1〜10)Variable・Function・数値微分・自動逆伝播・勾配チェック | `vol3/src/variable.rs`, `function.rs`, `utils.rs`(実例: `tests/step1_to_22.rs`) |
| 第2ステージ(11〜24)可変長引数・勾配累積・世代管理・no_grad・演算子オーバーロード・パッケージ化 | `vol3/src/functions.rs`, `config.rs`, `macros.rs`(実例: `tests/step1_to_22.rs`, `tests/step24.rs`) |

Python 版との主な設計差(Rust の所有権に合わせた意図的なもの):

- `Variable` は `Rc<RefCell<VariableInner>>` の薄いハンドル(Python の「全てが共有参照」の明示化)
- 関数の能力を trait で3分割: `Forward`(順伝播のみ・数値微分用、クロージャにも開放)/
  `Function: Forward`(+ 純関数の backward)/ `Creator`(グラフ遡行の最小界面)
- `Function::call(self)` は self を消費し、入力とともに `Node<F>` としてグラフに移る
  (Python の `self.input = input` に相当する状態を、型上「未設定になり得ない」形で持つ)
- f32 統一のため数値微分の刻みは eps=5e-3(∛ε_f32。本の 1e-4 は float64 用)

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
cargo test -p vol1    # 単一巻テスト(vol3 なら -p vol3)
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

同じ DeepConvNet の **GPU(wgpu/Metal)版**は 20 エポック約 4.5 分(21.8 ms/iter、CPU 比 ×18.8):

```sh
cargo test --release test_train_mnist_deep_gpu_adam -- --ignored --nocapture  # GPU 版(Adam、テスト精度 99.4% peak)
cargo test --release test_train_mnist_deep_gpu -- --ignored --nocapture       # GPU 版(素の SGD、98.97%)
```
