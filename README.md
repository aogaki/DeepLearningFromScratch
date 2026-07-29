# ゼロから作る Deep Learning — Rust 移植(学習用)

斎藤康毅『ゼロから作る Deep Learning』(オライリー・ジャパン)全6巻を、Python + NumPy から
**Rust** へ自分の手で移植しながら学ぶリポジトリ。成果物よりも「自分で書いて理解すること」が目的で、
Claude Code をガイド兼レビュアーとして一歩ずつ進めている。

長期目標は学習にとどまらない: vol3 で作る自動微分フレームワークを、自身の研究
(核物理 — ΔE-E 粒子識別の事後確率推定)の**本番ツールへ育てる**(下記ロードマップ)。

## 方針

- **Rust のイディオムを優先**して写経・移植する(`ndarray` を NumPy の代わりに使用)。
- 浮動小数は **`f32` で統一**。将来 [wgpu](https://wgpu.rs/) の compute シェーダに載せることを見据えている(WGSL に f64 が無いため)。
- 各巻を独立した Cargo クレート(`vol1`〜`vol6`)として実装し、共有クレートは作らない(巻ごとに再実装して理解を深める)。ただし vol3 のフレームワークは計画済みの例外で、**vol4 後半(7章〜)が path 依存で使用中**。将来は独自名のクレートに昇格させる予定(ロードマップ参照)。
- 常に `cargo test` を green に保ち、浮動小数の比較は誤差付きで行う。
- 本は各節のスクリプトをコマンドラインで実行していくスタイルだが、この移植では**実験も含めて基本 `#[test]` に書く**(学習ループなど遅いものは `#[ignore]` を付け、名前指定で明示的に実行する)。ファイル出力を伴う可視化などに限り `examples/` を使う。

## 進捗

| 巻   | 状態      | 内容                                                                                   |
| ---- | --------- | -------------------------------------------------------------------------------------- |
| vol1 | **完了** | 2章 パーセプトロン / 3章 順伝播・softmax・MNIST 推論(93.5%)/ 4章 損失関数・数値微分・ミニバッチ学習 / 5章 誤差逆伝播(レイヤ実装・勾配確認・高速学習: 損失 2.3→0.26 を約30秒/1000回)/ 6章 学習テクニック(Optimizer 4種・He 初期化・BatchNorm・Weight decay・Dropout・ハイパーパラメータ探索を rayon 並列化)/ 7章 CNN(im2col・Conv/Pooling レイヤ・SimpleConvNet で MNIST テスト精度 **98.75%**・フィルタ可視化)/ 8.1 ディープ CNN(Layer トレイトで全層を `Vec<Box<dyn Layer>>` に、conv 6 層の DeepConvNet で MNIST テスト精度 **99.32%**、本の ~99.4% と 1σ 以内)/ **wgpu GPU 化(8.3 の先の独自拡張)**: DeepConvNet の forward・backward・Optimizer を WGSL シェーダ 17 本で GPU 常駐化。カーネル最適化 3 段(vec4 レジスタタイル → workgroup リダクション → タイル×リダクション)で **1 iter 0.41 s → 21.8 ms(×18.8)**、20 エポック 82 分 → **4.5 分**、Adam で **テスト精度 99.41%(peak)** — CPU 版と同水準を達成。記録: [`vol1/docs/wgpu-journey.md`](vol1/docs/wgpu-journey.md) |
| vol2 | 未着手    | (個人的興味の巻として後回し)                                                        |
| vol3 | **本編完走** | **第3ステージ(〜ステップ36)完了**: `Variable`(`Rc<RefCell>` の薄いハンドル — Python の共有参照を Rust で明示)/ Function を `Forward`・`Function`・`Creator` の 3 trait に分割し、`call(self)` が関数を `Node<F>` としてグラフへ移す(「入力未設定」状態が型で排除される設計)/ 数値微分(f32 では eps ≈ ∛ε ≈ 5e-3 と導出)/ 自動逆伝播(ループ+世代管理のトポロジカル順+勾配累積)/ `no_grad`(thread_local + RAII ガード)/ 演算子オーバーロード(`std::ops` 4通り+スカラー混合をマクロ量産、`3.0 * &x` が `__rmul__` なしで書ける)/ Graphviz 可視化 / **高階微分(double backprop)** — grad を Variable にして backward 自体がグラフを作る(create_graph+no_grad ガード、Rc 循環リークは Weak テストで実証)。Goldstein-Price の勾配が本と厳密一致。**第4ステージ(37〜51)完了**: テンソル対応(reshape/transpose・broadcast_to/sum_to の双対ペア、勾配形状の不変条件 grad.shape == data.shape を debug_assert で確立)・行列積・線形回帰(loss がノイズ分散 1/12 の理論下限に到達)・**Layer trait**(params は名前なしホットパス/named_params は "l1/W" 階層名に分離)で Linear・TwoLayerNet・MLP — 同一の決定的軌道を step43〜46 の5実装がテストで相互に担保・Optimizer(SGD/MomentumSGD、`params()` の Rc ハンドルで所有権問題が消滅)・softmax 交差エントロピー(Gather/GatherGrad 双対+二階微分テスト、閉形式 (p−t)/N で検証)・spiral 分類 97%・**Dataset/DataLoader**(`IntoIterator for &mut` で1エポック=1 for ループ)・**MNIST**(IDX パーサ再実装・u8 保持+transform・ReLU)を MLP 5エポック **17秒** で test **97.7%** / **第5ステージ(52〜60)完了**: save/load(`ndarray-npy` の npz・2パス atomic load・**Rust→Python np.load のパリティ橋**)・Dropout(合成スタイル+`test_mode` RAII ガード)・CNN(im2col/col2im の双対・conv2d_simple/pooling_simple — gradient check では掴めない転置ずれを値テストで捕捉した教訓付き)・**VGG16(16層 forward が Python DeZero と 1e-4 で一致 — パリティ検証の初実戦、release 1.8s)**・RNN/LSTM(truncated BPTT を所有権 DAG の一点切断で実装、「ピンはレイヤの h と累積 loss の2本」の教訓、Adam)・SeqDataLoader+Dataset 関連型 Target。LSTM の cos 自己回帰 MSE **0.40**(SimpleRNN 1.5 超)を assert、100 epoch release 約1秒。52(GPU/CuPy)は設計読書のみ(GPU 化はロードマップ後段の wgpu v2)。**本編60ステップ完走(2026-07-25)** |
| vol4 | **本編完走** | 1章 バンディット(ε-greedy・非定常環境、ペアドシード比較で本の結論を assert)/ 2〜3章 読章 / 4章 動的計画法(GridWorld・policy/value iteration — 図4-13 パリティ・γ べき手計算・アルゴリズム間クロスチェックの三段検証)/ 5章 モンテカルロ法(MC↔DP クロスチェック 1万エピソード、重点サンプリングは b を π に近づけて分散 1/4 を実測)/ 6章 TD 法(SARSA・方策オフ SARSA・Q 学習 — 重点サンプリングの ρ=0 破壊的更新を α=0.1 で手当てする教訓込み)— ここまで素の Rust。**7章から vol3 フレームワークを path 依存で使用**(計画済みの最終判断を採択)/ 7章 ニューラルネットワークと Q 学習(GridWorld の one-hot 化・`gather` で q[:,a] 抽出・1000 エピソード学習後の greedy ロールアウトがゴール到達を assert)/ 8章 DQN: **CartPole-v0 環境を自作**し、本家 gymnasium 1.3.0 を実走した golden 軌道と 1e-4 で一致を assert(環境の転記ミスは自己一致テストでは検出不能のため)。ReplayBuffer+ターゲットネットワーク+Adam(QNet 4→128→128→2)で **10 シード全てが訓練中に満点 200 到達**、学習後の greedy 評価 ≥150 が 7/10(過半数成功を assert — 終盤の方策振動は DQN 本来の病理で 8.4 の伏線)/ 8.4 **Double DQN を式から実装**(本はコード非提供): 成績は 7/10 対 7/10 で差なし(行動 2 つの CartPole では感度不足 — 理論どおりの null)だが、**過大評価バイアスそのものを Q(s₀) で直接測定し Normal 14.4 > Double 13.5(9/10 シード、差 ≈4σ)を検出** — 機構レベルで効果を実証。8.3(Atari/Pong)は読章(本もコード非提供、wgpu GPU 化後の実戦候補)/ 9章 方策勾配法: simple PG(9.1)→ REINFORCE(9.2)→ **Actor-Critic**(9.4)を同一検証枠組みで三段比較 — 3000 エピソードの last-100 平均 **88 → 189 → 198**、プラトー到達 1400 → 700 エピソード。方策は Policy クラスを作らず MLP+softmax(サンプリングは共通部品 `sample_action_from_logits`)。π(a|s) の Variable を**グラフ付きで運ぶ**設計(DQN の切断習慣と真逆)を隔離テストで担保し、Actor-Critic の切断 2 点(V ターゲットと δ)は f32 境界で計算して本家 `unchain()` を「忘れられない」形に翻訳 / 10章(A3C・DDPG・TRPO・Rainbow 概説)は読章 — **本編完走(2026-07-29)** |
| vol5 | **進行中** | 生成モデル編(2026-07-29 開始)。vol4 パターンを踏襲し、前半(ステップ 1〜5: 統計・GMM・EM)は素の Rust、ステップ 6(NN/VAE 境界)入り口でクレート昇格を判断。ステップ 1〜3 完了: 正規分布・最尤推定・多変量正規分布(det/inv は BLAS に頼らず自作 — 部分ピボット付きガウス消去・ガウス・ジョルダン) |
| vol6 | 未着手    | —                                                                                      |

## ロードマップ

学習(写経)の先に、フレームワークを研究の本番ツールへ育てる計画(2026-07 策定)。
フェーズによってコードの書き手が変わる:

- **写経フェーズ**(本のステップ): 私が書き、Claude はガイド兼レビュアーに徹する(従来通り)。
- **本番ツール化フェーズ**: vibe coding(Claude が実装を書く)。品質の錨は逐行理解ではなく、
  **テストスイート + PyTorch パリティ検証**(同一データ・同一初期値での一致試験)に置く。

順路:

1. **vol3 第5ステージ**(53 save/load → 54 Dropout → 55〜 CNN、以降 RNN)。
   ステップ52(GPU/CuPy)は**設計読書のみ** — Rust に CuPy 相当は無く、GPU 化は後段で wgpu により行う。
2. **vol4(強化学習)**: 前半(バンディット〜TD 法)は素の Rust で完走(2026-07-28)。
   後半入り口の最終判断は「**vol3 を path 依存でそのまま使う**」を採択 —
   クレート昇格は DQN で実戦経験を積んでからの別フェーズとする。
   **vol5(生成モデル)**は SBI 研究に直結する巻で、フレームワーク成長の主要ドライバ。
3. **クレート昇格 + v1.0 凍結**: フレームワークに独自名を付けて独立クレート化。
4. **PyTorch パリティ検証**を「本番昇格の儀式」として整備。
5. **Tauri GUI**: 学習モニタ・研究用フロントエンド(フレームワーククレートを直接リンク、単一バイナリ)。
6. **wgpu GPU 化(v2)**: 実測で必要と感じてから。vol1 の GPU 化経験を汎用フレームワークに一般化する。

研究側の照準は 2027 年春に再開する PID(ΔE-E)解析。

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
├── vol4/            # 4巻目(強化学習編)のクレート — 7章から vol3 に path 依存
│   ├── src/
│   │   ├── bandit.rs     # 1章 バンディット
│   │   ├── grid_world.rs # 4〜7章の環境(GridWorld)
│   │   ├── dp.rs         # 4章 動的計画法
│   │   ├── mc.rs         # 5章 モンテカルロ法
│   │   ├── td.rs         # 6章 TD 法(SARSA・Q 学習)
│   │   ├── qlearn_nn.rs  # 7章 Q 学習のニューラルネット化
│   │   ├── cart_pole.rs  # 8章の環境(CartPole-v0 相当を自作、gymnasium パリティ)
│   │   ├── dqn.rs        # 8章 ReplayBuffer・DQNAgent(Double DQN 含む)
│   │   ├── pg.rs         # 9章 方策勾配法(PGAgent・ActorCriticAgent)
│   │   └── utils.rs      # argmax・approx_eq など
│   ├── tests/            # 章単位の実験+assert(ch01.rs〜ch09.rs)
│   └── tools/
│       └── gen_cartpole_golden.py # CartPole パリティ golden の一度きり生成
├── vol5/            # 5巻目(生成モデル編)のクレート — ステップ 1〜5 は素の Rust
│   ├── src/
│   │   ├── gaussian.rs   # ステップ1〜3 正規分布・最尤推定・多変量正規(det/inv 自作)
│   │   └── utils.rs      # approx_eq・random_normal_array(RNG 注入)
│   └── tests/            # ステップ単位の実験+assert(step1_3.rs〜)
├── books/           # 本の PDF(gitignore 済み)
└── Cargo.toml       # ワークスペース定義
```

各巻の `dataset/` にデータセットや変換後の重みを置く(gitignore 済み・再取得可能)。

## 本との対応

本のどの章がどのファイルに対応するかの鳥瞰。細かい対応(節・見出し)は各関数の
doc コメント `/// 本 X.Y「見出し」` に書いてあり、`cargo doc --open` で閲覧できる。
「4.5 のコードはどこ?」となったら `rg "本 4.5"` で該当箇所へ飛べる。

各巻は独立クレート(`vol1`〜`vol6`)。巻を進めたらこの節に対応表を追記していく。

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
| 第3ステージ(25〜36)Graphviz 可視化・テイラー展開・最適化(勾配降下/ニュートン法)・**高階微分(double backprop)** | `vol3/src/utils.rs`(get_dot_graph)、`variable.rs`/`function.rs`(grad の Variable 化・create_graph)(実例: `tests/step26〜36.rs`, `examples/`) |
| 第4ステージ(37〜51)テンソル対応・線形回帰・Layer/Parameter・Optimizer・softmax 交差エントロピー・Dataset/DataLoader・MNIST | `vol3/src/functions.rs`(Reshape〜MatMul・MSE・linear・Log/Clip/Gather・ReLU)、`layers.rs`、`optimizers.rs`、`datasets.rs`、`dataloaders.rs`、`mnist.rs`、`utils.rs`(sum_to・get_spiral・accuracy)(実例: `tests/step38〜51.rs`, `examples/step42〜51.rs`) |
| 第5ステージ(52〜60)save/load・Dropout・CNN・VGG16・RNN/LSTM | `vol3/src/layers.rs`(save/load_weights・Conv2d・RNN・LSTM)、`cnn.rs`(im2col/col2im・conv2d_simple/pooling_simple)、`models.rs`(VGG16・SimpleRNN・BetterRNN)、`functions.rs`(dropout・Max・TransposeAxes)、`datasets.rs`/`dataloaders.rs`(SinCurve・SeqDataLoader)、`optimizers.rs`(Adam)、`examples/fetch_vgg16.py`(重み・golden データの一度きり取得)(実例: `tests/step53〜60.rs`) |

Python 版との主な設計差(Rust の所有権に合わせた意図的なもの):

- `Variable` は `Rc<RefCell<VariableInner>>` の薄いハンドル(Python の「全てが共有参照」の明示化)
- 関数の能力を trait で3分割: `Forward`(順伝播のみ・数値微分用、クロージャにも開放)/
  `Function: Forward`(+ 純関数の backward)/ `Creator`(グラフ遡行の最小界面)
- `Function::call(self)` は self を消費し、入力とともに `Node<F>` としてグラフに移る
  (Python の `self.input = input` に相当する状態を、型上「未設定になり得ない」形で持つ)
- f32 統一のため数値微分の刻みは eps=5e-3(∛ε_f32。本の 1e-4 は float64 用)

### 第4巻 ― 強化学習編(`vol4`)

本編完走(2026-07-29)。前半(〜6章)はフレームワーク不要の素の Rust、
7章からは vol3(フレームワーク編の成果物)に path 依存する(計画済みの例外 — ロードマップ参照)。
実験は章単位の統合テスト `vol4/tests/chNN.rs` に置き、本の「目視で確認」を
できる限り assert(パリティ・クロスチェック・性能検証)へ翻訳している。

| 本の章                         | ファイル                          |
| ------------------------------ | --------------------------------- |
| 1章 バンディット問題           | `vol4/src/bandit.rs`(実験: `tests/ch01.rs`) |
| 2〜3章 MDP・ベルマン方程式     | (読章)                          |
| 4章 動的計画法                 | `vol4/src/dp.rs`, `grid_world.rs`(実験: `tests/ch04.rs`) |
| 5章 モンテカルロ法             | `vol4/src/mc.rs`(実験: `tests/ch05.rs`) |
| 6章 TD 法                      | `vol4/src/td.rs`(実験: `tests/ch06.rs`) |
| 7章 ニューラルネットワークと Q 学習 | `vol4/src/qlearn_nn.rs`(実験: `tests/ch07.rs`) |
| 8.1 OpenAI Gym(CartPole)     | `vol4/src/cart_pole.rs` — 環境を自作し、gymnasium 実走の golden 軌道とパリティ(生成: `vol4/tools/gen_cartpole_golden.py`) |
| 8.2 DQN(経験再生・ターゲットネットワーク) | `vol4/src/dqn.rs`(実験: `tests/ch08.rs`、10 シード検証) |
| 8.3 DQN と Atari(Pong)       | (読章 — 本もコード非提供。wgpu GPU 化後の実戦候補) |
| 8.4 DQN の拡張(Double DQN)  | `vol4/src/dqn.rs`(`use_double_dqn` フラグ。実験: `tests/ch08.rs` — 10 シード比較+Q(s₀) 過大評価の直接測定。優先度付き経験再生・Dueling DQN は読章) |
| 9.1〜9.2 方策勾配法・REINFORCE | `vol4/src/pg.rs`(`PGAgent` — `update_simple`(9.1)/`update`(9.2)の 2 変種。比較実験: `tests/ch09.rs`) |
| 9.3 ベースライン               | (読章 — 思想は 9.4 の TD 誤差 δ に実装として合流) |
| 9.4 Actor-Critic               | `vol4/src/pg.rs`(`ActorCriticAgent` — π と V の 2 ネット、ステップごとの TD 更新) |
| 10章 さらに先へ                | (読章 — A3C/DDPG/TRPO/Rainbow などの概説) |

### 第5巻 ― 生成モデル編(`vol5`)

進行中(2026-07-29 開始)。正規分布 → GMM → EM → VAE → 拡散モデルの全 10 ステップ。
本の Python は DeZero でなく **PyTorch 2.x** のため、ステップ 6 以降の移植はロードマップの
「PyTorch パリティ検証」の実地訓練を兼ねる。実験はステップ単位の統合テスト
`vol5/tests/stepNN.rs` に置く(vol4 の章単位テストと同じ流儀)。

| 本のステップ                   | ファイル                          |
| ------------------------------ | --------------------------------- |
| ステップ 1〜3 正規分布・最尤推定・多変量正規分布 | `vol5/src/gaussian.rs`(det/inv の自作線形代数含む)、`utils.rs`(実験: `tests/step1_3.rs`) |
| ステップ 4〜(GMM・EM・NN・VAE・拡散) | 進行中                        |

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

第3巻(vol3)の重いテスト(外部データ依存、学習など)も `#[ignore]` 付きです。VGG16 の推論パリティ確認は以下のように実行します(事前に `cd vol3 && python examples/fetch_vgg16.py` による重み取得が必要):

```sh
cargo test --release test_vgg16 -- --ignored --nocapture
```

第4巻(vol4)の DQN 学習(8.2、CartPole 300 エピソード × 10 シード)も `#[ignore]` 付き。
greedy 評価 150 点以上のシードが過半数(6/10)なら健全と判定する(約36秒、release 実測):

```sh
cargo test -p vol4 --release test_8_2_5_dqn_multi_seed -- --ignored --nocapture
cargo test -p vol4 --release test_8_4_double_dqn -- --ignored --nocapture  # Normal vs Double 比較+Q(s0) 測定(約85秒)
```

第4巻 9章の方策勾配法(simple PG vs REINFORCE の比較、Actor-Critic)も同様:

```sh
cargo test -p vol4 --release test_9_1_pg_cartpole -- --ignored --nocapture   # 9.1 vs 9.2 比較(約13秒)
cargo test -p vol4 --release test_9_4_actor_critic -- --ignored --nocapture  # 9.4 Actor-Critic(約15秒)
```
