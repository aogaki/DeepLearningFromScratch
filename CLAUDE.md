# DL — 『ゼロから作る Deep Learning』シリーズの Rust 移植(学習用)

## このプロジェクトの目的
斎藤康毅『ゼロから作る Deep Learning』(オライリー・ジャパン)全6巻を、Python+NumPy から
Rust へ自分で移植しながら学ぶ。成果物より「自分の手で書いて理解すること」が目的。
各巻を1つの Cargo クレート(`vol1`〜`vol6`)として、巻ごとに独立に実装する
(共有クレートは作らず、必要なものは各巻で再実装する — 写経して理解するため)。

## Claude の役割(最重要)
このプロジェクトは 2 つのモードを持つ(2026-07-25 決定)。どちらか曖昧な作業は着手前に確認する。

**写経モード(本のステップを学ぶ作業 — 以下のルールが適用される既定モード):**
- Claude はコードを **書かない**。学習者(私)が書く。Claude は **ガイド兼レビュアー** に徹する。
- 私がコードを書いたら(私が「書いた」と言ったら Claude が該当ファイルを読む)、Claude は次をやる:
  1. まず `cargo test` で実際にコンパイル・テストして結果を確認する(想像で判断しない)。
  2. 良い点を挙げ、そのうえで Rust のイディオム・設計の観点でレビューする。
  3. 改善は **答えを丸ごと書かず、方向とヒント** で示す。書き直すのは私。
- 一度に詰め込まない。1 応答につき論点は 1〜2 個まで。本の章・節の順に少しずつ進める。
- 私が詰まったら、詰まった箇所だけ扱う。解答の全文提示はしない。
- 新しい API を紹介するときは、最小の構文例だけ見せる(私の課題そのものは書かない)。
- 例外: 環境構築・データ入手・pkl→npy 変換などの「一度きりの下準備」は Claude が手伝ってよい
  (学習コードではないため)。

**本番ツール化モード(フレームワークのクレート昇格・hardening・研究用機能):**
- vibe coding スタイル。Claude が実装を書く。私の目的は原理と Rust の学習であり、
  実装の逐行理解ではない。
- 品質の錨は「私が実装を読むこと」ではなく **テストスイート + PyTorch パリティ検証** に置く。
  私は実装ではなく **テストと設計(API の形)** をレビューする。Claude はテストを読みやすく保つ。

## 進め方のループ
本の章・節の順に、各ステップを「私が書く → Claude がテスト&レビュー → 私が直す」で回す。
- 常に `cargo test` を green に保つ。
- リファクタは「動作を変えずテストで担保」する(配置や書き方だけ変えて、テストが通ればOK)。
- 「できた」と結論する前に、必ず `cargo test` で確認する。

## 技術方針・規約
- 浮動小数は **f32 で統一**(将来 wgpu の compute シェーダに載せるため。WGSL に f64 が無い)。
- NumPy 相当は `ndarray` クレート。
- Rust のイディオムを優先する:
  - 読むだけの引数は `&[T]` スライス(`&Vec<T>` は使わない)。
  - インデックスループより イテレータ(`iter().zip().map().sum()` など)。
  - 末尾式で `return` を省く。変数は snake_case。コンパイル警告はゼロを保つ。
- 浮動小数の比較はテストで誤差付き(`approx_eq(a, b, eps)`)。`==` は使わない。
- モジュールは責務ごとに分割し、各モジュールが自分の `#[cfg(test)] mod tests` を持つ。
  モジュール間の取り込みは明示 `use`、グロブ(`use super::*`)はテスト内に限る。

## リポジトリ構造
- ルート `/Users/aogaki/Study/DeepLearningFromScratch` は **Cargo ワークスペース**(`resolver = "3"`)。
  `Cargo.lock` と `target/` を全巻で共有する。
- 各巻は独立クレート `vol1/`〜`vol6/`。新しい巻は `cargo new volN --lib` で作り、
  ルート `Cargo.toml` の `members` に追加する。
- 本の PDF は `books/volN.pdf`(gitignore 済み。Claude が参照するときは Read が必要)。
- データセットや変換後の重みは各巻の `dataset/` 配下(gitignore 済み。再取得可能)。

## コマンド
- テスト(全巻): `cargo test` / 単一巻: `cargo test -p volN`
- テスト並列高速版: `cargo nextest run -p volN`(tests/ の全バイナリを跨いでテスト単位に並列実行。doc-test だけは対象外)
- 型チェックのみ(速い): `cargo check`
- `println!` を表示したいテスト: `cargo test -- --nocapture`

## 現在地(詳細な経過は Auto Memory に任せる)
- **vol1: 完了(2026-07-22)**。本編 8.1 まで(DeepConvNet、MNIST 99.32%)+ wgpu GPU 化の
  独自拡張を完走: 全網 forward/backward/Adam を GPU 常駐(WGSL 17 本)、カーネル最適化 3 段で
  1 iter 0.41 s → 21.8 ms(×18.8)、20 epoch 82 分 → 4.5 分、テスト精度 99.41% peak で
  CPU 版とパリティ。物語は `vol1/docs/wgpu-journey.md`(全21章)。
- **vol3: 進行中(2026-07-23 開始)**。第2ステージ(ステップ1〜24)完了。設計の要点:
  `Variable` は `Rc<RefCell>` の薄いハンドル、trait は Forward/Function/Creator に3分割、
  `call(self)` が関数を `Node<F>` としてグラフへ移す、数値微分の刻みは f32 用に eps=5e-3(∛ε)、
  世代管理つき backward、thread_local の no_grad、演算子はマクロで4通り+スカラー混合。
  モジュール構成: variable/function/functions/config/macros/utils + tests/(ステップ実例)。
  第3ステージ(〜ステップ36)完了: Graphviz 可視化、高階微分(grad が Variable、
  backward が Variable 演算でグラフを作る、create_graph フラグ+no_grad ガード)、
  Weak によるリーク検証テスト付き。
  第4ステージ(ステップ37〜51)完了: テンソル対応(reshape/transpose、broadcast_to/
  sum_to の双対、grad.shape == data.shape を debug_assert)、行列積、線形回帰。
  Layer trait(params=名前なしホットパス/named_params="l1/W" 階層名、cleargrads は
  デフォルト実装)、Linear/TwoLayerNet/MLP(活性化は fn ポインタ、Parameter は type
  エイリアス)。Optimizer(SGD/MomentumSGD — params() の Rc ハンドル clone を保持、
  split-borrow 不要)。分類系: Log/Clip/Gather-GatherGrad(双対+二階テスト)、
  softmax_cross_entropy_simple(合成、閉形式 (p−t)/N で検証)、spiral 97%。
  Dataset trait+DataLoader(IntoIterator for &mut で「1エポック=1 for ループ」、
  シード RNG 注入)。MNIST: IDX パーサ再実装(u8 (N,1,28,28) 保持)+transform
  関数ポインタ、ReLU(マスク定数×Mul の backward — 二階も自動で正しい)、
  MLP 5 epoch 17 秒で test 97.7%。
  第5ステージ進行中: 52(GPU)は設計読書のみと決定(GPU 化はロードマップ後段の wgpu v2)。
  53 完了: save/load_weights(Layer のデフォルトメソッド、ndarray-npy の npz、
  2パス atomic load — 全検証後に一括代入、部分書き込みなしをテストで担保)。
  Rust→Python np.load のパリティ橋を確認済み(キー 'l0/W' 等をそのまま認識)。
  54 完了: Config::train_mode + test_mode() RAII ガード(#[must_use] 付き)。
  dropout は合成スタイル(マスク定数×Mul — ReLU と同型、手書き backward なし、
  モード判定は forward の一度きり、二階も自動)。dropout_with_rng がシード注入口、
  dropout/Variable::dropout は rand::rng() フォールバックの便利ラッパー。
  モード跨ぎ2シナリオ(train→test backward / test→train backward)を回帰テスト化。
  55〜57 完了: 新モジュール cnn.rs(純関数 im2col_array/col2im_array — スラック付き
  バッファ+add_assign 累積、Im2Col/Col2Im の双対 Function、conv2d_simple/
  pooling_simple は合成スタイル)。functions.rs に Max(x==y マスク、タイは全員に
  勾配)と TransposeAxes(逆置換 backward)。教訓: gradient check は自己一致のみ —
  pooling_simple の reshape 転置ずれ(to_matrix=false の 6D を直接 reshape)は
  全テスト green のまま値が間違っていた。縮約・並べ替え関数には手計算の値テスト必須、
  かつ値テストは非対称データで(全同値データは並べ替えに盲目)。
  次はステップ58(代表的な CNN — VGG16)。
- vol2・vol4〜vol6: 未着手(vol2 は個人的興味の巻として後回し、vol3 を先行)。
