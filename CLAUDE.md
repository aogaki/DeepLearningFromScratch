# DL — 『ゼロから作る Deep Learning』シリーズの Rust 移植(学習用)

## このプロジェクトの目的
斎藤康毅『ゼロから作る Deep Learning』(オライリー・ジャパン)全6巻を、Python+NumPy から
Rust へ自分で移植しながら学ぶ。成果物より「自分の手で書いて理解すること」が目的。
各巻を1つの Cargo クレート(`vol1`〜`vol6`)として、巻ごとに独立に実装する
(共有クレートは作らず、必要なものは各巻で再実装する — 写経して理解するため)。

## Claude の役割(最重要)
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
  Weak によるリーク検証テスト付き。次はステップ37〜(第4ステージ: NN インフラ)。
- vol2・vol4〜vol6: 未着手(vol2 は個人的興味の巻として後回し、vol3 を先行)。
