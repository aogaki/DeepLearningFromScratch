# DL — 『ゼロから作る Deep Learning』シリーズの Rust 移植(学習用)

## このプロジェクトの目的
斎藤康毅『ゼロから作る Deep Learning』(オライリー・ジャパン)全5巻を、Python+NumPy から
Rust へ自分で移植しながら学ぶ。成果物より「自分の手で書いて理解すること」が目的。
各巻を1つの Cargo クレート(`vol1`〜`vol5`)として、巻ごとに独立に実装する
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
- 各巻は独立クレート `vol1/`〜`vol5/`。新しい巻は `cargo new volN --lib` で作り、
  ルート `Cargo.toml` の `members` に追加する。
- 本の PDF は `books/volN.pdf`(gitignore 済み。Claude が参照するときは Read が必要)。
- データセットや変換後の重みは各巻の `dataset/` 配下(gitignore 済み。再取得可能)。

## コマンド
- テスト(全巻): `cargo test` / 単一巻: `cargo test -p volN`
- 型チェックのみ(速い): `cargo check`
- `println!` を表示したいテスト: `cargo test -- --nocapture`

## 現在地(詳細な経過は Auto Memory に任せる)
- **vol1**: 8.1 完了(Layer トレイト + Vec<Box<dyn Layer>> の DeepConvNet、MNIST 99.32%)。
  wgpu 進行中(src/gpu.rs): vec4 matmul カーネル(CPU比 ×11)、GpuTensor 常駐チェーン
  (4層MLP forward ×11.9)。次は conv/im2col の GPU 化 → DeepConvNet forward を GPU に。
- vol2〜vol5: 未着手。
