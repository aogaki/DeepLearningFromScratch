# wgpu で CNN を GPU に載せるまでの記録

『ゼロから作る Deep Learning』vol1 の Rust 移植で、8 章の DeepConvNet(MNIST 99.32%)が
CPU で 0.41 s/iter・20 エポック 82 分かかったところから、wgpu による GPU 化を始めた記録。
ステップバイステップで進め、各段の**実測値**と**ハマりどころ**を残す。

- 環境: Apple M4 Pro(GPU 統合・ユニファイドメモリ)/ macOS / Metal バックエンド / wgpu **v30**
- コード: [`src/gpu.rs`](../src/gpu.rs)、シェーダは [`src/matmul.wgsl`](../src/matmul.wgsl) ほか
- 方針: f32 統一(WGSL に f64 が無い — プロジェクト開始時からの布石)

## 0. 前提となるメンタルモデル

- CPU = 賢いコア少数(逐次・不規則な仕事向き)。GPU = 単純な演算器数千個の一斉行進
  (同じ演算を大量データに)。
- GPU は**ビームライン**: 立ち上げ(転送・dispatch)の固定費が大きく、少量サンプルなら
  卓上装置(CPU)が勝つ。一度ビームが通れば桁違いの流量。
- 鍵となる量は**演算強度(arithmetic intensity)= 運んだバイトあたりの FLOP**。
  行列積は O(N²) のデータに O(N³) の演算 — GPU の理想形。ただし**細長い行列は不利**
  (後述、これが CNN の im2col で刺さる)。

## 1. 接続の確立(hello world 以前)

`Instance → Adapter(物理 GPU)→ Device(リソース工場)+ Queue(投入口)` の階層。
wgpu の一部 API は async だが、compute 用途なら `pollster::block_on` で同期化するだけでよい
(Rust の Future は `.await` されるまで何も実行されない「遅延実行の予約券」)。

```text
初期化コスト実測: Gpu::new() ≈ 0.5 秒(テストごとに払う)
max_buffer_size = 256 MB(M4 Pro / wgpu デフォルト limits)
```

**wgpu v30 の注意**(ネットの記事の大半は旧 API):

- `InstanceDescriptor` に `Default` が無い → compute 専用は `new_without_display_handle()`
- `request_adapter` / `request_device` は `Result` を返す(v25+)
- 待機は `device.poll(wgpu::PollType::wait_indefinitely())`
- `get_mapped_range()` は `Result<BufferView, _>` を返す

## 2. バッファ往復(シェーダなしでデータだけ運ぶ)

GPU が高速アクセスする `STORAGE` バッファは CPU から直接覗けない。読み戻しは
**staging バッファ**(`COPY_DST | MAP_READ`)に GPU 内コピー → `map_async` → `poll` →
`get_mapped_range` の「踊り」になる。ポイント:

- **usage フラグは前払いの契約**(ビームタイム申請書)。申告にない使い方は実行時エラー。
- コマンドは encoder に**記録**し `queue.submit` で**一括実行**。1 命令ずつ会話しない。
- `map_async` のコールバックは `device.poll()` の中で配達される。結果は
  `std::sync::mpsc::channel` でスレッドに持ち帰る(手組みのワーカー完了通知と同じ構図)。
- `BufferView` はバッファの借用 → **`drop(view)` してから `unmap()`**。ここは Rust の
  借用検査が守ってくれない領域(実行時に wgpu が検出する)。

テストは移動だけで算術ゼロなので `assert_eq!` 完全一致が正しい。

## 3. 初シェーダ(全要素 ×2)

WGSL は**実行時コンパイル**(naga が Metal に翻訳)。文法エラーは cargo build では捕まらず
実行時 panic — 「コンパイルが通った」の安全網がここでは効かない。必ずテストで実行する。

- 分身(invocation)は `@workgroup_size(64)` の組で起動され、dispatch には**組数**を渡す:
  `dispatch_workgroups(n.div_ceil(64))`。
- 端数の分身はシェーダ内の**番兵** `if (i < arrayLength(&data))` が黙らせる。
  **`div_ceil` と番兵は必ずペア** — 片方を忘れると取りこぼしか範囲外アクセス。
- ×2.0 は IEEE では指数部 +1 の厳密演算 → GPU/CPU が**ビット一致**する。exact assert 可。

## 4. matmul(素朴版)

1 スレッド = 出力 1 要素。新登場の道具:

- **uniform バッファ** = 少量・読み取り専用のパラメータ掲示板(行列サイズ)。16 バイト
  整列の作法で `_pad` を足す。
- `var<storage, read>` vs `read_write` — naga が強制する `&`/`&mut` 相当の契約。
- 2 次元 dispatch(`gid.x` = 列が最速軸)。row-major の添字算術 `a[row*k + i]` を手書きする。

テストの工夫:

- **整数値の小行列は exact assert**(f32 の整数演算は 2²⁴ まで厳密で加算順にも依存しない)。
- 乱数行列は ndarray の `dot` と突き合わせ、eps は「誤差 ~ |部分和| × K × 2⁻²⁴」から見積もる
  (K=53 で ~1e-5 のオーダー → 1e-3 は 100 倍の余裕)。
- サイズは**素数**(37,53)×(53,29)にして 2 次元番兵とタイル端を必ず踏ませる。

## 5. カーネル最適化サーガ(この記録の白眉)

2048² の行列積での実測。**教科書の定石が負けた**:

```text
naive(1 スレッド 1 出力)               : 302 GFLOP/s   ← 基準
workgroup 共有メモリ 16×16 タイリング    : 234 GFLOP/s   ← 定石の敗北
ローカル配列で 4×4 レジスタタイリング    : 151 GFLOP/s   ← さらに悪化
vec4 完全アンロール 4×4                 : 1170-1282 GFLOP/s ← 採用(CPU 比 ~×11)
```

敗因と勝因の分析:

1. **共有メモリタイリングが負けた理由**: あれは「global メモリが遅くキャッシュが貧弱な
   GPU」(CUDA 初期)の処方箋。Apple Silicon はキャッシュが優秀で、素朴版の重複読みは
   実質ヒットしていた。節約できる帯域が小さいところに `workgroupBarrier()` ×2/タイル
   (K=2048 で 256 回の全員整列)の同期コストを払って赤字。
2. **ローカル配列版が最悪だった理由**: `var acc: array<array<f32,4>,4>` を動的添字で
   触ると、レジスタではなく**スレッド私有メモリ(スタック相当)に落ちる**。
   レジスタのつもりがメモリアクセス。
3. **vec4 版が勝った理由**: ① 1 スレッドが 4×4 出力を担当 → ロード 1 回あたりの演算 4 倍
   (演算強度の向上)。② 蓄積を `vec4` 4 本 + 全添字コンパイル時定数 → **本当にレジスタに乗る**。

> **教訓: 最適化は呪文ではない。ハードウェアの模型を持ち、必ず測る。**
> 定石の由来(どの世代の、どんなメモリ階層の GPU の処方箋か)を問うこと。

なお共有メモリ版には「**バリアがあるカーネルでは early return 禁止**(全員がバリアに
到達する義務があるため、範囲外スレッドも 0 を詰めて参加させる)」という構造的複雑さが
あった。バリアの無い vec4 版では early return が合法に戻る。

シェーダ差し替えは binding インタフェースを変えなければ**純粋なリファクタ** —
既存テストが green のままなら移植成功(「動作を変えずテストで担保」の GPU 版)。

## 6. 端末間比較ベンチ(往復込みの正直な測定)

matmul 単体・**転送込み**での対 CPU(ndarray/matrixmultiply、単スレッドで一貫 ~110 GF/s):

```text
affine1  (100,1024)×(1024,50)   : GPU ×0.08  ← サブミリ秒領域は固定費が支配
conv1_2  (78400,144)×(144,16)   : GPU ×0.56  ← 転送律速(45MB 運んで 0.36 GFLOP)
square 2048                      : GPU ×11.25
```

**カーネルを 4 倍速くしても conv1_2 はほぼ動かなかった**(×0.50→×0.56)。
転送律速の仕事は演算をいくら速くしても救えない — 「dot をその場で GPU 呼び出しに
置き換える」路線はここで正式に棄却された。

## 7. GPU 常駐(GpuTensor)— 転送の削減

`GpuTensor`(`wgpu::Buffer` + shape)を導入し、`upload → matmul_gpu → add_bias_gpu →
relu_gpu → … → download` を**途中読み戻しゼロ**で連鎖させる。

- 同じ Queue への submit は**投入順に実行**され、バッファ依存(書いた→読む)には wgpu が
  **自動でメモリバリアを挿入**する。手動同期は不要(生 Metal/Vulkan との最大の違い)。
- in-place 演算は `&mut GpuTensor` で受ける。技術的には `&` でも通る(GPU バッファは
  内部可変)が、「書き換える」という意味論を型に載せ、Rust 側の別名借用も防ぐ。
- 重みは**ループ外で 1 回だけ** upload(訓練の現実: 重みは GPU に住む。動くのは
  バッチの上りと結果の下りだけ)。

4 層 1024 次元 MLP forward の実測(重み常駐、iter あたり):

```text
batch  100: CPU  8.17 ms ≈ GPU毎層往復 8.44 ms | GPU常駐 3.43 ms(CPU比 ×2.39)
batch 1024: CPU 74.6  ms | GPU毎層往復 13.3 ms | GPU常駐 6.27 ms(CPU比 ×11.9)
```

読み方:

- batch 100 で CPU ≈ 往復 — GPU の演算優位が転送コストと相殺している。転送を抜いた
  常駐版は勝ちに転じた。**「小さい仕事は GPU に不向き」は転送の罪であって仕事の罪ではない。**
- batch 1024 の ×11.9 は matmul 単体の ×11.25 と整合 — **層間転送さえなければ、連鎖しても
  カーネルの優位は保存される**。
- 常駐/往復 ×2.1〜2.5 が「転送そのものの値段」。カーネル最適化(×4)と常駐(×2)は
  **独立に効く直交するノブ**。

## 8. ハマりどころ総集編

1. **binding 番号の対応表は型システムの外。** bind group のエントリ欠落・席違いは
   コンパイルが通り、`create_bind_group` の実行時 validation で初めて落ちる。
   シェーダの `@binding` 宣言と Rust 側を目視で突き合わせ、**テストを先に書く**。
2. **呼ばれないコードの green は空虚。** 新関数を書いてスイートが green でも、
   どのテストもそれを呼んでいなければ何も検証されていない。ラッパ化リファクタ
   (旧 API を新 API の合成で書き直す)は、既存テストを新経路の検証に変える一石二鳥。
3. **`zip` は長さ不一致を黙って切り詰める。** 比較テストは先に `assert_eq!(dim)`、
   または期待値 Vec を作って `assert_eq!` 一発。
4. ベンチは `--release` 必須 + `std::hint::black_box` で最適化削除を防ぐ。
   release では行番号・`debug_assert!`・整数オーバーフロー検査が消えることも知っておく。
5. WGSL のエラーは実行時。シェーダは別ファイル + `include_str!` にするとエディタの
   支援が効く。
6. exact assert と誤差 assert の使い分けは「演算の種類」で決める: 移動・比較・×2・
   整数値演算 → exact / 加算順が変わる総和 → eps(ulp × 加算回数で見積もる)。

## 9. 現在地と次の課題

- 済み: 接続 / 往復 / 要素ごとカーネル / vec4 matmul(×11)/ 常駐チェーン(×11.9)
- 次: **conv の GPU 化**。im2col を WGSL シェーダにして(4 重ループは「1 スレッド =
  出力 1 要素」に素直に翻訳できる形をしている)、画像を GPU に置いたまま
  im2col → matmul → … と流す。最終目標は DeepConvNet の forward/backward を GPU 常駐で
  回し、0.41 s/iter がどこまで縮むかを測ること。
