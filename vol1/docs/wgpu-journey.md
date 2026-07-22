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

## 9. im2col の GPU 化

CPU 版の 4 重ループ(n / out_y / out_x / チャンネル×窓)は、GPU では
「**1 スレッド = col 行列の 1 要素**」+ 平坦添字からの `div`/`mod` 復元に翻訳できる
([`src/im2col.wgsl`](../src/im2col.wgsl))。ハマりどころ:

- **pad の添字は i32 で計算する。** `oy*stride + fy - pad` は負になり得るが、WGSL の
  u32 減算は panic せず**黙ってラップする**(Rust の debug ビルドより恐い)。
- **2D dispatch が必須。** conv1_2 実寸の col は 78400×144 ≈ 1130 万要素。1 次元だと
  workgroup 数が**1 次元あたり上限 65535** を超えて実行時エラーになる。
- テストは CPU 版 im2col と **`assert_eq!` 完全一致**(im2col は移動のみで算術ゼロ)。

## 10. conv forward 一式と GpuImage

conv には im2col・matmul のほかに「(N·OH·OW, FN) → NCHW への並べ替え」
([`src/nhwc_to_nchw.wgsl`](../src/nhwc_to_nchw.wgsl))が要る — CPU 版の
`permuted_axes([0,3,1,2])` に相当し、これがないと**次の層の im2col に食わせられない**。

4D の形情報の別送りが 3 箇所に達した時点で `GpuImage { tensor, dims }` を導入
(形はデータに同伴させる)。`conv_forward_gpu` = im2col → matmul → bias → 並べ替え。
検収は conv→ReLU→conv の 2 層連鎖を CPU と比較(チャンネル数を 3→4→5 と**全部変える** —
同数だとルーティングのバグが見えない)。

この過程で「**bind group のエントリ欠落はコンパイルが通り、実行時 validation で初めて
落ちる**」「**呼ばれない新関数はスイートが green でも何も検証されていない**」という
2 大教訓を実地で踏んだ(8 章の落とし穴 1・2 の実例)。

## 11. pooling シェーダ(初の自力 WGSL)

max-pooling は decode を NCHW で書くと**出力が最初から NCHW になり、並べ替え不要**
(conv との構造的な違い)。backward を見越して窓内 argmax も保存する
([`src/pooling.wgsl`](../src/pooling.wgsl))。

**意図的に本と挙動を変えた点**: pad 領域を −∞ 扱い(max から除外)にした。本の CPU 版は
im2col 流用のため 0 埋めで「窓内が全負のとき 0 が勝つ」— 標準的な max-pool の定義
(PyTorch 等)は −∞ 側。DeepConvNet は pool pad=0 なので実運用の差は出ないが、
pad>0 の CPU 突き合わせテストは成立しないため pad=0 に限定している。

注意: argmax は u32 なので f32 前提の `GpuTensor::download` に入れてはいけない
(bytemuck がビットパターンを f32 と誤読し、**黙って**無意味な数値になる)。

## 12. DeepConvNet forward 全載せ — ×20〜28

16 層(conv×6 + ReLU×7 相当 + pool×3 + affine×2)を同じ重みから CPU/GPU 両方で手組みし、
ロジットを比較。**Flatten は GPU ではタダ**(pool 出力の NCHW 平坦バッファを
そのまま (n, c·h·w) 行列と見なすだけ)。実測:

```text
batch 100: CPU 201 ms | GPU  9.8 ms | ×20.6   (logit 最大差 1e-5)
batch 200: CPU 398 ms | GPU 14.3 ms | ×27.7
```

**全網の速度比(×20+)が matmul 単体の ×11 を上回った**のがポイント:

- CPU の conv は「CPU で im2col(conv1_2 で 45MB の中間行列を実体化)→ dot」と
  メモリを激しく往復するのに対し、GPU 版は im2col がカーネル内の添字計算に溶けている。
- 転送は最初の画像(batch 100 で 300KB)上りとロジット 4KB 下りだけ。
  6 章で見た「転送律速 ×0.56」の構造は**消滅**した。

部品の比より系の比が良くなる — 常駐アーキテクチャの結論がここで完成する。

## 13. backward の部品 — atomics が無い世界の設計

conv backward = 「dout 並べ替え → dW = colᵀ·dout → db = 列和 → dcol = dout·w → col2im」。
新しい部品([`src/matmul_tn.wgsl`](../src/matmul_tn.wgsl)・[`src/column_sum.wgsl`](../src/column_sum.wgsl)・
[`src/relu_backward.wgsl`](../src/relu_backward.wgsl)・[`src/pool_backward.wgsl`](../src/pool_backward.wgsl)・
[`src/col2im.wgsl`](../src/col2im.wgsl))で得た知見:

- **WGSL に f32 の atomic 加算は無い**(atomic は u32/i32 のみ)。これが backward 設計全体を
  規定する: 「複数スレッドが同じ場所に足し込む」形は書けないので、**scatter を gather に
  裏返す**。col2im は「入力 1 画素 = 1 スレッドが、自分を覆う窓を**区間** oy∈[⌈(iy+pad−fh+1)/stride⌉, ⌊(iy+pad)/stride⌋]
  として列挙し集める」形に。u32 の ceil は負になり得る分子を場合分けで回避。
- **gather の加算順を CPU の蓄積順に一致させた結果、col2im は CPU と exact 0 一致**。
- ReLU backward はマスクを保存せず **forward の出力から復元**(out > 0)。境界 x==0 の
  意味論が CPU 版マスク(x<=0 を殺す)と一致することを確認してから採用。
- pool backward は保存済み argmax への散布。**窓が重ならない**(stride ≥ 窓)前提を
  assert で固定 — 重なると同一画素への競合加算になり、atomics の無い世界では書けない。
- 新品バッファは**仕様でゼロ初期化** — 散布先のクリアパスは不要。
- Rust 側の教訓: `into_shape_with_order` は **self を消費**する。`&` 越しには呼べず、
  clone(データコピー)か `view()` 経由(メタデータのみ)の二択。reshape は本来
  メタデータ操作なので view 経由が原則(ただし view 版は連続レイアウト必須)。

## 14. レイヤ構造体と子モジュール(gpu/layers.rs)

全網 backward には層ごとの状態(col キャッシュ・argmax・勾配置き場)が要る。タプルの
引き回しは 6 層で破綻するので、CPU 版と同型の構造体に束ねた
([`src/gpu/layers.rs`](../src/gpu/layers.rs))。設計の学び:

- **モジュール木**: `gpu.rs` と `gpu/` ディレクトリは共存でき(2018+)、`mod X;` の解決は
  「探索」ではなく**宣言側のモジュールパスからの一意な写像**。子モジュールは祖先の
  private が見える(可視性はファイル位置でなく木の祖先関係)— layers を gpu の**子**に
  した理由。
- **`wgpu::Buffer` は Clone 可能で、ハンドル共有**(Arc 的参照カウント。プローブで
  「clone 後に元へ書いた値が clone から見える」ことを確認)。ReLU 層の活性キャッシュは
  この**ハンドル clone** で持つ — GPU メモリコピーなし。
- ただしキャッシュと下流テンソルは**同一メモリのエイリアス**になる。安全性の根拠は
  借用検査ではなく**演算設計の規律**(下流は全部 read-only 消費+キュー投入順=実行順)。
  CPU 版との最大の思想差。

## 15. 全網 backward 一致 —「exact 0 は経路の性質」

16 層を CPU/GPU 両方で降ろし、dx・端の層の dW/db を比較。**dx と db は exact 0、
dW(matmul_tn 経由)だけ ~1e-5** という結果になった。この非対称こそが答え合わせ:

- f32 の積和は 1 演算ごとに正確に丸められるから、**ずれの源は「加算順序」と
  「FMA 縮約」の 2 つだけ**。選択・散布・コピー・同順加算の経路は exact、順序を変えた
  vec4 リダクション経路だけがノイズを持つ。理論と観測が一致。
- FMA の発見: SGD カーネル(`param -= lr·grad`)が CPU と**ちょうど 1 ULP** ずれた。
  Metal が積和を FMA 1 命令(丸め 1 回)に縮約するため。CPU 側を `mul_add` にしたら
  **ビット単位一致**(プローブで証明)。Rust は暗黙の FMA 縮約を決してしない。
- **ミューテーションテスト**: GPU 側の dout だけ 2 倍 → テストは diff=|dx| で即死。
  「比較が空回りしていない」ことと「dx の振幅 ~1.1」を 1 回の実験で同時に確認する技。
- のちに matmul_nt へ置換した際、dx の exact 0 は 3.6e-7 に、c1_1 の db も exact を失った
  (column_sum 自体は同順のまま — **入ってくる dout が既に bitwise 同一でない**)。
  精密な原理: **exact 0 は演算の性質ではなく「最後に bitwise 一致していた地点からの
  経路」の性質**。

## 16. matmul_nt で重み二重持ちを根絶

SGD 更新を入れると「forward 用 w_colt と backward 用 w_col の二重持ち」は**片方が陳腐化**
する。転置リフレッシュで延命する道もあったが、**NT カーネル(C = A·Bᵀ)を書けば二重持ち
自体が消える**([`src/matmul_nt.wgsl`](../src/matmul_nt.wgsl))— 陳腐化バグの構造的根絶。

- NT は両オペランドとも k 方向が行内連続 = **一番 vec4 に優しい形**(vec4 dot 蓄積 +
  水平和 + 端数スカラー)。
- Rust 側テストのバグから得た教訓: `Zip::from(&mut x.clone())` は**名前のない一時**を
  更新して捨てる(セミコロンで死ぬ)。`assign` 系メソッド呼び出しは「使用」に数えられる
  ため **unused 警告も出ない**。借用エラーを clone で黙らせるとき、それは大抵
  「設計が何かを教えようとしている」瞬間。

## 17. GpuDeepConvNet と学習ループ初動

網は `Vec<Box<dyn Layer>>` でなく**名前付きフィールド+手書きチェーン**にした
([`src/gpu/deep_conv_net.rs`](../src/gpu/deep_conv_net.rs))。理由は妥協ではない:

- GPU 層は forward の型が層種で違う(GpuImage/GpuTensor・借用/値渡し)。**型の不均一性は
  情報**で、手書きチェーンなら配線ミスがコンパイルエラーになる。dyn で揃えるには統一
  enum が要り、配線ミスは実行時 panic に格下げされる。閉じた集合(固定アーキテクチャ)に
  開いた集合の道具(dyn)は不要 — Rust の文化は「可変だと証明されるまで具体的に」。
- 重み注入(`GpuDeepConvNetParams`)を用意し「同じ重みを CPU と GPU に配る」テスト戦略を
  維持。**テストが呼ばない `new()` に 2 つのバグが眠っていた**(入力チャネル数・隠れ層幅)—
  「呼ばれないコードの green は空虚」の新変奏。
- 学習ループの分担: GPU = forward/backward/update 全部、CPU = バッチ抽出・softmax+CE
  (既存 SoftmaxWithLossLayer 再利用)・argmax。転送は上り 300KB+4KB、下り 4KB /iter。
- 初回実測: SGD で loss は健康に降下、1 epoch 96.4%。**しかし 0.24 s/iter — CPU 比 ×1.7
  しかない**。forward は 10ms のはずなのに。ここから最後の最適化アークが始まる。

## 18. 縮約カーネル三部作(前) — スレッド飢餓

まず計時の罠: **`queue.submit` は非同期**で、同期点(download の poll)まで実行を待たない。
素朴にフェーズ間で時刻を取ると、backward のコストが「次の iter の download」に請求される。
診断中は各フェーズ後に `device.poll(wait)` を入れて完了を強制する。

フェーズ分解の結果 **backward が 240ms/249ms** 。層別プローブで犯人確定:

```text
matmul_tn c1_1 (k=78400, 出力 9×16)  : 74.0 ms ← 起動スレッド 12 個!
column_sum c1  (78400×16)            : 21.7 ms ← 起動スレッド 16 個
参照: matmul_nt c1_1 (出力 78400×9)  :  0.15 ms ← スレッド 78400+
```

「出力 1 要素 = 1 スレッド」は**出力が巨大な forward では正解、出力が極小で k が巨大な
縮約(dW・db)では数千レーンを遊ばせる**。かつて batch 2 で「column_sum は negligible」と
判定していた — **スケールが変わるとレジームが変わる**。

対策 = **workgroup 協調リダクション**: 1 出力 = 1 workgroup(256 スレッド)の 3 幕構成。
①各スレッドが k を stride 256 で分担し私的部分和(通信なし)→ ②`var<workgroup>` の
共有黒板に書いて `workgroupBarrier()` → ③木構造の半減ループ(s=128,64,…,1、各ラウンドに
barrier)で log₂256 = 8 段で畳む。

- **バリアは一様制御フロー必須**: スレッド分岐の early return は残った全員を永久に待たせる
  (workgroup-uniform な分岐は合法だが、死んだガードは書かない)。
- 5 章で負けた共有メモリがここでは唯一解: あのときの用途は**キャッシュ**(Apple の
  キャッシュが既にやってくれる仕事)、今回は**通信路**(f32 atomics の無い世界でスレッドが
  部分和を合流させる唯一の手段)。**同じ道具でも仕事が違えば評決が変わる**。

効果: column_sum 21.7→0.46ms(×47)、iter 249→105ms。

## 19. 縮約カーネル三部作(後) — 帯域律速とタイル×リダクション

まだ backward に 95ms 残る。層別マップを取ると **時間が matmul_tn の積和数に綺麗に比例**
(~10 GMAC/s = 20 GF/s。forward の vec4 カーネルは 1170 GF/s)。スレッド飢餓は治ったが、
リダクション版は **1 積和につき 8 バイト読む(0.25 FLOP/byte)再利用ゼロ**の帯域大食いに
なっていた — 律速が「並列度」から「帯域」に移った。

最終形 = **既習 2 パターンの合成**: 5 章の 4×4 レジスタタイル × 18 章の k リダクション。
1 workgroup = 出力 4×4 タイル、各スレッドは k ステップごとに vec4×2(8 float)で 16 積和を
4 本の vec4 アキュムレータに(全添字コンパイル時 = レジスタ常駐)。黒板は vec4 配列 ×4 =
**16,384 バイト — WebGPU デフォルトの workgroup メモリ上限にぴったり(ヘッドルーム 0)**。
端数タイルは「**読みはクランプ、書きはガード**」— 全スレッドの制御フローが row/col の
有効性に依存しないので、バリアの一様性が**構造的に**保証される。

検収は形状砲撃(実戦 3 形状+境界 4 種を ndarray と突き合わせ)— 全形状で diff ≈ 1e-6·√k
= 純粋な加算順ノイズ。k=1 は exact 0(積 1 個なら順序の自由度が無い — ノイズモデルの検算)。

```text
iter: 249 ms → 105 ms → 22 ms(CPU 0.41 s 比 ×18.8)
三部作の教訓: ①仕事をレーンに配れ(飢餓)②配ったら運搬を削れ(帯域)— 測ってから。
```

## 20. 学習完走 — SGD 98.97% → Adam 99.4%

- **SGD 20 epoch: 279 秒(4.6 分)、最終 98.97%**。CPU(Adam)の 99.32% との差は
  パイプラインではなく**最適化アルゴリズムの差**(勾配一致は 1e-5 で証明済み、
  train loss 0.005 = 容量は足りている)。
- **Adam カーネル**([`src/adam.wgsl`](../src/adam.wgsl))= 初の**状態を持つカーネル**。
  m・v は param と同形の GPU 常駐バッファ(ゼロ初期化仕様がそのまま Adam の初期値)、
  バイアス補正係数は CPU 側で f32 `powi` 計算して uniform で渡す(CPU 実装と同じ計算経路)。
  **ε の位置(√v̂ の外)まで CPU 版 optimizer.rs を一字一句ミラー** — 流儀が違うと
  1e-6 の突き合わせが成立しない。テストは**状態が進化する 3 ステップ逐次比較**。
- **Adam 20 epoch: 272 秒(4.5 分)、peak 99.41% / 常用 99.2+** — CPU の 99.32% と
  統計的同水準。**パリティ達成**。

## 21. 総括

```text
              CPU(ch8 完成時)     GPU(完成形)
1 iter        0.41 s              21.8 ms   (×18.8)
20 epoch      82 分               4.5 分
最終精度      99.32%(Adam)      99.41% peak(Adam)
```

シェーダ 17 本(matmul 系 4・conv 系 4・pool 系 2・要素演算 5・縮約 2)、
lib テスト 91 本 green・警告ゼロ。大きな教訓を 5 つだけ選ぶなら:

1. **測ってから直す。** 定石(共有メモリタイリング)も直感(小さい仕事は GPU に不向き)も
   実測に負けた。犯人は毎回プローブで確定させ、予言を立ててから直す。
2. **転送・並列度・帯域は独立のノブ。** 常駐化(×2)→ 飢餓解消(×2.4)→ 再利用(×4.7)と、
   律速は解くたびに次の場所へ移る。
3. **型システムの外に落ちる場所を知る。** binding 対応表・バリアの一様性・エイリアスした
   ハンドル — そこを守るのはテストと設計の規律。
4. **exact 0 は経路の性質。** ずれの源は加算順序と FMA 縮約の 2 つだけ、という模型は
   全観測を説明した。誤差 assert の eps は見積もれる量であって、当て勘ではない。
5. **呼ばれないコードの green は空虚。** ミューテーションテスト・形状砲撃・「テストが
   本領を踏んでいるか」の点検が、green を信頼に変える。
