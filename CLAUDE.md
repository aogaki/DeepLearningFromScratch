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
  58 完了: Conv2d レイヤ、models.rs 新設(VGG16 — named_params が正、params は導出)。
  fetch_vgg16.py(DeZero pretrained から重み+golden ペア (x,y) を npz 化、一度きり下準備)。
  **VGG16 16層 forward が Python DeZero と 1e-4 で一致(パリティ儀式の初実戦)**。
  release 1.8s / debug 121s のため #[ignore]+README 手順化。重みは dataset/weights/
  (.gitignore に *.npz 追加済み — 528MB 事故防止)。
  59 完了: RNN レイヤ(RefCell<Option<Variable>> の隠れ状態、h2h は nobias)、
  Variable::unchain(一点切断 — 所有権 DAG なので本家のような walk 不要)+
  Layer::unchain_backward(デフォルト no-op、状態層が override、複合層は子へ委譲)。
  教訓: truncated BPTT のピンは2本 — レイヤの h と「累積 loss の Add 連鎖」。
  後者を切り忘れると silent full BPTT + 二次コスト(スクラッチ実測で発見、
  ウィンドウごと loss 変数+f32 集計に分離して解決)。SimpleRNN・SinCurve・
  Adam(本家準拠の lr_t 畳み込み — v1.0 要件を前倒しで充足)。
  sin 学習 100 epoch が release 2.8s、final loss < 0.02 を assert。
  境界テストは「勾配遮断」+「Weak で解放観測」の2本立て(レイヤ版+モデル版)。
  60 完了: LSTM(ゲート8 Linear、状態 h/c の2本 — reset/unchain とも両方処理)、
  SeqDataLoader(ジャンプ幅オフセットの並行ストリーム、1エポック=jump 回)、
  Dataset trait に関連型 Target 導入(usize 固定の借金を返済、既存実装も移行)、
  BetterRNN。LSTM の unchain_backward は「固有メソッドが trait を陰る」罠を踏みかけ
  (具象呼びは動くが dyn Layer 経由で no-op — スクラッチ実測で発見)、trait impl 側へ
  移して &dyn Layer 経由の回帰テストで封印。cos 自己回帰 MSE 0.40 を assert
  (SimpleRNN は 1.5 超)、100 epoch release 0.98s。
  **★ vol3 本編(60ステップ)完走(2026-07-25)。**次はロードマップ順で vol4 前半
  (素の Rust)、クレート昇格の判断は vol4 後半入り口。
- **vol4: 本編完走(2026-07-26〜07-29)**。前半(1〜6章: バンディット〜TD 法)はフレームワーク不要の
  素の Rust・写経モード。後半(7章〜: 関数近似・DQN)入り口でフレームワーク共有を最終判断。
  1章 バンディット問題 完了(2026-07-26): Bandit/NonStatBandit/Agent/AlphaAgent、
  update_q_step 一般化(1/n は特殊ケース)、utils::argmax(Option、max_by は同率で最後)、
  RNG は引数渡し。実験は tests/ch01.rs(章単位命名 chNN.rs、ゼロ埋め)で本の結論を assert
  (ペアドシード比較)。2〜3章は読章。
  4章 動的計画法 完了(2026-07-27): grid_world.rs(Action enum、None を HashMap+HashSet に
  分解、states() は壁除外・ゴールは含む、終端はゴールのみ — 爆弾は通過可)+ dp.rs
  (Policy 型、eval_onestep/policy_eval、action_values 共有ヘルパー、greedy_policy、
  policy_iter、value_iter)。検証は図4-13 パリティ+γべき手計算表+アルゴリズム間
  クロスチェック。教訓: 環境の転記ミスは自己一致テストで検出不能(2マス世界の壁 −1 抜け)。
  5章 モンテカルロ法 完了(2026-07-27): GridWorld に agent_state/reset/step(境界=脳、
  体は環境)、mc.rs(RandomAgent は update_q 再利用の every-visit MC、McAgent は
  Q+ε-greedy+固定α、greedy_probs)。MC↔DP クロスチェック(1万エピソード、tol 0.05)。
  MC 制御は「全状態最適」を assert せず greedy ロールアウトの性能で検証(オフ経路の
  局所解は診断出力に格下げ — ε-greedy オン方策の教科書的病理を実地観測)。
  重点サンプリングは推定量の分散を試行反復で測る設計(b を π に近づけ分散 1/4)。
  6章 TD 法 完了(2026-07-28): td.rs(TdAgent/SarsaAgent/OffPolicySarsaAgent/
  QLearningAgent)。SARSA の deque(2) は Option<前ステップ> に翻訳(本家の None ダミー
  呼び出しが構造ごと消滅)。方策オフ SARSA は IS の ρ=0 破壊的更新のため α=0.8→0.1
  (理由コメント付き — 本家は目視デモなので 0.8 で成立して見える)。Q 学習は 6.5.2 の
  サンプルモデル版に統合(分布を実体化しない ε-greedy on-the-fly、ε=1.0 で純ランダム
  学習の実証テスト)。3テストの α 履歴(0.8/0.1/0.8)が「IS の呪縛と Q 学習による解放」の
  自己文書化。**★ vol4 前半(素の Rust 編)完走。**
  フレームワーク判断(2026-07-28): **vol3 を path 依存でそのまま使う**を採択(クレート
  昇格は DQN で実戦経験を積んでから別フェーズで)。共有クレート禁止ルールのここでの
  例外は、予定されていた最終判断による。準備として vol3 の rand を 0.9→0.10 に統一
  (RngCore→Rng・Rng→RngExt 改名、ndarray-rand は 0.10 対応版が無く削除 →
  utils::random_array で代替、ChaCha8 の乱数列は不変でシード固定テスト無傷、
  87 テスト green)。
  7章 ニューラルネットワークと Q 学習 完了(2026-07-28): vol3 の初実戦投入(MLP/SGD/
  mean_squared_error/relu/Variable::gather が無改造で噛み合う)。本の 7.1〜7.3(DeZero
  復習・線形回帰・NN)は vol3 履修済みでスキップ、7.4 のみ実装: qlearn_nn.rs(one_hot
  ヘルパー+QLearningNNAgent — ε-greedy の get_action、update は qs.gather で q[:,action]
  切り出し(本家 p.228 準拠)、next_q_max は .data() の f32 取り出し=本家 unchain 相当、
  done で next_q=0)。Action::to_usize(as キャスト)/from_usize(match)を追加、all() は
  from_usize から導出して対応表の正を1箇所に、往復テストで宣言順との一致を封印。
  検証は性能ベース(本家は loss 曲線目視): 1000 エピソード学習→ε=0 greedy ロールアウトが
  20 歩以内ゴール・reward approx_eq 1.0。debug 3.4s で #[ignore] 不要。教訓: 非選択行動の
  Q も隠れ層共有で更新後微動(出力層の勾配は厳密 0 — ch8 target network への伏線)。
  rustc の unused_assignments は入れ子 for 内の dead store を見逃す(最小再現で確認)。
  8章 下準備(2026-07-28): CartPole 環境を Claude が移植(cart_pole.rs — 本は Gym を
  import するだけで実装は学習対象外のため「一度きりの下準備」扱い)。gymnasium 1.3.0 の
  cartpole.py に忠実(陽的 Euler、f32、RNG 引数渡し、v0=200 步上限、CartPoleAction は
  Action と同じ to_usize/from_usize/all 構成)。検証は golden 軌道 2 本
  (tools/gen_cartpole_golden.py が本家 gymnasium を実走して生成 — 物理 10 步+
  角度境界 18 步「閾値 0.2094 を 0.2029 でかすめて非終了→反対側で終了」、tol 1e-4)
  +位置終了両側・上限/reset カウンタ・reset 範囲の計 6 テスト。
  方針決定(2026-07-28): 8.3「DQN と Atari」(Pong)は読章扱い — 本自体がコード非提供
  (p.252 に明言)、実装には ALE+GPU 丸1日が必要。Pong はフレームワーク昇格+wgpu v2
  後の実戦検証候補として棚上げ。ε 線形減衰・報酬クリッピング(8.3.4)は環境非依存なので
  CartPole 上で味見できる拡張。
  8.2 DQN 完了(2026-07-28): ReplayBuffer<T>(VecDeque+満杯時 pop_front=deque(maxlen)、
  rand::seq::index::sample=random.sample、underfill は assert — should_panic(expected)は
  「パニック 1 呼び出しへ最小化」を教訓化)。DQNAgent(q_net/target_q_net、sync_qnet は
  params() zip+set_data コピーで deepcopy 相当 — 長さ assert 付き、update はバッチ gather+
  target 網 .data() 切断+(1-done) マスク、Adam lr=0.0005)。テスト: 同期一致・構造不一致
  panic・update 隔離(全層スナップショット: q_net は変化 assert_ne/target は全層不変 assert_eq)。
  8.2.5 は multi-seed 実験に発展(seed 引数のヘルパー+10 シード): 初回は隠れ 100×1 で
  成功 3/5+0/5 → **本家 QNet は隠れ 2 層×128**(p.245 実査で発見)へ修正後、
  **10/10 が訓練中に満点 200 到達**、greedy ≥150 は 7/10(過半数 6/10 を assert)。
  greedy 終盤劣化 3 シードは Q 過大評価の方策振動 — 8.4 Double DQN の before/after 素材として温存。
  教訓: 単一シードの DQN 曲線は「渋いが正常/微妙に壊れてる」を判定不能 — seed sweep
  (0.2〜3.6 s/本)+対照群で切り分ける。容量不足はシード運と紛らわしい。
  README を全面更新(vol4 の進捗表・構成ツリー・対応表・コマンド、2026-07-28)。
  8.4 完了(2026-07-28): Double DQN を式から実装(本はコード非提供)— use_double_dqn フラグで
  update のターゲット計算のみ分岐(選択=online の argmax・評価=target 網の該当列)。argmax は
  utils::argmax_f32(ArrayView1)へ共通化(同一ファイル内 2 実装目=三度目の法則)。10 シード比較:
  成績 7/10 vs 7/10・平均 167 vs 168 で**差なし(正しい null — 行動 2 つの CartPole は感度不足)**。
  教訓: 更新則が違えば同一シードでも経験列ごと別物=**ペアド比較は不成立**(弱シード改善・強シード
  悪化に見えるのは独立抽選間の平均への回帰)。追加測定で null を陽性に:
  **Q(s₀) で過大評価を直接測定 — Normal 14.41(SE .12)> Double 13.54(SE .16)、Δ≈0.87 ≈4σ・
  9/10 シード**。機構は検出、成績は不動 — 教科書の主張の定量化。絶対値は真値 ≈49(Σγ^t)より
  大幅に低い(伝播未収束+200 步打ち切り done 扱いの truncation バイアス — gymnasium が
  terminated/truncated を分けた理由の実地観測)。比較テストは両変種 ≥6/10 の健全性 assert 付き。
  8.4.2 優先度付き/8.4.3 Dueling は読章。**★ 8章完走。**
  9章 方策勾配法 完了(2026-07-29): pg.rs — PGAgent(update_simple=9.1/update=9.2 REINFORCE
  の 2 変種、_simple サフィックスは vol3 の家風)+ActorCriticAgent(9.4 — π と V の 2 ネット、
  lr 別々、ステップ更新)。Policy クラスは作らず MLP+softmax_simple、サンプリングは
  sample_action_from_logits に共通化(softmax 内蔵=忘れられない性質の部品化)。設計の要点:
  π(a|s) の Variable をグラフ付きで運ぶ(ch8 の .data() 切断習慣と真逆 — 「切れていれば
  Adam 不動」の隔離テストで両 update 経路を封印)。AC の切断 2 点(V ターゲットと δ)は
  f32 境界の計算で表現 — 本家の unchain() 2 回を「型で不可能」に翻訳。
  検証は 3 手法を同一枠組みで三段比較: 3000 エピソード last-100 平均 88→189→198、
  プラトー到達 1400→700 エピソード。事件簿: ①reduce への書き換えで 9.1 が偶然 9.2 に変質
  (G 畳み込みと項構築の融合 — 可変キャプチャ付き map は畳み込みの隠れ蓑)→ 9.1/9.2 の差分
  理解に昇華。②「アルゴリズムが上手くいきすぎると落ちる」上限 assert を排除(主張はギャップ
  assert に一本化)。リファクタ定石を実地演習: to_batch_var(utils)+sample_action_from_logits
  抽出→全呼び出し移行(dqn.rs にも展開)→旧経路削除→シード固定テストで ch08/ch09 全学習
  結果が小数点まで不変を証明。9.3 は読章(思想は δ に合流)。**★ 9章完走。**
  10章「さらに先へ」読章完了(2026-07-29)。**★ vol4 本編完走。**
  ロードマップ分岐(昇格 vs vol5)は「vol4 パターン」で解決: vol5 ステップ 1〜5 を素の
  Rust で先行し、クレート昇格の判断はステップ 6(NN/VAE 境界)入り口に繰り延べ。
  備考: clippy 更新で vol1 に後方警告 47 件が出現(vol3/vol4 は 0 — 別途扱い)。
- **vol5: 進行中(2026-07-29 開始)**。生成モデル編・全 10 ステップ(正規分布→最尤推定→
  多変量正規→GMM→EM→NN→VAE→階層型VAE→拡散→条件付き拡散)。**本の Python は DeZero
  でなく PyTorch 2.x** — ステップ 6 以降の移植は「PyTorch パリティ検証」の実地訓練を兼ねる。
  ステップ 1〜3 完了(2026-07-29): gaussian.rs(normal/normal_array/fit_normal/
  multivariate_normal+自作 determinant/invert_matrix — 部分ピボット付きガウス消去・
  ガウス・ジョルダン。ndarray-linalg はリンクエラー(BLAS バックエンド未指定)を機に排除)、
  utils.rs(approx_eq、random_normal_array — `&mut impl Rng` 注入)、tests/step1_3.rs。
  rand は 0.10+rand_distr 0.6 に一本化(ndarray-rand は rand 0.9 を引き込むため不採用 —
  vol3 合流準備)。検証の型: 線形代数は非対称 3×3 手計算値+ピボット交換強制+特異行列+
  A·A⁻¹≈I(単位行列だけでは消去が未踏になる — vol4 pooling 教訓の再演)、
  「図→統計量 assert」翻訳第 1 号として規格化条件のリーマン和 assert。
  次: ステップ 4(混合ガウスモデル)→ 5(EM)— 研究(Oslo/SBI)に最も近い区間。
- vol2・vol6: 未着手(vol2 は個人的興味の巻として後回し)。
