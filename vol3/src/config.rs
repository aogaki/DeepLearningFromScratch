use std::cell::Cell;

thread_local! {
    static ENABLE_BACKPROP: Cell<bool> = const { Cell::new(true) };
}

/// 本 ステップ18「メモリ使用量を減らすモード」の enable_backprop 設定。
///
/// 状態は thread_local に持つ: `cargo test` はテストを並列スレッドで走らせるため、
/// プロセス全体のグローバルにすると隣のテストのモード切替が漏れ込んでしまう。
pub struct Config;
impl Config {
    pub fn enable_backprop() -> bool {
        ENABLE_BACKPROP.with(|f| f.get())
    }
    pub fn set_enable_backprop(b: bool) {
        ENABLE_BACKPROP.with(|f| f.set(b));
    }
}

/// 逆伝播記録を一時停止する RAII ガード(Python の `with no_grad():` 相当)。
/// 生成時の値を覚え、Drop で復元するので入れ子にできる。
#[must_use = "ガードは変数に束縛してください(例: let _guard = no_grad();)。束縛しないと即座に drop されて無効になります。"]
pub struct NoGradGuard {
    prev: bool,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        Config::set_enable_backprop(self.prev);
    }
}

/// 本 ステップ18: このガードが生きている間、`call` はグラフを作らない(推論モード)。
#[must_use = "ガードは変数に束縛してください(例: let _guard = no_grad();)。束縛しないと即座に drop されて無効になります。"]
pub fn no_grad() -> NoGradGuard {
    let prev = Config::enable_backprop();
    Config::set_enable_backprop(false);
    NoGradGuard { prev }
}
