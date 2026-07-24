use std::cell::Cell;

thread_local! {
    static ENABLE_BACKPROP: Cell<bool> = const { Cell::new(true) };
}

pub struct Config;
impl Config {
    pub fn enable_backprop() -> bool {
        ENABLE_BACKPROP.with(|f| f.get())
    }
    pub fn set_enable_backprop(b: bool) {
        ENABLE_BACKPROP.with(|f| f.set(b));
    }
}

#[must_use = "ガードは変数に束縛してください(例: let _guard = no_grad();)。束縛しないと即座に drop されて無効になります。"]
pub struct NoGradGuard {
    prev: bool,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        Config::set_enable_backprop(self.prev);
    }
}

#[must_use = "ガードは変数に束縛してください(例: let _guard = no_grad();)。束縛しないと即座に drop されて無効になります。"]
pub fn no_grad() -> NoGradGuard {
    let prev = Config::enable_backprop();
    Config::set_enable_backprop(false);
    NoGradGuard { prev }
}
