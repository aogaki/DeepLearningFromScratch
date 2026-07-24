//! DeZero(『ゼロから作る Deep Learning ❸』)の Rust 移植 — 第2ステージ(〜ステップ24)まで。
//!
//! Python 版パッケージとのモジュール対応:
//! - dezero/core.py の Variable → `variable`(`Rc<RefCell>` ハンドル・backward・演算子)
//! - 同 Function / Config → `function`(Forward/Function/Creator trait と Node)・`config`
//! - dezero/functions.py → `functions`(Square, Exp, Add, Mul, Neg, Sub, Div, Pow)
//! - 補助 → `utils`(数値微分・近似比較)・`macros`(演算子 impl の量産)
//!
//! Python の「全てが共有参照」を、Rust では `Rc<RefCell<…>>` で明示する。
//! 使い方の実例は、ステップ番号付きの統合テスト(`tests/`)がそのまま目次になっている。
//! 下の `pub use` 群は dezero/__init__.py 相当のファサード(`use vol3::*` で全部届く)。

pub mod config;
pub mod function;
pub mod functions;
pub mod macros;
pub mod utils;
pub mod variable;

pub use config::*;
pub use function::*;
pub use functions::*;
pub use utils::*;
pub use variable::*;
