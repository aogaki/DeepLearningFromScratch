//! 『ゼロから作る Deep Learning ❹ 強化学習編』の Rust 移植。
//!
//! 章→モジュール対応: 1章 `bandit` / 4章 `dp`+`grid_world` / 5章 `mc` / 6章 `td` /
//! 7章 `qlearn_nn` / 8章 `cart_pole`+`dqn` / 9章 `pg`(2〜3章と 8.3 は読章)。
//! 本の実験は章単位の統合テスト `tests/chNN.rs` に置く。細かい節の対応は
//! 各項目の doc コメント `本 X.Y「見出し」` を `rg "本 5.4"` のように検索。

pub mod bandit;
pub mod cart_pole;
pub mod dp;
pub mod dqn;
pub mod grid_world;
pub mod mc;
pub mod pg;
pub mod qlearn_nn;
pub mod td;
pub mod utils;
