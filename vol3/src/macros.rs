/// 本 ステップ20「演算子のオーバーロード」の糖衣を量産するマクロ。
///
/// 基礎実装(`&Variable op &Variable`)は variable.rs に手書きし、このマクロは
/// 所有×所有・所有×参照・参照×所有の3通りを委譲で生成する。所有版は演算途中の
/// 一時変数のためのもので、名前を持つ葉変数は `&x` で渡す(ムーブさせない)。
/// `$crate::` の絶対パスにより、どのモジュールから呼んでも展開が壊れない。
#[macro_export]
macro_rules! impl_op_combinations {
    ($trait_name:ident, $method_name:ident) => {
        impl std::ops::$trait_name<$crate::Variable> for $crate::Variable {
            type Output = $crate::Variable;
            fn $method_name(self, rhs: $crate::Variable) -> $crate::Variable {
                std::ops::$trait_name::$method_name(&self, &rhs)
            }
        }

        impl std::ops::$trait_name<&$crate::Variable> for $crate::Variable {
            type Output = $crate::Variable;
            fn $method_name(self, rhs: &$crate::Variable) -> $crate::Variable {
                std::ops::$trait_name::$method_name(&self, rhs)
            }
        }

        impl std::ops::$trait_name<$crate::Variable> for &$crate::Variable {
            type Output = $crate::Variable;
            fn $method_name(self, rhs: $crate::Variable) -> $crate::Variable {
                std::ops::$trait_name::$method_name(self, &rhs)
            }
        }
    };
}

/// 本 ステップ21「スカラーとの混合演算」: f32 と Variable の4通りを量産するマクロ。
///
/// `impl Add<Variable> for f32` は「外部 trait × 外部型」だが、trait のジェネリック
/// 引数にローカル型 Variable が現れるため孤児ルールの例外で合法 — Python の
/// `__radd__` 群に相当する右側実装が、特別な仕組みなしに書ける。
#[macro_export]
macro_rules! impl_op_scalar {
    ($trait_name:ident, $method_name:ident) => {
        // f32 op Variable
        impl std::ops::$trait_name<$crate::Variable> for f32 {
            type Output = $crate::Variable;
            fn $method_name(self, rhs: $crate::Variable) -> $crate::Variable {
                std::ops::$trait_name::$method_name($crate::Variable::from(self), rhs)
            }
        }

        // f32 op &Variable
        impl std::ops::$trait_name<&$crate::Variable> for f32 {
            type Output = $crate::Variable;
            fn $method_name(self, rhs: &$crate::Variable) -> $crate::Variable {
                std::ops::$trait_name::$method_name($crate::Variable::from(self), rhs)
            }
        }

        // Variable op f32
        impl std::ops::$trait_name<f32> for $crate::Variable {
            type Output = $crate::Variable;
            fn $method_name(self, rhs: f32) -> $crate::Variable {
                std::ops::$trait_name::$method_name(self, $crate::Variable::from(rhs))
            }
        }

        // &Variable op f32
        impl std::ops::$trait_name<f32> for &$crate::Variable {
            type Output = $crate::Variable;
            fn $method_name(self, rhs: f32) -> $crate::Variable {
                std::ops::$trait_name::$method_name(self, $crate::Variable::from(rhs))
            }
        }
    };
}
