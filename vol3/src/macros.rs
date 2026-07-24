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
