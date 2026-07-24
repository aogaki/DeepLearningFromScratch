use crate::function::Forward;
use crate::variable::Variable;
use ndarray::ArrayD;

/// 本 ステップ4「数値微分」— 中心差分 (f(x+h) − f(x−h)) / 2h。
///
/// 注意: 本の eps=1e-4 は float64 用の値。f32 では丸め誤差 O(ε/h) が支配的に
/// なるため、h ≈ ∛ε_f32 ≈ 5e-3 が目安(テストの EPSILON_FOR_DIFF がこれ)。
pub fn numerical_diff<F>(f: F, x: &Variable, eps: f32) -> Variable
where
    F: Forward,
{
    let mut x0 = x.data();
    let mut x1 = x.data();

    x0 += eps;
    x1 -= eps;

    let y0 = f.forward(&[x0]);
    let y1 = f.forward(&[x1]);

    let diff_data = (y0 - y1) / (2.0 * eps);
    Variable::new(diff_data)
}

// 形の一致 + 全要素の絶対誤差で比較(浮動小数に == は使わない)
pub fn approx_equal_arrayd(a: &ArrayD<f32>, b: &ArrayD<f32>, tol: f32) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (*x - *y).abs() < tol)
}
