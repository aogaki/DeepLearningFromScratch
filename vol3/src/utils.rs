use crate::function::Forward;
use crate::variable::Variable;
use ndarray::ArrayD;

/// 数値微分用の微小な値。
/// f32 のマシンイプシロン ε (約 1.19e-7) に対して、丸め誤差 O(ε/h) と
/// 打ち切り誤差 O(h²) が釣り合う中心差分の最適な刻み幅の目安は ∛ε ≈ 5e-3。
pub const EPSILON_FOR_DIFF: f32 = 5e-3;

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
fn dot_var(v: &Variable, verbose: bool) -> String {
    let name = v.name().unwrap_or_default();
    let label = if verbose {
        let shape = v.shape();
        let shape_str = if shape.is_empty() {
            "()".to_string()
        } else {
            let s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", s.join(", "))
        };
        format!("{} {}", name, shape_str)
    } else {
        name
    };

    format!(
        "{} [label=\"{}\", color=orange, style=filled]\n",
        v.id(),
        label
    )
}
fn dot_func(func_id: usize, label: &str) -> String {
    format!(
        "{} [label=\"{}\", color=lightblue, style=filled, shape=box]\n",
        func_id, label
    )
}

/// 本 ステップ26「計算グラフの可視化」— 出力変数から遡って DOT 言語のテキストを作る。
///
/// 遡行は backward と同型だが、描くだけなので世代ソートは不要(seen による重複排除のみ)。
/// ノード ID は `Variable::id()`(Rc のポインタ)と creator の thin pointer を使う。
/// レンダリングの目安: `dot` コマンドが実用的なのは数千ノードまで(tanh の8階微分
/// ≈ 1.5万ノードで破綻する — examples/step35.rs の顛末を参照)。
pub fn get_dot_graph(output: &Variable, verbose: bool) -> String {
    let mut txt = String::from("digraph g {\n");
    let mut queue = vec![];
    let mut seen_set = std::collections::HashSet::new();

    let output_id = output.id();
    seen_set.insert(output_id);
    txt.push_str(&dot_var(output, verbose));
    queue.push(output.clone());

    while let Some(v) = queue.pop() {
        if let Some((creator_id, label, inputs)) = v.creator_info() {
            if !seen_set.contains(&creator_id) {
                seen_set.insert(creator_id);
                txt.push_str(&dot_func(creator_id, &label));
            }
            // Edge from function to output variable
            txt.push_str(&format!("{} -> {}\n", creator_id, v.id()));

            for input in inputs {
                let input_id = input.id();
                if !seen_set.contains(&input_id) {
                    seen_set.insert(input_id);
                    txt.push_str(&dot_var(&input, verbose));
                    queue.push(input.clone());
                }
                // Edge from input variable to function
                txt.push_str(&format!("{} -> {}\n", input_id, creator_id));
            }
        }
    }

    txt.push_str("}\n");
    txt
}

/// 本 ステップ40: 配列を target_shape まで和で畳むデータ層の実装(SumTo::forward の中身)。
/// 目標形を左から 1 で埋めて軸を対応させ、「目標が 1 で実際が >1」の軸だけ後ろから
/// sum_axis で潰す(後ろからなので軸番号がずれない)。最後の reshape が 1 の軸を整える。
pub fn sum_to(x: &ndarray::ArrayD<f32>, target_shape: &[usize]) -> ndarray::ArrayD<f32> {
    if x.shape() == target_shape {
        return x.clone();
    }
    let mut out = x.clone();
    let mut padded_target = vec![1; out.ndim().saturating_sub(target_shape.len())];
    padded_target.extend_from_slice(target_shape);

    for i in (0..out.ndim()).rev() {
        if padded_target[i] == 1 && out.shape()[i] > 1 {
            out = out.sum_axis(ndarray::Axis(i)).into_dyn();
        }
    }

    out.into_shape_with_order(target_shape).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sum_to() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();

        // Sum over axis 0 -> [3]
        let y = sum_to(&x, &[3]);
        assert_eq!(y, array![5.0, 7.0, 9.0].into_dyn());

        // Sum over axis 1 -> [2, 1]
        let y = sum_to(&x, &[2, 1]);
        assert_eq!(y, array![[6.0], [15.0]].into_dyn());

        // Sum all -> [] or [1] or whatever, let's say [1]
        let y = sum_to(&x, &[1]);
        assert_eq!(y, array![21.0].into_dyn());
    }
}
