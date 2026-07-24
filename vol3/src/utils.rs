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
