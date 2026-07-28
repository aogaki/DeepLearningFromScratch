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

/// ndarray-rand の `Array::random_using` 相当。
/// rand 0.10 系に対応した ndarray-rand が存在しないため自前実装。
pub fn random_array<Sh, D>(
    shape: Sh,
    dist: impl rand_distr::Distribution<f32>,
    rng: &mut impl rand::Rng,
) -> ndarray::Array<f32, D>
where
    Sh: ndarray::ShapeBuilder<Dim = D>,
    D: ndarray::Dimension,
{
    ndarray::Array::from_shape_simple_fn(shape, || dist.sample(rng))
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

/// 本 ステップ48: スパイラルデータセットの生成
/// train: 訓練用かテスト用かでシードを切り替える (本家 DeZero に準拠して 1984 / 2020)
pub fn get_spiral(train: bool) -> (ndarray::ArrayD<f32>, Vec<usize>) {
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let seed = if train { 1984 } else { 2020 };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let num_data = 100;
    let num_class = 3;
    let input_dim = 2;

    let mut x = ndarray::Array2::<f32>::zeros((num_data * num_class, input_dim));
    let mut t = vec![0; num_data * num_class];

    for j in 0..num_class {
        for i in 0..num_data {
            let rate = i as f32 / num_data as f32;
            let radius = 1.0 * rate;
            let theta = j as f32 * 4.0 + 4.0 * rate + rng.random_range(-0.2..0.2);

            let idx = num_data * j + i;
            x[[idx, 0]] = radius * theta.sin();
            x[[idx, 1]] = radius * theta.cos();
            t[idx] = j;
        }
    }

    (x.into_dyn(), t)
}

/// 本 ステップ48: 多クラス分類の正解率 (非微分・評価用)
pub fn accuracy(y: &Variable, t: &[usize]) -> f32 {
    let y_data = y.data();
    let y2d = y_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
    let mut correct = 0;
    for (i, &target) in t.iter().enumerate() {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for j in 0..y2d.shape()[1] {
            if y2d[[i, j]] > max_val {
                max_val = y2d[[i, j]];
                max_idx = j;
            }
        }
        if max_idx == target {
            correct += 1;
        }
    }
    correct as f32 / t.len() as f32
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

        let y = sum_to(&x, &[1]);
        assert_eq!(y, array![21.0].into_dyn());
    }
}
