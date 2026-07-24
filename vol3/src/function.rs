use crate::config::Config;
use crate::variable::Variable;
use ndarray::ArrayD;

/// 逆伝播がグラフ遡行に必要とする最小の界面(ステップ7の creator 参照の Rust 版)。
///
/// 出力 Variable が `Box<dyn Creator>` として所有する。参照は常に過去向き
/// (出力 → 関数 → 入力)なので、順伝播のグラフに Rc の循環は存在しない。
/// 例外はステップ32以降の `create_graph=true`: grad フィールド経由の循環が生じ得るが、
/// `cleargrad` が切断する(検証は tests/step33.rs のリークテスト)。
pub trait Creator {
    fn backward(&self, gy: &Variable) -> Vec<Variable>;
    fn get_inputs(&self) -> Vec<Variable>;
    fn label(&self) -> String;
}

/// 「関数と、その呼び出し時の入力」を束ねた計算グラフのノード。
///
/// `Function::call` が構築して出力の creator に渡すため、
/// 「入力が未設定の関数」という不正状態は型の上で存在しない。
pub struct Node<F> {
    inputs: Vec<Variable>,
    func: F,
}

impl<F: Function> Creator for Node<F> {
    fn backward(&self, gy: &Variable) -> Vec<Variable> {
        self.func.backward(&self.inputs, gy)
    }

    fn get_inputs(&self) -> Vec<Variable> {
        self.inputs.clone()
    }

    fn label(&self) -> String {
        let name = std::any::type_name::<F>();
        name.rsplit("::").next().unwrap_or(name).to_string()
    }
}

/// 順伝播だけの能力。数値微分(ステップ4)が関数に要求するのはここまで。
///
/// クロージャにはブランケット実装でこれだけを与える — `numerical_diff` には
/// 渡せるが、backward を持たないため `call` で計算グラフには入れない
/// (書こうとするとコンパイルエラーになる)。
pub trait Forward {
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32>;
}

/// 本 ステップ2「変数を生み出す関数」。
///
/// `call` が Python 版 `__call__` に相当するテンプレートメソッド。self を消費して
/// `Node` に移し、出力 Variable が creator として所有する(ステップ7)。
/// `where Self: Sized` により `call` は vtable から外れ、trait は dyn 互換のまま。
/// `backward` は「入力 x と gy から gx」の純関数(ステップ6)。ステップ32からは
/// Variable を受けて Variable 演算で書く — 勾配計算そのものが計算グラフになり、
/// `create_graph=true` の backward で高階微分が可能になる。
pub trait Function: Forward {
    fn call(self, inputs: &[Variable]) -> Variable
    where
        Self: Sized + 'static,
    {
        let xs: Vec<ArrayD<f32>> = inputs.iter().map(|x| x.data()).collect();
        let result_data = self.forward(&xs);
        let result = Variable::new(result_data);

        if Config::enable_backprop() {
            let max_gen = inputs.iter().map(|x| x.generation()).max().unwrap_or(0);
            result.set_generation(max_gen + 1);

            let node = Node {
                inputs: inputs.to_vec(),
                func: self,
            };

            result.set_creator(Box::new(node));
        }

        result
    }

    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable>;
}

impl<T> Forward for T
where
    T: Fn(&[ArrayD<f32>]) -> ArrayD<f32>,
{
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        self(xs)
    }
}
