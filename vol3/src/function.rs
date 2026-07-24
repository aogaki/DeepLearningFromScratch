use crate::config::Config;
use crate::variable::Variable;
use ndarray::ArrayD;

/// 逆伝播がグラフ遡行に必要とする最小の界面(ステップ7の creator 参照の Rust 版)。
///
/// 出力 Variable が `Box<dyn Creator>` として所有する。参照は常に過去向き
/// (出力 → 関数 → 入力)なので、現状のグラフに Rc の循環は存在しない。
pub trait Creator {
    fn backward(&self, gy: &ArrayD<f32>) -> Vec<ArrayD<f32>>;
    fn get_inputs(&self) -> Vec<Variable>;
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
    fn backward(&self, gy: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let xs: Vec<ArrayD<f32>> = self.inputs.iter().map(|v| v.data()).collect();
        self.func.backward(&xs, gy)
    }

    fn get_inputs(&self) -> Vec<Variable> {
        self.inputs.clone()
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
/// `backward` は「入力 x と gy から gx」の純関数(ステップ6)。
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

    fn backward(&self, xs: &[ArrayD<f32>], gy: &ArrayD<f32>) -> Vec<ArrayD<f32>>;
}

impl<T> Forward for T
where
    T: Fn(&[ArrayD<f32>]) -> ArrayD<f32>,
{
    fn forward(&self, xs: &[ArrayD<f32>]) -> ArrayD<f32> {
        self(xs)
    }
}
