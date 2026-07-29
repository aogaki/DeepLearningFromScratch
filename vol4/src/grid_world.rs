//! 本 4.2.1「GridWorld クラスの実装」— 3×4 グリッドワールド環境(4〜7章で使用)。
//! 5.3.1「step メソッド」で reset/step(エージェントの体 = agent_state)を追加。

use std::collections::{HashMap, HashSet};

/// 本 4.2.1: 上下左右の行動。usize との相互変換は 7章(NN の出力インデックス対応)で追加。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}
impl Action {
    // action_space() で返す順番に合わせる
    pub fn all() -> [Action; 4] {
        [
            Self::from_usize(0),
            Self::from_usize(1),
            Self::from_usize(2),
            Self::from_usize(3),
        ]
    }

    // 行動に対応する移動量 (dy, dx)
    fn delta(&self) -> (i32, i32) {
        match self {
            Action::Up => (-1, 0),
            Action::Down => (1, 0),
            Action::Left => (0, -1),
            Action::Right => (0, 1),
        }
    }

    // usize への変換（ここで enum のメモリ上の並びを利用）
    pub fn to_usize(self) -> usize {
        self as usize
    }
    // usize からの変換（ここを to_usize と対にする）
    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            3 => Action::Right,
            _ => panic!("Invalid action index"),
        }
    }
}

/// 本 4.2.1「GridWorld クラスの実装」: 図4-9 の 3×4 グリッドワールド。
/// 終端はゴールのみ(爆弾マスは通過可)、壁は states() から除外。
pub struct GridWorld {
    height: usize,
    width: usize,
    reward_map: HashMap<(usize, usize), f32>,
    wall_state: HashSet<(usize, usize)>,
    goal_state: (usize, usize),
    start_state: (usize, usize),
    agent_state: (usize, usize), // エージェントの体を含む、脳の外側は環境として扱う。
}
impl GridWorld {
    pub fn new(
        height: usize,
        width: usize,
        reward_map: HashMap<(usize, usize), f32>,
        wall_state: HashSet<(usize, usize)>,
        goal_state: (usize, usize),
        start_state: (usize, usize),
    ) -> Self {
        GridWorld {
            height,
            width,
            reward_map,
            wall_state,
            goal_state,
            start_state,
            agent_state: start_state,
        }
    }

    /// 本 図4-9 と同じ 3×4 グリッド(ゴール (0,3)・爆弾 (1,3)・壁 (1,1)・スタート (2,0))。
    pub fn make_default() -> Self {
        // 本と同じ 3 x 4 のグリッドワールドを作る
        let height = 3;
        let width = 4;
        let mut reward_map = HashMap::new();
        reward_map.insert((0, 3), 1.0); // ゴールマス
        reward_map.insert((1, 3), -1.0); // 罰マス
        let wall_state = HashSet::from([(1, 1)]); // 壁マス
        let goal_state = (0, 3);
        let start_state = (2, 0);

        GridWorld {
            height,
            width,
            reward_map,
            wall_state,
            goal_state,
            start_state,
            agent_state: start_state,
        }
    }

    pub fn states(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.height).flat_map(move |y| {
            (0..self.width).filter_map(move |x| {
                if self.wall_state.contains(&(y, x)) {
                    None // 壁マスは飛ばす
                } else {
                    Some((y, x))
                }
            })
        })
    }

    // 行動に対する次の状態を返す
    pub fn next_state(&self, state: (usize, usize), action: Action) -> (usize, usize) {
        let (dy, dx) = action.delta();
        let ny = state.0 as i32 + dy;
        let nx = state.1 as i32 + dx;
        let next = (ny as usize, nx as usize);
        // 範囲外 or 壁 → 元の位置にとどまる
        if ny < 0
            || ny >= self.height as i32
            || nx < 0
            || nx >= self.width as i32
            || self.wall_state.contains(&next)
        {
            state
        } else {
            next
        }
    }

    // 次の状態に対する報酬を返す
    pub fn reward(
        &self,
        _state: (usize, usize),
        _action: Action,
        next_state: (usize, usize),
    ) -> f32 {
        *self.reward_map.get(&next_state).unwrap_or(&0.0)
    }

    /// 本 5.3.1「step メソッド」: エージェントをスタート位置に戻す。
    pub fn reset(&mut self) -> (usize, usize) {
        self.agent_state = self.start_state;
        self.agent_state
    }

    /// 本 5.3.1「step メソッド」: 1 ステップ進めて (次状態, 報酬, done) を返す。
    pub fn step(&mut self, action: Action) -> ((usize, usize), f32, bool) {
        let state = self.agent_state;
        let next_state = self.next_state(state, action);
        let reward = self.reward(state, action, next_state);
        let done = self.is_goal(next_state);

        self.agent_state = next_state;
        (next_state, reward, done)
    }

    // ゴール状態（終端状態）かどうか
    pub fn is_goal(&self, state: (usize, usize)) -> bool {
        state == self.goal_state
    }
    pub fn goal_state(&self) -> (usize, usize) {
        self.goal_state
    }
    pub fn start_state(&self) -> (usize, usize) {
        self.start_state
    }

    pub fn is_terminal(&self, state: (usize, usize)) -> bool {
        self.is_goal(state)
    }

    pub fn height(&self) -> usize {
        self.height
    }
    pub fn width(&self) -> usize {
        self.width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_usize_roundtrip() {
        // 全バリアントについて、usize への変換と復元が完全に一致することを保証する
        // これにより、将来バリアントの宣言順が変更された際のサイレントなバグを検知する
        for &action in &Action::all() {
            let idx = action.to_usize();
            let restored = Action::from_usize(idx);
            assert_eq!(
                action, restored,
                "Action roundtrip failed for index {}. Check enum declaration order and from_usize match arms.",
                idx
            );
        }
    }

    #[test]
    fn test_next_state() {
        let gw = GridWorld::make_default();
        // 表: (現在状態, 行動, 期待される次状態)
        let cases = vec![
            // 通常移動
            ((2, 0), Action::Up, (1, 0)),
            ((2, 0), Action::Right, (2, 1)),
            // 場外（4辺）: 元の位置にとどまる
            ((0, 0), Action::Up, (0, 0)),    // 上端で上
            ((0, 0), Action::Left, (0, 0)),  // 左端で左
            ((2, 3), Action::Down, (2, 3)),  // 下端で下
            ((2, 3), Action::Right, (2, 3)), // 右端で右
            // 壁(1,1)への進入ブロック: 4方向から
            ((0, 1), Action::Down, (0, 1)),  // 上から壁へ
            ((2, 1), Action::Up, (2, 1)),    // 下から壁へ
            ((1, 0), Action::Right, (1, 0)), // 左から壁へ
            ((1, 2), Action::Left, (1, 2)),  // 右から壁へ
        ];
        for (state, action, expected) in cases {
            assert_eq!(
                gw.next_state(state, action),
                expected,
                "state={:?}, action={:?}",
                state,
                action
            );
        }
    }

    // --- 2. states() の個数と壁の除外 ---
    #[test]
    fn test_states() {
        let gw = GridWorld::make_default();
        let states: Vec<_> = gw.states().collect();
        // 3×4 = 12 マス − 壁1マス = 11 状態
        assert_eq!(states.len(), 11);
        // 壁(1,1)が含まれていないこと
        assert!(!states.contains(&(1, 1)));
    }

    // --- 3. reward: ゴール / 爆弾 / 空きマスの3点 ---
    #[test]
    fn test_reward() {
        let gw = GridWorld::make_default();
        // ゴール(0,3)に入る → +1
        assert_eq!(gw.reward((0, 2), Action::Right, (0, 3)), 1.0);
        // 爆弾(1,3)に入る → -1
        assert_eq!(gw.reward((1, 2), Action::Right, (1, 3)), -1.0);
        // 普通のマスに入る → 0
        assert_eq!(gw.reward((2, 0), Action::Right, (2, 1)), 0.0);
    }

    // --- 4. 終端状態の扱い ---
    #[test]
    fn test_terminal_states() {
        let gw = GridWorld::make_default();
        // ゴールだけが終端
        assert!(gw.is_goal((0, 3)));
        // 爆弾は終端ではない（−1を食らうが通過できる普通のマス）
        assert!(!gw.is_goal((1, 3)));
        // 通常マスも非終端
        assert!(!gw.is_goal((2, 0)));
    }

    #[test]
    fn test_reset_and_step() {
        let mut gw = GridWorld::make_default();
        let initial_state = gw.reset();
        assert_eq!(initial_state, (2, 0));
        // ゴールまでの最短ルート
        gw.step(Action::Right); // (2,1)
        gw.step(Action::Right); // (2,2)
        gw.step(Action::Up); // (1,2)
        gw.step(Action::Up); // (0,2)

        // ゴール到達
        let (next_state, reward, done) = gw.step(Action::Right);
        assert_eq!(next_state, (0, 3));
        assert_eq!(reward, 1.0);
        assert!(done);
        // 1. 【追加】エピソード終了後の reset の検出力
        // ゴール(0,3)にいる状態から、ちゃんとスタート地点に戻るかを確認
        let reset_state = gw.reset();
        assert_eq!(reset_state, (2, 0));
    }

    #[test]
    fn test_step_bomb_is_not_terminal() {
        // 2. 【追加】爆弾マスは「ペナルティを食らうが通過できる」ことの封印
        let mut gw = GridWorld::make_default();
        gw.reset();
        // 爆弾マス(1,3)へのルート（壁を避けて下から回る）
        gw.step(Action::Right); // (2,1)
        gw.step(Action::Right); // (2,2)
        gw.step(Action::Up); // (1,2)

        // 爆弾へ踏み込む
        let (next_state, reward, done) = gw.step(Action::Right);
        assert_eq!(next_state, (1, 3));
        assert_eq!(reward, -1.0); // ペナルティ
        assert!(!done); // ★重要：ここでエピソードは終わらない
        // 爆弾マスから抜け出してゴールできることも確認
        let (next_state, reward, done) = gw.step(Action::Up);
        assert_eq!(next_state, (0, 3));
        assert_eq!(reward, 1.0);
        assert!(done);
    }
}
