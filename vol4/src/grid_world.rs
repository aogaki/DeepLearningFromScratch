use std::collections::{HashMap, HashSet};

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
        [Action::Up, Action::Down, Action::Left, Action::Right]
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
}

pub struct GridWorld {
    height: usize,
    width: usize,
    reward_map: HashMap<(usize, usize), f32>,
    wall_state: HashSet<(usize, usize)>,
    goal_state: (usize, usize),
    start_state: (usize, usize),
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
        }
    }

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
}
