use vol4::dp::{eval_onestep, policy_eval, random_policy};
use vol4::grid_world::GridWorld;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// 2マスグリッドワールドの状態遷移と報酬を返すヘルパー関数
// 状態: 0 (L1), 1 (L2)
// 行動: 0 (Left), 1 (Right)
fn get_next_state_and_reward(state: usize, action: usize) -> (usize, f32) {
    match (state, action) {
        (0, 0) => (0, -1.0), // L1で左: 壁にぶつかる → -1
        (0, 1) => (1, 1.0),  // L1で右: L2へ移動、リンゴ → +1
        (1, 0) => (0, 0.0),  // L2で左: L1へ移動 → 0
        (1, 1) => (1, -1.0), // L2で右: 壁にぶつかる → -1
        _ => panic!("Invalid state or action"),
    }
}

// 4.1.2 同期更新（新しいVを計算してから一斉に差し替える）
fn policy_eval_sync() -> [f32; 2] {
    let mut v = [0.0, 0.0];
    let gamma = 0.9;
    let pi = 0.5; // ランダム方策 (左:0.5, 右:0.5)

    let iters = 100; // 本家同様、固定回数回す
    for _ in 0..iters {
        let mut new_v = [0.0, 0.0];
        for (state, val) in new_v.iter_mut().enumerate() {
            let mut action_value_sum = 0.0;
            for action in 0..2 {
                let (next_state, reward) = get_next_state_and_reward(state, action);
                action_value_sum += pi * (reward + gamma * v[next_state]);
            }
            *val = action_value_sum;
        }

        // スイープが終わってから一斉更新
        v = new_v;
    }
    v
}

// 4.1.3 インプレース更新（その場でVを上書きし、収束判定で止める）
fn policy_eval_inplace() -> [f32; 2] {
    let mut v = [0.0, 0.0];
    let gamma = 0.9;
    let pi = 0.5;
    let threshold = 1e-4; // 収束判定のしきい値
    loop {
        let mut max_delta: f32 = 0.0;

        for state in 0..2 {
            let mut action_value_sum = 0.0;
            for action in 0..2 {
                let (next_state, reward) = get_next_state_and_reward(state, action);
                // その時点での最新の v を参照する
                action_value_sum += pi * (reward + gamma * v[next_state]);
            }

            // 更新量の絶対値を計算し、最大値を更新
            let delta = (v[state] - action_value_sum).abs();
            if delta > max_delta {
                max_delta = delta;
            }

            // その場で上書き（直後の L2 の計算で、更新されたての L1 の値が使われる）
            v[state] = action_value_sum;
        }

        // 更新量の最大値がしきい値を下回ったら終了
        if max_delta < threshold {
            break;
        }
    }
    v
}

#[test]
fn test_iterative_policy_evaluation() {
    let v_sync = policy_eval_sync();
    let v_inplace = policy_eval_inplace();

    println!("Sync Update V: {:?}", v_sync);
    println!("In-place Update V: {:?}", v_inplace);

    assert!((v_sync[0] - (-2.25)).abs() < 1e-3);
    assert!((v_sync[1] - (-2.75)).abs() < 1e-3);

    assert!((v_inplace[0] - (-2.25)).abs() < 1e-3);
    assert!((v_inplace[1] - (-2.75)).abs() < 1e-3);

    assert!((v_sync[0] - v_inplace[0]).abs() < 1e-3);
    assert!((v_sync[1] - v_inplace[1]).abs() < 1e-3);
}

#[test]
fn test_policy_eval() {
    let env = GridWorld::make_default();
    let pi = random_policy(&env);
    let v = policy_eval(&pi, &env, 0.9, 0.001);
    // まず値を確認（cargo test -- --nocapture で出力）
    for y in 0..3 {
        for x in 0..4 {
            if let Some(&val) = v.get(&(y, x)) {
                print!("{:8.3} ", val);
            } else {
                print!("  WALL   "); // 壁マスは V に存在しない
            }
        }
        println!();
    }

    // 実際の出力
    //   0.030    0.098    0.207    0.000
    //   -0.027   WALL     -0.496   -0.371
    //   -0.099   -0.217   -0.434   -0.784
    // 本の値（図4-13）と比較して、十分近いことを確認
    let tolerance = 1e-2;
    assert!(approx_eq(v[&(0, 0)], 0.030, tolerance));
    assert!(approx_eq(v[&(0, 1)], 0.1, tolerance));
    assert!(approx_eq(v[&(0, 2)], 0.21, tolerance));
    assert!(approx_eq(v[&(0, 3)], 0.000, tolerance));
    assert!(approx_eq(v[&(1, 0)], -0.03, tolerance));
    assert!(approx_eq(v[&(1, 2)], -0.5, tolerance));
    assert!(approx_eq(v[&(1, 3)], -0.37, tolerance));
    assert!(approx_eq(v[&(2, 0)], -0.1, tolerance));
    assert!(approx_eq(v[&(2, 1)], -0.22, tolerance));
    assert!(approx_eq(v[&(2, 2)], -0.43, tolerance));
    assert!(approx_eq(v[&(2, 3)], -0.78, tolerance));
}

#[test]
fn test_policy_eval_is_fixed_point() {
    let env = GridWorld::make_default();
    let pi = random_policy(&env);
    let v = policy_eval(&pi, &env, 0.9, 0.001);
    // 収束した V に対してもう1回更新
    let mut v_check = v.clone();
    eval_onestep(&pi, &mut v_check, &env, 0.9);
    // 変化量がほぼゼロであること = 不動点に到達している
    for state in env.states() {
        assert!(
            (v[&state] - v_check[&state]).abs() < 0.01,
            "V not converged at {:?}: before={}, after={}",
            state,
            v[&state],
            v_check[&state]
        );
    }
}
