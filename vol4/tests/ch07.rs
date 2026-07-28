use ndarray::Array;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vol3::functions::{mean_squared_error, relu};
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Optimizer, SGD};
use vol3::variable::Variable;
use vol4::grid_world::GridWorld;
use vol4::qlearn_nn::{QLearningNNAgent, one_hot};
use vol4::utils::approx_eq;

// --- 7.4.1 DeZero の復習 / 7.4.2 Q関数とニューラルネットワーク ---
#[test]
fn test_7_4_1_qnet_forward() {
    let mut rng = StdRng::seed_from_u64(42);

    // GridWorld (3x4 = 12状態) を想定した one-hot ベクトル
    let x = one_hot((2, 0), 4, 3);
    // QNet (MLP): 入力12 -> 隠れ層100 -> 出力4
    let q_net = MLP::new(&[12, 100, 4], relu, &mut rng);
    let qs = q_net.forward(&x);

    assert_eq!(qs.shape(), &[1, 4]);
    println!("State (2,0) Initial Q-values:\n{:?}", qs.data());
}

// --- 7.4.3 Q学習とニューラルネットワーク ---
#[test]
fn test_7_4_3_q_learning_update() {
    let mut rng = StdRng::seed_from_u64(42);

    let q_net = MLP::new(&[12, 100, 4], relu, &mut rng);
    let mut optimizer = SGD::new(0.01);
    optimizer.setup(&q_net);
    let action = 0; // 例: UP (0)
    let reward = 1.0;
    let gamma = 0.9;

    let state_var = one_hot((2, 0), 4, 3);
    let next_state_var = one_hot((1, 0), 4, 3);
    // 1. 現在のQ値（全4行動）を予測
    let qs = q_net.forward(&state_var);
    let q_old_action = qs.data()[[0, action]];
    // 2. vol3の Gather を使って、選んだ行動のQ値だけを抽出 (qs[:, action] 相当)
    // 形状は [1, 4] から [1] になる
    let q = qs.gather(&[action]);
    // 3. 次状態の最大Q値を予測
    let next_qs = q_net.forward(&next_state_var);
    let next_q_max = next_qs
        .data()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    // 4. TDターゲットの計算 (スカラーを形状 [1] の Variable に)
    let target_val = reward + gamma * next_q_max;
    let target_var = Variable::new(
        Array::from_shape_vec(vec![1], vec![target_val])
            .unwrap()
            .into_dyn(),
    );
    // 5. 損失の計算と逆伝播
    q_net.cleargrads();
    let loss = mean_squared_error(&q, &target_var);
    loss.backward(false, false);
    // 6. 更新
    optimizer.update();

    let qs_new = q_net.forward(&state_var);
    let q_new_action = qs_new.data()[[0, action]];
    println!("Target TD value: {}", target_val);
    println!("Old Q-values: {:?}", qs.data());
    println!("New Q-values: {:?}", qs_new.data());
    // アサート: 選択した行動の Q 値だけは、確実にターゲットに近づいていることを保証する
    let old_diff = (q_old_action - target_val).abs();
    let new_diff = (q_new_action - target_val).abs();
    assert!(
        new_diff < old_diff,
        "Q-value did not move closer to target!"
    );

    // (出力結果の print を見ると、更新していない残りの3行動のQ値も
    // 隠れ層の共有によって微妙に変動していることが確認できます)
}

// ---------------------------------------------------------
// 7.4.3 QLearningNNAgent の学習と評価（本家 p.227〜230 相当）
// ---------------------------------------------------------
#[test]
fn test_7_4_3_q_learning_nn_agent_rollout() {
    let mut env = GridWorld::make_default();
    let mut rng = StdRng::seed_from_u64(42);

    // GridWorldの幅と高さ (3x4 = 12)
    let width = 4;
    let height = 3;
    let action_size = 4;
    // パラメータ (本家準拠: lr=0.01, gamma=0.9, epsilon=0.1)
    let mut agent =
        QLearningNNAgent::new(width * height, 100, action_size, 0.9, 0.01, 0.1, &mut rng);
    let episodes = 1000;

    // 学習ループ
    for _ in 0..episodes {
        let state = env.reset();
        let mut state_var = one_hot(state, width, height);
        loop {
            let action = agent.get_action(&state_var, &mut rng);
            let (next_state, reward, done) = env.step(action);
            let next_state_var = one_hot(next_state, width, height);

            agent.update(&state_var, action, reward, &next_state_var, done);

            if done {
                break;
            }
            state_var = next_state_var;
        }
    }
    // --- ロールアウト評価 ---
    let state = env.reset();
    let mut state_var = one_hot(state, width, height);
    let mut total_reward = 0.0;
    let mut steps = 0;

    // 学習完了後は ε=0 にして完全なGreedy行動で最適方策をたどる
    agent.epsilon = 0.0;

    let loop_state = loop {
        let action = agent.get_action(&state_var, &mut rng);
        let (next_state, reward, done) = env.step(action);
        let next_state_var = one_hot(next_state, width, height);

        total_reward += reward;
        steps += 1;

        if done || steps >= 20 {
            break done;
        }
        state_var = next_state_var;
    };

    assert!(
        loop_state,
        "Agent did not reach a terminal state within 20 steps!"
    );
    assert!(
        steps < 20,
        "Agent got stuck or didn't learn the optimal path!"
    );
    assert!(
        approx_eq(total_reward, 1.0, 1e-5),
        "Agent failed to reach the correct goal state!"
    );
}
