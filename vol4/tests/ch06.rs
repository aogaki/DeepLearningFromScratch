use rand::SeedableRng;
use rand::rngs::StdRng;
use vol4::dp::{policy_eval, random_policy};
use vol4::grid_world::{Action, GridWorld};
use vol4::td::{OffPolicySarsaAgent, QLearningAgent, SarsaAgent, TdAgent};

#[test]
fn test_td_eval_vs_dp() {
    let mut env = GridWorld::make_default();

    // DP側のランダム方策を生成し、それをTdAgentに評価させる
    let dp_pi = random_policy(&env);

    // 学習率 alpha を MC の時より小さめ(0.01)にして、変動をマイルドに収束させる
    let mut agent = TdAgent::new(0.9, 0.01, dp_pi.clone());
    let mut rng = StdRng::seed_from_u64(42);

    let episodes = 10000;

    for _ in 0..episodes {
        let mut state = env.reset();

        loop {
            let action = agent.get_action(state, &mut rng);
            let (next_state, reward, done) = env.step(action);

            // エピソード末尾を待たず、1歩ごとにオンライン更新！
            agent.update(state, reward, next_state, done);

            if done {
                break;
            }
            state = next_state;
        }
    }

    // オラクル（DP）とのクロスチェック
    let dp_v = policy_eval(&dp_pi, &env, 0.9, 0.001);

    for state in env.states() {
        if env.is_goal(state) {
            continue;
        }

        let td_val = *agent.v.get(&state).expect("TD should visit every state");
        let dp_val = *dp_v
            .get(&state)
            .expect("DP V is initialized for all states");

        assert!(
            (td_val - dp_val).abs() < 0.05,
            "Value mismatch at {:?}: TD={:.3}, DP={:.3}",
            state,
            td_val,
            dp_val
        );
    }
}

#[test]
fn test_sarsa_control() {
    let mut env = GridWorld::make_default();
    let mut sarsa_agent = SarsaAgent::new(0.9, 0.8, 0.1);
    let mut rng = StdRng::seed_from_u64(42);
    let episodes = 10000;
    // --- 1. 学習フェーズ ---
    for _ in 0..episodes {
        let mut state = env.reset();
        sarsa_agent.reset(); // エピソード開始時に必ず記憶をクリア

        loop {
            let action = sarsa_agent.get_action(state, &mut rng);
            let (next_state, reward, done) = env.step(action);

            // オンラインで更新 (SARSA)
            sarsa_agent.update(state, action, reward, done);

            if done {
                break;
            }
            state = next_state;
        }
    }
    // --- 2. 評価フェーズ（探索オフの Greedy ロールアウト） ---
    let mut state = env.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;
    let max_steps = 20;
    let loop_state = loop {
        let probs = sarsa_agent
            .pi
            .get(&state)
            .expect("Unvisited state on greedy path!");
        let (&action, _) = probs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (next_state, reward, done) = env.step(action);
        total_reward += reward;
        steps += 1;
        if done || steps >= max_steps {
            break done;
        }
        state = next_state;
    };
    assert!(
        loop_state,
        "SarsaAgent did not reach the goal state within {} steps!",
        max_steps
    );
    assert!(
        steps < max_steps,
        "SarsaAgent got stuck in a loop! (Steps >= {})",
        max_steps
    );
    assert_eq!(
        total_reward, 1.0,
        "SarsaAgent failed to find optimal path! (Reward was {})",
        total_reward
    );
}

#[test]
fn test_off_policy_sarsa_control() {
    let mut env = GridWorld::make_default();
    // 本家は α=0.8 だが、IS の重み ρ が探索行動で 0 になり target=0 の破壊的更新が 7.5% の頻度で走るため、α=0.8 では Q が安定せずロールアウトが落ちる。分散を平均化するため α=0.1 に下げた
    let mut agent = OffPolicySarsaAgent::new(0.9, 0.1, 0.1);
    let mut rng = StdRng::seed_from_u64(42);
    let episodes = 10000;
    // --- 1. 学習フェーズ（挙動方策 b で探索） ---
    for _ in 0..episodes {
        let mut state = env.reset();
        agent.reset();

        loop {
            // get_action は内部で `b` からサンプリングしている
            let action = agent.get_action(state, &mut rng);
            let (next_state, reward, done) = env.step(action);

            agent.update(state, action, reward, done);

            if done {
                break;
            }
            state = next_state;
        }
    }
    // --- 2. 評価フェーズ（目標方策 pi による Greedy ロールアウト） ---
    let mut state = env.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;
    let max_steps = 20;
    let loop_state = loop {
        // オフポリシーSARSAが学習した「目標方策」pi に従う
        let probs = agent
            .pi
            .get(&state)
            .expect("Unvisited state on greedy path!");
        let (&action, _) = probs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (next_state, reward, done) = env.step(action);
        total_reward += reward;
        steps += 1;
        if done || steps >= max_steps {
            break done;
        }
        state = next_state;
    };
    assert!(
        loop_state,
        "Agent did not reach the goal state within {} steps!",
        max_steps
    );
    assert!(
        steps < max_steps,
        "Agent got stuck in a loop! (Steps >= {})",
        max_steps
    );
    assert_eq!(
        total_reward, 1.0,
        "Agent failed to find optimal path! (Reward was {})",
        total_reward
    );
}

// ヘルパー: Q値から純粋なGreedy行動を引く（ロールアウト用）
fn get_greedy_action(agent: &QLearningAgent, state: (usize, usize)) -> Action {
    let q_s = agent
        .q
        .get(&state)
        .expect("Unvisited state on greedy path!");
    *Action::all()
        .iter()
        .max_by(|&&a1, &&a2| {
            let q1 = q_s.get(&a1).unwrap_or(&0.0);
            let q2 = q_s.get(&a2).unwrap_or(&0.0);
            q1.partial_cmp(q2).unwrap()
        })
        .unwrap()
}
// ---------------------------------------------------------
// 1. 本家の再現：ε-greedy による Q学習
// ---------------------------------------------------------
#[test]
fn test_q_learning_control_epsilon_greedy() {
    let mut env = GridWorld::make_default();
    let mut agent = QLearningAgent::new(0.9, 0.8, 0.1);
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10000 {
        let mut state = env.reset();
        loop {
            let action = agent.get_action(state, &mut rng);
            let (next_state, reward, done) = env.step(action);
            agent.update(state, action, reward, next_state, done);
            if done {
                break;
            }
            state = next_state;
        }
    }
    let mut state = env.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;
    let loop_state = loop {
        let action = get_greedy_action(&agent, state);
        let (next_state, reward, done) = env.step(action);
        total_reward += reward;
        steps += 1;
        if done || steps >= 20 {
            break done;
        }
        state = next_state;
    };
    assert!(
        loop_state,
        "Agent did not reach the goal state within 20 steps!"
    );
    assert!(steps < 20);
    assert_eq!(total_reward, 1.0);
}
// ---------------------------------------------------------
// 2. Q学習の真髄：完全ランダム（ε=1.0）でも最適方策を獲得できる証明
// ---------------------------------------------------------
#[test]
fn test_q_learning_control_pure_random() {
    let mut env = GridWorld::make_default();
    // ε=1.0（完全なランダム徘徊）で行動させる
    let mut agent = QLearningAgent::new(0.9, 0.8, 1.0);
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10000 {
        let mut state = env.reset();
        loop {
            let action = agent.get_action(state, &mut rng);
            let (next_state, reward, done) = env.step(action);
            agent.update(state, action, reward, next_state, done);
            if done {
                break;
            }
            state = next_state;
        }
    }
    let mut state = env.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;
    let loop_state = loop {
        // ロールアウト時はQ値から最適行動を選ぶ
        let action = get_greedy_action(&agent, state);
        let (next_state, reward, done) = env.step(action);
        total_reward += reward;
        steps += 1;
        if done || steps >= 20 {
            break done;
        }
        state = next_state;
    };
    assert!(
        loop_state,
        "Agent did not reach the goal state within 20 steps!"
    );
    assert!(steps < 20, "Agent got stuck despite pure random training");
    assert_eq!(
        total_reward, 1.0,
        "Agent failed optimal path despite pure random training"
    );
}
