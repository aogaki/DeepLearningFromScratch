use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use vol4::cart_pole::{CartPole, CartPoleAction};
use vol4::dqn::{DQNAgent, Experience, ReplayBuffer};

// ---------------------------------------------------------
// 8.1.2 ランダムなエージェント (gym_play.py 相当)
// ---------------------------------------------------------
#[test]
fn test_8_1_2_random_agent() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut env = CartPole::v0();

    let mut total_reward = 0.0;
    let mut steps = 0;

    // 取るべき行動の選択肢 [Left, Right]
    let actions = CartPoleAction::all();

    loop {
        // ランダムに行動を選択 (本家の action = env.action_space.sample() に相当)
        let action = *actions.choose(&mut rng).unwrap();

        let (next_state, reward, done) = env.step(action);

        total_reward += reward;
        steps += 1;

        if done {
            println!(
                "Episode finished after {} steps. Total Reward: {}",
                steps, total_reward
            );
            println!("Final state: {:?}", next_state);
            break;
        }
    }

    // CartPole-v0 において完全なランダム行動をとった場合、通常は10〜30ステップ程度でポールが倒れて終了します。
    // （シード 42 の場合は 10 数ステップで終了するはずです）
    assert!(
        steps > 5 && steps < 50,
        "Random agent steps should be small, usually ~10-20."
    );

    // CartPole はバランスを保っている間、1ステップごとに報酬 1.0 がもらえる仕様です。
    // f32 の整数回加算は誤差が生じないため、ステップ数と報酬の合計は完全に一致するはずです。
    assert_eq!(
        total_reward, steps as f32,
        "Reward should exactly equal the number of steps."
    );
}

// ループ本体を括り出したヘルパー関数
// (訓練中の最大報酬, 学習完了後のGreedy報酬) を返す
fn run_dqn_training(seed: u64) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = CartPole::v0();

    // ハイパーパラメータ (本家 ch08/dqn.py 準拠)
    let episodes = 300;
    let sync_interval = 20;
    let buffer_size = 10000;
    let batch_size = 32;
    let gamma = 0.98;
    let lr = 0.0005;
    let epsilon = 0.1;
    let action_size = 2;
    let state_size = 4;
    let hidden_sizes = &[128, 128];
    let mut agent = DQNAgent::new(
        state_size,
        hidden_sizes,
        action_size,
        gamma,
        lr,
        epsilon,
        &mut rng,
    );
    let mut buffer = ReplayBuffer::<Experience>::new(buffer_size, batch_size);

    let mut max_reward_so_far: f32 = 0.0;
    // --- 訓練ループ ---
    for episode in 1..=episodes {
        let mut state = env.reset(&mut rng);
        let mut total_reward: f32 = 0.0;
        loop {
            let action = agent.get_action(&state, &mut rng);
            let (next_state, reward, done) = env.step(action);

            buffer.add((state, action, reward, next_state, done));

            if buffer.len() >= buffer.batch_size {
                let batch = buffer.get_batch(&mut rng);
                agent.update(&batch);
            }

            total_reward += reward;

            if done {
                break;
            }
            state = next_state;
        }
        if episode % sync_interval == 0 {
            agent.sync_qnet();
        }
        max_reward_so_far = max_reward_so_far.max(total_reward);
    }
    // --- ロールアウト評価 (本家 p.250 の締め) ---
    agent.epsilon = 0.0; // 完全な Greedy 方策に切り替え
    let mut state = env.reset(&mut rng);
    let mut greedy_reward = 0.0;

    loop {
        let action = agent.get_action(&state, &mut rng);
        let (next_state, reward, done) = env.step(action);
        greedy_reward += reward;

        if done {
            break;
        }
        state = next_state;
    }
    (max_reward_so_far, greedy_reward)
}
// ---------------------------------------------------------
// 8.2.5 DQNの複数シード評価テスト
// 実行手順: cargo test test_8_2_5_dqn_multi_seed --release -- --ignored --nocapture
// ---------------------------------------------------------
#[test]
#[ignore]
fn test_8_2_5_dqn_multi_seed() {
    let seeds = [42, 43, 44, 45, 46, 303, 404, 606, 808, 909];
    let mut success_count = 0;

    println!("Running DQN training on {} seeds...", seeds.len());

    for &seed in &seeds {
        let (max_train, greedy) = run_dqn_training(seed);
        println!(
            "Seed {}: Max Train Reward = {}, Final Greedy Reward = {}",
            seed, max_train, greedy
        );

        // 最終的な Greedy スコアが 150 以上（または満点の 200）なら成功とみなす
        if greedy >= 150.0 {
            success_count += 1;
        }
    }

    println!("Success rate: {} / {}", success_count, seeds.len());

    // 強化学習の分散を考慮し、過半数（6/10）が成功すれば実装は健全と判定する
    assert!(
        success_count >= 6,
        "DQN implementation failed. Only {} out of {} seeds learned successfully.",
        success_count,
        seeds.len()
    );
}
