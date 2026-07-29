use rand::SeedableRng;
use rand::rngs::StdRng;
use vol4::cart_pole::CartPole;
use vol4::pg::{ActorCriticAgent, PGAgent};

fn run_pg_training(use_simple: bool, seed: u64, episodes: usize) -> f32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = CartPole::v0();
    let mut agent = PGAgent::new(4, &[128], 2, 0.98, 0.0002, &mut rng);
    let mut total_rewards = Vec::with_capacity(episodes);
    for episode in 1..=episodes {
        let mut state = env.reset(&mut rng);
        let mut total_reward = 0.0;
        loop {
            let (action, prob) = agent.get_action(&state, &mut rng);
            let (next_state, reward, done) = env.step(action);
            agent.add(reward, prob);
            total_reward += reward;
            state = next_state;
            if done {
                break;
            }
        }
        // フラグに応じて更新メソッドを切り替え
        if use_simple {
            agent.update_simple();
        } else {
            agent.update();
        }
        total_rewards.push(total_reward);
        // 途中経過を 100 エピソードごとに表示
        if episode % 100 == 0 {
            let avg_reward: f32 = total_rewards.iter().rev().take(100).sum::<f32>() / 100.0;
            println!(
                "  Episode {:>4}: Average Reward: {:.1}",
                episode, avg_reward
            );
        }
    }
    // 最後の100エピソードの平均報酬を計算
    let take_count = 100.min(total_rewards.len());
    let final_sum: f32 = total_rewards.iter().rev().take(take_count).sum();
    final_sum / take_count as f32
}

// ---------------------------------------------------------
// 9.1 PG の効果測定テスト
// 目的: Simple PG と REINFORCE の性能差を確認する
// 実行手順: cargo test test_9_1_pg_cartpole --release -- --ignored --nocapture
// ---------------------------------------------------------
#[test]
#[ignore]
fn test_9_1_pg_cartpole() {
    let seed = 42;

    let episodes = 3000;
    println!("Running Simple PG (update_simple)...");
    let simple_score = run_pg_training(true, seed, episodes);
    println!("Simple PG Final Average: {:.1}", simple_score);
    println!("Running REINFORCE (update)...");
    let reinforce_score = run_pg_training(false, seed, episodes);
    println!("REINFORCE Final Average: {:.1}", reinforce_score);

    // REINFORCE は G_t を使うことで分散が抑えられ、しっかり学習できる
    assert!(
        reinforce_score > 150.0,
        "REINFORCE should perform well, but got {}",
        reinforce_score
    );
    // 両者の間に明確な性能差（例えば 50 以上のスコア差）があることを保証
    assert!(
        reinforce_score > simple_score + 50.0,
        "REINFORCE must be significantly better than Simple PG"
    );
}

// ---------------------------------------------------------
// 9.4 Actor-Critic の効果測定テスト
// 実行手順: cargo test test_9_4_actor_critic --release -- --ignored --nocapture
// ---------------------------------------------------------
#[test]
#[ignore]
fn test_9_4_actor_critic() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut env = CartPole::v0();

    // lr_pi=0.0002, lr_v=0.0005 と本家通りに学習率を分ける
    let mut agent = ActorCriticAgent::new(4, &[128], 2, 0.98, 0.0002, 0.0005, &mut rng);

    let episodes = 3000;
    let mut total_rewards = Vec::with_capacity(episodes);

    for episode in 1..=episodes {
        let mut state = env.reset(&mut rng);
        let mut total_reward = 0.0;

        loop {
            // 方策(行動)と、その確率ノード(prob)を取り出す
            let (action, prob) = agent.get_action(&state, &mut rng);
            let (next_state, reward, done) = env.step(action);

            // ステップごとに即時学習 (メモリへの add は不要)
            agent.update(&state, &prob, reward, &next_state, done);

            total_reward += reward;
            state = next_state;

            if done {
                break;
            }
        }

        total_rewards.push(total_reward);

        if episode % 100 == 0 {
            let avg_reward: f32 = total_rewards.iter().rev().take(100).sum::<f32>() / 100.0;
            println!("Episode {:>4}: Average Reward: {:.1}", episode, avg_reward);
        }
    }

    // 検証: 最後の100エピソードの平均が十分高水準(例: 150以上)であることを確認
    let take_count = 100.min(total_rewards.len());
    let final_avg: f32 =
        total_rewards.iter().rev().take(take_count).sum::<f32>() / take_count as f32;
    assert!(
        final_avg > 150.0,
        "Actor-Critic should reach near 200, got {}",
        final_avg
    );
}
