use rand::SeedableRng;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::StdRng;
use vol4::dp::{policy_eval, random_policy};
use vol4::grid_world::{Action, GridWorld};
use vol4::mc::{McAgent, RandomAgent};

#[test]
fn test_mc_eval_vs_dp() {
    let mut env = GridWorld::make_default();
    let mut agent = RandomAgent::new(0.9);
    let mut rng = StdRng::seed_from_u64(42); // シード固定

    let episodes = 10000;

    // --- 1. MCによる状態価値の推定 ---
    for _ in 0..episodes {
        let mut state = env.reset();
        agent.reset();

        loop {
            let action = agent.get_action(state, &mut rng);
            let (next_state, reward, done) = env.step(action);

            agent.add(state, action, reward);

            if done {
                break;
            }
            state = next_state;
        }

        // エピソード終了後に価値を更新
        agent.eval();
    }

    // --- 2. DPによる厳密な状態価値（オラクル） ---
    let dp_pi = random_policy(&env);
    let dp_v = policy_eval(&dp_pi, &env, 0.9, 0.001);

    // --- 3. クロスチェック ---
    for state in env.states() {
        if env.is_goal(state) {
            continue; // ゴールの価値は0固定なので除外
        }

        let mc_val = *agent.v.get(&state).expect("MC should visit every state");
        let dp_val = *dp_v.get(&state).expect("All states are initialized");

        // 許容誤差 0.05 で一致することを確認(数シードで最悪 0.03 を観測、余裕をみて 0.05)
        assert!(
            (mc_val - dp_val).abs() < 0.05,
            "Value mismatch at {:?}: MC={:.3}, DP={:.3}",
            state,
            mc_val,
            dp_val
        );
    }
}

#[test]
fn test_mc_control() {
    use std::collections::HashMap;
    let mut env = GridWorld::make_default();
    let mut mc_agent = McAgent::new(0.9, 0.1, 0.1);
    let mut rng = StdRng::seed_from_u64(42);
    let episodes = 10000;
    // --- 1. 学習フェーズ ---
    for _ in 0..episodes {
        let mut state = env.reset();
        mc_agent.reset();
        loop {
            let action = mc_agent.get_action(state, &mut rng);
            let (next_state, reward, done) = env.step(action);
            mc_agent.add(state, action, reward);
            if done {
                break;
            }
            state = next_state;
        }
        mc_agent.eval();
    }
    // --- 2. 診断出力用（--nocapture で見る用） ---
    println!("--- Learned Policy (Diagnostic) ---");
    for state in env.states() {
        if env.is_goal(state) {
            continue;
        }
        if let Some(probs) = mc_agent.pi.get(&state) {
            let (best_action, max_prob) = probs
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            println!(
                "State {:?}: best action = {:?} (prob: {:.3})",
                state, best_action, max_prob
            );
        } else {
            println!("State {:?}: UNVISITED", state);
        }
    }
    println!("-----------------------------------");
    // --- 3. 期待方策テーブル（ロールアウトの軌跡上でのみ照合） ---
    let expected_policy: HashMap<_, _> = vec![
        ((0, 0), vec![Action::Right]),
        ((0, 1), vec![Action::Right]),
        ((0, 2), vec![Action::Right]),
        ((1, 0), vec![Action::Up]),
        ((1, 2), vec![Action::Up]),
        ((1, 3), vec![Action::Up]),
        ((2, 0), vec![Action::Right, Action::Up]), // タイを許容
        ((2, 1), vec![Action::Right]),
        ((2, 2), vec![Action::Up]),
        ((2, 3), vec![Action::Left]),
    ]
    .into_iter()
    .collect();
    // --- 4. 評価フェーズ（探索オフの Greedy ロールアウト） ---
    let mut state = env.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;
    let max_steps = 20; // 循環バグに対するフェイルセーフ
    let result = loop {
        // 学習済みの方策から、最も確率の高い（greedyな）行動を選ぶ
        let probs = mc_agent
            .pi
            .get(&state)
            .expect("Unvisited state on greedy path!");
        let (&action, _) = probs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        // 実際に通った場所だけ、局所的にテーブルと一致しているかをアサート
        if let Some(expected_actions) = expected_policy.get(&state) {
            assert!(
                expected_actions.contains(&action),
                "Rollout deviated at {:?}: expected one of {:?}, but took {:?}",
                state,
                expected_actions,
                action
            );
        }
        let (next_state, reward, done) = env.step(action);
        total_reward += reward;
        steps += 1;
        if done || steps >= max_steps {
            break done;
        }
        state = next_state;
    };
    // --- 5. 最終性能のアサート（ここが最も重要） ---
    assert!(result, "Agent did not reach the goal! (done={})", result);
    assert!(
        steps < max_steps,
        "Agent got stuck in a loop! (Steps >= {})",
        max_steps
    );
    assert_eq!(
        total_reward, 1.0,
        "Agent failed to find optimal path! (Reward was {}, expected 1.0. Stepped on bomb?)",
        total_reward
    );
}

#[test]
fn test_importance_sampling_variance() {
    let x: [f32; 3] = [1.0, 2.0, 3.0];
    let p: [f32; 3] = [0.1, 0.1, 0.8];

    let b1: [f32; 3] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; // πから遠い
    let b2: [f32; 3] = [0.2, 0.2, 0.6]; // πに近い
    // 重点サンプリングで期待値を 1 回推定するヘルパー関数
    let estimate_expectation = |b: &[f32; 3], rng: &mut StdRng| -> f32 {
        let n_samples = 100;
        let dist = WeightedIndex::new(b).unwrap();
        let mut sum = 0.0;

        for _ in 0..n_samples {
            // 挙動分布 b に従ってサンプリング
            let idx = dist.sample(rng);
            // 重点重み ρ = π(x) / b(x)
            let rho = p[idx] / b[idx];
            // ρ * x を足し込む
            sum += rho * x[idx];
        }
        sum / n_samples as f32
    };
    let mut rng = StdRng::seed_from_u64(42);
    let n_trials = 100; // 推定を100回繰り返して分散を測る
    let mut estimates_b1 = Vec::new();
    let mut estimates_b2 = Vec::new();
    for _ in 0..n_trials {
        estimates_b1.push(estimate_expectation(&b1, &mut rng));
        estimates_b2.push(estimate_expectation(&b2, &mut rng));
    }
    // 平均と分散を計算するクロージャ
    let calc_stats = |estimates: &[f32]| -> (f32, f32) {
        let mean = estimates.iter().sum::<f32>() / estimates.len() as f32;
        let variance =
            estimates.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / estimates.len() as f32;
        (mean, variance)
    };
    let (mean_b1, var_b1) = calc_stats(&estimates_b1);
    let (mean_b2, var_b2) = calc_stats(&estimates_b2);
    println!("True Expectation = 2.7");
    println!(
        "b1 (far)   : mean = {:.3}, variance = {:.3}",
        mean_b1, var_b1
    );
    println!(
        "b2 (close) : mean = {:.3}, variance = {:.3}",
        mean_b2, var_b2
    );
    // 1. 無偏性の確認（どちらの分布からサンプリングしても真の期待値2.7に近づく）
    assert!((mean_b1 - 2.7).abs() < 0.1);
    assert!((mean_b2 - 2.7).abs() < 0.1);
    // 2. 分散の比較（目標分布に近い b2 の方が、圧倒的に分散が小さい！）
    assert!(
        var_b2 < var_b1,
        "Variance of b2 ({:.3}) should be strictly less than b1 ({:.3})",
        var_b2,
        var_b1
    );
}
