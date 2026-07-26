use ndarray::Array1;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use vol4::bandit::{Agent, AlphaAgent, Bandit, NonStatBandit};
use vol4::utils::argmax;

#[test]
fn test_simulation() {
    let arms = 10;
    let steps = 1000;
    let epsilon = 0.1;
    // 1. テスト側でシードを固定し、各腕の勝率（真の期待値）をランダム生成
    let mut rng = StdRng::seed_from_u64(42);
    let mut rates = Array1::zeros(arms);
    for i in 0..arms {
        rates[i] = rng.random::<f32>(); // 0.0 ~ 1.0
    }

    // 真の最良腕（argmax）を事前に求めておく
    let best_arm = argmax(&rates).unwrap();
    let mut bandit = Bandit::new(rates);
    let mut agent = Agent::new(epsilon, arms);

    let mut total_reward = 0.0;
    let mut rates_history = Vec::with_capacity(steps);
    let mut action_counts = vec![0; arms];
    // 2. 1000ステップの学習ループ
    for step in 0..steps {
        // Agentが行動を選び、Banditから報酬を得て、Agentが学習する
        let action = agent.get_action(&mut rng);
        action_counts[action] += 1;

        let reward = bandit.play(action, &mut rng);
        agent.update(action, reward);

        // 累積報酬とステップごとの平均勝率を記録
        total_reward += reward;
        rates_history.push(total_reward / (step + 1) as f32);
    }
    // 検証用: 実際に一番多く選ばれた腕を取得
    let most_frequent_arm = action_counts
        .iter()
        .enumerate()
        .max_by_key(|a| a.1)
        .map(|(i, _)| i)
        .unwrap();

    // 値を出力して確認する場合は cargo test -- --nocapture を使います
    println!("Best arm: {}", best_arm);
    println!("Most frequent arm: {}", most_frequent_arm);
    println!("Final win rate: {}", rates_history.last().unwrap());
    // 3. 学習が成功していることの Assert
    // ① Agentが最も多く選んだ腕が、真の最良腕と一致しているか
    assert_eq!(
        best_arm, most_frequent_arm,
        "Agent should learn to choose the best arm most frequently"
    );

    // ② 最終的な勝率が十分に高いか
    assert!(
        *rates_history.last().unwrap() > 0.9,
        "Final win rate should be reasonably high"
    );
}

fn run_simulation(epsilon: f32, arms: usize, steps: usize, seed: u64) -> Array1<f32> {
    // 実行(run)番号をシードにして再現性を持たせる
    let mut rng = StdRng::seed_from_u64(seed);

    let mut rates = Array1::zeros(arms);
    for i in 0..arms {
        rates[i] = rng.random::<f32>();
    }
    let mut bandit = Bandit::new(rates);
    let mut agent = Agent::new(epsilon, arms);

    let mut total_reward = 0.0;
    let mut rates_history = Array1::zeros(steps);
    for step in 0..steps {
        let action = agent.get_action(&mut rng);
        let reward = bandit.play(action, &mut rng);
        agent.update(action, reward);

        total_reward += reward;
        rates_history[step] = total_reward / (step + 1) as f32;
    }

    rates_history
}

// 指定した ε で runs 回シミュレーションを回し、勝率カーブの「平均」を返す関数
fn get_average_rates(epsilon: f32, arms: usize, steps: usize, runs: usize) -> Array1<f32> {
    let mut all_rates = Array1::zeros(steps);
    for run in 0..runs {
        let history = run_simulation(epsilon, arms, steps, run as u64);
        // ndarray を使えば、配列同士の足し算が一括で書けます
        all_rates += &history;
    }
    // 合計を回数で割って平均化
    all_rates / (runs as f32)
}

#[test]
fn test_average_behavior() {
    let arms = 10;
    let steps = 1000;
    let runs = 200; // 本の通り 200 回回す
    // --- ① ε = 0.1 の平均的な性質のテスト ---
    let avg_rates_01 = get_average_rates(0.1, arms, steps, runs);
    let final_rate_01 = *avg_rates_01.last().unwrap();

    // 200回平均ならブレが少ないため、少し強気なしきい値（例：0.7）を置けます
    println!("ε=0.1 Final average win rate: {}", final_rate_01);
    assert!(
        final_rate_01 > 0.7,
        "Average final win rate for ε=0.1 should be high enough"
    );
    // --- ② 【余力】 ε = 0.01 / 0.1 / 0.3 の比較（本家の結論検証） ---
    let avg_rates_001 = get_average_rates(0.01, arms, steps, runs);
    let avg_rates_03 = get_average_rates(0.3, arms, steps, runs);

    let final_rate_001 = *avg_rates_001.last().unwrap();
    let final_rate_03 = *avg_rates_03.last().unwrap();
    println!("ε=0.01 Final average win rate: {}", final_rate_001);
    println!("ε=0.30 Final average win rate: {}", final_rate_03);
    // 本書の結論：「1000ステップ時点では ε=0.1 が他の2つに勝つ」
    // 0.3 は探索しすぎてスコアが伸び悩み、0.01 は探索が足りずにアタリを見つけきれない
    assert!(
        final_rate_01 > final_rate_001,
        "At step 1000, ε=0.1 should be better than ε=0.01"
    );
    assert!(
        final_rate_01 > final_rate_03,
        "At step 1000, ε=0.1 should be better than ε=0.3"
    );
}

fn run_nonstat_agent(epsilon: f32, arms: usize, steps: usize, seed: u64) -> Array1<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rates = Array1::zeros(arms);
    for i in 0..arms {
        rates[i] = rng.random::<f32>();
    }

    let mut bandit = NonStatBandit::new(rates);
    let mut agent = Agent::new(epsilon, arms); // 標本平均

    let mut total_reward = 0.0;
    let mut rates_history = Array1::zeros(steps);
    for step in 0..steps {
        let action = agent.get_action(&mut rng);
        let reward = bandit.play(action, &mut rng);
        agent.update(action, reward);

        total_reward += reward;
        rates_history[step] = total_reward / (step + 1) as f32;
    }
    rates_history
}

// --- AlphaAgent用の非定常シミュレーション ---
fn run_nonstat_alpha(
    epsilon: f32,
    alpha: f32,
    arms: usize,
    steps: usize,
    seed: u64,
) -> Array1<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rates = Array1::zeros(arms);
    for i in 0..arms {
        rates[i] = rng.random::<f32>();
    }

    let mut bandit = NonStatBandit::new(rates);
    let mut agent = AlphaAgent::new(epsilon, alpha, arms); // 固定値α

    // （ループの中身は上の関数と全く同じ）
    let mut total_reward = 0.0;
    let mut rates_history = Array1::zeros(steps);
    for step in 0..steps {
        let action = agent.get_action(&mut rng);
        let reward = bandit.play(action, &mut rng);
        agent.update(action, reward);

        total_reward += reward;
        rates_history[step] = total_reward / (step + 1) as f32;
    }
    rates_history
}

#[test]
fn test_non_stationary_comparison() {
    let arms = 10;
    let steps = 1000;
    let runs = 200;
    let epsilon = 0.1;
    let alpha = 0.8; // 本家のデフォルト値
    let mut avg_rates_sample = Array1::zeros(steps);
    let mut avg_rates_alpha = Array1::zeros(steps);
    // 200回のシミュレーションを回して平均を取る
    for run in 0..runs {
        avg_rates_sample += &run_nonstat_agent(epsilon, arms, steps, run as u64);
        avg_rates_alpha += &run_nonstat_alpha(epsilon, alpha, arms, steps, run as u64);
    }
    avg_rates_sample /= runs as f32;
    avg_rates_alpha /= runs as f32;
    let final_sample = *avg_rates_sample.last().unwrap();
    let final_alpha = *avg_rates_alpha.last().unwrap();
    println!("Sample Average (Agent) Final Win Rate: {}", final_sample);
    println!("Alpha = 0.8 (AlphaAgent) Final Win Rate: {}", final_alpha);
    // 本の結論：非定常問題では AlphaAgent の方が適応力が高く、勝率が高くなる
    assert!(
        final_alpha > final_sample,
        "AlphaAgent should outperform standard Agent in non-stationary environment"
    );
}
