//! Phase 2.1: Multi-seed evaluation and aggregated reporting.
//!
//! Provides infrastructure for running experiments across multiple seeds
//! and computing aggregate statistics (mean ± std).

/// Per-seed run metrics.
#[derive(Clone, Debug, Default)]
pub struct SeedRun {
    pub seed: u64,

    // Core metrics
    pub coverage_pos: f64,
    pub selective_accuracy: f64,
    pub false_positive: f64,
    pub stable_share: f64,

    // Mode rates
    pub explore_rate: f64,
    pub exploit_rate: f64,
    pub reset_rate: f64,

    // Action rates
    pub scan_rate: f64,
    pub focus_rate: f64,
    pub perturb_rate: f64,

    // Regret/recovery metrics (optional)
    pub regret_rate: Option<f64>,
    pub recovery_improve: Option<f64>,
    pub bad_state_share: Option<f64>,
}

impl SeedRun {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }
}

/// Aggregated statistics across seeds.
#[derive(Clone, Debug, Default)]
pub struct Aggregate {
    pub num_seeds: usize,

    // Core metrics (mean, std)
    pub coverage_pos_mean: f64,
    pub coverage_pos_std: f64,
    pub selective_accuracy_mean: f64,
    pub selective_accuracy_std: f64,
    pub false_positive_mean: f64,
    pub false_positive_std: f64,
    pub stable_share_mean: f64,
    pub stable_share_std: f64,

    // Mode rates (mean, std)
    pub explore_rate_mean: f64,
    pub explore_rate_std: f64,
    pub exploit_rate_mean: f64,
    pub exploit_rate_std: f64,
    pub reset_rate_mean: f64,
    pub reset_rate_std: f64,

    // Action rates (mean, std)
    pub scan_rate_mean: f64,
    pub scan_rate_std: f64,
    pub focus_rate_mean: f64,
    pub focus_rate_std: f64,
    pub perturb_rate_mean: f64,
    pub perturb_rate_std: f64,

    // Regret/recovery (mean, std) - optional
    pub regret_rate_mean: Option<f64>,
    pub regret_rate_std: Option<f64>,
    pub recovery_improve_mean: Option<f64>,
    pub recovery_improve_std: Option<f64>,
    pub bad_state_share_mean: Option<f64>,
    pub bad_state_share_std: Option<f64>,

    // Seeds that failed regression guard
    pub failed_seeds: Vec<u64>,
}

/// Compute mean of values.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute standard deviation of values.
fn std_dev(values: &[f64], mean_val: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance: f64 = values.iter()
        .map(|&x| (x - mean_val).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Compute mean and std for optional values.
fn mean_std_opt(values: &[Option<f64>]) -> (Option<f64>, Option<f64>) {
    let valid: Vec<f64> = values.iter().filter_map(|&x| x).collect();
    if valid.is_empty() {
        return (None, None);
    }
    let m = mean(&valid);
    let s = std_dev(&valid, m);
    (Some(m), Some(s))
}

/// Aggregate metrics across multiple seed runs.
pub fn aggregate(runs: &[SeedRun]) -> Aggregate {
    if runs.is_empty() {
        return Aggregate::default();
    }

    let n = runs.len();

    // Extract arrays
    let coverage_pos: Vec<f64> = runs.iter().map(|r| r.coverage_pos).collect();
    let selective_accuracy: Vec<f64> = runs.iter().map(|r| r.selective_accuracy).collect();
    let false_positive: Vec<f64> = runs.iter().map(|r| r.false_positive).collect();
    let stable_share: Vec<f64> = runs.iter().map(|r| r.stable_share).collect();

    let explore_rate: Vec<f64> = runs.iter().map(|r| r.explore_rate).collect();
    let exploit_rate: Vec<f64> = runs.iter().map(|r| r.exploit_rate).collect();
    let reset_rate: Vec<f64> = runs.iter().map(|r| r.reset_rate).collect();

    let scan_rate: Vec<f64> = runs.iter().map(|r| r.scan_rate).collect();
    let focus_rate: Vec<f64> = runs.iter().map(|r| r.focus_rate).collect();
    let perturb_rate: Vec<f64> = runs.iter().map(|r| r.perturb_rate).collect();

    let regret_rate: Vec<Option<f64>> = runs.iter().map(|r| r.regret_rate).collect();
    let recovery_improve: Vec<Option<f64>> = runs.iter().map(|r| r.recovery_improve).collect();
    let bad_state_share: Vec<Option<f64>> = runs.iter().map(|r| r.bad_state_share).collect();

    // Compute means
    let coverage_pos_mean = mean(&coverage_pos);
    let selective_accuracy_mean = mean(&selective_accuracy);
    let false_positive_mean = mean(&false_positive);
    let stable_share_mean = mean(&stable_share);

    let explore_rate_mean = mean(&explore_rate);
    let exploit_rate_mean = mean(&exploit_rate);
    let reset_rate_mean = mean(&reset_rate);

    let scan_rate_mean = mean(&scan_rate);
    let focus_rate_mean = mean(&focus_rate);
    let perturb_rate_mean = mean(&perturb_rate);

    // Compute stds
    let coverage_pos_std = std_dev(&coverage_pos, coverage_pos_mean);
    let selective_accuracy_std = std_dev(&selective_accuracy, selective_accuracy_mean);
    let false_positive_std = std_dev(&false_positive, false_positive_mean);
    let stable_share_std = std_dev(&stable_share, stable_share_mean);

    let explore_rate_std = std_dev(&explore_rate, explore_rate_mean);
    let exploit_rate_std = std_dev(&exploit_rate, exploit_rate_mean);
    let reset_rate_std = std_dev(&reset_rate, reset_rate_mean);

    let scan_rate_std = std_dev(&scan_rate, scan_rate_mean);
    let focus_rate_std = std_dev(&focus_rate, focus_rate_mean);
    let perturb_rate_std = std_dev(&perturb_rate, perturb_rate_mean);

    // Optional metrics
    let (regret_rate_mean, regret_rate_std) = mean_std_opt(&regret_rate);
    let (recovery_improve_mean, recovery_improve_std) = mean_std_opt(&recovery_improve);
    let (bad_state_share_mean, bad_state_share_std) = mean_std_opt(&bad_state_share);

    // Find seeds that fail regression guard
    let failed_seeds: Vec<u64> = runs.iter()
        .filter(|r| r.coverage_pos < 0.70 || r.selective_accuracy < 0.80 || r.false_positive > 0.0)
        .map(|r| r.seed)
        .collect();

    Aggregate {
        num_seeds: n,
        coverage_pos_mean,
        coverage_pos_std,
        selective_accuracy_mean,
        selective_accuracy_std,
        false_positive_mean,
        false_positive_std,
        stable_share_mean,
        stable_share_std,
        explore_rate_mean,
        explore_rate_std,
        exploit_rate_mean,
        exploit_rate_std,
        reset_rate_mean,
        reset_rate_std,
        scan_rate_mean,
        scan_rate_std,
        focus_rate_mean,
        focus_rate_std,
        perturb_rate_mean,
        perturb_rate_std,
        regret_rate_mean,
        regret_rate_std,
        recovery_improve_mean,
        recovery_improve_std,
        bad_state_share_mean,
        bad_state_share_std,
        failed_seeds,
    }
}

/// Print per-seed table.
pub fn print_per_seed_table(runs: &[SeedRun]) {
    println!("  Per-seed results:");
    println!("  {:>10} | {:>8} | {:>8} | {:>6} | {:>7} | {:>7} | {:>7}",
        "seed", "cov%", "sel_acc%", "FP%", "scan%", "focus%", "pert%");
    println!("  {}", "-".repeat(72));

    for run in runs {
        println!("  {:>10} | {:>7.1}% | {:>7.1}% | {:>5.1}% | {:>6.1}% | {:>6.1}% | {:>5.2}%",
            format!("{:08X}", run.seed),
            run.coverage_pos * 100.0,
            run.selective_accuracy * 100.0,
            run.false_positive * 100.0,
            run.scan_rate * 100.0,
            run.focus_rate * 100.0,
            run.perturb_rate * 100.0);
    }
}

/// Print aggregate table with mean ± std.
pub fn print_aggregate_table(label: &str, agg: &Aggregate) {
    println!("  {} (n={}):", label, agg.num_seeds);
    println!("  {:>18} | {:>15}", "Metric", "Mean ± Std");
    println!("  {}", "-".repeat(38));

    println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
        "coverage_pos", agg.coverage_pos_mean * 100.0, agg.coverage_pos_std * 100.0);
    println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
        "selective_accuracy", agg.selective_accuracy_mean * 100.0, agg.selective_accuracy_std * 100.0);
    println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
        "false_positive", agg.false_positive_mean * 100.0, agg.false_positive_std * 100.0);
    println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
        "stable_share", agg.stable_share_mean * 100.0, agg.stable_share_std * 100.0);
    println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
        "explore_rate", agg.explore_rate_mean * 100.0, agg.explore_rate_std * 100.0);
    println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
        "exploit_rate", agg.exploit_rate_mean * 100.0, agg.exploit_rate_std * 100.0);
    println!("  {:>18} | {:>6.2}% ± {:>5.2}%",
        "reset_rate", agg.reset_rate_mean * 100.0, agg.reset_rate_std * 100.0);
    println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
        "scan_rate", agg.scan_rate_mean * 100.0, agg.scan_rate_std * 100.0);
    println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
        "focus_rate", agg.focus_rate_mean * 100.0, agg.focus_rate_std * 100.0);
    println!("  {:>18} | {:>6.2}% ± {:>5.2}%",
        "perturb_rate", agg.perturb_rate_mean * 100.0, agg.perturb_rate_std * 100.0);

    if let Some(rr) = agg.regret_rate_mean {
        let std = agg.regret_rate_std.unwrap_or(0.0);
        println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
            "regret_rate", rr * 100.0, std * 100.0);
    }
    if let Some(ri) = agg.recovery_improve_mean {
        let std = agg.recovery_improve_std.unwrap_or(0.0);
        println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
            "recovery_improve", ri * 100.0, std * 100.0);
    }
    if let Some(bs) = agg.bad_state_share_mean {
        let std = agg.bad_state_share_std.unwrap_or(0.0);
        println!("  {:>18} | {:>6.1}% ± {:>5.1}%",
            "bad_state_share", bs * 100.0, std * 100.0);
    }

    if !agg.failed_seeds.is_empty() {
        println!();
        println!("  ⚠ Failed seeds: {:?}", agg.failed_seeds.iter()
            .map(|s| format!("{:08X}", s))
            .collect::<Vec<_>>());
    }
}

/// Check if aggregate passes regression guard.
pub fn passes_regression_guard(agg: &Aggregate) -> bool {
    agg.coverage_pos_mean >= 0.70
        && agg.selective_accuracy_mean >= 0.80
        && agg.false_positive_mean < 0.001 // Use epsilon for floating-point comparison
}
