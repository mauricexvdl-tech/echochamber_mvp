//! Demo 13: Phase 2.1 - Multi-seed evaluation + lift metrics
//! Phase 2.1b: Adds seed-robust policy stabilization with guardrails.
//!
//! Runs experiments across multiple seeds with aggregated reporting.
//! Compares FULL policy against RANDOM_BUDGETED baseline.

use crate::action::{Action, ActionConfig, ActionPolicy};
use crate::anchor::{
    AnchorBank, ConfidenceInfo, GateParams, KeyedMemoryConfig, KeyedMemoryMetrics,
    KeyedMemoryStore, KeyedRecallDecision, MemoryKey, ANCHOR_MARGIN_MIN,
};
use crate::causes::{get_top_k, Causes};
use crate::config::Config;
use crate::echo::EchoChamber;
use crate::lift::{self, LiftConfig, LiftStats};
use crate::memory::RollingWindow;
use crate::mode::{Mode, ModePolicy, ModePolicyConfig};
use crate::multiseed::{self, SeedRun};
use crate::results::{
    write_json, AcceptanceResult, AggregateResult, Demo13Result, LiftResult, ResultMeta,
    SeedRunResult, VariantResult,
};
use crate::rng::Rng;

/// Phase 2.1b/c: Per-seed diagnostics for collapse detection.
#[derive(Clone, Debug, Default)]
pub struct SeedDiagnostics {
    pub seed: u64,
    // Distribution stats
    pub proto_align_mean: f32,
    pub proto_align_p50: f32,
    pub proto_align_p90: f32,
    pub margin_mean: f64,
    pub margin_p50: f64,
    pub margin_p90: f64,
    // Rates
    pub stable_share: f64,
    pub bad_state_share: f64,
    pub explore_rate: f64,
    pub exploit_rate: f64,
    pub reset_rate: f64,
    // Collapse indicators
    pub explore_streak_max: u32,
    pub exploit_streak_max: u32,
    pub gate_fail_streak_max: u32,
    pub rescue_count: usize,
    // Adaptive thresholds used
    pub adaptive_proto_min: f32,
    pub adaptive_margin_min: f64,
    // Phase 2.1c: Thrash metrics
    pub rescues_per_10k: f64,
    pub post_rescue_lock_share: f64,
    pub total_ticks: usize,
    // Phase 2.1h: Chronic clamp metrics
    pub chronic_lock_share: f64,
    pub chronic_lock_total_ticks: usize,
    pub chronic_enter_count: u32,
    pub chronic_exit_count: u32,
    pub chronic_enter_by_bad: u32,
    pub chronic_enter_by_unstable: u32,
    pub chronic_exit_by_watchdog: u32,
    // Phase 2.1h: EMA shares at end of run
    pub chronic_stable_ema_final: f32,
    pub chronic_bad_ema_final: f32,
    // Phase 2.1e: Perturb rate
    pub perturb_rate: f64,
    // Phase 2.1k: Exploit quality metrics
    pub exploit_hard_count: usize,
    pub exploit_soft_count: usize,
    pub exploit_soft_share: f64,
    pub bad_in_explore_share: f64,
    pub bad_in_exploit_share: f64,
}

/// Phase 2.1b: Warmup stats collector for adaptive thresholds.
#[derive(Clone, Debug)]
struct WarmupStats {
    proto_samples: Vec<f32>,
    margin_samples: Vec<f64>,
}

impl WarmupStats {
    fn new() -> Self {
        Self {
            proto_samples: Vec::with_capacity(5000),
            margin_samples: Vec::with_capacity(5000),
        }
    }

    fn push(&mut self, proto: f32, margin: f64) {
        if self.proto_samples.len() < 5000 {
            self.proto_samples.push(proto);
            self.margin_samples.push(margin);
        }
    }

    fn compute_p50(&self) -> (f32, f64) {
        if self.proto_samples.is_empty() {
            return (0.0, 0.0);
        }

        let mut proto_sorted = self.proto_samples.clone();
        let mut margin_sorted = self.margin_samples.clone();
        proto_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        margin_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = proto_sorted.len() / 2;
        (proto_sorted[mid], margin_sorted[mid])
    }
}

/// Print per-seed diagnostics table.
fn print_diagnostics_table(diagnostics: &[SeedDiagnostics]) {
    println!();
    println!("Per-Seed Diagnostics (Phase 2.1h):");
    println!(
        "────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"
    );
    println!(
        "  Seed       | explore% | exploit% | stable% | bad%  | perturb% | chronic% | enters | mean_len | ema_stab | ema_bad | by_bad | by_unstab"
    );
    println!(
        "────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"
    );
    for d in diagnostics {
        let collapse_marker = if d.explore_rate > 0.25
            || d.stable_share < 0.55
            || d.rescue_count > 15
            || d.perturb_rate > 0.05
            || d.chronic_lock_share > 0.50
        {
            " ⚠"
        } else {
            ""
        };
        let mean_lock_len = if d.chronic_enter_count > 0 {
            d.chronic_lock_total_ticks as f64 / d.chronic_enter_count as f64
        } else {
            0.0
        };
        println!(
            "  0x{:08X} | {:6.1}%  | {:6.1}%  | {:5.1}%  | {:4.1}% | {:7.1}%  | {:7.1}%  | {:6} | {:8.1} | {:8.1}% | {:7.1}% | {:6} | {:9}{}",
            d.seed,
            d.explore_rate * 100.0,
            d.exploit_rate * 100.0,
            d.stable_share * 100.0,
            d.bad_state_share * 100.0,
            d.perturb_rate * 100.0,
            d.chronic_lock_share * 100.0,
            d.chronic_enter_count,
            mean_lock_len,
            d.chronic_stable_ema_final * 100.0,
            d.chronic_bad_ema_final * 100.0,
            d.chronic_enter_by_bad,
            d.chronic_enter_by_unstable,
            collapse_marker,
        );
    }
    println!(
        "────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"
    );

    // Phase 2.1k: Print exploit quality metrics table
    println!();
    println!("Exploit Quality Metrics (Phase 2.1k):");
    println!(
        "────────────────────────────────────────────────────────────────────────────────────────────"
    );
    println!(
        "  Seed       | hard_exploit | soft_exploit | soft_share | bad_in_expl% | bad_in_exp%"
    );
    println!(
        "────────────────────────────────────────────────────────────────────────────────────────────"
    );
    for d in diagnostics {
        println!(
            "  0x{:08X} | {:12} | {:12} | {:9.1}% | {:11.1}% | {:10.1}%",
            d.seed,
            d.exploit_hard_count,
            d.exploit_soft_count,
            d.exploit_soft_share * 100.0,
            d.bad_in_exploit_share * 100.0,
            d.bad_in_explore_share * 100.0,
        );
    }
    println!(
        "────────────────────────────────────────────────────────────────────────────────────────────"
    );
}

/// Options for running Demo 13.
#[derive(Clone, Debug, Default)]
pub struct Demo13Options {
    /// Custom seeds (overrides config.demo13_num_seeds).
    pub seeds: Option<Vec<u64>>,
    /// Quick mode: reduced tick budgets.
    pub quick: bool,
    /// Output path for JSON results.
    pub out_path: Option<String>,
}

/// Run Demo 13: Multi-seed evaluation with lift metrics.
pub fn run(config: &Config) {
    run_with_options(config, Demo13Options::default());
}

/// Run Demo 13 with options.
pub fn run_with_options(config: &Config, options: Demo13Options) {
    // Apply quick mode overrides
    let mut config = config.clone();
    if options.quick {
        // Reduce tick budgets for faster CI runs while keeping metrics meaningful
        config.competitive_episodes = 200; // Was 400 (50% reduction)
        config.competitive_episode_ticks = 300; // Was 500 (40% reduction)
                                                // Total ticks: 200*300 = 60,000 (vs 200,000 normally) - 70% faster
    }
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 13: Phase 2.1 - MULTI-SEED EVALUATION + LIFT METRICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if !config.enable_mode_policy || !config.enable_action_policy {
        println!("Mode or action policy disabled. Skipping Demo 13.");
        return;
    }

    // Get config hash for reproducibility
    let config_hash = config.config_hash();

    // Use custom seeds from options, or generate from config
    let seeds: Vec<u64> = if let Some(ref custom_seeds) = options.seeds {
        custom_seeds.clone()
    } else {
        let base_seed = config.seed;
        (0..config.demo13_num_seeds)
            .map(|i| base_seed.wrapping_add(0x1000_0000 * i as u64))
            .collect()
    };
    let num_seeds = seeds.len();

    println!("Configuration:");
    println!("  config_hash: {}", config_hash);
    println!("  num_seeds: {}", num_seeds);
    println!("  seeds: {:?}", seeds);
    println!("  episodes_per_seed: {}", config.competitive_episodes);
    println!("  ticks_per_episode: {}", config.competitive_episode_ticks);
    if options.quick {
        println!("  mode: QUICK (reduced ticks for CI)");
    }
    println!();

    let lift_config = LiftConfig {
        bad_margin: config.lift_bad_margin,
        bad_proto: config.lift_bad_proto,
        bad_value: config.lift_bad_value,
        recovery_window: 10,
    };

    // ==========================================================================
    // Run FULL variant across all seeds
    // ==========================================================================
    println!("Running FULL variant across {} seeds...", num_seeds);
    let mut full_runs: Vec<SeedRun> = Vec::new();
    let mut full_lifts: Vec<LiftStats> = Vec::new();

    let mut full_diagnostics: Vec<SeedDiagnostics> = Vec::new();

    for (i, &seed) in seeds.iter().enumerate() {
        print!("  Seed {}/{} (0x{:08X})... ", i + 1, num_seeds, seed);
        let (run, lift_stats, diag) = run_single_seed_full(&config, &lift_config, seed);
        println!(
            "cov={:.1}% sel={:.1}% FP={:.1}% rescues={}",
            run.coverage_pos * 100.0,
            run.selective_accuracy * 100.0,
            run.false_positive * 100.0,
            diag.rescue_count,
        );
        full_runs.push(run);
        full_lifts.push(lift_stats);
        full_diagnostics.push(diag);
    }

    let full_agg = multiseed::aggregate(&full_runs);
    let full_lift_agg = lift::aggregate_lift(&full_lifts);

    // ==========================================================================
    // Run RANDOM_BUDGETED variant across all seeds
    // ==========================================================================
    println!();
    println!(
        "Running RANDOM_BUDGETED variant across {} seeds...",
        num_seeds
    );

    // Use average action rates from FULL as budget targets
    let target_scan_rate = full_agg.scan_rate_mean;
    let target_perturb_rate = full_agg.perturb_rate_mean;

    let mut budgeted_runs: Vec<SeedRun> = Vec::new();
    let mut budgeted_lifts: Vec<LiftStats> = Vec::new();

    for (i, &seed) in seeds.iter().enumerate() {
        print!("  Seed {}/{} (0x{:08X})... ", i + 1, num_seeds, seed);
        let (run, lift_stats) = run_single_seed_budgeted(
            &config,
            &lift_config,
            seed,
            target_scan_rate,
            target_perturb_rate,
        );
        println!(
            "cov={:.1}% sel={:.1}% FP={:.1}%",
            run.coverage_pos * 100.0,
            run.selective_accuracy * 100.0,
            run.false_positive * 100.0
        );
        budgeted_runs.push(run);
        budgeted_lifts.push(lift_stats);
    }

    let budgeted_agg = multiseed::aggregate(&budgeted_runs);
    let budgeted_lift_agg = lift::aggregate_lift(&budgeted_lifts);

    // ==========================================================================
    // Print Results
    // ==========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.1 RESULTS: MULTI-SEED EVALUATION");
    println!("═══════════════════════════════════════════════════════════════════");

    // Per-seed tables
    println!();
    println!("FULL per-seed:");
    multiseed::print_per_seed_table(&full_runs);

    println!();
    println!("RANDOM_BUDGETED per-seed:");
    multiseed::print_per_seed_table(&budgeted_runs);

    // Phase 2.1b: Diagnostics table
    print_diagnostics_table(&full_diagnostics);

    // Aggregate tables
    println!();
    println!("───────────────────────────────────────────────────────────────────");
    println!("Aggregate Metrics (Mean ± Std):");
    println!("───────────────────────────────────────────────────────────────────");
    println!();
    multiseed::print_aggregate_table("FULL", &full_agg);
    println!();
    multiseed::print_aggregate_table("RANDOM_BUDGETED", &budgeted_agg);

    // Lift tables
    println!();
    println!("───────────────────────────────────────────────────────────────────");
    println!("Lift Metrics (Policy Advantage):");
    println!("───────────────────────────────────────────────────────────────────");
    println!();
    lift::print_lift_table("FULL", &full_lift_agg);
    println!();
    lift::print_lift_table("RANDOM_BUDGETED", &budgeted_lift_agg);

    // Lift comparison
    println!();
    println!("───────────────────────────────────────────────────────────────────");
    println!("Lift Comparison (FULL vs RANDOM_BUDGETED):");
    println!("───────────────────────────────────────────────────────────────────");

    let (lift_wins, comparisons) = lift::compare_lift(&full_lift_agg, &budgeted_lift_agg);
    println!();
    for (metric, full_wins) in &comparisons {
        let symbol = if *full_wins { "✓" } else { "✗" };
        let direction = if *metric == "bad_state_share" {
            "lower"
        } else {
            "higher"
        };
        println!(
            "  [{}] FULL {} on {}: {} is better",
            symbol,
            if *full_wins { "wins" } else { "loses" },
            metric,
            direction
        );
    }
    println!();
    println!("  FULL wins on {}/3 lift metrics", lift_wins);

    // ==========================================================================
    // Acceptance Criteria
    // ==========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.1 ACCEPTANCE:");
    println!("═══════════════════════════════════════════════════════════════════");

    // C1) Regression guard on FULL
    println!();
    println!("C1) Regression guard (FULL mean across seeds):");

    let coverage_ok = full_agg.coverage_pos_mean >= 0.70;
    let selective_ok = full_agg.selective_accuracy_mean >= 0.80;
    let fp_ok = full_agg.false_positive_mean < 0.001; // Use epsilon for floating-point comparison

    println!(
        "  [{}] coverage_pos_mean >= 70%: {:.1}%",
        if coverage_ok { "✓" } else { "✗" },
        full_agg.coverage_pos_mean * 100.0
    );
    println!(
        "  [{}] selective_accuracy_mean >= 80%: {:.1}%",
        if selective_ok { "✓" } else { "✗" },
        full_agg.selective_accuracy_mean * 100.0
    );
    println!(
        "  [{}] false_positive_mean == 0%: {:.1}%",
        if fp_ok { "✓" } else { "✗" },
        full_agg.false_positive_mean * 100.0
    );

    let regression_ok = coverage_ok && selective_ok && fp_ok;

    // C2) Policy advantage (FULL beats RANDOM_BUDGETED on >= 2 lift metrics)
    println!();
    println!("C2) Policy advantage (FULL vs RANDOM_BUDGETED):");

    let policy_advantage_ok = lift_wins >= 2;
    println!(
        "  [{}] FULL beats RANDOM_BUDGETED on >= 2 lift metrics: {}/3",
        if policy_advantage_ok { "✓" } else { "✗" },
        lift_wins
    );

    // C3) Variability reporting
    println!();
    println!("C3) Variability (std devs reported):");

    let low_variability =
        full_agg.coverage_pos_std < 0.15 && full_agg.selective_accuracy_std < 0.15;
    println!(
        "  [{}] coverage_pos_std < 15%: {:.1}%",
        if full_agg.coverage_pos_std < 0.15 {
            "✓"
        } else {
            "~"
        },
        full_agg.coverage_pos_std * 100.0
    );
    println!(
        "  [{}] selective_accuracy_std < 15%: {:.1}%",
        if full_agg.selective_accuracy_std < 0.15 {
            "✓"
        } else {
            "~"
        },
        full_agg.selective_accuracy_std * 100.0
    );

    if !full_agg.failed_seeds.is_empty() {
        println!();
        println!(
            "  ⚠ Seeds violating regression guard: {}",
            full_agg
                .failed_seeds
                .iter()
                .map(|s| format!("0x{:08X}", s))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Phase 2.1c: Thrash reduction checks
    println!();
    println!("C4) Thrash reduction (Phase 2.1c):");

    let max_rescues = full_diagnostics
        .iter()
        .map(|d| d.rescue_count)
        .max()
        .unwrap_or(0);
    let max_rescues_ok = max_rescues <= 15;
    println!(
        "  [{}] max_rescues_per_seed <= 15: {}",
        if max_rescues_ok { "✓" } else { "✗" },
        max_rescues
    );

    // C5) Worst-seed floor
    println!();
    println!("C5) Worst-seed floor (Phase 2.1c):");

    let worst_coverage = full_runs
        .iter()
        .map(|r| r.coverage_pos)
        .fold(f64::INFINITY, f64::min);
    let worst_sel_acc = full_runs
        .iter()
        .map(|r| r.selective_accuracy)
        .fold(f64::INFINITY, f64::min);

    let worst_coverage_ok = worst_coverage >= 0.65;
    let worst_sel_acc_ok = worst_sel_acc >= 0.75;

    println!(
        "  [{}] worst_seed_coverage_pos >= 65%: {:.1}%",
        if worst_coverage_ok { "✓" } else { "✗" },
        worst_coverage * 100.0
    );
    println!(
        "  [{}] worst_seed_selective_accuracy >= 75%: {:.1}%",
        if worst_sel_acc_ok { "✓" } else { "✗" },
        worst_sel_acc * 100.0
    );

    let thrash_ok = max_rescues_ok;
    let worst_seed_ok = worst_coverage_ok && worst_sel_acc_ok;

    // C6) Phase 2.1e: Perturb control + chronic control
    println!();
    println!("C6) Perturb & chronic control (Phase 2.1f):");

    let perturb_rate_mean = full_agg.perturb_rate_mean;
    let perturb_ok = perturb_rate_mean <= 0.05; // <= 5%
    println!(
        "  [{}] perturb_rate_mean <= 5%: {:.1}%",
        if perturb_ok { "✓" } else { "✗" },
        perturb_rate_mean * 100.0
    );

    let chronic_mean = full_diagnostics
        .iter()
        .map(|d| d.chronic_lock_share)
        .sum::<f64>()
        / full_diagnostics.len().max(1) as f64;
    let chronic_ok = chronic_mean <= 0.50; // <= 50%
    println!(
        "  [{}] chronic_lock_share_mean <= 50%: {:.1}%",
        if chronic_ok { "✓" } else { "✗" },
        chronic_mean * 100.0
    );

    let perturb_chronic_ok = perturb_ok && chronic_ok;

    // Summary
    let all_ok =
        regression_ok && policy_advantage_ok && thrash_ok && worst_seed_ok && perturb_chronic_ok;
    println!();
    if all_ok {
        println!("  → Phase 2.1f: ALL ACCEPTANCE CRITERIA MET!");
        if low_variability {
            println!("  → Low variability across seeds - results are robust.");
        }
    } else {
        let mut issues = Vec::new();
        if !regression_ok {
            issues.push("regression guard");
        }
        if !policy_advantage_ok {
            issues.push("policy advantage");
        }
        if !thrash_ok {
            issues.push("thrash reduction");
        }
        if !worst_seed_ok {
            issues.push("worst-seed floor");
        }
        if !perturb_chronic_ok {
            issues.push("perturb/chronic control");
        }
        println!("  → Phase 2.1f: Failed checks: {}", issues.join(", "));
    }

    // ==========================================================================
    // JSON Export (if --out specified)
    // ==========================================================================
    if let Some(ref out_path) = options.out_path {
        let meta = ResultMeta::new(13, &config_hash, Some(seeds.clone()), options.quick);

        let full_result = VariantResult {
            name: "FULL".to_string(),
            runs: full_runs.iter().map(SeedRunResult::from).collect(),
            aggregate: AggregateResult::from(&full_agg),
            lift: LiftResult::from(&full_lift_agg),
        };

        let budgeted_result = VariantResult {
            name: "RANDOM_BUDGETED".to_string(),
            runs: budgeted_runs.iter().map(SeedRunResult::from).collect(),
            aggregate: AggregateResult::from(&budgeted_agg),
            lift: LiftResult::from(&budgeted_lift_agg),
        };

        let acceptance = AcceptanceResult {
            regression_guard_ok: regression_ok,
            coverage_ok,
            selective_accuracy_ok: selective_ok,
            false_positive_ok: fp_ok,
            policy_advantage_ok,
            lift_wins,
            all_pass: all_ok,
        };

        let result = Demo13Result {
            meta,
            full: full_result,
            random_budgeted: budgeted_result,
            acceptance,
        };

        match write_json(&result, out_path) {
            Ok(()) => println!("\nJSON written to: {}", out_path),
            Err(e) => eprintln!("\nError writing JSON: {}", e),
        }
    }
}

/// Run a single seed with FULL policy (Phase 2.1b with guardrails).
fn run_single_seed_full(
    config: &Config,
    lift_config: &LiftConfig,
    seed: u64,
) -> (SeedRun, LiftStats, SeedDiagnostics) {
    // Phase 2.1b: Create mode policy config with guardrails
    let mode_policy_config = ModePolicyConfig::from_config(config);
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    // Phase 2.1b: Warmup stats for adaptive thresholds
    let mut warmup_stats = WarmupStats::new();

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    // Phase 2.1e: Use constructor with both floor and budget cap
    let mut action_policy = ActionPolicy::new_with_floor_and_budget(
        action_config,
        config.perturb_floor_window,
        config.perturb_floor_min_rate,
        config.perturb_budget_window,
        config.perturb_cap,
    );

    let mut rng = Rng::new(seed.wrapping_add(0x7A7A_7A7A));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(config, &mut rng);

    // Pre-train
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

    let mut anchor_bank = AnchorBank::new();
    let keyed_config = KeyedMemoryConfig {
        label_min_p: 0.50,
        label_margin: 0.10,
        alpha: 0.5,
        num_labels: config.num_ctx,
    };
    let mut keyed_memory = KeyedMemoryStore::new(keyed_config);
    let mut metrics = KeyedMemoryMetrics::new();

    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);
    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;

    // Value learning state
    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    // Mode/action counters
    let mut explore_count: usize = 0;
    let mut exploit_count: usize = 0;
    let mut reset_count: usize = 0;
    let mut stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;

    // Lift stats
    let mut lift_stats = LiftStats::new();

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            total_ticks += 1;
            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            // Periodic merges
            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            if is_stable {
                stable_ticks += 1;
            }

            // Compute TD
            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);

            let proto_align = if anchor_id != 0xFFFF {
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    anchor.proto_score(&base_topk, config.proto_m)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Phase 2.1b: Collect warmup stats for adaptive thresholds
            if config.demo13_enable_adaptive_thresholds && total_ticks <= 5000 {
                warmup_stats.push(proto_align, topk_margin);

                // After warmup, update mode policy with adaptive thresholds
                if total_ticks == 5000 {
                    let (proto_p50, margin_p50) = warmup_stats.compute_p50();
                    let adaptive_proto = (proto_p50 * config.exploit_proto_p50_scale)
                        .max(config.exploit_proto_min_floor);
                    let adaptive_margin = (margin_p50 * config.exploit_margin_p50_scale)
                        .max(config.exploit_margin_min_floor);
                    mode_policy.config.exploit_proto_min = adaptive_proto;
                    mode_policy.config.exploit_margin_min = adaptive_margin;
                }
            }

            // Mode policy with extended observation
            mode_policy.observe_extended(
                global_tick,
                anchor_value,
                abs_td as f32,
                base_gate_passed,
                proto_align,
                topk_margin,
                is_stable,
            );

            // Phase 2.1b: Use guardrails if enabled
            let mode = if config.demo13_enable_rescue {
                let (m, _rescue_fired) = mode_policy.choose_mode_with_guardrails(global_tick);
                m
            } else {
                mode_policy.choose_mode(global_tick)
            };

            match mode {
                Mode::Explore => explore_count += 1,
                Mode::Exploit => exploit_count += 1,
                Mode::Reset => reset_count += 1,
            }

            // Phase 2.1c/d: Action policy with combined lock bias (post-rescue + chronic)
            let post_rescue_active = mode_policy.is_post_rescue_lock_active();
            let post_rescue_bias = mode_policy.get_lock_focus_bias();
            let chronic_active = mode_policy.is_chronic_lock_active();
            let chronic_bias = mode_policy.get_chronic_focus_bias();

            // First get base action with triggers
            let (mut action, trigger_reason) = action_policy.choose_action_with_triggers(
                mode,
                abs_td as f32,
                base_gate_passed,
                topk_margin as f32,
                proto_align,
                anchor_value,
                config,
            );

            // Phase 2.1d: Override with combined lock bias if active (except for Perturb from triggers)
            if (post_rescue_active || chronic_active) && trigger_reason.is_none() {
                action = action_policy.choose_action_with_combined_lock(
                    mode,
                    post_rescue_active,
                    post_rescue_bias,
                    chronic_active,
                    chronic_bias,
                    abs_td as f32,
                    config.mode_reset_td_min,
                );
            }

            // Phase 2.1e: Enforce perturb budget cap (BEFORE min perturb guard)
            // If over budget and action is Perturb (not from Reset mode), downgrade to Focus
            if action == Action::Perturb && mode != Mode::Reset {
                if action_policy.is_perturb_over_budget() {
                    action = Action::Focus; // Downgrade to Focus
                }
                // Also check chronic perturb disallow
                if mode_policy.is_chronic_perturb_disallowed() {
                    action = Action::Focus; // Downgrade to Focus during chronic
                }
            }

            // Phase 2.1b: Min perturb guard (only if not over budget and not chronic-disallowed)
            let perturb_allowed = !action_policy.is_perturb_over_budget()
                && !mode_policy.is_chronic_perturb_disallowed();
            if config.demo13_enable_min_perturb_guard
                && action != Action::Perturb
                && perturb_allowed
            {
                if action_policy.should_force_perturb_guard(
                    config.demo13_min_perturb_rate,
                    base_gate_passed,
                    topk_margin as f32,
                    proto_align,
                    config.perturb_trig_margin_min * 1.5,
                    config.perturb_trig_proto_min * 1.2,
                ) {
                    action = Action::Perturb;
                    action_policy
                        .triggers
                        .on_perturb(config.perturb_cooldown_ticks);
                    action_policy.trigger_stats.by_floor += 1;
                }
            }

            action_policy.record_trigger(trigger_reason);
            action_policy.record_action_for_floor(action);

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            // Phase 2.1c/d: Apply lock margin scale during post-rescue or chronic lock
            if post_rescue_active {
                let lock_margin_scale = mode_policy.get_lock_margin_scale();
                adjusted_gate_params.margin_mult *= lock_margin_scale as f64;
            }
            if chronic_active {
                let chronic_margin_scale = mode_policy.get_chronic_margin_scale();
                adjusted_gate_params.margin_mult *= chronic_margin_scale as f64;
            }

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Apply perturb noise
            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            // Update lift stats
            lift_stats.observe_tick(
                lift_config,
                global_tick,
                mode,
                action,
                anchor_id,
                abs_td as f32,
                topk_margin,
                proto_align,
                anchor_value,
                gate_passed,
            );

            // Update partition info
            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            // Update prototype
            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            // Value update
            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            // Update previous state
            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            // Memory store/recall
            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    lift_stats.finalize(0.0);

    let action_stats = &action_policy.stats;
    let mode_total = explore_count + exploit_count + reset_count;

    let mut run = SeedRun::new(seed);
    run.coverage_pos = metrics.coverage_pos();
    run.selective_accuracy = metrics.selective_accuracy();
    run.false_positive = metrics.false_positive_rate();
    run.stable_share = if total_ticks > 0 {
        stable_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    run.explore_rate = if mode_total > 0 {
        explore_count as f64 / mode_total as f64
    } else {
        0.0
    };
    run.exploit_rate = if mode_total > 0 {
        exploit_count as f64 / mode_total as f64
    } else {
        0.0
    };
    run.reset_rate = if mode_total > 0 {
        reset_count as f64 / mode_total as f64
    } else {
        0.0
    };

    run.scan_rate = action_stats.scan_rate();
    run.focus_rate = action_stats.focus_rate();
    run.perturb_rate = action_stats.perturb_rate();

    run.bad_state_share = Some(lift_stats.bad_state_share());
    run.recovery_improve = Some(lift_stats.recovery_after_perturb());

    // Phase 2.1b/c: Collect diagnostics
    let mode_stats = mode_policy.mode_stats();
    let (proto_p50, margin_p50) = warmup_stats.compute_p50();

    // Phase 2.1c: Compute thrash metrics
    let rescues_per_10k = if total_ticks > 0 {
        (mode_stats.rescue_count as f64 / total_ticks as f64) * 10000.0
    } else {
        0.0
    };
    let post_rescue_lock_share = if total_ticks > 0 {
        mode_stats.post_rescue_lock_total_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    // Phase 2.1d: Compute chronic lock share
    let chronic_lock_share = if total_ticks > 0 {
        mode_stats.chronic_lock_total_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    let diag = SeedDiagnostics {
        seed,
        proto_align_mean: if !warmup_stats.proto_samples.is_empty() {
            warmup_stats.proto_samples.iter().sum::<f32>() / warmup_stats.proto_samples.len() as f32
        } else {
            0.0
        },
        proto_align_p50: proto_p50,
        proto_align_p90: {
            if warmup_stats.proto_samples.len() > 10 {
                let mut sorted = warmup_stats.proto_samples.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sorted[(sorted.len() * 9) / 10]
            } else {
                proto_p50
            }
        },
        margin_mean: if !warmup_stats.margin_samples.is_empty() {
            warmup_stats.margin_samples.iter().sum::<f64>()
                / warmup_stats.margin_samples.len() as f64
        } else {
            0.0
        },
        margin_p50,
        margin_p90: {
            if warmup_stats.margin_samples.len() > 10 {
                let mut sorted = warmup_stats.margin_samples.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sorted[(sorted.len() * 9) / 10]
            } else {
                margin_p50
            }
        },
        stable_share: run.stable_share,
        bad_state_share: lift_stats.bad_state_share(),
        explore_rate: run.explore_rate,
        exploit_rate: run.exploit_rate,
        reset_rate: run.reset_rate,
        explore_streak_max: mode_stats.explore_streak_max,
        exploit_streak_max: mode_stats.exploit_streak_max,
        gate_fail_streak_max: mode_stats.gate_fail_streak_max,
        rescue_count: mode_stats.rescue_count,
        adaptive_proto_min: mode_policy.config.exploit_proto_min,
        adaptive_margin_min: mode_policy.config.exploit_margin_min,
        // Phase 2.1c: Thrash metrics
        rescues_per_10k,
        post_rescue_lock_share,
        total_ticks,
        // Phase 2.1h: Chronic clamp metrics
        chronic_lock_share,
        chronic_lock_total_ticks: mode_stats.chronic_lock_total_ticks,
        chronic_enter_count: mode_stats.chronic_enter_count,
        chronic_exit_count: mode_stats.chronic_exit_count,
        chronic_enter_by_bad: mode_stats.chronic_enter_by_bad,
        chronic_enter_by_unstable: mode_stats.chronic_enter_by_unstable,
        chronic_exit_by_watchdog: mode_stats.chronic_exit_by_watchdog,
        // Phase 2.1h: EMA final values
        chronic_stable_ema_final: mode_stats.chronic_stable_ema_final,
        chronic_bad_ema_final: mode_stats.chronic_bad_ema_final,
        // Phase 2.1e: Perturb rate
        perturb_rate: run.perturb_rate,
        // Phase 2.1k: Exploit quality metrics
        exploit_hard_count: mode_stats.exploit_hard_count,
        exploit_soft_count: mode_stats.exploit_soft_count,
        exploit_soft_share: mode_stats.exploit_soft_share,
        bad_in_explore_share: if mode_stats.explore_count > 0 {
            mode_stats.bad_in_explore_count as f64 / mode_stats.explore_count as f64
        } else {
            0.0
        },
        bad_in_exploit_share: if mode_stats.exploit_count > 0 {
            mode_stats.bad_in_exploit_count as f64 / mode_stats.exploit_count as f64
        } else {
            0.0
        },
    };

    (run, lift_stats, diag)
}

/// Run a single seed with RANDOM_BUDGETED policy.
fn run_single_seed_budgeted(
    config: &Config,
    lift_config: &LiftConfig,
    seed: u64,
    target_scan_rate: f64,
    target_perturb_rate: f64,
) -> (SeedRun, LiftStats) {
    let mode_policy_config = ModePolicyConfig::from_config(config);
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);

    let mut rng = Rng::new(seed.wrapping_add(0x7A7A_7A7A));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(config, &mut rng);

    // Pre-train
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

    let mut anchor_bank = AnchorBank::new();
    let keyed_config = KeyedMemoryConfig {
        label_min_p: 0.50,
        label_margin: 0.10,
        alpha: 0.5,
        num_labels: config.num_ctx,
    };
    let mut keyed_memory = KeyedMemoryStore::new(keyed_config);
    let mut metrics = KeyedMemoryMetrics::new();

    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);
    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;

    // Value learning state
    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    // Mode/action counters
    let mut explore_count: usize = 0;
    let mut exploit_count: usize = 0;
    let mut reset_count: usize = 0;
    let mut stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;

    // Lift stats
    let mut lift_stats = LiftStats::new();

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            total_ticks += 1;
            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            // Periodic merges
            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            if is_stable {
                stable_ticks += 1;
            }

            // Compute TD
            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);

            let proto_align = if anchor_id != 0xFFFF {
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    anchor.proto_score(&base_topk, config.proto_m)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Mode policy (still use real mode for tracking)
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);
            let mode = mode_policy.choose_mode(global_tick);

            match mode {
                Mode::Explore => explore_count += 1,
                Mode::Exploit => exploit_count += 1,
                Mode::Reset => reset_count += 1,
            }

            // RANDOM_BUDGETED: choose action randomly with budget-matched rates
            let r = rng.next_f64();
            let action = if r < target_perturb_rate {
                Action::Perturb
            } else if r < target_perturb_rate + target_scan_rate {
                Action::Scan
            } else {
                Action::Focus
            };

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Apply perturb noise
            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            // Update lift stats
            lift_stats.observe_tick(
                lift_config,
                global_tick,
                mode,
                action,
                anchor_id,
                abs_td as f32,
                topk_margin,
                proto_align,
                anchor_value,
                gate_passed,
            );

            // Update partition info
            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            // Update prototype
            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            // Value update
            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            // Update previous state
            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            // Memory store/recall
            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    lift_stats.finalize(0.0);

    let action_stats = &action_policy.stats;
    let mode_total = explore_count + exploit_count + reset_count;

    let mut run = SeedRun::new(seed);
    run.coverage_pos = metrics.coverage_pos();
    run.selective_accuracy = metrics.selective_accuracy();
    run.false_positive = metrics.false_positive_rate();
    run.stable_share = if total_ticks > 0 {
        stable_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    run.explore_rate = if mode_total > 0 {
        explore_count as f64 / mode_total as f64
    } else {
        0.0
    };
    run.exploit_rate = if mode_total > 0 {
        exploit_count as f64 / mode_total as f64
    } else {
        0.0
    };
    run.reset_rate = if mode_total > 0 {
        reset_count as f64 / mode_total as f64
    } else {
        0.0
    };

    run.scan_rate = action_stats.scan_rate();
    run.focus_rate = action_stats.focus_rate();
    run.perturb_rate = action_stats.perturb_rate();

    run.bad_state_share = Some(lift_stats.bad_state_share());
    run.recovery_improve = Some(lift_stats.recovery_after_perturb());

    (run, lift_stats)
}

/// Compute reward for value learning.
fn compute_reward(
    delta_power: f64,
    prev_topk_margin: f64,
    prev_proto_align: f32,
    config: &Config,
) -> f32 {
    let r_power = (delta_power / config.r_p_clip as f64).clamp(-1.0, 1.0) as f32;
    let r_margin = (prev_topk_margin as f32 / config.margin_norm).clamp(0.0, 1.0);
    let r_proto = prev_proto_align.clamp(0.0, 1.0);

    config.r_w_power * r_power + config.r_w_margin * r_margin + config.r_w_proto * r_proto
}

/// Simple bit flip for negative queries.
fn flip_bits_simple(mask: u64, num_flips: u32, rand_val: u64) -> u64 {
    let mut result = mask;
    let mut r = rand_val;
    for _ in 0..num_flips {
        let bit_pos = (r % 64) as u8;
        result ^= 1u64 << bit_pos;
        r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    result
}
