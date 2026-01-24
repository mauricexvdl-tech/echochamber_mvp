//! Phase 2.1: Lift Metrics
//!
//! Provides metrics that measure policy advantage beyond standard end-metrics.
//! These help distinguish between policies that achieve similar coverage/accuracy
//! but have different behavioral characteristics.

use crate::action::Action;
use crate::mode::Mode;

/// Configuration for lift metrics.
#[derive(Clone, Debug)]
pub struct LiftConfig {
    /// Margin threshold for "bad state" (below = bad).
    pub bad_margin: f64,
    /// Proto alignment threshold for "bad state" (below = bad).
    pub bad_proto: f32,
    /// Value threshold for "bad state" (below = bad).
    pub bad_value: f32,
    /// Window size around perturb events for recovery measurement.
    pub recovery_window: usize,
}

impl Default for LiftConfig {
    fn default() -> Self {
        Self {
            bad_margin: 0.02,
            bad_proto: 0.10,
            bad_value: 0.15,
            recovery_window: 10,
        }
    }
}

/// Lift statistics computed during a run.
#[derive(Clone, Debug, Default)]
pub struct LiftStats {
    // Exploit-Focus lift tracking
    pub focus_in_exploit: usize,
    pub exploit_ticks: usize,
    pub focus_in_explore: usize,
    pub explore_ticks: usize,

    // Recovery after perturb tracking
    pub perturb_events: usize,
    pub pre_perturb_td_sum: f64,
    pub post_perturb_td_sum: f64,
    perturb_pending: Vec<(u64, f32)>, // (tick, pre_td)

    // Bad state tracking
    pub total_ticks: usize,
    pub bad_state_ticks: usize,
    gate_fail_streak: usize,

    // Scan diversity tracking (unique anchors seen during Explore + Scan)
    pub scan_explore_ticks: usize,
    scan_anchors_seen: Vec<u16>,
}

impl LiftStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe a tick and update counters.
    pub fn observe_tick(
        &mut self,
        config: &LiftConfig,
        tick: u64,
        mode: Mode,
        action: Action,
        anchor_id: u16,
        abs_td: f32,
        topk_margin: f64,
        proto_align: f32,
        anchor_value: f32,
        gate_passed: bool,
    ) {
        self.total_ticks += 1;

        // Update gate fail streak for bad state detection
        if gate_passed {
            self.gate_fail_streak = 0;
        } else {
            self.gate_fail_streak += 1;
        }

        // Check bad state conditions
        let is_bad_state = self.gate_fail_streak > 0
            || topk_margin < config.bad_margin
            || proto_align < config.bad_proto
            || anchor_value < config.bad_value;

        if is_bad_state {
            self.bad_state_ticks += 1;
        }

        // Track Focus rate per mode for exploit_focus_lift
        match mode {
            Mode::Exploit => {
                self.exploit_ticks += 1;
                if action == Action::Focus {
                    self.focus_in_exploit += 1;
                }
            }
            Mode::Explore => {
                self.explore_ticks += 1;
                if action == Action::Focus {
                    self.focus_in_explore += 1;
                }
            }
            Mode::Reset => {}
        }

        // Track Scan diversity during Explore
        if mode == Mode::Explore && action == Action::Scan {
            self.scan_explore_ticks += 1;
            if anchor_id != 0xFFFF && !self.scan_anchors_seen.contains(&anchor_id) {
                self.scan_anchors_seen.push(anchor_id);
            }
        }

        // Track perturb events for recovery measurement
        if action == Action::Perturb {
            self.perturb_events += 1;
            self.perturb_pending.push((tick, abs_td));
        }

        // Check pending perturb events for recovery measurement
        self.check_perturb_recovery(config, tick, abs_td);
    }

    fn check_perturb_recovery(&mut self, config: &LiftConfig, current_tick: u64, current_td: f32) {
        let window = config.recovery_window as u64;

        self.perturb_pending.retain(|&(perturb_tick, pre_td)| {
            if current_tick >= perturb_tick + window {
                // Measure recovery
                self.pre_perturb_td_sum += pre_td as f64;
                self.post_perturb_td_sum += current_td as f64;
                false // Remove from pending
            } else {
                true // Keep
            }
        });
    }

    /// Finalize any pending perturb events at end of run.
    pub fn finalize(&mut self, final_td: f32) {
        for (_, pre_td) in self.perturb_pending.drain(..) {
            self.pre_perturb_td_sum += pre_td as f64;
            self.post_perturb_td_sum += final_td as f64;
        }
    }

    /// Reset for a new run.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    // Computed lift metrics

    /// Exploit-Focus lift: Focus%(Exploit) - Focus%(Explore)
    /// Higher is better: means policy concentrates Focus when exploiting.
    pub fn exploit_focus_lift(&self) -> f64 {
        let focus_rate_exploit = if self.exploit_ticks > 0 {
            self.focus_in_exploit as f64 / self.exploit_ticks as f64
        } else {
            0.0
        };
        let focus_rate_explore = if self.explore_ticks > 0 {
            self.focus_in_explore as f64 / self.explore_ticks as f64
        } else {
            0.0
        };
        focus_rate_exploit - focus_rate_explore
    }

    /// Recovery after perturb: mean(|TD|_pre - |TD|_post) / mean(|TD|_pre)
    /// Higher is better: means perturb reduces TD error.
    pub fn recovery_after_perturb(&self) -> f64 {
        if self.perturb_events == 0 || self.pre_perturb_td_sum < 0.001 {
            return 0.0;
        }
        let pre_mean = self.pre_perturb_td_sum / self.perturb_events as f64;
        let post_mean = self.post_perturb_td_sum / self.perturb_events as f64;
        if pre_mean > 0.001 {
            (pre_mean - post_mean) / pre_mean
        } else {
            0.0
        }
    }

    /// Bad state share: % ticks in "bad state".
    /// Lower is better.
    pub fn bad_state_share(&self) -> f64 {
        if self.total_ticks > 0 {
            self.bad_state_ticks as f64 / self.total_ticks as f64
        } else {
            0.0
        }
    }

    /// Scan diversity: unique anchors seen per 1000 Scan-in-Explore ticks.
    /// Higher is better: means Scan actually explores diverse states.
    pub fn scan_diversity(&self) -> f64 {
        if self.scan_explore_ticks == 0 {
            return 0.0;
        }
        let unique = self.scan_anchors_seen.len() as f64;
        (unique / self.scan_explore_ticks as f64) * 1000.0
    }
}

/// Aggregated lift statistics across multiple runs.
#[derive(Clone, Debug, Default)]
pub struct LiftAggregate {
    pub num_runs: usize,

    pub exploit_focus_lift_mean: f64,
    pub exploit_focus_lift_std: f64,

    pub recovery_after_perturb_mean: f64,
    pub recovery_after_perturb_std: f64,

    pub bad_state_share_mean: f64,
    pub bad_state_share_std: f64,

    pub scan_diversity_mean: f64,
    pub scan_diversity_std: f64,
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
    let variance: f64 =
        values.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Aggregate lift stats across multiple runs.
pub fn aggregate_lift(runs: &[LiftStats]) -> LiftAggregate {
    if runs.is_empty() {
        return LiftAggregate::default();
    }

    let exploit_focus_lift: Vec<f64> = runs.iter().map(|r| r.exploit_focus_lift()).collect();
    let recovery_after_perturb: Vec<f64> =
        runs.iter().map(|r| r.recovery_after_perturb()).collect();
    let bad_state_share: Vec<f64> = runs.iter().map(|r| r.bad_state_share()).collect();
    let scan_diversity: Vec<f64> = runs.iter().map(|r| r.scan_diversity()).collect();

    let exploit_focus_lift_mean = mean(&exploit_focus_lift);
    let recovery_after_perturb_mean = mean(&recovery_after_perturb);
    let bad_state_share_mean = mean(&bad_state_share);
    let scan_diversity_mean = mean(&scan_diversity);

    LiftAggregate {
        num_runs: runs.len(),
        exploit_focus_lift_mean,
        exploit_focus_lift_std: std_dev(&exploit_focus_lift, exploit_focus_lift_mean),
        recovery_after_perturb_mean,
        recovery_after_perturb_std: std_dev(&recovery_after_perturb, recovery_after_perturb_mean),
        bad_state_share_mean,
        bad_state_share_std: std_dev(&bad_state_share, bad_state_share_mean),
        scan_diversity_mean,
        scan_diversity_std: std_dev(&scan_diversity, scan_diversity_mean),
    }
}

/// Print lift table for a single variant.
pub fn print_lift_table(label: &str, agg: &LiftAggregate) {
    println!("  {} Lift Metrics (n={}):", label, agg.num_runs);
    println!("  {:>24} | {:>15}", "Metric", "Mean ± Std");
    println!("  {}", "-".repeat(44));

    println!(
        "  {:>24} | {:>+6.1}% ± {:>5.1}%",
        "exploit_focus_lift",
        agg.exploit_focus_lift_mean * 100.0,
        agg.exploit_focus_lift_std * 100.0
    );
    println!(
        "  {:>24} | {:>+6.1}% ± {:>5.1}%",
        "recovery_after_perturb",
        agg.recovery_after_perturb_mean * 100.0,
        agg.recovery_after_perturb_std * 100.0
    );
    println!(
        "  {:>24} | {:>6.1}% ± {:>5.1}%",
        "bad_state_share",
        agg.bad_state_share_mean * 100.0,
        agg.bad_state_share_std * 100.0
    );
    println!(
        "  {:>24} | {:>6.1} ± {:>5.1}",
        "scan_diversity", agg.scan_diversity_mean, agg.scan_diversity_std
    );
}

/// Compare two lift aggregates and return number of wins for first.
/// Returns (wins_for_a, metric_comparisons)
pub fn compare_lift(a: &LiftAggregate, b: &LiftAggregate) -> (usize, Vec<(&'static str, bool)>) {
    let mut comparisons = Vec::new();
    let mut wins = 0;

    // exploit_focus_lift: higher is better
    let efl_win = a.exploit_focus_lift_mean > b.exploit_focus_lift_mean;
    if efl_win {
        wins += 1;
    }
    comparisons.push(("exploit_focus_lift", efl_win));

    // recovery_after_perturb: higher is better
    let rap_win = a.recovery_after_perturb_mean > b.recovery_after_perturb_mean;
    if rap_win {
        wins += 1;
    }
    comparisons.push(("recovery_after_perturb", rap_win));

    // bad_state_share: lower is better
    let bss_win = a.bad_state_share_mean < b.bad_state_share_mean;
    if bss_win {
        wins += 1;
    }
    comparisons.push(("bad_state_share", bss_win));

    (wins, comparisons)
}
