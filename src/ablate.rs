//! Phase 2.0b: Ablation Study + Per-Mode Metrics + Targeted Reset
//!
//! Provides:
//! - AblationConfig for enabling/disabling modes
//! - PerModeStats for tracking metrics per mode
//! - ResetTargetMode for targeted reset selection
//! - VariantReport for ablation results

use crate::mode::Mode;

/// Configuration for ablation variants.
#[derive(Clone, Debug)]
pub struct AblationConfig {
    /// Enable Reset mode (if false, Reset never fires)
    pub enable_reset: bool,
    /// Enable Explore mode (if false, Explore forced to Exploit)
    pub enable_explore: bool,
    /// Target mode for reset dampening
    pub target_mode: ResetTargetMode,
    /// Max nodes to dampen per reset
    pub reset_max_nodes: usize,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            enable_reset: true,
            enable_explore: true,
            target_mode: ResetTargetMode::TopK,
            reset_max_nodes: 6,
        }
    }
}

impl AblationConfig {
    /// Full variant: all modes enabled
    pub fn full() -> Self {
        Self::default()
    }

    /// NO_RESET variant: Explore + Exploit only
    pub fn no_reset() -> Self {
        Self {
            enable_reset: false,
            enable_explore: true,
            target_mode: ResetTargetMode::TopK,
            reset_max_nodes: 6,
        }
    }

    /// NO_EXPLORE variant: Exploit + Reset only
    pub fn no_explore() -> Self {
        Self {
            enable_reset: true,
            enable_explore: false,
            target_mode: ResetTargetMode::TopK,
            reset_max_nodes: 6,
        }
    }

    /// Full variant with BadActors targeting
    pub fn full_bad_actors() -> Self {
        Self {
            enable_reset: true,
            enable_explore: true,
            target_mode: ResetTargetMode::BadActors,
            reset_max_nodes: 6,
        }
    }
}

/// Reset target selection mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResetTargetMode {
    /// Dampen all top-K nodes (existing behavior)
    TopK,
    /// Dampen only "bad actors" - nodes not in anchor prototype
    BadActors,
}

/// Per-mode statistics tracking.
#[derive(Clone, Debug, Default)]
pub struct PerModeStats {
    // Tick counts per mode
    pub explore_ticks: usize,
    pub exploit_ticks: usize,
    pub reset_ticks: usize,

    // Gate pass tracking per mode
    pub explore_gate_pass: usize,
    pub explore_gate_total: usize,
    pub exploit_gate_pass: usize,
    pub exploit_gate_total: usize,
    pub reset_gate_pass: usize,
    pub reset_gate_total: usize,

    // Sum of |TD| per mode (for computing mean)
    pub explore_abs_td_sum: f64,
    pub exploit_abs_td_sum: f64,
    pub reset_abs_td_sum: f64,

    // Sum of anchor value per mode
    pub explore_value_sum: f64,
    pub exploit_value_sum: f64,
    pub reset_value_sum: f64,

    // Stable ticks per mode
    pub explore_stable_ticks: usize,
    pub exploit_stable_ticks: usize,
    pub reset_stable_ticks: usize,

    // Per-mode recall outcomes (for coverage/accuracy)
    pub explore_pos_total: usize,
    pub explore_pos_covered: usize,
    pub explore_pos_correct: usize,
    pub explore_neg_total: usize,
    pub explore_neg_abstain: usize,

    pub exploit_pos_total: usize,
    pub exploit_pos_covered: usize,
    pub exploit_pos_correct: usize,
    pub exploit_neg_total: usize,
    pub exploit_neg_abstain: usize,

    pub reset_pos_total: usize,
    pub reset_pos_covered: usize,
    pub reset_pos_correct: usize,
    pub reset_neg_total: usize,
    pub reset_neg_abstain: usize,
}

impl PerModeStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a tick for the given mode.
    pub fn record_tick(
        &mut self,
        mode: Mode,
        gate_passed: bool,
        abs_td: f32,
        anchor_value: f32,
        is_stable: bool,
    ) {
        match mode {
            Mode::Explore => {
                self.explore_ticks += 1;
                self.explore_gate_total += 1;
                if gate_passed {
                    self.explore_gate_pass += 1;
                }
                self.explore_abs_td_sum += abs_td as f64;
                self.explore_value_sum += anchor_value as f64;
                if is_stable {
                    self.explore_stable_ticks += 1;
                }
            }
            Mode::Exploit => {
                self.exploit_ticks += 1;
                self.exploit_gate_total += 1;
                if gate_passed {
                    self.exploit_gate_pass += 1;
                }
                self.exploit_abs_td_sum += abs_td as f64;
                self.exploit_value_sum += anchor_value as f64;
                if is_stable {
                    self.exploit_stable_ticks += 1;
                }
            }
            Mode::Reset => {
                self.reset_ticks += 1;
                self.reset_gate_total += 1;
                if gate_passed {
                    self.reset_gate_pass += 1;
                }
                self.reset_abs_td_sum += abs_td as f64;
                self.reset_value_sum += anchor_value as f64;
                if is_stable {
                    self.reset_stable_ticks += 1;
                }
            }
        }
    }

    /// Record a recall outcome for the given mode.
    pub fn record_recall(&mut self, mode: Mode, is_positive: bool, covered: bool, correct: bool) {
        match mode {
            Mode::Explore => {
                if is_positive {
                    self.explore_pos_total += 1;
                    if covered {
                        self.explore_pos_covered += 1;
                        if correct {
                            self.explore_pos_correct += 1;
                        }
                    }
                } else {
                    self.explore_neg_total += 1;
                    if !covered {
                        self.explore_neg_abstain += 1;
                    }
                }
            }
            Mode::Exploit => {
                if is_positive {
                    self.exploit_pos_total += 1;
                    if covered {
                        self.exploit_pos_covered += 1;
                        if correct {
                            self.exploit_pos_correct += 1;
                        }
                    }
                } else {
                    self.exploit_neg_total += 1;
                    if !covered {
                        self.exploit_neg_abstain += 1;
                    }
                }
            }
            Mode::Reset => {
                if is_positive {
                    self.reset_pos_total += 1;
                    if covered {
                        self.reset_pos_covered += 1;
                        if correct {
                            self.reset_pos_correct += 1;
                        }
                    }
                } else {
                    self.reset_neg_total += 1;
                    if !covered {
                        self.reset_neg_abstain += 1;
                    }
                }
            }
        }
    }

    // Computed metrics for each mode

    pub fn gate_pass_rate(&self, mode: Mode) -> f64 {
        match mode {
            Mode::Explore => {
                if self.explore_gate_total > 0 {
                    self.explore_gate_pass as f64 / self.explore_gate_total as f64
                } else {
                    0.0
                }
            }
            Mode::Exploit => {
                if self.exploit_gate_total > 0 {
                    self.exploit_gate_pass as f64 / self.exploit_gate_total as f64
                } else {
                    0.0
                }
            }
            Mode::Reset => {
                if self.reset_gate_total > 0 {
                    self.reset_gate_pass as f64 / self.reset_gate_total as f64
                } else {
                    0.0
                }
            }
        }
    }

    pub fn mean_abs_td(&self, mode: Mode) -> f64 {
        match mode {
            Mode::Explore => {
                if self.explore_ticks > 0 {
                    self.explore_abs_td_sum / self.explore_ticks as f64
                } else {
                    0.0
                }
            }
            Mode::Exploit => {
                if self.exploit_ticks > 0 {
                    self.exploit_abs_td_sum / self.exploit_ticks as f64
                } else {
                    0.0
                }
            }
            Mode::Reset => {
                if self.reset_ticks > 0 {
                    self.reset_abs_td_sum / self.reset_ticks as f64
                } else {
                    0.0
                }
            }
        }
    }

    pub fn mean_value(&self, mode: Mode) -> f64 {
        match mode {
            Mode::Explore => {
                if self.explore_ticks > 0 {
                    self.explore_value_sum / self.explore_ticks as f64
                } else {
                    0.0
                }
            }
            Mode::Exploit => {
                if self.exploit_ticks > 0 {
                    self.exploit_value_sum / self.exploit_ticks as f64
                } else {
                    0.0
                }
            }
            Mode::Reset => {
                if self.reset_ticks > 0 {
                    self.reset_value_sum / self.reset_ticks as f64
                } else {
                    0.0
                }
            }
        }
    }

    pub fn stable_share(&self, mode: Mode) -> f64 {
        match mode {
            Mode::Explore => {
                if self.explore_ticks > 0 {
                    self.explore_stable_ticks as f64 / self.explore_ticks as f64
                } else {
                    0.0
                }
            }
            Mode::Exploit => {
                if self.exploit_ticks > 0 {
                    self.exploit_stable_ticks as f64 / self.exploit_ticks as f64
                } else {
                    0.0
                }
            }
            Mode::Reset => {
                if self.reset_ticks > 0 {
                    self.reset_stable_ticks as f64 / self.reset_ticks as f64
                } else {
                    0.0
                }
            }
        }
    }

    pub fn ticks_pct(&self, mode: Mode) -> f64 {
        let total = self.explore_ticks + self.exploit_ticks + self.reset_ticks;
        if total == 0 {
            return 0.0;
        }
        match mode {
            Mode::Explore => self.explore_ticks as f64 / total as f64,
            Mode::Exploit => self.exploit_ticks as f64 / total as f64,
            Mode::Reset => self.reset_ticks as f64 / total as f64,
        }
    }

    pub fn coverage_pos(&self, mode: Mode) -> f64 {
        match mode {
            Mode::Explore => {
                if self.explore_pos_total > 0 {
                    self.explore_pos_covered as f64 / self.explore_pos_total as f64
                } else {
                    0.0
                }
            }
            Mode::Exploit => {
                if self.exploit_pos_total > 0 {
                    self.exploit_pos_covered as f64 / self.exploit_pos_total as f64
                } else {
                    0.0
                }
            }
            Mode::Reset => {
                if self.reset_pos_total > 0 {
                    self.reset_pos_covered as f64 / self.reset_pos_total as f64
                } else {
                    0.0
                }
            }
        }
    }

    pub fn selective_accuracy(&self, mode: Mode) -> f64 {
        match mode {
            Mode::Explore => {
                if self.explore_pos_covered > 0 {
                    self.explore_pos_correct as f64 / self.explore_pos_covered as f64
                } else {
                    0.0
                }
            }
            Mode::Exploit => {
                if self.exploit_pos_covered > 0 {
                    self.exploit_pos_correct as f64 / self.exploit_pos_covered as f64
                } else {
                    0.0
                }
            }
            Mode::Reset => {
                if self.reset_pos_covered > 0 {
                    self.reset_pos_correct as f64 / self.reset_pos_covered as f64
                } else {
                    0.0
                }
            }
        }
    }
}

/// Reset targeting statistics.
#[derive(Clone, Debug, Default)]
pub struct ResetTargetStats {
    /// Total number of resets
    pub reset_count: usize,
    /// Total nodes dampened
    pub total_nodes_dampened: usize,
    /// Nodes dampened that were off-prototype
    pub off_proto_dampened: usize,
    /// Nodes dampened that were on-prototype
    pub on_proto_dampened: usize,
}

impl ResetTargetStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_reset(&mut self, nodes_dampened: usize, off_proto: usize, on_proto: usize) {
        self.reset_count += 1;
        self.total_nodes_dampened += nodes_dampened;
        self.off_proto_dampened += off_proto;
        self.on_proto_dampened += on_proto;
    }

    pub fn avg_dampened_per_reset(&self) -> f64 {
        if self.reset_count > 0 {
            self.total_nodes_dampened as f64 / self.reset_count as f64
        } else {
            0.0
        }
    }

    pub fn off_proto_fraction(&self) -> f64 {
        if self.total_nodes_dampened > 0 {
            self.off_proto_dampened as f64 / self.total_nodes_dampened as f64
        } else {
            0.0
        }
    }
}

/// Report for a single ablation variant run.
#[derive(Clone, Debug)]
pub struct VariantReport {
    pub label: String,
    pub ablation_config: AblationConfig,

    // Mode rates
    pub explore_rate: f64,
    pub exploit_rate: f64,
    pub reset_rate: f64,

    // Global metrics
    pub coverage_pos: f64,
    pub selective_accuracy: f64,
    pub false_positive_rate: f64,
    pub stable_drop_ratio: f64,
    pub merges_done_proto: usize,
    pub avg_merge_score: f32,

    // Reset effectiveness
    pub reset_effectiveness_mean: f64,
    pub reset_effectiveness_count: usize,

    // Per-mode stats
    pub per_mode: PerModeStats,

    // Reset target stats
    pub reset_target_stats: ResetTargetStats,
}

impl VariantReport {
    pub fn new(label: &str, config: AblationConfig) -> Self {
        Self {
            label: label.to_string(),
            ablation_config: config,
            explore_rate: 0.0,
            exploit_rate: 0.0,
            reset_rate: 0.0,
            coverage_pos: 0.0,
            selective_accuracy: 0.0,
            false_positive_rate: 0.0,
            stable_drop_ratio: 0.0,
            merges_done_proto: 0,
            avg_merge_score: 0.0,
            reset_effectiveness_mean: 0.0,
            reset_effectiveness_count: 0,
            per_mode: PerModeStats::new(),
            reset_target_stats: ResetTargetStats::new(),
        }
    }

    /// Print per-mode diagnostics table.
    pub fn print_per_mode_table(&self) {
        println!(
            "  {:8} | {:>7} | {:>10} | {:>8} | {:>8} | {:>12}",
            "Mode", "ticks%", "gate_pass%", "mean|TD|", "meanV", "stable_share%"
        );
        println!("  {}", "-".repeat(65));

        for mode in [Mode::Explore, Mode::Exploit, Mode::Reset] {
            let ticks_pct = self.per_mode.ticks_pct(mode) * 100.0;
            if ticks_pct < 0.01 {
                continue; // Skip modes with no ticks
            }
            let mode_name = match mode {
                Mode::Explore => "Explore",
                Mode::Exploit => "Exploit",
                Mode::Reset => "Reset",
            };
            println!(
                "  {:8} | {:6.1}% | {:9.1}% | {:8.4} | {:+7.3} | {:11.1}%",
                mode_name,
                ticks_pct,
                self.per_mode.gate_pass_rate(mode) * 100.0,
                self.per_mode.mean_abs_td(mode),
                self.per_mode.mean_value(mode),
                self.per_mode.stable_share(mode) * 100.0
            );
        }
    }
}

/// Select which nodes to dampen based on target mode.
pub fn select_reset_targets(
    topk: &[(usize, f64)],
    proto_nodes: &[u8],
    proto_weights: &[f32],
    proto_m: usize,
    target_mode: ResetTargetMode,
    max_nodes: usize,
    _margin: f32,
    _abs_td: f32,
) -> (Vec<usize>, usize, usize) {
    // Returns: (node_ids_to_dampen, off_proto_count, on_proto_count)

    match target_mode {
        ResetTargetMode::TopK => {
            // Original behavior: dampen all top-K up to max
            let nodes: Vec<usize> = topk.iter().take(max_nodes).map(|(id, _)| *id).collect();
            let _count = nodes.len();
            // Count how many are on-proto vs off-proto
            let mut on_proto = 0;
            let mut off_proto = 0;
            for &node_id in &nodes {
                if is_in_proto(node_id as u8, proto_nodes, proto_weights, proto_m) {
                    on_proto += 1;
                } else {
                    off_proto += 1;
                }
            }
            (nodes, off_proto, on_proto)
        }
        ResetTargetMode::BadActors => {
            // Dampen only nodes NOT in the prototype (off-prototype energy)
            let mut off_proto_nodes = Vec::new();
            let mut on_proto_nodes = Vec::new();

            for (node_id, _amp) in topk.iter().take(max_nodes * 2) {
                // Look at more than max_nodes to find bad actors
                if is_in_proto(*node_id as u8, proto_nodes, proto_weights, proto_m) {
                    on_proto_nodes.push(*node_id);
                } else {
                    off_proto_nodes.push(*node_id);
                }
            }

            // Prioritize off-proto nodes
            let mut result = Vec::new();
            for node_id in off_proto_nodes.iter().take(max_nodes) {
                result.push(*node_id);
            }

            // If we haven't filled max_nodes, add some on-proto nodes
            let remaining = max_nodes.saturating_sub(result.len());
            for node_id in on_proto_nodes.iter().take(remaining) {
                result.push(*node_id);
            }

            let off_count = result
                .iter()
                .filter(|&&id| !is_in_proto(id as u8, proto_nodes, proto_weights, proto_m))
                .count();
            let on_count = result.len() - off_count;

            (result, off_count, on_count)
        }
    }
}

/// Check if a node is in the prototype.
fn is_in_proto(node_id: u8, proto_nodes: &[u8], proto_weights: &[f32], proto_m: usize) -> bool {
    for i in 0..proto_m.min(proto_nodes.len()) {
        if proto_nodes[i] == node_id && proto_weights[i] > 0.0 {
            return true;
        }
    }
    false
}
