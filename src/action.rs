//! Phase 2.0c: Mode → Action Binding
//!
//! Introduces explicit ACTION SELECTION driven by ModePolicy,
//! forming the first closed perception-action loop.
//!
//! Actions are controlled, reversible perturbations of sampling
//! and gating — executive control only.

use crate::mode::Mode;

/// The three executive actions available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Broaden perception (widen Top-K, loosen gating)
    Scan,
    /// Concentrate on current concept (tighten Top-K, stricter gating)
    Focus,
    /// Mild controlled disruption (tiny noise or dampening)
    Perturb,
}

impl Action {
    /// Deterministic Mode → Action mapping.
    pub fn from_mode(mode: Mode) -> Self {
        match mode {
            Mode::Explore => Action::Scan,
            Mode::Exploit => Action::Focus,
            Mode::Reset => Action::Perturb,
        }
    }
}

/// Configuration for action effects.
#[derive(Clone, Debug)]
pub struct ActionConfig {
    /// Scale factor for Top-K during Scan (> 1.0 = more nodes)
    pub scan_topk_scale: f32,
    /// Scale factor for Top-K during Focus (< 1.0 = fewer nodes)
    pub focus_topk_scale: f32,
    /// Scale factor for margin during Scan (< 1.0 = looser)
    pub scan_margin_scale: f32,
    /// Scale factor for margin during Focus (> 1.0 = tighter)
    pub focus_margin_scale: f32,
    /// Noise amplitude for Perturb action
    pub perturb_noise_amp: f32,
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            scan_topk_scale: 1.25,
            focus_topk_scale: 0.80,
            scan_margin_scale: 0.80,
            focus_margin_scale: 1.30,
            perturb_noise_amp: 0.02,
        }
    }
}

/// Overrides to apply for the current action.
#[derive(Clone, Debug, Default)]
pub struct ActionOverrides {
    /// Scale factor for Top-K count (1.0 = no change)
    pub topk_scale: f32,
    /// Scale factor for margin threshold (1.0 = no change)
    pub margin_scale: f32,
    /// Whether to apply noise perturbation
    pub apply_noise: bool,
    /// Noise amplitude if apply_noise is true
    pub noise_amp: f32,
}

/// Statistics tracking for action policy.
#[derive(Clone, Debug, Default)]
pub struct ActionPolicyStats {
    // Action counts
    pub scan_count: usize,
    pub focus_count: usize,
    pub perturb_count: usize,

    // Gate pass tracking per action
    pub scan_gate_pass: usize,
    pub scan_gate_total: usize,
    pub focus_gate_pass: usize,
    pub focus_gate_total: usize,
    pub perturb_gate_pass: usize,
    pub perturb_gate_total: usize,

    // Sum of |TD| per action (for computing mean)
    pub scan_abs_td_sum: f64,
    pub focus_abs_td_sum: f64,
    pub perturb_abs_td_sum: f64,

    // Sum of anchor value per action
    pub scan_value_sum: f64,
    pub focus_value_sum: f64,
    pub perturb_value_sum: f64,

    // Stable ticks per action
    pub scan_stable_ticks: usize,
    pub focus_stable_ticks: usize,
    pub perturb_stable_ticks: usize,

    // Perturb effectiveness tracking (reuse reset metric)
    pub perturb_pre_td_sum: f64,
    pub perturb_post_td_sum: f64,
    pub perturb_effectiveness_samples: usize,
}

impl ActionPolicyStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a tick for the given action.
    pub fn record_tick(
        &mut self,
        action: Action,
        gate_passed: bool,
        abs_td: f32,
        anchor_value: f32,
        is_stable: bool,
    ) {
        match action {
            Action::Scan => {
                self.scan_count += 1;
                self.scan_gate_total += 1;
                if gate_passed {
                    self.scan_gate_pass += 1;
                }
                self.scan_abs_td_sum += abs_td as f64;
                self.scan_value_sum += anchor_value as f64;
                if is_stable {
                    self.scan_stable_ticks += 1;
                }
            }
            Action::Focus => {
                self.focus_count += 1;
                self.focus_gate_total += 1;
                if gate_passed {
                    self.focus_gate_pass += 1;
                }
                self.focus_abs_td_sum += abs_td as f64;
                self.focus_value_sum += anchor_value as f64;
                if is_stable {
                    self.focus_stable_ticks += 1;
                }
            }
            Action::Perturb => {
                self.perturb_count += 1;
                self.perturb_gate_total += 1;
                if gate_passed {
                    self.perturb_gate_pass += 1;
                }
                self.perturb_abs_td_sum += abs_td as f64;
                self.perturb_value_sum += anchor_value as f64;
                if is_stable {
                    self.perturb_stable_ticks += 1;
                }
            }
        }
    }

    /// Record perturb effectiveness (pre/post TD).
    pub fn record_perturb_effectiveness(&mut self, pre_td: f32, post_td: f32) {
        self.perturb_pre_td_sum += pre_td as f64;
        self.perturb_post_td_sum += post_td as f64;
        self.perturb_effectiveness_samples += 1;
    }

    // Computed metrics

    pub fn total_count(&self) -> usize {
        self.scan_count + self.focus_count + self.perturb_count
    }

    pub fn scan_rate(&self) -> f64 {
        let total = self.total_count();
        if total > 0 {
            self.scan_count as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn focus_rate(&self) -> f64 {
        let total = self.total_count();
        if total > 0 {
            self.focus_count as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn perturb_rate(&self) -> f64 {
        let total = self.total_count();
        if total > 0 {
            self.perturb_count as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn gate_pass_rate(&self, action: Action) -> f64 {
        match action {
            Action::Scan => {
                if self.scan_gate_total > 0 {
                    self.scan_gate_pass as f64 / self.scan_gate_total as f64
                } else {
                    0.0
                }
            }
            Action::Focus => {
                if self.focus_gate_total > 0 {
                    self.focus_gate_pass as f64 / self.focus_gate_total as f64
                } else {
                    0.0
                }
            }
            Action::Perturb => {
                if self.perturb_gate_total > 0 {
                    self.perturb_gate_pass as f64 / self.perturb_gate_total as f64
                } else {
                    0.0
                }
            }
        }
    }

    pub fn mean_abs_td(&self, action: Action) -> f64 {
        match action {
            Action::Scan => {
                if self.scan_count > 0 {
                    self.scan_abs_td_sum / self.scan_count as f64
                } else {
                    0.0
                }
            }
            Action::Focus => {
                if self.focus_count > 0 {
                    self.focus_abs_td_sum / self.focus_count as f64
                } else {
                    0.0
                }
            }
            Action::Perturb => {
                if self.perturb_count > 0 {
                    self.perturb_abs_td_sum / self.perturb_count as f64
                } else {
                    0.0
                }
            }
        }
    }

    pub fn mean_value(&self, action: Action) -> f64 {
        match action {
            Action::Scan => {
                if self.scan_count > 0 {
                    self.scan_value_sum / self.scan_count as f64
                } else {
                    0.0
                }
            }
            Action::Focus => {
                if self.focus_count > 0 {
                    self.focus_value_sum / self.focus_count as f64
                } else {
                    0.0
                }
            }
            Action::Perturb => {
                if self.perturb_count > 0 {
                    self.perturb_value_sum / self.perturb_count as f64
                } else {
                    0.0
                }
            }
        }
    }

    pub fn stable_share(&self, action: Action) -> f64 {
        match action {
            Action::Scan => {
                if self.scan_count > 0 {
                    self.scan_stable_ticks as f64 / self.scan_count as f64
                } else {
                    0.0
                }
            }
            Action::Focus => {
                if self.focus_count > 0 {
                    self.focus_stable_ticks as f64 / self.focus_count as f64
                } else {
                    0.0
                }
            }
            Action::Perturb => {
                if self.perturb_count > 0 {
                    self.perturb_stable_ticks as f64 / self.perturb_count as f64
                } else {
                    0.0
                }
            }
        }
    }

    /// Perturb effectiveness: mean TD reduction ratio.
    pub fn perturb_effectiveness(&self) -> f64 {
        if self.perturb_effectiveness_samples > 0 && self.perturb_pre_td_sum > 0.001 {
            let pre_mean = self.perturb_pre_td_sum / self.perturb_effectiveness_samples as f64;
            let post_mean = self.perturb_post_td_sum / self.perturb_effectiveness_samples as f64;
            (pre_mean - post_mean) / pre_mean
        } else {
            0.0
        }
    }
}

/// The action policy controller.
#[derive(Clone, Debug)]
pub struct ActionPolicy {
    pub config: ActionConfig,
    pub stats: ActionPolicyStats,
}

impl ActionPolicy {
    pub fn new(config: ActionConfig) -> Self {
        Self {
            config,
            stats: ActionPolicyStats::new(),
        }
    }

    /// Choose action based on current mode.
    pub fn choose_action(&self, mode: Mode) -> Action {
        Action::from_mode(mode)
    }

    /// Choose action with ablation configuration applied.
    pub fn choose_action_with_ablation(
        &self,
        mode: Mode,
        variant_config: &crate::action_ablate::VariantConfig,
        rng_value: f64,
    ) -> Action {
        let base_action = Action::from_mode(mode);
        crate::action_ablate::choose_action_with_ablation(base_action, variant_config, rng_value)
    }

    /// Reset stats for a new run.
    pub fn reset_stats(&mut self) {
        self.stats = ActionPolicyStats::new();
    }

    /// Get the overrides to apply for the given action.
    pub fn get_overrides(&self, action: Action) -> ActionOverrides {
        match action {
            Action::Scan => ActionOverrides {
                topk_scale: self.config.scan_topk_scale,
                margin_scale: self.config.scan_margin_scale,
                apply_noise: false,
                noise_amp: 0.0,
            },
            Action::Focus => ActionOverrides {
                topk_scale: self.config.focus_topk_scale,
                margin_scale: self.config.focus_margin_scale,
                apply_noise: false,
                noise_amp: 0.0,
            },
            Action::Perturb => ActionOverrides {
                topk_scale: 1.0,   // No Top-K change for perturb
                margin_scale: 1.0, // No margin change for perturb
                apply_noise: true,
                noise_amp: self.config.perturb_noise_amp,
            },
        }
    }

    /// Record a tick observation.
    pub fn record_tick(
        &mut self,
        action: Action,
        gate_passed: bool,
        abs_td: f32,
        anchor_value: f32,
        is_stable: bool,
    ) {
        self.stats
            .record_tick(action, gate_passed, abs_td, anchor_value, is_stable);
    }

    /// Record perturb effectiveness.
    pub fn record_perturb_effectiveness(&mut self, pre_td: f32, post_td: f32) {
        self.stats.record_perturb_effectiveness(pre_td, post_td);
    }

    /// Get computed Top-K count for action.
    pub fn effective_topk(&self, base_topk: usize, action: Action) -> usize {
        let overrides = self.get_overrides(action);
        let scaled = (base_topk as f32 * overrides.topk_scale).round() as usize;
        scaled.max(1) // At least 1 node
    }
}
