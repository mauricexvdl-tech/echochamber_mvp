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

// =============================================================================
// Phase 2.0c-FIX: Perturb Trigger System
// =============================================================================

/// Tracks conditions that indicate "bad states" requiring intervention.
#[derive(Clone, Debug, Default)]
pub struct ActionTriggers {
    /// Consecutive ticks where gate failed.
    pub gate_fail_streak: u32,
    /// Consecutive ticks with low margin (weak winner separation).
    pub low_margin_streak: u32,
    /// Consecutive ticks with low proto alignment.
    pub off_proto_streak: u32,
    /// Consecutive ticks with declining value.
    pub value_drop_streak: u32,
    /// Cooldown counter (ticks since last perturb).
    pub cooldown: u32,
    /// Rolling value buffer for drop detection.
    value_buffer: [f32; 32],
    /// Write index into value buffer.
    value_idx: usize,
    /// Number of values written.
    value_count: usize,
}

/// Statistics for perturb trigger breakdown.
#[derive(Clone, Debug, Default)]
pub struct PerturbTriggerStats {
    pub by_mode_reset: usize, // Triggered by Mode::Reset
    pub by_high_td: usize,    // Triggered by high |TD|
    pub by_gate_fail: usize,  // Triggered by gate fail streak
    pub by_low_margin: usize, // Triggered by low margin streak
    pub by_off_proto: usize,  // Triggered by off-proto streak
    pub by_value_drop: usize, // Triggered by value drop streak
    pub by_floor: usize,      // Triggered by floor mechanism
}

impl PerturbTriggerStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total(&self) -> usize {
        self.by_mode_reset
            + self.by_high_td
            + self.by_gate_fail
            + self.by_low_margin
            + self.by_off_proto
            + self.by_value_drop
            + self.by_floor
    }
}

/// Rolling window for perturb floor calculation.
#[derive(Clone, Debug)]
pub struct PerturbFloor {
    /// Rolling action counts: 0=Scan, 1=Focus, 2=Perturb.
    actions: Vec<u8>,
    /// Window size.
    window_size: usize,
    /// Write index.
    idx: usize,
    /// Number of entries written.
    count: usize,
    /// Target minimum perturb rate.
    min_rate: f32,
}

impl PerturbFloor {
    pub fn new(window_size: usize, min_rate: f32) -> Self {
        Self {
            actions: vec![0; window_size],
            window_size,
            idx: 0,
            count: 0,
            min_rate,
        }
    }

    /// Record an action.
    pub fn record(&mut self, action: Action) {
        let code = match action {
            Action::Scan => 0,
            Action::Focus => 1,
            Action::Perturb => 2,
        };
        self.actions[self.idx] = code;
        self.idx = (self.idx + 1) % self.window_size;
        if self.count < self.window_size {
            self.count += 1;
        }
    }

    /// Get current perturb rate in window.
    pub fn perturb_rate(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let perturb_count = self.actions[..self.count]
            .iter()
            .filter(|&&a| a == 2)
            .count();
        perturb_count as f32 / self.count as f32
    }

    /// Check if perturb rate is below floor.
    pub fn below_floor(&self) -> bool {
        // Only enforce floor after warm-up period
        if self.count < self.window_size / 4 {
            return false;
        }
        self.perturb_rate() < self.min_rate
    }
}

impl ActionTriggers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update trigger state based on current tick observations.
    pub fn update(
        &mut self,
        gate_passed: bool,
        topk_margin: f32,
        proto_align: f32,
        anchor_value: f32,
        margin_min: f32,
        proto_min: f32,
        value_drop_threshold: f32,
    ) {
        // Update gate fail streak
        if gate_passed {
            self.gate_fail_streak = 0;
        } else {
            self.gate_fail_streak += 1;
        }

        // Update low margin streak
        if topk_margin < margin_min {
            self.low_margin_streak += 1;
        } else {
            self.low_margin_streak = 0;
        }

        // Update off-proto streak
        if proto_align < proto_min {
            self.off_proto_streak += 1;
        } else {
            self.off_proto_streak = 0;
        }

        // Update value buffer and compute drop streak
        self.value_buffer[self.value_idx] = anchor_value;
        self.value_idx = (self.value_idx + 1) % 32;
        if self.value_count < 32 {
            self.value_count += 1;
        }

        // Check for value drop (compare recent avg to older avg)
        if self.value_count >= 16 {
            let recent_start = (self.value_idx + 32 - 8) % 32;
            let older_start = (self.value_idx + 32 - 24) % 32;

            let mut recent_sum = 0.0f32;
            let mut older_sum = 0.0f32;
            for i in 0..8 {
                recent_sum += self.value_buffer[(recent_start + i) % 32];
                older_sum += self.value_buffer[(older_start + i) % 32];
            }
            let recent_avg = recent_sum / 8.0;
            let older_avg = older_sum / 8.0;

            if older_avg - recent_avg > value_drop_threshold {
                self.value_drop_streak += 1;
            } else {
                self.value_drop_streak = 0;
            }
        }

        // Decrement cooldown
        if self.cooldown > 0 {
            self.cooldown -= 1;
        }
    }

    /// Check if any trigger condition is met.
    pub fn should_perturb(
        &self,
        abs_td: f32,
        td_threshold: f32,
        fail_streak_threshold: u32,
        margin_streak_threshold: u32,
        offproto_streak_threshold: u32,
        value_streak_threshold: u32,
    ) -> Option<&'static str> {
        // Cooldown active - no trigger
        if self.cooldown > 0 {
            return None;
        }

        // Check triggers in priority order
        if abs_td >= td_threshold {
            return Some("high_td");
        }
        if self.gate_fail_streak >= fail_streak_threshold {
            return Some("gate_fail");
        }
        if self.low_margin_streak >= margin_streak_threshold {
            return Some("low_margin");
        }
        if self.off_proto_streak >= offproto_streak_threshold {
            return Some("off_proto");
        }
        if self.value_drop_streak >= value_streak_threshold {
            return Some("value_drop");
        }

        None
    }

    /// Reset trigger after perturb is applied.
    pub fn on_perturb(&mut self, cooldown_ticks: u32) {
        self.gate_fail_streak = 0;
        self.low_margin_streak = 0;
        self.off_proto_streak = 0;
        self.value_drop_streak = 0;
        self.cooldown = cooldown_ticks;
    }
}

/// The action policy controller.
#[derive(Clone, Debug)]
pub struct ActionPolicy {
    pub config: ActionConfig,
    pub stats: ActionPolicyStats,
    pub triggers: ActionTriggers,
    pub trigger_stats: PerturbTriggerStats,
    pub floor: Option<PerturbFloor>,
}

impl ActionPolicy {
    pub fn new(config: ActionConfig) -> Self {
        Self {
            config,
            stats: ActionPolicyStats::new(),
            triggers: ActionTriggers::new(),
            trigger_stats: PerturbTriggerStats::new(),
            floor: None,
        }
    }

    /// Create with perturb floor enabled.
    pub fn new_with_floor(config: ActionConfig, floor_window: usize, floor_min_rate: f32) -> Self {
        Self {
            config,
            stats: ActionPolicyStats::new(),
            triggers: ActionTriggers::new(),
            trigger_stats: PerturbTriggerStats::new(),
            floor: Some(PerturbFloor::new(floor_window, floor_min_rate)),
        }
    }

    /// Choose action based on current mode (basic, no triggers).
    pub fn choose_action(&self, mode: Mode) -> Action {
        Action::from_mode(mode)
    }

    /// Choose action with trigger system enabled.
    /// Returns (action, trigger_reason) where trigger_reason is Some if perturb was triggered.
    pub fn choose_action_with_triggers(
        &mut self,
        mode: Mode,
        abs_td: f32,
        gate_passed: bool,
        topk_margin: f32,
        proto_align: f32,
        anchor_value: f32,
        config: &crate::config::Config,
    ) -> (Action, Option<&'static str>) {
        // Update trigger state
        self.triggers.update(
            gate_passed,
            topk_margin as f32,
            proto_align,
            anchor_value,
            config.perturb_trig_margin_min,
            config.perturb_trig_proto_min,
            config.perturb_trig_value_drop,
        );

        // Base action from mode
        let base_action = Action::from_mode(mode);

        // If mode already wants Perturb (Reset mode), use it
        if base_action == Action::Perturb {
            self.triggers.on_perturb(config.perturb_cooldown_ticks);
            return (Action::Perturb, Some("mode_reset"));
        }

        // Check if extra triggers are enabled
        if !config.perturb_extra_triggers {
            return (base_action, None);
        }

        // Check trigger conditions
        if let Some(reason) = self.triggers.should_perturb(
            abs_td,
            config.mode_reset_td_min,
            config.perturb_trig_fail_streak,
            config.perturb_trig_margin_streak,
            config.perturb_trig_offproto_streak,
            config.perturb_trig_value_streak,
        ) {
            self.triggers.on_perturb(config.perturb_cooldown_ticks);
            return (Action::Perturb, Some(reason));
        }

        // Check floor mechanism
        if config.perturb_floor_enabled {
            if let Some(ref floor) = self.floor {
                if floor.below_floor() && self.triggers.cooldown == 0 {
                    // Floor trigger: use weaker conditions
                    // Trigger if any of: gate failed, low margin, or off-proto (single tick)
                    if !gate_passed
                        || topk_margin < config.perturb_trig_margin_min * 2.0
                        || proto_align < config.perturb_trig_proto_min * 1.5
                    {
                        self.triggers.on_perturb(config.perturb_cooldown_ticks);
                        return (Action::Perturb, Some("floor"));
                    }
                }
            }
        }

        (base_action, None)
    }

    /// Record action in floor window (call after choosing action).
    pub fn record_action_for_floor(&mut self, action: Action) {
        if let Some(ref mut floor) = self.floor {
            floor.record(action);
        }
    }

    /// Update trigger stats based on trigger reason.
    pub fn record_trigger(&mut self, reason: Option<&'static str>) {
        if let Some(r) = reason {
            match r {
                "mode_reset" => self.trigger_stats.by_mode_reset += 1,
                "high_td" => self.trigger_stats.by_high_td += 1,
                "gate_fail" => self.trigger_stats.by_gate_fail += 1,
                "low_margin" => self.trigger_stats.by_low_margin += 1,
                "off_proto" => self.trigger_stats.by_off_proto += 1,
                "value_drop" => self.trigger_stats.by_value_drop += 1,
                "floor" => self.trigger_stats.by_floor += 1,
                _ => {}
            }
        }
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

    /// Phase 2.1c: Choose action with post-rescue lock bias toward Focus.
    /// During post-rescue lock, strongly prefer Focus over Scan unless TD spike.
    pub fn choose_action_with_lock(
        &self,
        mode: Mode,
        lock_active: bool,
        lock_focus_bias: f32,
        abs_td: f32,
        td_threshold: f32,
    ) -> Action {
        let base_action = Action::from_mode(mode);

        // If not in lock or base action is already Perturb, return base
        if !lock_active || base_action == Action::Perturb {
            return base_action;
        }

        // During lock: check if TD spike warrants Perturb despite lock
        if abs_td >= td_threshold {
            return Action::Perturb;
        }

        // During lock: bias toward Focus
        // If mode says Scan (Explore mode), override to Focus if bias is strong enough
        if base_action == Action::Scan && lock_focus_bias >= 1.0 {
            return Action::Focus;
        }

        base_action
    }

    /// Phase 2.1b: Check if min perturb guard should force a perturb action.
    /// Returns true if perturb should be forced due to low perturb rate + bad state.
    pub fn should_force_perturb_guard(
        &self,
        min_rate: f32,
        gate_passed: bool,
        topk_margin: f32,
        proto_align: f32,
        margin_threshold: f32,
        proto_threshold: f32,
    ) -> bool {
        // Only check if floor exists and cooldown is clear
        if self.triggers.cooldown > 0 {
            return false;
        }

        if let Some(ref floor) = self.floor {
            // Check if rate is below minimum
            if floor.perturb_rate() < min_rate {
                // Check for bad state indicators
                let is_bad_state =
                    !gate_passed || topk_margin < margin_threshold || proto_align < proto_threshold;

                if is_bad_state {
                    return true;
                }
            }
        }

        false
    }
}
