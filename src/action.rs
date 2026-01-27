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

    // Phase 2.2b: Helper methods to reduce match duplication
    /// Get raw stats for an action: (count, gate_pass, gate_total, abs_td_sum, value_sum, stable_ticks)
    fn get_action_raw(&self, action: Action) -> (usize, usize, usize, f64, f64, usize) {
        match action {
            Action::Scan => (
                self.scan_count,
                self.scan_gate_pass,
                self.scan_gate_total,
                self.scan_abs_td_sum,
                self.scan_value_sum,
                self.scan_stable_ticks,
            ),
            Action::Focus => (
                self.focus_count,
                self.focus_gate_pass,
                self.focus_gate_total,
                self.focus_abs_td_sum,
                self.focus_value_sum,
                self.focus_stable_ticks,
            ),
            Action::Perturb => (
                self.perturb_count,
                self.perturb_gate_pass,
                self.perturb_gate_total,
                self.perturb_abs_td_sum,
                self.perturb_value_sum,
                self.perturb_stable_ticks,
            ),
        }
    }

    /// Safe division helper: returns 0.0 if denominator is 0.
    #[inline]
    fn safe_div(num: f64, denom: usize) -> f64 {
        if denom > 0 {
            num / denom as f64
        } else {
            0.0
        }
    }

    pub fn gate_pass_rate(&self, action: Action) -> f64 {
        let (_, gate_pass, gate_total, _, _, _) = self.get_action_raw(action);
        Self::safe_div(gate_pass as f64, gate_total)
    }

    pub fn mean_abs_td(&self, action: Action) -> f64 {
        let (count, _, _, abs_td_sum, _, _) = self.get_action_raw(action);
        Self::safe_div(abs_td_sum, count)
    }

    pub fn mean_value(&self, action: Action) -> f64 {
        let (count, _, _, _, value_sum, _) = self.get_action_raw(action);
        Self::safe_div(value_sum, count)
    }

    pub fn stable_share(&self, action: Action) -> f64 {
        let (count, _, _, _, _, stable_ticks) = self.get_action_raw(action);
        Self::safe_div(stable_ticks as f64, count)
    }

    // Legacy methods kept for backwards compatibility
    pub fn _stable_share_legacy(&self, action: Action) -> f64 {
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
/// Phase 2.2b: Buffer sizes are now configurable instead of hardcoded.
#[derive(Clone, Debug)]
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
    /// Rolling value buffer for drop detection (configurable size).
    value_buffer: Vec<f32>,
    /// Write index into value buffer.
    value_idx: usize,
    /// Number of values written.
    value_count: usize,
    /// Buffer size (from config).
    buffer_size: usize,
    /// Minimum samples before drop detection (from config).
    min_samples: usize,
    /// Window size for average comparison (from config).
    window_size: usize,
}

impl Default for ActionTriggers {
    fn default() -> Self {
        Self::new(32, 16, 8) // Legacy defaults
    }
}

impl ActionTriggers {
    /// Create with configurable buffer parameters.
    pub fn new(buffer_size: usize, min_samples: usize, window_size: usize) -> Self {
        Self {
            gate_fail_streak: 0,
            low_margin_streak: 0,
            off_proto_streak: 0,
            value_drop_streak: 0,
            cooldown: 0,
            value_buffer: vec![0.0; buffer_size],
            value_idx: 0,
            value_count: 0,
            buffer_size,
            min_samples,
            window_size,
        }
    }

    /// Create from config.
    pub fn from_config(config: &crate::config::Config) -> Self {
        Self::new(
            config.action_trigger_buffer_size,
            config.action_trigger_min_samples,
            config.action_trigger_window_size,
        )
    }
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

/// Phase 2.1e: Rolling window for perturb budget cap.
#[derive(Clone, Debug)]
pub struct PerturbBudget {
    /// Rolling action counts: true=Perturb, false=other.
    actions: Vec<bool>,
    /// Window size.
    window_size: usize,
    /// Write index.
    idx: usize,
    /// Number of entries written.
    count: usize,
    /// Maximum perturb rate (hard cap).
    max_rate: f32,
}

impl PerturbBudget {
    pub fn new(window_size: usize, max_rate: f32) -> Self {
        Self {
            actions: vec![false; window_size],
            window_size,
            idx: 0,
            count: 0,
            max_rate,
        }
    }

    /// Record an action.
    pub fn record(&mut self, is_perturb: bool) {
        self.actions[self.idx] = is_perturb;
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
        let perturb_count = self.actions[..self.count].iter().filter(|&&a| a).count();
        perturb_count as f32 / self.count as f32
    }

    /// Check if perturb rate is over cap.
    pub fn over_cap(&self) -> bool {
        // Only enforce after warm-up period
        if self.count < self.window_size / 4 {
            return false;
        }
        self.perturb_rate() >= self.max_rate
    }
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
    /// Update trigger state based on current tick observations.
    /// Phase 2.2b: Uses configurable buffer_size, min_samples, window_size.
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
        // Phase 2.2b: Uses configurable buffer_size instead of hardcoded 32
        self.value_buffer[self.value_idx] = anchor_value;
        self.value_idx = (self.value_idx + 1) % self.buffer_size;
        if self.value_count < self.buffer_size {
            self.value_count += 1;
        }

        // Check for value drop (compare recent avg to older avg)
        // Phase 2.2b: Uses configurable min_samples and window_size
        if self.value_count >= self.min_samples {
            let window = self.window_size;
            let buffer_size = self.buffer_size;
            // Recent window: last `window` samples
            let recent_start = (self.value_idx + buffer_size - window) % buffer_size;
            // Older window: samples from `window*3` to `window*2` ago
            let older_start = (self.value_idx + buffer_size - window * 3) % buffer_size;

            let mut recent_sum = 0.0f32;
            let mut older_sum = 0.0f32;
            for i in 0..window {
                recent_sum += self.value_buffer[(recent_start + i) % buffer_size];
                older_sum += self.value_buffer[(older_start + i) % buffer_size];
            }
            let recent_avg = recent_sum / window as f32;
            let older_avg = older_sum / window as f32;

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

// =============================================================================
// Phase 2.1s: Episodic Burst Effectiveness Tracking
// =============================================================================

/// Record for a single burst episode (Phase 2.1s).
#[derive(Clone, Debug, Default)]
pub struct BurstEpisodeRecord {
    /// Tick when burst started.
    pub start_tick: u64,
    /// Tick when burst ended.
    pub end_tick: u64,
    /// Burst duration in ticks.
    pub duration: u32,
    /// Burst probability used.
    pub prob_used: f32,
    /// Pre-burst metrics
    pub pre_abs_td_mean: f32,
    pub pre_stable_share: f32,
    pub pre_bad_share: f32,
    /// Post-burst metrics
    pub post_abs_td_mean: f32,
    pub post_stable_share: f32,
    pub post_bad_share: f32,
    /// Whether this burst was successful.
    pub success: bool,
    /// Success reason (if any).
    pub success_reason: Option<&'static str>,
}

/// Aggregate burst effectiveness statistics (Phase 2.1s).
#[derive(Clone, Debug, Default)]
pub struct BurstEffectivenessStats {
    /// Total burst episodes completed (with post-window measured).
    pub episodes_completed: u32,
    /// Episodes that were successful.
    pub episodes_success: u32,
    /// Sum of TD improvements (pre - post) for all episodes.
    pub td_improve_sum: f32,
    /// Sum of bad share improvements (pre - post) for all episodes.
    pub bad_improve_sum: f32,
    /// Sum of stable share gains (post - pre) for all episodes.
    pub stable_gain_sum: f32,
    /// Count of successes by reason.
    pub success_by_td: u32,
    pub success_by_bad: u32,
    pub success_by_stable: u32,
    /// Individual TD improvements for percentile calculation (Phase 2.1u diagnostic).
    pub td_improve_values: Vec<f32>,
    /// Sum of pre-burst TD values for mean_pre_td calculation.
    pub pre_td_sum: f32,
}

impl BurstEffectivenessStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Success rate as fraction.
    pub fn success_rate(&self) -> f64 {
        if self.episodes_completed > 0 {
            self.episodes_success as f64 / self.episodes_completed as f64
        } else {
            0.0
        }
    }

    /// Mean TD improvement as percentage.
    pub fn mean_td_improve_pct(&self) -> f64 {
        if self.episodes_completed > 0 {
            self.td_improve_sum as f64 / self.episodes_completed as f64 * 100.0
        } else {
            0.0
        }
    }

    /// Mean bad share improvement (absolute).
    pub fn mean_bad_improve(&self) -> f64 {
        if self.episodes_completed > 0 {
            self.bad_improve_sum as f64 / self.episodes_completed as f64
        } else {
            0.0
        }
    }

    /// Mean stable share gain (absolute).
    pub fn mean_stable_gain(&self) -> f64 {
        if self.episodes_completed > 0 {
            self.stable_gain_sum as f64 / self.episodes_completed as f64
        } else {
            0.0
        }
    }

    /// Mean pre-burst TD (for normalization diagnostic).
    pub fn mean_pre_td(&self) -> f64 {
        if self.episodes_completed > 0 {
            self.pre_td_sum as f64 / self.episodes_completed as f64
        } else {
            0.0
        }
    }

    /// TD improve percentile (p50 or p90). Returns 0 if no data.
    pub fn td_improve_percentile(&self, pct: f64) -> f64 {
        if self.td_improve_values.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f32> = self.td_improve_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)] as f64 * 100.0 // return as percentage
    }

    /// Record a completed episode.
    pub fn record_episode(&mut self, record: &BurstEpisodeRecord, config: &crate::config::Config) {
        self.episodes_completed += 1;

        // Compute improvements
        let td_improve = record.pre_abs_td_mean - record.post_abs_td_mean;
        let bad_improve = record.pre_bad_share - record.post_bad_share;
        let stable_gain = record.post_stable_share - record.pre_stable_share;

        self.td_improve_sum += td_improve;
        self.bad_improve_sum += bad_improve;
        self.stable_gain_sum += stable_gain;

        // Track for percentile calculation (Phase 2.1u)
        self.td_improve_values.push(td_improve);
        self.pre_td_sum += record.pre_abs_td_mean;

        // Check success criteria
        let td_success =
            record.post_abs_td_mean <= record.pre_abs_td_mean * config.burst_td_success_ratio;
        let bad_success = bad_improve >= config.burst_bad_improve_abs;
        let stable_success = stable_gain >= config.burst_stable_gain_abs;

        if td_success || bad_success || stable_success {
            self.episodes_success += 1;
            if td_success {
                self.success_by_td += 1;
            }
            if bad_success {
                self.success_by_bad += 1;
            }
            if stable_success {
                self.success_by_stable += 1;
            }
        }
    }
}

/// Rolling metric buffer for pre/post burst measurement (Phase 2.1s).
#[derive(Clone, Debug)]
pub struct BurstMetricBuffer {
    /// Circular buffer for abs_td samples.
    pub abs_td_buffer: Vec<f32>,
    /// Circular buffer for stable_share samples.
    pub stable_share_buffer: Vec<f32>,
    /// Circular buffer for bad_share samples.
    pub bad_share_buffer: Vec<f32>,
    /// Write index.
    pub idx: usize,
    /// Count of samples written.
    pub count: usize,
    /// Buffer capacity.
    pub capacity: usize,
}

impl BurstMetricBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            abs_td_buffer: vec![0.0; capacity],
            stable_share_buffer: vec![0.0; capacity],
            bad_share_buffer: vec![0.0; capacity],
            idx: 0,
            count: 0,
            capacity,
        }
    }

    /// Push a sample into the buffer.
    pub fn push(&mut self, abs_td: f32, stable_share: f32, bad_share: f32) {
        self.abs_td_buffer[self.idx] = abs_td;
        self.stable_share_buffer[self.idx] = stable_share;
        self.bad_share_buffer[self.idx] = bad_share;
        self.idx = (self.idx + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Compute mean of most recent N samples.
    pub fn mean_last_n(&self, n: usize) -> (f32, f32, f32) {
        if self.count == 0 {
            return (0.0, 0.0, 0.0);
        }
        let n = n.min(self.count);
        let mut td_sum = 0.0f32;
        let mut stable_sum = 0.0f32;
        let mut bad_sum = 0.0f32;
        for i in 0..n {
            let pos = (self.idx + self.capacity - 1 - i) % self.capacity;
            td_sum += self.abs_td_buffer[pos];
            stable_sum += self.stable_share_buffer[pos];
            bad_sum += self.bad_share_buffer[pos];
        }
        (td_sum / n as f32, stable_sum / n as f32, bad_sum / n as f32)
    }

    /// Get count of samples.
    pub fn len(&self) -> usize {
        self.count
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
    /// Phase 2.1e: Perturb budget cap.
    pub budget: Option<PerturbBudget>,

    // Phase 2.1r: Bad-Regime Quality Repair state
    /// Ticks that bad-regime condition has been held.
    pub repair_bad_hold: u32,
    /// Ticks that clear condition has been held.
    pub repair_clear_hold: u32,
    /// Remaining ticks in current burst.
    pub repair_burst_remaining: u32,
    /// Cooldown remaining after burst.
    pub repair_burst_cooldown: u32,
    /// Total bursts triggered.
    pub repair_burst_trigger_count: u32,
    /// Total ticks spent in burst mode.
    pub repair_burst_total_ticks: u32,
    /// RNG state for burst probability (seeded).
    pub repair_rng_state: u64,

    // Phase 2.1s: Episodic burst state
    /// Tick when last burst ended (for gap enforcement).
    pub burst_last_end_tick: u64,
    /// Current burst probability (can escalate).
    pub burst_current_prob: f32,
    /// Current burst duration (can escalate).
    pub burst_current_ticks: u32,
    /// Consecutive failed bursts (for escalation).
    pub burst_consecutive_failures: u32,
    /// Pre-burst metrics snapshot.
    pub burst_pre_metrics: Option<(f32, f32, f32)>,
    /// Tick when current burst started.
    pub burst_start_tick: u64,
    /// Rolling metric buffer for measurements.
    pub burst_metric_buffer: BurstMetricBuffer,
    /// Ticks since burst ended (for post-window measurement).
    pub burst_post_window_remaining: u32,
    /// Burst episode records (for analysis).
    pub burst_episodes: Vec<BurstEpisodeRecord>,
    /// Aggregate effectiveness stats.
    pub burst_effectiveness: BurstEffectivenessStats,
}

impl ActionPolicy {
    pub fn new(config: ActionConfig) -> Self {
        Self::new_with_options(config, None, None)
    }

    /// Create with perturb floor enabled.
    pub fn new_with_floor(config: ActionConfig, floor_window: usize, floor_min_rate: f32) -> Self {
        Self::new_with_options(
            config,
            Some(PerturbFloor::new(floor_window, floor_min_rate)),
            None,
        )
    }

    /// Phase 2.1e: Create with perturb floor and budget cap enabled.
    pub fn new_with_floor_and_budget(
        config: ActionConfig,
        floor_window: usize,
        floor_min_rate: f32,
        budget_window: usize,
        budget_max_rate: f32,
    ) -> Self {
        Self::new_with_options(
            config,
            Some(PerturbFloor::new(floor_window, floor_min_rate)),
            Some(PerturbBudget::new(budget_window, budget_max_rate)),
        )
    }

    /// Phase 2.2b: Internal constructor to reduce duplication.
    fn new_with_options(
        config: ActionConfig,
        floor: Option<PerturbFloor>,
        budget: Option<PerturbBudget>,
    ) -> Self {
        Self {
            config,
            stats: ActionPolicyStats::new(),
            triggers: ActionTriggers::default(),
            trigger_stats: PerturbTriggerStats::new(),
            floor,
            budget,
            // Phase 2.1r: Repair burst state
            repair_bad_hold: 0,
            repair_clear_hold: 0,
            repair_burst_remaining: 0,
            repair_burst_cooldown: 0,
            repair_burst_trigger_count: 0,
            repair_burst_total_ticks: 0,
            repair_rng_state: 0x12345678,
            // Phase 2.1s: Episodic burst state
            burst_last_end_tick: 0,
            burst_current_prob: 0.40,
            burst_current_ticks: 30,
            burst_consecutive_failures: 0,
            burst_pre_metrics: None,
            burst_start_tick: 0,
            burst_metric_buffer: BurstMetricBuffer::new(500),
            burst_post_window_remaining: 0,
            burst_episodes: Vec::new(),
            burst_effectiveness: BurstEffectivenessStats::new(),
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

    /// Record action in floor window and budget (call after choosing action).
    pub fn record_action_for_floor(&mut self, action: Action) {
        if let Some(ref mut floor) = self.floor {
            floor.record(action);
        }
        // Phase 2.1e: Also record in budget
        if let Some(ref mut budget) = self.budget {
            budget.record(action == Action::Perturb);
        }
    }

    /// Phase 2.1e: Check if perturb is over budget cap.
    pub fn is_perturb_over_budget(&self) -> bool {
        if let Some(ref budget) = self.budget {
            budget.over_cap()
        } else {
            false
        }
    }

    /// Phase 2.1e: Get current perturb rate from budget window.
    pub fn budget_perturb_rate(&self) -> f32 {
        if let Some(ref budget) = self.budget {
            budget.perturb_rate()
        } else {
            0.0
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

    /// Phase 2.1d: Choose action with combined lock bias (post-rescue + chronic).
    /// Combines both lock types for Focus bias.
    pub fn choose_action_with_combined_lock(
        &self,
        mode: Mode,
        post_rescue_active: bool,
        post_rescue_bias: f32,
        chronic_active: bool,
        chronic_bias: f32,
        abs_td: f32,
        td_threshold: f32,
    ) -> Action {
        let base_action = Action::from_mode(mode);

        // Combine lock states
        let any_lock_active = post_rescue_active || chronic_active;
        let combined_bias = if post_rescue_active && chronic_active {
            post_rescue_bias.max(chronic_bias)
        } else if post_rescue_active {
            post_rescue_bias
        } else if chronic_active {
            chronic_bias
        } else {
            0.0
        };

        // If not in any lock or base action is already Perturb, return base
        if !any_lock_active || base_action == Action::Perturb {
            return base_action;
        }

        // During lock: check if TD spike warrants Perturb despite lock
        if abs_td >= td_threshold {
            return Action::Perturb;
        }

        // During lock: bias toward Focus
        // If mode says Scan (Explore mode), override to Focus if bias is strong enough
        if base_action == Action::Scan && combined_bias >= 1.0 {
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

    // =========================================================================
    // Phase 2.1r/2.1s: Bad-Regime Quality Repair (Episodic Perturb Burst)
    // =========================================================================

    /// Set the repair RNG seed for deterministic burst probability.
    pub fn set_repair_seed(&mut self, seed: u64) {
        self.repair_rng_state = seed;
    }

    /// Initialize burst parameters from config (call at start of run).
    pub fn init_burst_params(&mut self, config: &crate::config::Config) {
        self.burst_current_prob = config.burst_base_prob;
        self.burst_current_ticks = config.burst_base_ticks;
        self.burst_consecutive_failures = 0;
    }

    /// Simple LCG RNG for burst probability (deterministic).
    fn repair_rng_next(&mut self) -> f32 {
        // LCG: state = (a * state + c) mod m
        self.repair_rng_state = self
            .repair_rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Return value in [0, 1)
        (self.repair_rng_state >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Push metrics into the rolling buffer for pre/post measurement.
    pub fn push_burst_metrics(&mut self, abs_td: f32, stable_share: f32, bad_share: f32) {
        self.burst_metric_buffer
            .push(abs_td, stable_share, bad_share);
    }

    /// Update the repair burst state based on rolling stats.
    /// Call this every tick with current rolling stats from ModePolicy.
    /// Phase 2.1s: Adds episodic logic with gap enforcement, max per run, and escalation.
    /// Phase 2.1v: Returns true if a new burst was triggered this tick.
    pub fn update_repair_burst(
        &mut self,
        stable_share: f32,
        bad_share: f32,
        rescue_rate: f32, // rescues per tick in rolling window
        global_tick: u64,
        config: &crate::config::Config,
    ) -> bool {
        if !config.repair_enabled {
            return false;
        }

        // Push current metrics to buffer
        // Note: abs_td is pushed separately via push_burst_metrics() with actual TD value
        // Here we just track stable/bad shares
        // The abs_td gets pushed by the caller with the real value

        // Decrement cooldown
        if self.repair_burst_cooldown > 0 {
            self.repair_burst_cooldown -= 1;
        }

        // Phase 2.1s: Handle post-burst window measurement
        if self.burst_post_window_remaining > 0 {
            self.burst_post_window_remaining -= 1;

            // When post-window completes, finalize the episode
            if self.burst_post_window_remaining == 0 {
                self.finalize_burst_episode(config);
            }
        }

        // Decrement burst remaining and detect burst end
        if self.repair_burst_remaining > 0 {
            self.repair_burst_remaining -= 1;
            self.repair_burst_total_ticks += 1;

            // Burst just ended
            if self.repair_burst_remaining == 0 {
                self.burst_last_end_tick = global_tick;
                // Start post-window measurement
                self.burst_post_window_remaining = config.burst_post_window;
            }
        }

        // Check bad regime conditions (OR logic: any condition can trigger)
        let stable_bad = stable_share < config.repair_bad_stable_lo;
        let bad_high = bad_share > config.repair_bad_share_hi;
        let rescue_high = rescue_rate > config.repair_rescue_rate_hi;
        let in_bad_regime = stable_bad || bad_high || rescue_high;

        // Check clear conditions
        let clear_ok =
            stable_share > config.repair_clear_stable_hi && bad_share < config.repair_clear_bad_lo;

        // Update hold counters
        if in_bad_regime {
            self.repair_bad_hold += 1;
            self.repair_clear_hold = 0;
        } else if clear_ok {
            self.repair_clear_hold += 1;
            // Only reset bad hold after sustained clear
            if self.repair_clear_hold >= config.repair_clear_hold_ticks {
                self.repair_bad_hold = 0;
            }
        } else {
            // Neither bad nor clear - decay holds slowly
            self.repair_clear_hold = 0;
            // Keep bad_hold - don't reset unless clear conditions met
        }

        // Phase 2.1s: Check if we can trigger a new burst
        // Conditions:
        // 1. bad_hold exceeded threshold
        // 2. not currently in burst
        // 3. cooldown expired
        // 4. gap since last burst exceeded (NEW in 2.1s)
        // 5. haven't hit max bursts per run (NEW in 2.1s)
        let gap_ok = global_tick >= self.burst_last_end_tick + config.burst_min_gap_ticks as u64;
        let under_max = self.repair_burst_trigger_count < config.burst_max_per_run;

        if self.repair_bad_hold >= config.repair_bad_hold_ticks
            && self.repair_burst_remaining == 0
            && self.repair_burst_cooldown == 0
            && gap_ok
            && under_max
        {
            // Snapshot pre-burst metrics from rolling buffer
            let (pre_td, pre_stable, pre_bad) = self
                .burst_metric_buffer
                .mean_last_n(config.burst_pre_window as usize);
            self.burst_pre_metrics = Some((pre_td, pre_stable, pre_bad));
            self.burst_start_tick = global_tick;

            // Start burst with current (possibly escalated) parameters
            self.repair_burst_remaining = self.burst_current_ticks;
            self.repair_burst_cooldown = config.repair_burst_cooldown;
            self.repair_burst_trigger_count += 1;
            self.repair_bad_hold = 0; // Reset hold after triggering
            return true; // Phase 2.1v: Signal that burst was triggered
        }
        false
    }

    /// Finalize a burst episode after post-window measurement completes.
    fn finalize_burst_episode(&mut self, config: &crate::config::Config) {
        // Get post-burst metrics
        let (post_td, post_stable, post_bad) = self
            .burst_metric_buffer
            .mean_last_n(config.burst_post_window as usize);

        // Get pre-burst metrics (should have been captured at burst start)
        if let Some((pre_td, pre_stable, pre_bad)) = self.burst_pre_metrics.take() {
            // Determine success
            let td_success = post_td <= pre_td * config.burst_td_success_ratio;
            let bad_success = (pre_bad - post_bad) >= config.burst_bad_improve_abs;
            let stable_success = (post_stable - pre_stable) >= config.burst_stable_gain_abs;
            let is_success = td_success || bad_success || stable_success;

            let success_reason = if td_success {
                Some("td")
            } else if bad_success {
                Some("bad")
            } else if stable_success {
                Some("stable")
            } else {
                None
            };

            // Create episode record
            let record = BurstEpisodeRecord {
                start_tick: self.burst_start_tick,
                end_tick: self.burst_last_end_tick,
                duration: self.burst_current_ticks,
                prob_used: self.burst_current_prob,
                pre_abs_td_mean: pre_td,
                pre_stable_share: pre_stable,
                pre_bad_share: pre_bad,
                post_abs_td_mean: post_td,
                post_stable_share: post_stable,
                post_bad_share: post_bad,
                success: is_success,
                success_reason,
            };

            // Update effectiveness stats
            self.burst_effectiveness.record_episode(&record, config);
            self.burst_episodes.push(record);

            // Phase 2.1s: Escalation logic
            if is_success {
                // Reset to base parameters on success
                self.burst_current_prob = config.burst_base_prob;
                self.burst_current_ticks = config.burst_base_ticks;
                self.burst_consecutive_failures = 0;
            } else {
                // Escalate on failure
                self.burst_consecutive_failures += 1;
                self.burst_current_prob = (self.burst_current_prob + config.burst_prob_escalation)
                    .min(config.burst_max_prob);
                self.burst_current_ticks = (self.burst_current_ticks
                    + config.burst_ticks_escalation)
                    .min(config.burst_max_ticks);
            }
        }
    }

    /// Check if burst is currently active.
    pub fn is_repair_burst_active(&self) -> bool {
        self.repair_burst_remaining > 0
    }

    /// Apply burst override to action selection.
    /// Returns Some(Action::Perturb) if burst should override, None otherwise.
    /// Phase 2.1s: Uses current escalated probability.
    pub fn apply_repair_burst_override(
        &mut self,
        is_soft_exploit: bool,
        is_bad_state: bool,
        config: &crate::config::Config,
    ) -> Option<Action> {
        if !config.repair_enabled || self.repair_burst_remaining == 0 {
            return None;
        }

        // Only override during soft exploit OR when in bad state
        if !is_soft_exploit && !is_bad_state {
            return None;
        }

        // Check budget cap - don't burst if already over perturb cap
        if let Some(ref budget) = self.budget {
            if budget.perturb_rate() >= config.repair_perturb_cap_mean {
                return None;
            }
        }

        // Apply burst probability (using current escalated value)
        let rand_val = self.repair_rng_next();
        if rand_val < self.burst_current_prob {
            Some(Action::Perturb)
        } else {
            None
        }
    }

    /// Get repair burst statistics.
    pub fn repair_burst_stats(&self) -> (u32, u32, u32) {
        (
            self.repair_burst_trigger_count,
            self.repair_burst_total_ticks,
            self.repair_burst_remaining,
        )
    }

    /// Get current repair state for diagnostics.
    pub fn repair_state(&self) -> (u32, u32, u32, u32) {
        (
            self.repair_bad_hold,
            self.repair_clear_hold,
            self.repair_burst_remaining,
            self.repair_burst_cooldown,
        )
    }

    /// Get burst effectiveness stats (Phase 2.1s).
    pub fn burst_effectiveness_stats(&self) -> &BurstEffectivenessStats {
        &self.burst_effectiveness
    }

    /// Get number of burst episodes recorded.
    pub fn burst_episode_count(&self) -> usize {
        self.burst_episodes.len()
    }

    // =========================================================================
    // Phase 2.2a: Post-Rescue Quality Repair Override
    // =========================================================================

    /// Phase 2.2a: Apply post-rescue repair override to action selection.
    /// Returns Some(Action::Perturb) if repair should override, None otherwise.
    /// Called when ModePolicy.is_post_rescue_repair_active() is true.
    pub fn apply_post_rescue_repair_override(
        &mut self,
        repair_quality_bad: bool,
        perturb_prob: f32,
        perturb_cap: f32,
    ) -> Option<Action> {
        // Only override if quality is bad
        if !repair_quality_bad {
            return None;
        }

        // Check budget cap - don't perturb if already over cap
        if let Some(ref budget) = self.budget {
            if budget.perturb_rate() >= perturb_cap {
                return None;
            }
        }

        // Apply perturb probability
        let rand_val = self.repair_rng_next();
        if rand_val < perturb_prob {
            Some(Action::Perturb)
        } else {
            None
        }
    }
}
