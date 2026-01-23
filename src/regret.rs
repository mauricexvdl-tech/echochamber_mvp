//! Phase 2.0e: Regret and Recovery Metrics
//!
//! Provides metrics that amplify differences between action policies
//! beyond standard end-metrics like coverage and selective accuracy.

use crate::action::Action;

/// Configuration for regret/recovery metrics.
#[derive(Clone, Debug)]
pub struct RegretConfig {
    /// Margin threshold for "bad state" (topk_margin < margin_bad)
    pub margin_bad: f64,
    /// Proto alignment threshold for "bad state"
    pub proto_bad: f32,
    /// Value threshold for "bad state"
    pub v_bad: f32,
    /// TD spike threshold (abs_td > td_spike counts as spike)
    pub td_spike: f32,
    /// Window size for pre-action measurement
    pub pre_window: usize,
    /// Window size for post-action measurement
    pub post_window: usize,
    /// Window size for gate pass rate tracking
    pub post_gate_window: usize,
    /// Minimum improvement ratio to count as "good" recovery
    pub recovery_good_threshold: f64,
}

impl Default for RegretConfig {
    fn default() -> Self {
        Self {
            margin_bad: 0.02,
            proto_bad: 0.20,
            v_bad: 0.15,
            td_spike: 0.55,
            pre_window: 10,
            post_window: 10,
            post_gate_window: 50,
            recovery_good_threshold: 0.10,
        }
    }
}

/// Rolling window for tracking recent values.
#[derive(Clone, Debug)]
pub struct RollingBuffer {
    buffer: Vec<f32>,
    position: usize,
    count: usize,
}

impl RollingBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            position: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, value: f32) {
        self.buffer[self.position] = value;
        self.position = (self.position + 1) % self.buffer.len();
        if self.count < self.buffer.len() {
            self.count += 1;
        }
    }

    pub fn mean(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let sum: f32 = if self.count < self.buffer.len() {
            self.buffer[..self.count].iter().sum()
        } else {
            self.buffer.iter().sum()
        };
        sum / self.count as f32
    }

    pub fn is_full(&self) -> bool {
        self.count >= self.buffer.len()
    }

    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.position = 0;
        self.count = 0;
    }
}

/// Pending action event awaiting post-measurement.
#[derive(Clone, Debug)]
struct PendingAction {
    tick: u64,
    action: Action,
    pre_mean_td: f32,
    pre_gate_pass_rate: f32,
}

/// Statistics for regret and recovery metrics.
#[derive(Clone, Debug, Default)]
pub struct RegretStats {
    // Bad state tracking
    pub total_ticks: usize,
    pub bad_state_ticks: usize,
    pub gate_fail_streak: usize,

    // TD spike tracking
    pub td_spikes: usize,

    // Action tracking
    pub action_ticks: usize, // Ticks where non-Focus action taken
    pub regret_ticks: usize, // Actions that led to worse outcome

    // Recovery tracking
    pub recovery_count: usize,
    pub recovery_good_count: usize,
    pub sum_recovery_improve: f64,

    // Rolling buffers for measurement
    td_buffer: Option<RollingBuffer>,
    gate_buffer: Option<RollingBuffer>,

    // Pending actions awaiting post-measurement
    pending_actions: Vec<PendingAction>,
}

impl RegretStats {
    pub fn new(config: &RegretConfig) -> Self {
        Self {
            td_buffer: Some(RollingBuffer::new(config.pre_window)),
            gate_buffer: Some(RollingBuffer::new(config.post_gate_window)),
            pending_actions: Vec::new(),
            ..Default::default()
        }
    }

    /// Observe a tick and update bad-state and TD spike counters.
    pub fn observe_tick(
        &mut self,
        config: &RegretConfig,
        _tick: u64,
        abs_td: f32,
        topk_margin: f64,
        proto_align: f32,
        anchor_value: f32,
        gate_passed: bool,
    ) {
        self.total_ticks += 1;

        // Update gate fail streak
        if gate_passed {
            self.gate_fail_streak = 0;
        } else {
            self.gate_fail_streak += 1;
        }

        // Check bad state conditions
        let is_bad_state = self.gate_fail_streak > 0
            || topk_margin < config.margin_bad
            || proto_align < config.proto_bad
            || anchor_value < config.v_bad;

        if is_bad_state {
            self.bad_state_ticks += 1;
        }

        // Check TD spike
        if abs_td > config.td_spike {
            self.td_spikes += 1;
        }

        // Update rolling buffers
        if let Some(ref mut buf) = self.td_buffer {
            buf.push(abs_td);
        }
        if let Some(ref mut buf) = self.gate_buffer {
            buf.push(if gate_passed { 1.0 } else { 0.0 });
        }
    }

    /// Record an action event (Scan or Perturb) for regret/recovery tracking.
    pub fn observe_action(&mut self, tick: u64, action: Action) {
        if action == Action::Focus {
            return; // Only track non-Focus actions
        }

        self.action_ticks += 1;

        // Capture pre-action stats
        let pre_mean_td = self.td_buffer.as_ref().map(|b| b.mean()).unwrap_or(0.0);
        let pre_gate_pass_rate = self.gate_buffer.as_ref().map(|b| b.mean()).unwrap_or(0.0);

        self.pending_actions.push(PendingAction {
            tick,
            action,
            pre_mean_td,
            pre_gate_pass_rate,
        });
    }

    /// Check pending actions and finalize recovery/regret measurements.
    /// Call this periodically or after post_window ticks.
    pub fn check_pending_actions(&mut self, config: &RegretConfig, current_tick: u64) {
        let post_mean_td = self.td_buffer.as_ref().map(|b| b.mean()).unwrap_or(0.0);
        let post_gate_pass_rate = self.gate_buffer.as_ref().map(|b| b.mean()).unwrap_or(0.0);

        // Process actions that have had enough post-window time
        let ready_tick = current_tick.saturating_sub(config.post_window as u64);

        self.pending_actions.retain(|pending| {
            if pending.tick > ready_tick {
                return true; // Keep, not ready yet
            }

            // Evaluate recovery (for Perturb actions)
            if pending.action == Action::Perturb && pending.pre_mean_td > 0.001 {
                let improve = (pending.pre_mean_td - post_mean_td) / pending.pre_mean_td;
                self.recovery_count += 1;
                self.sum_recovery_improve += improve as f64;
                if improve >= config.recovery_good_threshold as f32 {
                    self.recovery_good_count += 1;
                }
            }

            // Evaluate regret (action led to worse outcome)
            let td_worse = post_mean_td > pending.pre_mean_td * 1.05; // 5% tolerance
            let gate_worse = post_gate_pass_rate < pending.pre_gate_pass_rate - 0.05;

            if td_worse || gate_worse {
                self.regret_ticks += 1;
            }

            false // Remove from pending
        });
    }

    /// Finalize all pending actions at end of run.
    pub fn finalize(&mut self, config: &RegretConfig) {
        // Force-process all remaining pending actions
        let post_mean_td = self.td_buffer.as_ref().map(|b| b.mean()).unwrap_or(0.0);
        let post_gate_pass_rate = self.gate_buffer.as_ref().map(|b| b.mean()).unwrap_or(0.0);

        for pending in self.pending_actions.drain(..) {
            // Evaluate recovery
            if pending.action == Action::Perturb && pending.pre_mean_td > 0.001 {
                let improve = (pending.pre_mean_td - post_mean_td) / pending.pre_mean_td;
                self.recovery_count += 1;
                self.sum_recovery_improve += improve as f64;
                if improve >= config.recovery_good_threshold as f32 {
                    self.recovery_good_count += 1;
                }
            }

            // Evaluate regret
            let td_worse = post_mean_td > pending.pre_mean_td * 1.05;
            let gate_worse = post_gate_pass_rate < pending.pre_gate_pass_rate - 0.05;

            if td_worse || gate_worse {
                self.regret_ticks += 1;
            }
        }
    }

    // Computed metrics

    /// Bad state share as fraction of total ticks.
    pub fn bad_state_share(&self) -> f64 {
        if self.total_ticks > 0 {
            self.bad_state_ticks as f64 / self.total_ticks as f64
        } else {
            0.0
        }
    }

    /// TD spike rate per 10k ticks.
    pub fn td_spike_rate(&self) -> f64 {
        if self.total_ticks > 0 {
            (self.td_spikes as f64 / self.total_ticks as f64) * 10000.0
        } else {
            0.0
        }
    }

    /// Mean recovery improvement ratio.
    pub fn recovery_improve_mean(&self) -> f64 {
        if self.recovery_count > 0 {
            self.sum_recovery_improve / self.recovery_count as f64
        } else {
            0.0
        }
    }

    /// Fraction of recovery events with good improvement.
    pub fn recovery_good_rate(&self) -> f64 {
        if self.recovery_count > 0 {
            self.recovery_good_count as f64 / self.recovery_count as f64
        } else {
            0.0
        }
    }

    /// Regret rate as fraction of action ticks.
    pub fn regret_rate(&self) -> f64 {
        if self.action_ticks > 0 {
            self.regret_ticks as f64 / self.action_ticks as f64
        } else {
            0.0
        }
    }

    /// Reset for a new run.
    pub fn reset(&mut self, config: &RegretConfig) {
        self.total_ticks = 0;
        self.bad_state_ticks = 0;
        self.gate_fail_streak = 0;
        self.td_spikes = 0;
        self.action_ticks = 0;
        self.regret_ticks = 0;
        self.recovery_count = 0;
        self.recovery_good_count = 0;
        self.sum_recovery_improve = 0.0;
        self.pending_actions.clear();
        if let Some(ref mut buf) = self.td_buffer {
            *buf = RollingBuffer::new(config.pre_window);
        }
        if let Some(ref mut buf) = self.gate_buffer {
            *buf = RollingBuffer::new(config.post_gate_window);
        }
    }
}

/// Report combining standard metrics with regret metrics.
#[derive(Clone, Debug, Default)]
pub struct RegretReport {
    pub label: String,

    // Action rates
    pub scan_rate: f64,
    pub focus_rate: f64,
    pub perturb_rate: f64,
    pub trigger_count: usize, // Non-Focus actions

    // Standard metrics
    pub coverage_pos: f64,
    pub selective_accuracy: f64,
    pub false_positive_rate: f64,
    pub stable_time_share: f64,

    // Regret metrics
    pub bad_state_share: f64,
    pub td_spike_rate: f64,
    pub recovery_improve_mean: f64,
    pub recovery_good_rate: f64,
    pub regret_rate: f64,
}

impl RegretReport {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            ..Default::default()
        }
    }
}
