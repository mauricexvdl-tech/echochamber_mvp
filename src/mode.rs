// Phase 2.0a: Mode Policy for Explore/Exploit/Reset
// Turns stable anchor/value/prototype signals into a simple mode-policy loop
// WITHOUT changing core echo dynamics.

/// The three operating modes for the system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Loosen gating, encourage diversity
    Explore,
    /// Tighten gating, follow stable prototypes (default behavior)
    Exploit,
    /// Targeted disruption when dynamics indicate stuck/bad state
    Reset,
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Exploit
    }
}

/// Configuration for the mode policy.
#[derive(Clone, Debug)]
pub struct ModePolicyConfig {
    /// Max anchor value to trigger Explore mode (low V = explore)
    pub explore_v_max: f32,
    /// Min anchor value to stay in Exploit mode (high V = exploit)
    pub exploit_v_min: f32,
    /// Min |TD| to consider Reset (high TD = unstable)
    pub reset_td_min: f32,
    /// Value drop threshold to trigger Reset
    pub reset_value_drop: f32,
    /// Consecutive gate fails to trigger Reset
    pub reset_fail_streak: u32,
    /// Cooldown ticks after Reset before another Reset can fire
    pub post_reset_cooldown: u32,
    /// Scale factor for margin_min in Explore mode (< 1.0 = looser)
    pub explore_margin_min_scale: f32,
    /// Scale factor for margin_min in Exploit mode (> 1.0 = tighter)
    pub exploit_margin_min_scale: f32,
    /// Dampening factor for Reset (buffer *= factor)
    pub reset_dampen: f32,
    /// Number of top-K nodes to apply dampening to
    pub reset_dampen_top_k: usize,
    /// Window size for tracking recent values/TD
    pub window_size: usize,
}

impl Default for ModePolicyConfig {
    fn default() -> Self {
        Self {
            explore_v_max: 0.25,
            exploit_v_min: 0.50,
            reset_td_min: 0.55,
            reset_value_drop: 0.12,
            reset_fail_streak: 6,
            post_reset_cooldown: 40,
            explore_margin_min_scale: 0.5,
            exploit_margin_min_scale: 1.2,
            reset_dampen: 0.65,
            reset_dampen_top_k: 8,
            window_size: 32,
        }
    }
}

/// Ring buffer for tracking recent values.
#[derive(Clone, Debug)]
pub struct RingBuffer {
    data: Vec<f32>,
    head: usize,
    len: usize,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            head: 0,
            len: 0,
            capacity,
        }
    }

    pub fn push(&mut self, value: f32) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    pub fn mean(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        let sum: f32 = self.data.iter().take(self.len).sum();
        sum / self.len as f32
    }

    pub fn last(&self) -> Option<f32> {
        if self.len == 0 {
            None
        } else {
            let idx = if self.head == 0 {
                self.capacity - 1
            } else {
                self.head - 1
            };
            Some(self.data[idx])
        }
    }

    pub fn first(&self) -> Option<f32> {
        if self.len == 0 {
            None
        } else if self.len < self.capacity {
            Some(self.data[0])
        } else {
            Some(self.data[self.head])
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Get the last N values as a slice (for computing windowed stats)
    pub fn last_n(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.len);
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let idx = if self.head >= i + 1 {
                self.head - i - 1
            } else {
                self.capacity - (i + 1 - self.head)
            };
            result.push(self.data[idx]);
        }
        result.reverse();
        result
    }
}

/// State maintained by the mode policy across ticks.
#[derive(Clone, Debug)]
pub struct ModePolicyState {
    /// Last selected mode
    pub last_mode: Mode,
    /// Cooldown counter (decrements each tick, reset fires only when 0)
    pub cooldown: u32,
    /// Consecutive gate failure streak
    pub gate_fail_streak: u32,
    /// Recent anchor values
    pub recent_values: RingBuffer,
    /// Recent |TD| values
    pub recent_abs_td: RingBuffer,
    /// Tick of last reset (for effectiveness measurement)
    pub last_reset_tick: Option<u64>,
    /// Pre-reset mean |TD| for effectiveness tracking
    pub pre_reset_td_mean: f32,
    /// Accumulator for reset effectiveness
    pub reset_effectiveness_sum: f64,
    pub reset_effectiveness_count: usize,
    /// Mode counters
    pub explore_count: usize,
    pub exploit_count: usize,
    pub reset_count: usize,
    /// Gate pass counts per mode (for diagnostics)
    pub gate_pass_explore: usize,
    pub gate_pass_exploit: usize,
    pub gate_total_explore: usize,
    pub gate_total_exploit: usize,
}

impl ModePolicyState {
    pub fn new(window_size: usize) -> Self {
        Self {
            last_mode: Mode::Exploit,
            cooldown: 0,
            gate_fail_streak: 0,
            recent_values: RingBuffer::new(window_size),
            recent_abs_td: RingBuffer::new(window_size),
            last_reset_tick: None,
            pre_reset_td_mean: 0.0,
            reset_effectiveness_sum: 0.0,
            reset_effectiveness_count: 0,
            explore_count: 0,
            exploit_count: 0,
            reset_count: 0,
            gate_pass_explore: 0,
            gate_pass_exploit: 0,
            gate_total_explore: 0,
            gate_total_exploit: 0,
        }
    }
}

/// Gate overrides returned by apply_mode_overrides.
#[derive(Clone, Debug, Default)]
pub struct GateOverrides {
    /// Scale factor for margin_min (1.0 = no change)
    pub margin_min_scale: f32,
}

/// Action to take after mode selection.
#[derive(Clone, Debug)]
pub enum ModeAction {
    /// No special action, just use gate overrides
    None,
    /// Dampen top-K nodes with given factor
    Dampen { factor: f32, top_k: usize },
}

/// The mode policy controller.
#[derive(Clone, Debug)]
pub struct ModePolicy {
    pub config: ModePolicyConfig,
    pub state: ModePolicyState,
}

impl ModePolicy {
    pub fn new(config: ModePolicyConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            state: ModePolicyState::new(window_size),
        }
    }

    /// Observe current tick state. Call before choose_mode.
    pub fn observe(
        &mut self,
        current_tick: u64,
        anchor_value: f32,
        abs_td: f32,
        gate_passed: bool,
    ) {
        // Update ring buffers
        self.state.recent_values.push(anchor_value);
        self.state.recent_abs_td.push(abs_td);

        // Update gate fail streak
        if gate_passed {
            self.state.gate_fail_streak = 0;
        } else {
            self.state.gate_fail_streak += 1;
        }

        // Decrement cooldown
        if self.state.cooldown > 0 {
            self.state.cooldown -= 1;
        }

        // Check reset effectiveness after 10 ticks
        if let Some(reset_tick) = self.state.last_reset_tick {
            if current_tick == reset_tick + 10 {
                // Compute post-reset mean |TD|
                let post_td_values = self.state.recent_abs_td.last_n(10);
                if !post_td_values.is_empty() {
                    let post_mean: f32 =
                        post_td_values.iter().sum::<f32>() / post_td_values.len() as f32;
                    let pre_mean = self.state.pre_reset_td_mean;
                    if pre_mean > 0.001 {
                        // Compute improvement ratio (positive = improved)
                        let improvement = (pre_mean - post_mean) / pre_mean;
                        self.state.reset_effectiveness_sum += improvement as f64;
                        self.state.reset_effectiveness_count += 1;
                    }
                }
                self.state.last_reset_tick = None;
            }
        }
    }

    /// Choose the mode for this tick based on observed state.
    pub fn choose_mode(&mut self, current_tick: u64) -> Mode {
        // Check for Reset conditions (highest priority, but respects cooldown)
        let should_reset = self.state.cooldown == 0 && self.check_reset_conditions();

        let cfg = &self.config;
        let state = &mut self.state;

        let mode = if should_reset {
            // Record pre-reset TD mean for effectiveness measurement
            let pre_td_values = state.recent_abs_td.last_n(10);
            if !pre_td_values.is_empty() {
                state.pre_reset_td_mean =
                    pre_td_values.iter().sum::<f32>() / pre_td_values.len() as f32;
            }
            state.last_reset_tick = Some(current_tick);
            state.cooldown = cfg.post_reset_cooldown;
            state.gate_fail_streak = 0; // Reset the streak
            Mode::Reset
        } else {
            // Check Explore vs Exploit based on recent value
            let recent_v = state.recent_values.mean();
            if recent_v < cfg.explore_v_max {
                Mode::Explore
            } else if recent_v >= cfg.exploit_v_min {
                Mode::Exploit
            } else {
                // Middle ground: default to last mode or Exploit
                state.last_mode
            }
        };

        // Update counters
        match mode {
            Mode::Explore => state.explore_count += 1,
            Mode::Exploit => state.exploit_count += 1,
            Mode::Reset => state.reset_count += 1,
        }

        state.last_mode = mode;
        mode
    }

    /// Check if reset conditions are met.
    fn check_reset_conditions(&self) -> bool {
        let cfg = &self.config;
        let state = &self.state;

        // Condition 1: High |TD| (unstable dynamics)
        let recent_td = state.recent_abs_td.mean();
        if recent_td >= cfg.reset_td_min {
            return true;
        }

        // Condition 2: Value drop over window
        if state.recent_values.is_full() {
            if let (Some(first), Some(last)) =
                (state.recent_values.first(), state.recent_values.last())
            {
                let drop = first - last;
                if drop >= cfg.reset_value_drop {
                    return true;
                }
            }
        }

        // Condition 3: Consecutive gate failures
        if state.gate_fail_streak >= cfg.reset_fail_streak {
            return true;
        }

        false
    }

    /// Get the gate overrides and action for the current mode.
    pub fn apply_mode_overrides(&self, mode: Mode) -> (GateOverrides, ModeAction) {
        let cfg = &self.config;

        let overrides = match mode {
            Mode::Explore => GateOverrides {
                margin_min_scale: cfg.explore_margin_min_scale,
            },
            Mode::Exploit => GateOverrides {
                margin_min_scale: cfg.exploit_margin_min_scale,
            },
            Mode::Reset => GateOverrides {
                margin_min_scale: 1.0, // Normal gating during reset tick
            },
        };

        let action = match mode {
            Mode::Reset => ModeAction::Dampen {
                factor: cfg.reset_dampen,
                top_k: cfg.reset_dampen_top_k,
            },
            _ => ModeAction::None,
        };

        (overrides, action)
    }

    /// Record gate pass/fail for mode-specific stats.
    pub fn record_gate_outcome(&mut self, mode: Mode, passed: bool) {
        match mode {
            Mode::Explore => {
                self.state.gate_total_explore += 1;
                if passed {
                    self.state.gate_pass_explore += 1;
                }
            }
            Mode::Exploit => {
                self.state.gate_total_exploit += 1;
                if passed {
                    self.state.gate_pass_exploit += 1;
                }
            }
            Mode::Reset => {
                // Don't track Reset separately
            }
        }
    }

    /// Get mode usage statistics.
    pub fn mode_stats(&self) -> ModeStats {
        let total = self.state.explore_count + self.state.exploit_count + self.state.reset_count;
        let total_f = total.max(1) as f64;

        ModeStats {
            explore_count: self.state.explore_count,
            exploit_count: self.state.exploit_count,
            reset_count: self.state.reset_count,
            explore_rate: self.state.explore_count as f64 / total_f,
            exploit_rate: self.state.exploit_count as f64 / total_f,
            reset_rate: self.state.reset_count as f64 / total_f,
            reset_effectiveness_mean: if self.state.reset_effectiveness_count > 0 {
                self.state.reset_effectiveness_sum / self.state.reset_effectiveness_count as f64
            } else {
                0.0
            },
            reset_effectiveness_count: self.state.reset_effectiveness_count,
            gate_pass_rate_explore: if self.state.gate_total_explore > 0 {
                self.state.gate_pass_explore as f64 / self.state.gate_total_explore as f64
            } else {
                0.0
            },
            gate_pass_rate_exploit: if self.state.gate_total_exploit > 0 {
                self.state.gate_pass_exploit as f64 / self.state.gate_total_exploit as f64
            } else {
                0.0
            },
        }
    }
}

/// Summary statistics for mode usage.
#[derive(Clone, Debug, Default)]
pub struct ModeStats {
    pub explore_count: usize,
    pub exploit_count: usize,
    pub reset_count: usize,
    pub explore_rate: f64,
    pub exploit_rate: f64,
    pub reset_rate: f64,
    pub reset_effectiveness_mean: f64,
    pub reset_effectiveness_count: usize,
    pub gate_pass_rate_explore: f64,
    pub gate_pass_rate_exploit: f64,
}

// ============================================================================
// Phase 2.0b: Ablation-aware mode selection
// ============================================================================

impl ModePolicy {
    /// Choose mode with ablation config (Phase 2.0b).
    /// If enable_reset=false, Reset never fires.
    /// If enable_explore=false, Explore is forced to Exploit.
    pub fn choose_mode_with_ablation(
        &mut self,
        current_tick: u64,
        enable_reset: bool,
        enable_explore: bool,
    ) -> Mode {
        // Check for Reset conditions (highest priority, but respects cooldown)
        let should_reset =
            enable_reset && self.state.cooldown == 0 && self.check_reset_conditions();

        let cfg = &self.config;
        let state = &mut self.state;

        let mode = if should_reset {
            // Record pre-reset TD mean for effectiveness measurement
            let pre_td_values = state.recent_abs_td.last_n(10);
            if !pre_td_values.is_empty() {
                state.pre_reset_td_mean =
                    pre_td_values.iter().sum::<f32>() / pre_td_values.len() as f32;
            }
            state.last_reset_tick = Some(current_tick);
            state.cooldown = cfg.post_reset_cooldown;
            state.gate_fail_streak = 0;
            Mode::Reset
        } else {
            // Check Explore vs Exploit based on recent value
            let recent_v = state.recent_values.mean();
            let base_mode = if recent_v < cfg.explore_v_max {
                Mode::Explore
            } else if recent_v >= cfg.exploit_v_min {
                Mode::Exploit
            } else {
                state.last_mode
            };

            // Apply ablation: force Explore -> Exploit if disabled
            if !enable_explore && base_mode == Mode::Explore {
                Mode::Exploit
            } else {
                base_mode
            }
        };

        // Update counters
        match mode {
            Mode::Explore => state.explore_count += 1,
            Mode::Exploit => state.exploit_count += 1,
            Mode::Reset => state.reset_count += 1,
        }

        state.last_mode = mode;
        mode
    }

    /// Get the current mean |TD| from recent observations.
    pub fn current_abs_td(&self) -> f32 {
        self.state.recent_abs_td.mean()
    }

    /// Get the last observed anchor value.
    pub fn last_value(&self) -> f32 {
        self.state.recent_values.last().unwrap_or(0.0)
    }

    /// Reset the policy state (for ablation runs with same initial state).
    pub fn reset_state(&mut self) {
        self.state = ModePolicyState::new(self.config.window_size);
    }
}
