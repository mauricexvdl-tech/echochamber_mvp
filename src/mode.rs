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

    // Phase 2.0f-E: Natural Exploit Emergence
    /// Minimum proto_align for Exploit mode
    pub exploit_proto_min: f32,
    /// Minimum topk_margin for Exploit mode
    pub exploit_margin_min: f64,
    /// Require anchor to be stable for Exploit mode
    pub exploit_requires_stable: bool,

    // Phase 2.1b: Seed-Robust Policy Stabilization
    /// Minimum ticks to stay in Exploit once entered (anti-collapse).
    pub min_exploit_ticks_on: u32,
    /// Explore streak threshold to trigger rescue.
    pub explore_streak_rescue: u32,
    /// Gate fail streak threshold to trigger rescue.
    pub fail_streak_rescue: u32,
    /// Cooldown after rescue.
    pub rescue_cooldown: u32,
    /// |TD| threshold to escape exploit lock (catastrophic).
    pub catastrophic_abs_td: f32,
    /// Value drop threshold to escape exploit lock.
    pub catastrophic_value_drop: f32,
    /// Ticks after Reset/Perturb to apply tighter margin.
    pub post_reset_boost_ticks: u32,
    /// Scale factor for margin during post-reset boost.
    pub post_reset_margin_scale: f32,

    // Phase 2.1c: Anti-Thrash Post-Rescue Lock
    /// Ticks to lock in Exploit mode after a rescue.
    pub post_rescue_lock_ticks: u32,
    /// Margin min scale during post-rescue lock.
    pub lock_margin_min_scale: f32,
    /// Focus bias during post-rescue lock.
    pub lock_focus_bias: f32,
    /// Require bad_state for rescue to fire.
    pub rescue_requires_bad_state: bool,
    /// Bad state proto threshold.
    pub rescue_bad_proto: f32,
    /// Bad state margin threshold.
    pub rescue_bad_margin: f64,
    /// Bad state value threshold.
    pub rescue_bad_value: f32,
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
            // Phase 2.0f-E: Natural Exploit Emergence
            exploit_proto_min: 0.55,
            exploit_margin_min: 0.04,
            exploit_requires_stable: true,

            // Phase 2.1b: Seed-Robust Policy Stabilization
            min_exploit_ticks_on: 20,
            explore_streak_rescue: 250,
            fail_streak_rescue: 12,
            rescue_cooldown: 80,
            catastrophic_abs_td: 0.75,
            catastrophic_value_drop: 0.20,
            post_reset_boost_ticks: 40,
            post_reset_margin_scale: 1.15,

            // Phase 2.1c: Anti-Thrash Post-Rescue Lock
            post_rescue_lock_ticks: 80,
            lock_margin_min_scale: 1.20,
            lock_focus_bias: 2.0,
            rescue_requires_bad_state: true,
            rescue_bad_proto: 0.12,
            rescue_bad_margin: 0.03,
            rescue_bad_value: 0.10,
        }
    }
}

impl ModePolicyConfig {
    /// Create a ModePolicyConfig from a Config.
    pub fn from_config(config: &crate::config::Config) -> Self {
        Self {
            explore_v_max: config.mode_explore_v_max,
            exploit_v_min: config.mode_exploit_v_min,
            reset_td_min: config.mode_reset_td_min,
            reset_value_drop: config.mode_reset_value_drop,
            reset_fail_streak: config.mode_reset_fail_streak,
            post_reset_cooldown: config.mode_post_reset_cooldown,
            explore_margin_min_scale: config.mode_explore_margin_scale,
            exploit_margin_min_scale: config.mode_exploit_margin_scale,
            reset_dampen: config.mode_reset_dampen,
            reset_dampen_top_k: config.mode_reset_dampen_top_k,
            window_size: config.mode_window_size,
            exploit_proto_min: config.exploit_proto_min,
            exploit_margin_min: config.exploit_margin_min,
            exploit_requires_stable: config.exploit_requires_stable,
            // Phase 2.1b: Seed-Robust Policy Stabilization
            min_exploit_ticks_on: config.min_exploit_ticks_on,
            explore_streak_rescue: config.explore_streak_rescue,
            fail_streak_rescue: config.fail_streak_rescue,
            rescue_cooldown: config.rescue_cooldown,
            catastrophic_abs_td: config.catastrophic_abs_td,
            catastrophic_value_drop: config.catastrophic_value_drop,
            post_reset_boost_ticks: config.post_reset_exploit_boost_ticks,
            post_reset_margin_scale: config.post_reset_exploit_margin_scale,

            // Phase 2.1c: Anti-Thrash Post-Rescue Lock
            post_rescue_lock_ticks: config.post_rescue_lock_ticks,
            lock_margin_min_scale: config.lock_margin_min_scale,
            lock_focus_bias: config.lock_focus_bias,
            rescue_requires_bad_state: config.rescue_requires_bad_state,
            rescue_bad_proto: config.rescue_bad_proto,
            rescue_bad_margin: config.rescue_bad_margin,
            rescue_bad_value: config.rescue_bad_value,
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

    // Phase 2.0f-E: Extended observation state
    /// Last observed proto_align
    pub last_proto_align: f32,
    /// Last observed topk_margin
    pub last_topk_margin: f64,
    /// Last observed is_stable flag
    pub last_is_stable: bool,
    /// Last observed gate_passed
    pub last_gate_passed: bool,

    // Phase 2.1b: Seed-Robust Policy Stabilization state
    /// Remaining ticks in exploit lock (countdown).
    pub exploit_lock_remaining: u32,
    /// Consecutive explore mode ticks.
    pub explore_streak: u32,
    /// Maximum explore streak observed.
    pub explore_streak_max: u32,
    /// Maximum exploit streak observed.
    pub exploit_streak_max: u32,
    /// Current exploit streak.
    pub exploit_streak: u32,
    /// Rescue cooldown counter.
    pub rescue_cooldown: u32,
    /// Number of rescues triggered.
    pub rescue_count: usize,
    /// Ticks remaining in post-reset boost period.
    pub post_reset_boost_remaining: u32,
    /// Last anchor value (for drop detection).
    pub prev_anchor_value: f32,
    /// Maximum consecutive gate fails observed.
    pub gate_fail_streak_max: u32,

    // Phase 2.1c: Post-rescue lock state
    /// Ticks remaining in post-rescue lock.
    pub post_rescue_lock_remaining: u32,
    /// Total ticks spent in post-rescue lock.
    pub post_rescue_lock_total_ticks: usize,
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
            // Phase 2.0f-E: Extended observation state
            last_proto_align: 0.0,
            last_topk_margin: 0.0,
            last_is_stable: false,
            last_gate_passed: false,

            // Phase 2.1b: Seed-Robust Policy Stabilization state
            exploit_lock_remaining: 0,
            explore_streak: 0,
            explore_streak_max: 0,
            exploit_streak_max: 0,
            exploit_streak: 0,
            rescue_cooldown: 0,
            rescue_count: 0,
            post_reset_boost_remaining: 0,
            prev_anchor_value: 0.0,
            gate_fail_streak_max: 0,

            // Phase 2.1c: Post-rescue lock state
            post_rescue_lock_remaining: 0,
            post_rescue_lock_total_ticks: 0,
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
        // Delegate to extended observe with default extended values
        self.observe_extended(
            current_tick,
            anchor_value,
            abs_td,
            gate_passed,
            0.0,
            0.0,
            false,
        );
    }

    /// Extended observe with proto_align, margin, and stable flag for natural Exploit emergence.
    /// Phase 2.0f-E: Use this method for natural mode selection based on signal quality.
    pub fn observe_extended(
        &mut self,
        current_tick: u64,
        anchor_value: f32,
        abs_td: f32,
        gate_passed: bool,
        proto_align: f32,
        topk_margin: f64,
        is_stable: bool,
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

        // Phase 2.0f-E: Store extended observation state
        self.state.last_proto_align = proto_align;
        self.state.last_topk_margin = topk_margin;
        self.state.last_is_stable = is_stable;
        self.state.last_gate_passed = gate_passed;

        // Phase 2.1b: Update max gate fail streak
        if self.state.gate_fail_streak > self.state.gate_fail_streak_max {
            self.state.gate_fail_streak_max = self.state.gate_fail_streak;
        }

        // Phase 2.1b: Decrement rescue cooldown
        if self.state.rescue_cooldown > 0 {
            self.state.rescue_cooldown -= 1;
        }

        // Phase 2.1b: Decrement exploit lock
        if self.state.exploit_lock_remaining > 0 {
            self.state.exploit_lock_remaining -= 1;
        }

        // Phase 2.1b: Decrement post-reset boost
        if self.state.post_reset_boost_remaining > 0 {
            self.state.post_reset_boost_remaining -= 1;
        }

        // Phase 2.1c: Decrement post-rescue lock
        if self.state.post_rescue_lock_remaining > 0 {
            self.state.post_rescue_lock_remaining -= 1;
            self.state.post_rescue_lock_total_ticks += 1;
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
    /// Phase 2.0f-E: Uses extended signals for natural Exploit emergence.
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
            // Phase 2.0f-E: Natural Exploit emergence based on signal quality
            // Exploit triggers when: gate_passed AND (stable OR !requires_stable)
            //                        AND proto_align >= min AND margin >= min
            let stable_ok = !cfg.exploit_requires_stable || state.last_is_stable;
            let proto_ok = state.last_proto_align >= cfg.exploit_proto_min;
            let margin_ok = state.last_topk_margin >= cfg.exploit_margin_min;
            let gate_ok = state.last_gate_passed;

            let can_exploit = gate_ok && stable_ok && proto_ok && margin_ok;

            // Check Explore vs Exploit
            let recent_v = state.recent_values.mean();

            if can_exploit {
                // Strong signal quality -> Exploit mode
                Mode::Exploit
            } else if recent_v < cfg.explore_v_max {
                // Low value -> Explore mode
                Mode::Explore
            } else if recent_v >= cfg.exploit_v_min {
                // High value alone can also trigger Exploit (original condition)
                Mode::Exploit
            } else {
                // Middle ground: default to Explore for more diversity
                Mode::Explore
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

    /// Choose mode with Phase 2.1b/c guardrails (exploit lock + explore rescue + post-rescue lock).
    /// Returns (mode, rescue_triggered).
    pub fn choose_mode_with_guardrails(&mut self, current_tick: u64) -> (Mode, bool) {
        // Store previous value for drop detection
        let current_value = self.state.recent_values.last().unwrap_or(0.0);
        let value_drop = self.state.prev_anchor_value - current_value;
        self.state.prev_anchor_value = current_value;

        // Check for catastrophic conditions (escape exploit lock and post-rescue lock)
        let recent_td = self.state.recent_abs_td.mean();
        let catastrophic = recent_td >= self.config.catastrophic_abs_td
            || value_drop >= self.config.catastrophic_value_drop;

        // Phase 2.1c: Check post-rescue lock (takes priority over exploit lock)
        if self.state.post_rescue_lock_remaining > 0 && !catastrophic {
            // Forced to stay in Exploit mode during post-rescue lock
            self.state.exploit_count += 1;
            self.state.exploit_streak += 1;
            if self.state.exploit_streak > self.state.exploit_streak_max {
                self.state.exploit_streak_max = self.state.exploit_streak;
            }
            self.state.explore_streak = 0;
            self.state.last_mode = Mode::Exploit;
            return (Mode::Exploit, false);
        }

        // Check if we're in exploit lock
        if self.state.exploit_lock_remaining > 0 && !catastrophic {
            // Forced to stay in Exploit mode
            self.state.exploit_count += 1;
            self.state.exploit_streak += 1;
            if self.state.exploit_streak > self.state.exploit_streak_max {
                self.state.exploit_streak_max = self.state.exploit_streak;
            }
            self.state.explore_streak = 0;
            self.state.last_mode = Mode::Exploit;
            return (Mode::Exploit, false);
        }

        // Phase 2.1c: Check for bad_state condition for stricter rescue
        let is_bad_state = self.state.last_proto_align < self.config.rescue_bad_proto
            && self.state.last_topk_margin < self.config.rescue_bad_margin
            && current_value < self.config.rescue_bad_value;

        // Check for rescue conditions
        // Phase 2.1c: Require (streak condition) AND (td OR value_drop OR bad_state) if rescue_requires_bad_state
        let streak_condition = self.state.explore_streak >= self.config.explore_streak_rescue
            || self.state.gate_fail_streak >= self.config.fail_streak_rescue;

        let secondary_condition = if self.config.rescue_requires_bad_state {
            recent_td >= self.config.reset_td_min
                || value_drop >= self.config.catastrophic_value_drop * 0.5
                || is_bad_state
        } else {
            true
        };

        let rescue_needed = self.state.rescue_cooldown == 0 && streak_condition && secondary_condition;

        if rescue_needed {
            // Trigger rescue: force Reset mode
            self.state.rescue_count += 1;
            self.state.rescue_cooldown = self.config.rescue_cooldown;
            self.state.explore_streak = 0;
            self.state.exploit_streak = 0;
            self.state.gate_fail_streak = 0;
            self.state.reset_count += 1;
            self.state.last_mode = Mode::Reset;
            self.state.post_reset_boost_remaining = self.config.post_reset_boost_ticks;

            // Phase 2.1c: Set post-rescue lock
            self.state.post_rescue_lock_remaining = self.config.post_rescue_lock_ticks;

            // Record pre-reset TD for effectiveness
            let pre_td_values = self.state.recent_abs_td.last_n(10);
            if !pre_td_values.is_empty() {
                self.state.pre_reset_td_mean =
                    pre_td_values.iter().sum::<f32>() / pre_td_values.len() as f32;
            }
            self.state.last_reset_tick = Some(current_tick);
            self.state.cooldown = self.config.post_reset_cooldown;

            return (Mode::Reset, true);
        }

        // Normal mode selection with reset check
        let should_reset = self.state.cooldown == 0 && self.check_reset_conditions();

        let mode = if should_reset {
            let pre_td_values = self.state.recent_abs_td.last_n(10);
            if !pre_td_values.is_empty() {
                self.state.pre_reset_td_mean =
                    pre_td_values.iter().sum::<f32>() / pre_td_values.len() as f32;
            }
            self.state.last_reset_tick = Some(current_tick);
            self.state.cooldown = self.config.post_reset_cooldown;
            self.state.gate_fail_streak = 0;
            self.state.post_reset_boost_remaining = self.config.post_reset_boost_ticks;
            Mode::Reset
        } else {
            // Natural Exploit emergence
            let stable_ok = !self.config.exploit_requires_stable || self.state.last_is_stable;
            let proto_ok = self.state.last_proto_align >= self.config.exploit_proto_min;
            let margin_ok = self.state.last_topk_margin >= self.config.exploit_margin_min;
            let gate_ok = self.state.last_gate_passed;
            let can_exploit = gate_ok && stable_ok && proto_ok && margin_ok;

            let recent_v = self.state.recent_values.mean();

            if can_exploit {
                Mode::Exploit
            } else if recent_v < self.config.explore_v_max {
                Mode::Explore
            } else if recent_v >= self.config.exploit_v_min {
                Mode::Exploit
            } else {
                Mode::Explore
            }
        };

        // Update streaks and counters
        match mode {
            Mode::Explore => {
                self.state.explore_count += 1;
                self.state.explore_streak += 1;
                self.state.exploit_streak = 0;
                if self.state.explore_streak > self.state.explore_streak_max {
                    self.state.explore_streak_max = self.state.explore_streak;
                }
            }
            Mode::Exploit => {
                self.state.exploit_count += 1;
                self.state.exploit_streak += 1;
                self.state.explore_streak = 0;
                if self.state.exploit_streak > self.state.exploit_streak_max {
                    self.state.exploit_streak_max = self.state.exploit_streak;
                }
                // Start exploit lock on transition from non-Exploit
                if self.state.last_mode != Mode::Exploit {
                    self.state.exploit_lock_remaining = self.config.min_exploit_ticks_on;
                }
            }
            Mode::Reset => {
                self.state.reset_count += 1;
                self.state.explore_streak = 0;
                self.state.exploit_streak = 0;
            }
        }

        self.state.last_mode = mode;
        (mode, false)
    }

    /// Get margin scale with post-reset boost applied.
    pub fn get_margin_scale_with_boost(&self, base_scale: f32) -> f32 {
        if self.state.post_reset_boost_remaining > 0 {
            base_scale * self.config.post_reset_margin_scale
        } else {
            base_scale
        }
    }

    /// Phase 2.1c: Check if post-rescue lock is active.
    pub fn is_post_rescue_lock_active(&self) -> bool {
        self.state.post_rescue_lock_remaining > 0
    }

    /// Phase 2.1c: Get the Focus bias to apply during post-rescue lock.
    pub fn get_lock_focus_bias(&self) -> f32 {
        if self.state.post_rescue_lock_remaining > 0 {
            self.config.lock_focus_bias
        } else {
            0.0
        }
    }

    /// Phase 2.1c: Get margin scale during post-rescue lock.
    pub fn get_lock_margin_scale(&self) -> f32 {
        if self.state.post_rescue_lock_remaining > 0 {
            self.config.lock_margin_min_scale
        } else {
            1.0
        }
    }

    /// Phase 2.1c: Get total ticks spent in post-rescue lock.
    pub fn post_rescue_lock_total_ticks(&self) -> usize {
        self.state.post_rescue_lock_total_ticks
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
            // Phase 2.1b: Guardrail metrics
            explore_streak_max: self.state.explore_streak_max,
            exploit_streak_max: self.state.exploit_streak_max,
            gate_fail_streak_max: self.state.gate_fail_streak_max,
            rescue_count: self.state.rescue_count,
            // Phase 2.1c: Post-rescue lock metrics
            post_rescue_lock_total_ticks: self.state.post_rescue_lock_total_ticks,
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
    // Phase 2.1b: Guardrail metrics
    pub explore_streak_max: u32,
    pub exploit_streak_max: u32,
    pub gate_fail_streak_max: u32,
    pub rescue_count: usize,
    // Phase 2.1c: Post-rescue lock metrics
    pub post_rescue_lock_total_ticks: usize,
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
