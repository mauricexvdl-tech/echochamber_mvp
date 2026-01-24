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

// ============================================================================
// Phase 2.1g: True sliding window for chronic stats
// ============================================================================

/// Ring buffer entry for chronic window tracking.
#[derive(Clone, Copy, Debug, Default)]
struct ChronicTick {
    was_bad: bool,
    was_stable: bool,
    was_active: bool,
}

/// True sliding window buffer for chronic instability detection.
/// Maintains exact counts via push/pop for consistent shares.
#[derive(Clone, Debug)]
pub struct ChronicWindowBuffer {
    buffer: Vec<ChronicTick>,
    head: usize,
    len: usize,
    capacity: usize,
    // Running counts for O(1) share computation
    bad_count: usize,
    stable_count: usize,
    active_count: usize,
}

impl ChronicWindowBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![ChronicTick::default(); capacity],
            head: 0,
            len: 0,
            capacity,
            bad_count: 0,
            stable_count: 0,
            active_count: 0,
        }
    }

    /// Push a new tick observation, popping oldest if at capacity.
    pub fn push(&mut self, was_bad: bool, was_stable: bool, was_active: bool) {
        // If at capacity, subtract the oldest entry's counts
        if self.len == self.capacity {
            let oldest_idx = (self.head + self.capacity - self.len) % self.capacity;
            let oldest = self.buffer[oldest_idx];
            if oldest.was_bad {
                self.bad_count = self.bad_count.saturating_sub(1);
            }
            if oldest.was_stable {
                self.stable_count = self.stable_count.saturating_sub(1);
            }
            if oldest.was_active {
                self.active_count = self.active_count.saturating_sub(1);
            }
        } else {
            self.len += 1;
        }

        // Add new entry
        let tick = ChronicTick {
            was_bad,
            was_stable,
            was_active,
        };
        self.buffer[self.head] = tick;
        self.head = (self.head + 1) % self.capacity;

        // Update counts
        if was_bad {
            self.bad_count += 1;
        }
        if was_stable {
            self.stable_count += 1;
        }
        if was_active {
            self.active_count += 1;
        }
    }

    /// Get bad state share (0.0 to 1.0).
    pub fn bad_share(&self) -> f32 {
        if self.len == 0 {
            0.0
        } else {
            self.bad_count as f32 / self.len as f32
        }
    }

    /// Get stable share (0.0 to 1.0).
    pub fn stable_share(&self) -> f32 {
        if self.len == 0 {
            0.0
        } else {
            self.stable_count as f32 / self.len as f32
        }
    }

    /// Get chronic active share (0.0 to 1.0).
    pub fn active_share(&self) -> f32 {
        if self.len == 0 {
            0.0
        } else {
            self.active_count as f32 / self.len as f32
        }
    }

    /// Check if buffer is filled to capacity.
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Current number of samples in buffer.
    pub fn len(&self) -> usize {
        self.len
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

    // Phase 2.1g: Chronic Instability Clamp v4 (enter-hold + true sliding window)
    /// Window size for chronic instability detection (true sliding window).
    pub chronic_window_ticks: usize,
    /// Bad state share threshold to ENTER clamp.
    pub chronic_bad_share_hi: f32,
    /// Stable share threshold to ENTER clamp.
    pub chronic_stable_share_lo: f32,
    /// Ticks in enter-hold window before decision.
    pub chronic_enter_hold_ticks: u32,
    /// Fraction of ticks that can fail in enter-hold window.
    pub chronic_enter_hold_tolerance: f32,
    /// Bad state share threshold to EXIT clamp (exit if below).
    pub chronic_exit_bad_max: f32,
    /// Stable share threshold to EXIT clamp (exit if above).
    pub chronic_exit_stable_min: f32,
    /// Ticks to hold exit conditions before exiting.
    pub chronic_exit_hold_ticks: u32,
    /// Fraction of ticks that can fail in hold window.
    pub chronic_exit_hold_tolerance: f32,
    /// Max Explore rate during clamp.
    pub chronic_explore_cap: f32,
    /// Minimum ticks to keep clamp active.
    pub chronic_lock_ticks: u32,
    /// Margin scale during chronic clamp.
    pub chronic_exploit_margin_scale: f32,
    /// Focus bias during chronic clamp.
    pub chronic_focus_bias: f32,
    /// Minimum ticks before enabling chronic detection.
    pub chronic_min_ticks_before_enable: u32,
    /// Cooldown after natural expiry (prevents immediate re-arm).
    pub chronic_rearm_cooldown: u32,
    /// Maximum chronic active share (watchdog).
    pub chronic_max_share: f32,
    /// Cooldown after watchdog release.
    pub chronic_release_cooldown: u32,
    /// Continuous ticks before escape pulse.
    pub chronic_escape_after: u32,
    /// Escape pulse duration.
    pub chronic_escape_ticks: u32,
    /// Disallow Perturb during chronic (except Reset).
    pub chronic_disallow_perturb: bool,
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

            // Phase 2.1g: Chronic Instability Clamp v4 (hysteresis + strict enter)
            chronic_window_ticks: 500,
            chronic_bad_share_hi: 0.35,
            chronic_stable_share_lo: 0.45,
            chronic_enter_hold_ticks: 50,
            chronic_enter_hold_tolerance: 0.10,
            chronic_exit_bad_max: 0.28,
            chronic_exit_stable_min: 0.60,
            chronic_exit_hold_ticks: 100,
            chronic_exit_hold_tolerance: 0.15,
            chronic_explore_cap: 0.10,
            chronic_lock_ticks: 150,
            chronic_exploit_margin_scale: 1.20,
            chronic_focus_bias: 2.5,
            chronic_min_ticks_before_enable: 5000,
            chronic_rearm_cooldown: 900,
            chronic_max_share: 0.50,
            chronic_release_cooldown: 200,
            chronic_escape_after: 1000,
            chronic_escape_ticks: 50,
            chronic_disallow_perturb: true,
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

            // Phase 2.1g: Chronic Instability Clamp v4 (hysteresis + enter-hold)
            chronic_window_ticks: config.chronic_window_ticks,
            chronic_bad_share_hi: config.chronic_bad_share_hi,
            chronic_stable_share_lo: config.chronic_stable_share_lo,
            chronic_enter_hold_ticks: config.chronic_enter_hold_ticks,
            chronic_enter_hold_tolerance: config.chronic_enter_hold_tolerance,
            chronic_exit_bad_max: config.chronic_exit_bad_max,
            chronic_exit_stable_min: config.chronic_exit_stable_min,
            chronic_exit_hold_ticks: config.chronic_exit_hold_ticks,
            chronic_exit_hold_tolerance: config.chronic_exit_hold_tolerance,
            chronic_explore_cap: config.chronic_explore_cap,
            chronic_lock_ticks: config.chronic_lock_ticks,
            chronic_exploit_margin_scale: config.chronic_exploit_margin_scale,
            chronic_focus_bias: config.chronic_focus_bias,
            chronic_min_ticks_before_enable: config.chronic_min_ticks_before_enable,
            chronic_rearm_cooldown: config.chronic_rearm_cooldown,
            chronic_max_share: config.chronic_max_share,
            chronic_release_cooldown: config.chronic_release_cooldown,
            chronic_escape_after: config.chronic_escape_after,
            chronic_escape_ticks: config.chronic_escape_ticks,
            chronic_disallow_perturb: config.chronic_disallow_perturb,
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

    // Phase 2.1g: Chronic Instability Clamp v4 state (true sliding window)
    /// True sliding window buffer for chronic stats.
    pub chronic_window: ChronicWindowBuffer,
    /// Ticks remaining in chronic lock.
    pub chronic_lock_remaining: u32,
    /// Total ticks spent in chronic lock.
    pub chronic_lock_total_ticks: usize,
    /// Rolling explore count for soft cap.
    pub chronic_explore_count: u32,
    /// Total ticks for global tracking.
    pub global_tick_count: u64,
    /// Ticks entry condition satisfied in enter-hold window.
    pub chronic_enter_hold_count: u32,
    /// Total ticks in enter-hold window.
    pub chronic_enter_hold_window: u32,
    /// Ticks exit conditions have been satisfied (pass count for tolerance).
    pub chronic_exit_hold_count: u32,
    /// Ticks in exit hold window (total count for tolerance).
    pub chronic_exit_hold_window: u32,
    /// Continuous ticks in chronic lock (for escape pulse).
    pub chronic_continuous_ticks: u32,
    /// Ticks in escape pulse mode.
    pub chronic_escape_remaining: u32,
    /// Cooldown after watchdog release.
    pub chronic_release_cooldown: u32,
    /// Cooldown after natural expiry (re-arm prevention).
    pub chronic_rearm_cooldown_remaining: u32,

    // Phase 2.1g: Chronic diagnostics
    /// Number of times chronic lock was entered.
    pub chronic_enter_count: u32,
    /// Number of times chronic lock was exited.
    pub chronic_exit_count: u32,
    /// Enters triggered by bad_share > threshold.
    pub chronic_enter_by_bad: u32,
    /// Enters triggered by stable_share < threshold.
    pub chronic_enter_by_unstable: u32,
    /// Exits forced by watchdog (chronic_active_share > max_share).
    pub chronic_exit_by_watchdog: u32,
}

impl ModePolicyState {
    pub fn new(window_size: usize, chronic_window_size: usize) -> Self {
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

            // Phase 2.1g: Chronic Instability Clamp v4 state (hysteresis + enter-hold)
            chronic_window: ChronicWindowBuffer::new(chronic_window_size),
            chronic_lock_remaining: 0,
            chronic_lock_total_ticks: 0,
            chronic_explore_count: 0,
            global_tick_count: 0,
            chronic_enter_hold_count: 0,
            chronic_enter_hold_window: 0,
            chronic_exit_hold_count: 0,
            chronic_exit_hold_window: 0,
            chronic_continuous_ticks: 0,
            chronic_escape_remaining: 0,
            chronic_release_cooldown: 0,
            chronic_rearm_cooldown_remaining: 0,

            // Phase 2.1g: Chronic diagnostics
            chronic_enter_count: 0,
            chronic_exit_count: 0,
            chronic_enter_by_bad: 0,
            chronic_enter_by_unstable: 0,
            chronic_exit_by_watchdog: 0,
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
        let chronic_window_size = config.chronic_window_ticks;
        Self {
            state: ModePolicyState::new(window_size, chronic_window_size),
            config,
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

        // Phase 2.1g: Update chronic lock state
        let is_chronic_active = self.state.chronic_lock_remaining > 0;
        if is_chronic_active {
            let was_active = self.state.chronic_lock_remaining > 1;
            self.state.chronic_lock_remaining -= 1;
            self.state.chronic_lock_total_ticks += 1;
            self.state.chronic_continuous_ticks += 1;

            // Phase 2.1f: When lock naturally expires, set rearm cooldown to prevent immediate re-entry
            if was_active && self.state.chronic_lock_remaining == 0 {
                self.state.chronic_rearm_cooldown_remaining = self.config.chronic_rearm_cooldown;
                self.state.chronic_exit_count += 1;
            }
        } else {
            self.state.chronic_continuous_ticks = 0;
        }

        // Phase 2.1e: Decrement escape pulse
        if self.state.chronic_escape_remaining > 0 {
            self.state.chronic_escape_remaining -= 1;
        }

        // Phase 2.1e: Decrement release cooldown
        if self.state.chronic_release_cooldown > 0 {
            self.state.chronic_release_cooldown -= 1;
        }

        // Phase 2.1g: Track chronic metrics with true sliding window
        self.state.global_tick_count += 1;

        // Determine if current tick is "bad state"
        let is_bad_state = !gate_passed
            || (topk_margin < self.config.rescue_bad_margin
                && proto_align < self.config.rescue_bad_proto
                && anchor_value < self.config.rescue_bad_value);

        // Phase 2.1g: Push observation to true sliding window buffer
        self.state
            .chronic_window
            .push(is_bad_state, is_stable, is_chronic_active);

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

        let rescue_needed =
            self.state.rescue_cooldown == 0 && streak_condition && secondary_condition;

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

        // Phase 2.1g: Chronic clamp v4 with enter-hold + true sliding window
        let chronic_enabled =
            self.state.global_tick_count >= self.config.chronic_min_ticks_before_enable as u64;

        // Get shares from true sliding window buffer
        let bad_share = self.state.chronic_window.bad_share();
        let stable_share = self.state.chronic_window.stable_share();
        let chronic_active_share = self.state.chronic_window.active_share();

        // Decrement cooldowns
        if self.state.chronic_release_cooldown > 0 {
            self.state.chronic_release_cooldown -= 1;
        }
        if self.state.chronic_rearm_cooldown_remaining > 0 {
            self.state.chronic_rearm_cooldown_remaining -= 1;
        }

        // Watchdog: force release if chronic is on too much
        if self.state.chronic_lock_remaining > 0
            && chronic_active_share > self.config.chronic_max_share
        {
            self.state.chronic_lock_remaining = 0;
            self.state.chronic_release_cooldown = self.config.chronic_release_cooldown;
            self.state.chronic_enter_hold_count = 0;
            self.state.chronic_enter_hold_window = 0;
            self.state.chronic_exit_hold_count = 0;
            self.state.chronic_exit_hold_window = 0;
            self.state.chronic_continuous_ticks = 0;
            self.state.chronic_exit_count += 1;
            self.state.chronic_exit_by_watchdog += 1;
        }

        // Escape pulse: if chronic lock continuous too long, allow brief escape
        if self.state.chronic_continuous_ticks >= self.config.chronic_escape_after
            && self.state.chronic_escape_remaining == 0
        {
            self.state.chronic_escape_remaining = self.config.chronic_escape_ticks;
        }

        if chronic_enabled && self.state.chronic_release_cooldown == 0 {
            if self.state.chronic_lock_remaining > 0 {
                // Phase 2.1f: Check EXIT conditions with OR (easier to exit)
                // Exit if bad_share improved OR stable_share improved
                let exit_ok = bad_share < self.config.chronic_exit_bad_max
                    || stable_share > self.config.chronic_exit_stable_min;

                // Track exit hold with tolerance
                self.state.chronic_exit_hold_window += 1;
                if exit_ok {
                    self.state.chronic_exit_hold_count += 1;
                }

                // Check if enough ticks in hold window passed the condition
                if self.state.chronic_exit_hold_window >= self.config.chronic_exit_hold_ticks {
                    let pass_rate = self.state.chronic_exit_hold_count as f32
                        / self.state.chronic_exit_hold_window as f32;
                    let min_pass_rate = 1.0 - self.config.chronic_exit_hold_tolerance;

                    if pass_rate >= min_pass_rate {
                        // Exit chronic clamp
                        self.state.chronic_lock_remaining = 0;
                        self.state.chronic_rearm_cooldown_remaining =
                            self.config.chronic_rearm_cooldown;
                        self.state.chronic_exit_count += 1;
                    }
                    // Reset hold window for next attempt
                    self.state.chronic_exit_hold_count = 0;
                    self.state.chronic_exit_hold_window = 0;
                    self.state.chronic_continuous_ticks = 0;
                }
            } else if self.state.chronic_rearm_cooldown_remaining == 0 {
                // Phase 2.1g: Check ENTER with strict consecutive streak requirement
                let enter_by_bad = bad_share > self.config.chronic_bad_share_hi;
                let enter_by_unstable = stable_share < self.config.chronic_stable_share_lo;
                let enter_cond = (enter_by_bad || enter_by_unstable)
                    && self.state.chronic_window.is_full();

                // Strict streak: reset on any false tick (including when window not full)
                if enter_cond {
                    self.state.chronic_enter_hold_count += 1;
                } else {
                    self.state.chronic_enter_hold_count = 0;
                }

                // Enter only if streak meets requirement
                if self.state.chronic_enter_hold_count >= self.config.chronic_enter_hold_ticks {
                    // Trigger chronic clamp
                    self.state.chronic_lock_remaining = self.config.chronic_lock_ticks;
                    self.state.chronic_explore_count = 0;
                    self.state.chronic_exit_hold_count = 0;
                    self.state.chronic_exit_hold_window = 0;
                    self.state.chronic_enter_hold_count = 0;
                    self.state.chronic_enter_count += 1;

                    // Track enter reason
                    if enter_by_bad {
                        self.state.chronic_enter_by_bad += 1;
                    }
                    if enter_by_unstable {
                        self.state.chronic_enter_by_unstable += 1;
                    }
                }
            }
        }

        // Normal mode selection with reset check
        let should_reset = self.state.cooldown == 0 && self.check_reset_conditions();

        let mut mode = if should_reset {
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

        // Phase 2.1e: Apply chronic clamp - during chronic lock, enforce Explore cap
        // Exception: during escape pulse, allow slightly more Explore
        if self.state.chronic_lock_remaining > 0 && mode == Mode::Explore {
            let total_modes =
                (self.state.explore_count + self.state.exploit_count + self.state.reset_count)
                    .max(1);
            let actual_explore_rate = self.state.explore_count as f32 / total_modes as f32;

            // During escape pulse, use relaxed cap (2x normal)
            let effective_cap = if self.state.chronic_escape_remaining > 0 {
                self.config.chronic_explore_cap * 2.0
            } else {
                self.config.chronic_explore_cap
            };

            // If actual explore rate is over cap, force Exploit
            if actual_explore_rate >= effective_cap {
                mode = Mode::Exploit;
            }
        }

        // Update streaks and counters
        match mode {
            Mode::Explore => {
                self.state.explore_count += 1;
                self.state.explore_streak += 1;
                self.state.exploit_streak = 0;
                if self.state.explore_streak > self.state.explore_streak_max {
                    self.state.explore_streak_max = self.state.explore_streak;
                }
                // Phase 2.1d: Track explore for chronic soft cap
                if self.state.chronic_lock_remaining > 0 {
                    self.state.chronic_explore_count += 1;
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

    /// Phase 2.1d: Check if chronic clamp is active.
    pub fn is_chronic_lock_active(&self) -> bool {
        self.state.chronic_lock_remaining > 0
    }

    /// Phase 2.1d: Get Focus bias during chronic clamp.
    pub fn get_chronic_focus_bias(&self) -> f32 {
        if self.state.chronic_lock_remaining > 0 {
            self.config.chronic_focus_bias
        } else {
            0.0
        }
    }

    /// Phase 2.1d: Get margin scale during chronic clamp.
    pub fn get_chronic_margin_scale(&self) -> f32 {
        if self.state.chronic_lock_remaining > 0 {
            self.config.chronic_exploit_margin_scale
        } else {
            1.0
        }
    }

    /// Phase 2.1e: Check if Perturb is disallowed during chronic lock.
    pub fn is_chronic_perturb_disallowed(&self) -> bool {
        self.state.chronic_lock_remaining > 0 && self.config.chronic_disallow_perturb
    }

    /// Phase 2.1e: Check if we're in escape pulse mode.
    pub fn is_escape_pulse_active(&self) -> bool {
        self.state.chronic_escape_remaining > 0
    }

    /// Phase 2.1d: Get total ticks spent in chronic lock.
    pub fn chronic_lock_total_ticks(&self) -> usize {
        self.state.chronic_lock_total_ticks
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
            // Phase 2.1f: Chronic clamp metrics
            chronic_lock_total_ticks: self.state.chronic_lock_total_ticks,
            chronic_enter_count: self.state.chronic_enter_count,
            chronic_exit_count: self.state.chronic_exit_count,
            chronic_enter_by_bad: self.state.chronic_enter_by_bad,
            chronic_enter_by_unstable: self.state.chronic_enter_by_unstable,
            chronic_exit_by_watchdog: self.state.chronic_exit_by_watchdog,
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
    // Phase 2.1f: Chronic clamp metrics
    pub chronic_lock_total_ticks: usize,
    pub chronic_enter_count: u32,
    pub chronic_exit_count: u32,
    pub chronic_enter_by_bad: u32,
    pub chronic_enter_by_unstable: u32,
    pub chronic_exit_by_watchdog: u32,
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
        self.state =
            ModePolicyState::new(self.config.window_size, self.config.chronic_window_ticks);
    }
}
