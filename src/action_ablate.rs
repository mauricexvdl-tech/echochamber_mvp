//! Phase 2.0d: Action Policy Ablations + Sensitivity Sweep
//!
//! Validates that ActionPolicy (Scan/Focus/Perturb) is causal, not decorative.
//! Provides ablation variants and sensitivity sweep infrastructure.

use crate::action::Action;

/// Ablation variant identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionAblationVariant {
    /// All actions enabled (baseline)
    Full,
    /// Disable Scan action (force Focus when Explore mode)
    NoScan,
    /// Disable Perturb action (force Focus when Reset mode)
    NoPerturb,
    /// Disable Focus action (use Scan when Exploit mode)
    NoFocus,
    /// Choose actions uniformly at random (unfair baseline - different budget)
    Random,
    /// Budget-matched random: same action rates as FULL, but random timing
    RandomBudgeted,
    /// Trigger-matched random: same trigger moments as FULL, but random action choice
    RandomTriggerMatched,
}

impl ActionAblationVariant {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Full => "FULL",
            Self::NoScan => "NO_SCAN",
            Self::NoPerturb => "NO_PERTURB",
            Self::NoFocus => "NO_FOCUS",
            Self::Random => "RANDOM",
            Self::RandomBudgeted => "RANDOM_BUDGETED",
            Self::RandomTriggerMatched => "RANDOM_TRIGGER",
        }
    }

    pub fn all_variants() -> Vec<Self> {
        vec![
            Self::Full,
            Self::NoScan,
            Self::NoPerturb,
            Self::NoFocus,
            Self::RandomBudgeted, // Use budgeted random instead of pure random
        ]
    }

    /// Variants for Demo 11 (trigger-matched comparison).
    pub fn demo11_variants() -> Vec<Self> {
        vec![Self::Full, Self::RandomBudgeted, Self::RandomTriggerMatched]
    }
}

/// Configuration for an ablation variant run.
#[derive(Clone, Debug)]
pub struct VariantConfig {
    /// Disable Scan action
    pub disable_scan: bool,
    /// Disable Focus action
    pub disable_focus: bool,
    /// Disable Perturb action
    pub disable_perturb: bool,
    /// Use random action selection
    pub random_actions: bool,
    /// Use budget-matched random action selection
    pub budgeted_random: bool,
    /// Use trigger-matched random action selection
    pub trigger_matched: bool,
}

impl VariantConfig {
    pub fn from_variant(variant: ActionAblationVariant) -> Self {
        match variant {
            ActionAblationVariant::Full => Self {
                disable_scan: false,
                disable_focus: false,
                disable_perturb: false,
                random_actions: false,
                budgeted_random: false,
                trigger_matched: false,
            },
            ActionAblationVariant::NoScan => Self {
                disable_scan: true,
                disable_focus: false,
                disable_perturb: false,
                random_actions: false,
                budgeted_random: false,
                trigger_matched: false,
            },
            ActionAblationVariant::NoPerturb => Self {
                disable_scan: false,
                disable_focus: false,
                disable_perturb: true,
                random_actions: false,
                budgeted_random: false,
                trigger_matched: false,
            },
            ActionAblationVariant::NoFocus => Self {
                disable_scan: false,
                disable_focus: true,
                disable_perturb: false,
                random_actions: false,
                budgeted_random: false,
                trigger_matched: false,
            },
            ActionAblationVariant::Random => Self {
                disable_scan: false,
                disable_focus: false,
                disable_perturb: false,
                random_actions: true,
                budgeted_random: false,
                trigger_matched: false,
            },
            ActionAblationVariant::RandomBudgeted => Self {
                disable_scan: false,
                disable_focus: false,
                disable_perturb: false,
                random_actions: false,
                budgeted_random: true,
                trigger_matched: false,
            },
            ActionAblationVariant::RandomTriggerMatched => Self {
                disable_scan: false,
                disable_focus: false,
                disable_perturb: false,
                random_actions: false,
                budgeted_random: false,
                trigger_matched: true,
            },
        }
    }
}

/// Report from running an ablation variant.
#[derive(Clone, Debug, Default)]
pub struct VariantReport {
    pub label: String,

    // Action counts and rates
    pub scan_count: usize,
    pub focus_count: usize,
    pub perturb_count: usize,
    pub scan_rate: f64,
    pub focus_rate: f64,
    pub perturb_rate: f64,

    // Core performance metrics
    pub coverage_pos: f64,
    pub selective_accuracy: f64,
    pub false_positive_rate: f64,
    pub stable_drop_ratio: f64,

    // Merge metrics
    pub merges_done_proto: usize,
    pub avg_merge_score: f32,

    // Stability and TD metrics
    pub mean_abs_td: f64,
    pub stable_time_share: f64,

    // Perturb effectiveness
    pub perturb_effectiveness: f64,
    pub perturb_effectiveness_samples: usize,

    // Per-action stable shares for differentiation checks
    pub scan_stable_share: f64,
    pub focus_stable_share: f64,
    pub perturb_stable_share: f64,
}

impl VariantReport {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            ..Default::default()
        }
    }

    pub fn total_actions(&self) -> usize {
        self.scan_count + self.focus_count + self.perturb_count
    }
}

/// Sweep point result for sensitivity analysis.
#[derive(Clone, Debug, Default)]
pub struct SweepPoint {
    pub param_name: String,
    pub param_value: f64,

    // Action rates
    pub scan_rate: f64,
    pub focus_rate: f64,
    pub perturb_rate: f64,

    // Performance metrics
    pub coverage_pos: f64,
    pub selective_accuracy: f64,
    pub false_positive_rate: f64,
    pub stable_drop_ratio: f64,
    pub mean_abs_td: f64,
    pub stable_time_share: f64,
}

impl SweepPoint {
    pub fn new(param_name: &str, param_value: f64) -> Self {
        Self {
            param_name: param_name.to_string(),
            param_value,
            ..Default::default()
        }
    }
}

/// Choose action with ablation configuration.
/// Returns the modified action based on variant rules.
pub fn choose_action_with_ablation(
    base_action: Action,
    variant_config: &VariantConfig,
    rng_value: f64, // 0.0 to 1.0 for random selection
) -> Action {
    if variant_config.random_actions {
        // Uniform random selection among enabled actions
        let mut enabled = Vec::new();
        if !variant_config.disable_scan {
            enabled.push(Action::Scan);
        }
        if !variant_config.disable_focus {
            enabled.push(Action::Focus);
        }
        if !variant_config.disable_perturb {
            enabled.push(Action::Perturb);
        }
        if enabled.is_empty() {
            // Fallback: if all disabled, use Focus
            return Action::Focus;
        }
        let idx = (rng_value * enabled.len() as f64).floor() as usize;
        return enabled[idx.min(enabled.len() - 1)];
    }

    // Apply ablation rules
    match base_action {
        Action::Scan => {
            if variant_config.disable_scan {
                // Scan disabled -> use Focus instead
                if variant_config.disable_focus {
                    Action::Perturb // Last resort
                } else {
                    Action::Focus
                }
            } else {
                Action::Scan
            }
        }
        Action::Focus => {
            if variant_config.disable_focus {
                // Focus disabled -> use Scan instead
                if variant_config.disable_scan {
                    Action::Perturb // Last resort
                } else {
                    Action::Scan
                }
            } else {
                Action::Focus
            }
        }
        Action::Perturb => {
            if variant_config.disable_perturb {
                // Perturb disabled -> use Focus instead
                if variant_config.disable_focus {
                    Action::Scan // Last resort
                } else {
                    Action::Focus
                }
            } else {
                Action::Perturb
            }
        }
    }
}

/// Sweep parameter configuration.
#[derive(Clone, Debug)]
pub struct SweepConfig {
    /// Target scan rates to sweep
    pub scan_rate_targets: Vec<f64>,
    /// Target perturb rates to sweep
    pub perturb_rate_targets: Vec<f64>,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            scan_rate_targets: vec![0.0, 0.02, 0.05, 0.08, 0.12],
            perturb_rate_targets: vec![0.0, 0.003, 0.007, 0.015, 0.030],
        }
    }
}

/// Budget-matched random action selector.
/// Uses a sliding window to enforce budget constraints matching FULL's action rates.
#[derive(Clone, Debug)]
pub struct BudgetedRandomAction {
    /// Sliding window size
    window_size: usize,
    /// Target scan rate (from FULL baseline)
    scan_target: f64,
    /// Target perturb rate (from FULL baseline)
    perturb_target: f64,
    /// Ring buffer of recent actions (0=Focus, 1=Scan, 2=Perturb)
    action_history: Vec<u8>,
    /// Current position in ring buffer
    position: usize,
    /// Current scan count in window
    scan_count: usize,
    /// Current perturb count in window
    perturb_count: usize,
    /// Whether buffer is fully populated
    buffer_full: bool,
}

impl BudgetedRandomAction {
    /// Create a new budgeted random action selector.
    /// scan_target and perturb_target are the rates observed from FULL baseline.
    pub fn new(scan_target: f64, perturb_target: f64) -> Self {
        const WINDOW_SIZE: usize = 2000;
        Self {
            window_size: WINDOW_SIZE,
            scan_target,
            perturb_target,
            action_history: vec![0; WINDOW_SIZE], // All Focus initially
            position: 0,
            scan_count: 0,
            perturb_count: 0,
            buffer_full: false,
        }
    }

    /// Maximum allowed scans in the window.
    fn max_scan(&self) -> usize {
        (self.window_size as f64 * self.scan_target).ceil() as usize
    }

    /// Maximum allowed perturbs in the window.
    fn max_perturb(&self) -> usize {
        (self.window_size as f64 * self.perturb_target).ceil() as usize
    }

    /// Choose next action with budget enforcement.
    /// rng_value should be in [0.0, 1.0).
    pub fn choose(&mut self, rng_value: f64) -> Action {
        // Build list of allowed actions based on current budget
        let mut allowed = Vec::with_capacity(3);

        // Scan allowed if under budget
        if self.scan_count < self.max_scan() {
            allowed.push(Action::Scan);
        }

        // Perturb allowed if under budget
        if self.perturb_count < self.max_perturb() {
            allowed.push(Action::Perturb);
        }

        // Focus is always allowed (fallback)
        allowed.push(Action::Focus);

        // Choose uniformly among allowed actions
        let idx = (rng_value * allowed.len() as f64).floor() as usize;
        let chosen = allowed[idx.min(allowed.len() - 1)];

        // Record the action in the ring buffer
        self.record_action(chosen);

        chosen
    }

    /// Record an action in the ring buffer and update counts.
    fn record_action(&mut self, action: Action) {
        // If buffer is full, we need to remove the effect of the oldest action
        if self.buffer_full {
            let oldest = self.action_history[self.position];
            match oldest {
                1 => self.scan_count = self.scan_count.saturating_sub(1),
                2 => self.perturb_count = self.perturb_count.saturating_sub(1),
                _ => {} // Focus (0)
            }
        }

        // Record new action
        let action_code = match action {
            Action::Focus => 0,
            Action::Scan => 1,
            Action::Perturb => 2,
        };
        self.action_history[self.position] = action_code;

        // Update counts
        match action {
            Action::Scan => self.scan_count += 1,
            Action::Perturb => self.perturb_count += 1,
            Action::Focus => {}
        }

        // Advance position
        self.position = (self.position + 1) % self.window_size;
        if self.position == 0 {
            self.buffer_full = true;
        }
    }

    /// Reset the selector for a new run.
    pub fn reset(&mut self) {
        self.action_history.fill(0);
        self.position = 0;
        self.scan_count = 0;
        self.perturb_count = 0;
        self.buffer_full = false;
    }
}

// =============================================================================
// Phase 2.0e: Trigger-Matched Random Baseline
// =============================================================================

/// Records FULL policy's trigger decisions for replay by trigger-matched baseline.
#[derive(Clone, Debug)]
pub struct TriggerTrace {
    /// Per-tick: true if FULL chose a non-Focus action (Scan or Perturb)
    pub should_act: Vec<bool>,
    /// Per-tick: the actual action FULL chose (for reference, optional matching)
    pub actions: Vec<Action>,
}

impl TriggerTrace {
    pub fn new() -> Self {
        Self {
            should_act: Vec::new(),
            actions: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            should_act: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
        }
    }

    /// Record FULL's action at a tick.
    pub fn record(&mut self, action: Action) {
        let is_trigger = action != Action::Focus;
        self.should_act.push(is_trigger);
        self.actions.push(action);
    }

    /// Get whether this tick should trigger an action.
    pub fn should_act_at(&self, tick: usize) -> bool {
        self.should_act.get(tick).copied().unwrap_or(false)
    }

    /// Get FULL's action at this tick (for type-matched variant if needed).
    pub fn action_at(&self, tick: usize) -> Option<Action> {
        self.actions.get(tick).copied()
    }

    /// Count of trigger ticks.
    pub fn trigger_count(&self) -> usize {
        self.should_act.iter().filter(|&&x| x).count()
    }

    /// Total ticks recorded.
    pub fn len(&self) -> usize {
        self.should_act.len()
    }

    pub fn is_empty(&self) -> bool {
        self.should_act.is_empty()
    }
}

impl Default for TriggerTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Trigger-matched random action selector.
/// Uses FULL's trigger trace to decide WHEN to act, but chooses randomly WHAT action.
#[derive(Clone, Debug)]
pub struct TriggerMatchedRandom {
    /// Reference to the trigger trace from FULL run
    trigger_trace: TriggerTrace,
    /// Current tick position
    tick: usize,
}

impl TriggerMatchedRandom {
    pub fn new(trigger_trace: TriggerTrace) -> Self {
        Self {
            trigger_trace,
            tick: 0,
        }
    }

    /// Choose action for current tick, then advance.
    /// rng_value should be in [0.0, 1.0).
    pub fn choose(&mut self, rng_value: f64) -> Action {
        let should_act = self.trigger_trace.should_act_at(self.tick);
        self.tick += 1;

        if should_act {
            // Choose randomly between Scan and Perturb
            if rng_value < 0.5 {
                Action::Scan
            } else {
                Action::Perturb
            }
        } else {
            Action::Focus
        }
    }

    /// Reset for a new run.
    pub fn reset(&mut self) {
        self.tick = 0;
    }

    /// Get trigger count from the trace.
    pub fn trigger_count(&self) -> usize {
        self.trigger_trace.trigger_count()
    }
}
