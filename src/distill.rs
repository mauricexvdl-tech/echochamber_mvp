//! Phase 2.0f-B: Action Policy Distillation (Fair Budgeted)
//!
//! Trains a lightweight linear-softmax student to imitate the teacher ActionPolicy.
//! Uses only existing per-tick diagnostics as features.
//!
//! Key fairness guarantees:
//! - Single teacher action path used for both training and evaluation
//! - BudgetLimiter enforces identical action rate constraints on teacher AND student
//! - Calibration phase determines target rates before evaluation

use crate::action::Action;
use crate::rng::Rng;
use std::collections::VecDeque;

/// Number of input features for the student policy.
pub const NUM_FEATURES: usize = 10;

/// Number of output actions (Scan=0, Focus=1, Perturb=2).
pub const NUM_ACTIONS: usize = 3;

/// Configuration for policy distillation.
#[derive(Clone, Debug)]
pub struct DistillConfig {
    /// Enable distillation in Demo 12.
    pub enable: bool,
    /// Learning rate for SGD.
    pub lr: f32,
    /// L2 regularization coefficient.
    pub l2: f32,
    /// Softmax temperature.
    pub temperature: f32,
    /// Warmup ticks before training starts.
    pub train_warmup_ticks: u64,
    /// Replay buffer capacity.
    pub replay_capacity: usize,
    /// Mini-batch size for SGD.
    pub batch_size: usize,
    /// Train every N ticks.
    pub train_every: u64,
    /// RNG seed for replay sampling.
    pub seed: u64,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            enable: true,
            lr: 0.03,
            l2: 1e-4,
            temperature: 1.0,
            train_warmup_ticks: 5_000,
            replay_capacity: 50_000,
            batch_size: 128,
            train_every: 5,
            seed: 0xD1571,
        }
    }
}

/// Per-tick diagnostic values used as features for the student.
/// Keep this minimal - just the scalars already computed in the main loop.
#[derive(Clone, Debug, Default)]
pub struct TickDiag {
    pub gate_pass: bool,
    pub topk_margin: f64,
    pub proto_align: f32,
    pub anchor_value: f32,
    pub abs_td: f32,
    pub stable: bool,
    pub fail_streak: u32,
    pub total_power: f64,
    pub mode_bucket: u8, // 0=Explore, 1=Exploit, 2=Reset
}

/// Extract normalized feature vector from tick diagnostics.
pub fn extract_features(diag: &TickDiag) -> [f32; NUM_FEATURES] {
    let mut x = [0.0f32; NUM_FEATURES];

    // Feature 0: gate_pass (0 or 1)
    x[0] = if diag.gate_pass { 1.0 } else { 0.0 };

    // Feature 1: topk_margin (clipped to [0, 1])
    x[1] = (diag.topk_margin as f32).clamp(0.0, 1.0);

    // Feature 2: proto_align (already in [0, 1])
    x[2] = diag.proto_align.clamp(0.0, 1.0);

    // Feature 3: anchor_value (clipped to [0, 1])
    x[3] = diag.anchor_value.clamp(0.0, 1.0);

    // Feature 4: abs_td (clipped and scaled, typical range 0-0.5)
    x[4] = (diag.abs_td * 2.0).clamp(0.0, 1.0);

    // Feature 5: stable flag (0 or 1)
    x[5] = if diag.stable { 1.0 } else { 0.0 };

    // Feature 6: fail_streak (normalized, capped at 10)
    x[6] = (diag.fail_streak.min(10) as f32) / 10.0;

    // Feature 7: total_power (scaled, typical range 0-50)
    x[7] = ((diag.total_power as f32) / 50.0).clamp(0.0, 1.0);

    // Feature 8-9: mode one-hot (Explore, Exploit, Reset -> 2 bits suffice)
    x[8] = if diag.mode_bucket == 0 { 1.0 } else { 0.0 }; // Explore
    x[9] = if diag.mode_bucket == 2 { 1.0 } else { 0.0 }; // Reset

    x
}

/// Training sample: feature vector + teacher action label.
#[derive(Clone, Debug)]
pub struct Sample {
    pub x: [f32; NUM_FEATURES],
    pub y: u8, // 0=Scan, 1=Focus, 2=Perturb
}

/// Ring buffer for experience replay.
#[derive(Clone, Debug)]
pub struct ReplayBuffer {
    buf: Vec<Sample>,
    head: usize,
    capacity: usize,
    len: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity.min(1024)), // Don't pre-alloc huge
            head: 0,
            capacity,
            len: 0,
        }
    }

    /// Push a sample into the replay buffer.
    pub fn push(&mut self, x: [f32; NUM_FEATURES], y: u8) {
        let sample = Sample { x, y };
        if self.buf.len() < self.capacity {
            self.buf.push(sample);
            self.len = self.buf.len();
        } else {
            self.buf[self.head] = sample;
        }
        self.head = (self.head + 1) % self.capacity;
    }

    /// Sample a random batch from the buffer.
    pub fn sample_batch(&self, batch_size: usize, rng: &mut Rng) -> Vec<&Sample> {
        if self.len == 0 {
            return Vec::new();
        }
        let actual_batch = batch_size.min(self.len);
        let mut batch = Vec::with_capacity(actual_batch);
        for _ in 0..actual_batch {
            let idx = (rng.next_u64() as usize) % self.len;
            batch.push(&self.buf[idx]);
        }
        batch
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Linear softmax classifier: logits[a] = b[a] + sum_i W[a,i] * x[i]
#[derive(Clone, Debug)]
pub struct LinearSoftmax {
    /// Weights: W[action][feature]
    pub w: [[f32; NUM_FEATURES]; NUM_ACTIONS],
    /// Biases: b[action]
    pub b: [f32; NUM_ACTIONS],
}

impl LinearSoftmax {
    pub fn new() -> Self {
        Self {
            w: [[0.0; NUM_FEATURES]; NUM_ACTIONS],
            b: [0.0; NUM_ACTIONS],
        }
    }

    /// Compute logits for each action.
    pub fn logits(&self, x: &[f32; NUM_FEATURES]) -> [f32; NUM_ACTIONS] {
        let mut logits = self.b;
        for a in 0..NUM_ACTIONS {
            for i in 0..NUM_FEATURES {
                logits[a] += self.w[a][i] * x[i];
            }
        }
        logits
    }

    /// Compute softmax probabilities with temperature.
    pub fn probs(&self, x: &[f32; NUM_FEATURES], temperature: f32) -> [f32; NUM_ACTIONS] {
        let logits = self.logits(x);
        // Apply temperature
        let mut scaled = [0.0f32; NUM_ACTIONS];
        for i in 0..NUM_ACTIONS {
            scaled[i] = logits[i] / temperature.max(0.01);
        }
        // Stable softmax: subtract max
        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut probs = [0.0f32; NUM_ACTIONS];
        for i in 0..NUM_ACTIONS {
            probs[i] = (scaled[i] - max_logit).exp();
            exp_sum += probs[i];
        }
        for i in 0..NUM_ACTIONS {
            probs[i] /= exp_sum;
        }
        probs
    }

    /// Predict action (argmax of logits).
    pub fn predict(&self, x: &[f32; NUM_FEATURES]) -> usize {
        let logits = self.logits(x);
        let mut best = 0;
        let mut best_val = logits[0];
        for a in 1..NUM_ACTIONS {
            if logits[a] > best_val {
                best_val = logits[a];
                best = a;
            }
        }
        best
    }

    /// Train on a mini-batch with cross-entropy loss.
    /// Returns (mean_loss, accuracy).
    pub fn train_step(
        &mut self,
        batch: &[&Sample],
        lr: f32,
        l2: f32,
        temperature: f32,
    ) -> (f32, f32) {
        if batch.is_empty() {
            return (0.0, 0.0);
        }

        let batch_size = batch.len() as f32;
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;

        // Accumulate gradients
        let mut grad_w = [[0.0f32; NUM_FEATURES]; NUM_ACTIONS];
        let mut grad_b = [0.0f32; NUM_ACTIONS];

        for sample in batch {
            let probs = self.probs(&sample.x, temperature);
            let pred = self.predict(&sample.x);
            if pred == sample.y as usize {
                correct += 1;
            }

            // Cross-entropy loss: -log(prob[y])
            let prob_y = probs[sample.y as usize].max(1e-10);
            total_loss -= prob_y.ln();

            // Gradient: d_loss/d_logit[a] = prob[a] - 1{a == y}
            for a in 0..NUM_ACTIONS {
                let grad_logit = probs[a] - if a == sample.y as usize { 1.0 } else { 0.0 };
                // Scale by temperature for proper gradient
                let grad_logit_scaled = grad_logit / temperature.max(0.01);
                grad_b[a] += grad_logit_scaled;
                for i in 0..NUM_FEATURES {
                    grad_w[a][i] += grad_logit_scaled * sample.x[i];
                }
            }
        }

        // Apply gradients with L2 regularization
        for a in 0..NUM_ACTIONS {
            self.b[a] -= lr * (grad_b[a] / batch_size);
            for i in 0..NUM_FEATURES {
                self.w[a][i] -= lr * (grad_w[a][i] / batch_size + l2 * self.w[a][i]);
            }
        }

        (total_loss / batch_size, correct as f32 / batch_size)
    }

    /// Compute L2 norm of weights (for diagnostics).
    pub fn weight_norm(&self) -> f32 {
        let mut sum = 0.0f32;
        for a in 0..NUM_ACTIONS {
            sum += self.b[a] * self.b[a];
            for i in 0..NUM_FEATURES {
                sum += self.w[a][i] * self.w[a][i];
            }
        }
        sum.sqrt()
    }

    /// Total number of parameters.
    pub fn param_count() -> usize {
        NUM_ACTIONS * (NUM_FEATURES + 1)
    }
}

impl Default for LinearSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for distillation training and evaluation.
#[derive(Clone, Debug, Default)]
pub struct DistillStats {
    // Teacher action counts
    pub teacher_scan: usize,
    pub teacher_focus: usize,
    pub teacher_perturb: usize,

    // Student action counts
    pub student_scan: usize,
    pub student_focus: usize,
    pub student_perturb: usize,

    // Imitation stats
    pub total_samples: usize,
    pub correct_predictions: usize,

    // Confusion matrix: confusion[teacher][student]
    pub confusion: [[usize; NUM_ACTIONS]; NUM_ACTIONS],

    // Training stats
    pub loss_ema: f32,
    pub train_steps: usize,
}

impl DistillStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a teacher-student pair.
    pub fn record(&mut self, teacher_action: usize, student_action: usize) {
        // Teacher counts
        match teacher_action {
            0 => self.teacher_scan += 1,
            1 => self.teacher_focus += 1,
            _ => self.teacher_perturb += 1,
        }

        // Student counts
        match student_action {
            0 => self.student_scan += 1,
            1 => self.student_focus += 1,
            _ => self.student_perturb += 1,
        }

        // Imitation
        self.total_samples += 1;
        if teacher_action == student_action {
            self.correct_predictions += 1;
        }

        // Confusion
        if teacher_action < NUM_ACTIONS && student_action < NUM_ACTIONS {
            self.confusion[teacher_action][student_action] += 1;
        }
    }

    /// Update loss EMA after a training step.
    pub fn update_loss(&mut self, loss: f32) {
        const BETA: f32 = 0.99;
        if self.train_steps == 0 {
            self.loss_ema = loss;
        } else {
            self.loss_ema = BETA * self.loss_ema + (1.0 - BETA) * loss;
        }
        self.train_steps += 1;
    }

    /// Imitation accuracy.
    pub fn imitation_accuracy(&self) -> f64 {
        if self.total_samples > 0 {
            self.correct_predictions as f64 / self.total_samples as f64
        } else {
            0.0
        }
    }

    /// Teacher scan rate.
    pub fn teacher_scan_rate(&self) -> f64 {
        let total = self.teacher_scan + self.teacher_focus + self.teacher_perturb;
        if total > 0 {
            self.teacher_scan as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Teacher perturb rate.
    pub fn teacher_perturb_rate(&self) -> f64 {
        let total = self.teacher_scan + self.teacher_focus + self.teacher_perturb;
        if total > 0 {
            self.teacher_perturb as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Student scan rate.
    pub fn student_scan_rate(&self) -> f64 {
        let total = self.student_scan + self.student_focus + self.student_perturb;
        if total > 0 {
            self.student_scan as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Student perturb rate.
    pub fn student_perturb_rate(&self) -> f64 {
        let total = self.student_scan + self.student_focus + self.student_perturb;
        if total > 0 {
            self.student_perturb as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Reset for a new run.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Budget-enforced student action selector.
/// Allows student choice only if within budget, else falls back to Focus.
#[derive(Clone, Debug)]
pub struct BudgetedStudent {
    window_size: usize,
    scan_target: f64,
    perturb_target: f64,
    action_history: Vec<u8>, // 0=Scan, 1=Focus, 2=Perturb
    position: usize,
    scan_count: usize,
    perturb_count: usize,
    buffer_full: bool,
}

impl BudgetedStudent {
    /// Create with target rates from teacher.
    pub fn new(scan_target: f64, perturb_target: f64) -> Self {
        const WINDOW: usize = 2000;
        Self {
            window_size: WINDOW,
            scan_target,
            perturb_target,
            action_history: vec![1; WINDOW], // All Focus initially
            position: 0,
            scan_count: 0,
            perturb_count: 0,
            buffer_full: false,
        }
    }

    fn max_scan(&self) -> usize {
        ((self.window_size as f64) * self.scan_target * 1.1).ceil() as usize // 10% slack
    }

    fn max_perturb(&self) -> usize {
        ((self.window_size as f64) * self.perturb_target * 1.1).ceil() as usize
    }

    /// Apply student's preferred action, constrained by budget.
    /// Returns the actual action to execute.
    pub fn apply(&mut self, student_choice: usize) -> usize {
        let action = match student_choice {
            0 if self.scan_count < self.max_scan() => 0, // Scan allowed
            2 if self.perturb_count < self.max_perturb() => 2, // Perturb allowed
            0 | 2 => 1,                                  // Over budget, fall back to Focus
            _ => 1,                                      // Focus always allowed
        };

        // Update history
        if self.buffer_full {
            let oldest = self.action_history[self.position];
            match oldest {
                0 => self.scan_count = self.scan_count.saturating_sub(1),
                2 => self.perturb_count = self.perturb_count.saturating_sub(1),
                _ => {}
            }
        }

        self.action_history[self.position] = action as u8;
        match action {
            0 => self.scan_count += 1,
            2 => self.perturb_count += 1,
            _ => {}
        }

        self.position = (self.position + 1) % self.window_size;
        if self.position == 0 {
            self.buffer_full = true;
        }

        action
    }

    pub fn reset(&mut self) {
        self.action_history.fill(1);
        self.position = 0;
        self.scan_count = 0;
        self.perturb_count = 0;
        self.buffer_full = false;
    }
}

// =============================================================================
// Phase 2.0f-B: Fair BudgetLimiter for both Teacher and Student
// =============================================================================

/// Target action rates from calibration phase.
#[derive(Clone, Debug, Default)]
pub struct CalibrationRates {
    pub scan_rate: f64,
    pub focus_rate: f64,
    pub perturb_rate: f64,
    pub total_ticks: usize,
}

impl CalibrationRates {
    pub fn from_counts(scan: usize, focus: usize, perturb: usize) -> Self {
        let total = scan + focus + perturb;
        if total == 0 {
            return Self::default();
        }
        Self {
            scan_rate: scan as f64 / total as f64,
            focus_rate: focus as f64 / total as f64,
            perturb_rate: perturb as f64 / total as f64,
            total_ticks: total,
        }
    }
}

/// Budget limiter that enforces target action rates over a sliding window.
/// Used by BOTH teacher and student during evaluation for fair comparison.
#[derive(Clone, Debug)]
pub struct BudgetLimiter {
    /// Sliding window size
    window_size: usize,
    /// Target scan rate
    scan_target: f64,
    /// Target perturb rate
    perturb_target: f64,
    /// Ring buffer of recent actions
    history: VecDeque<u8>, // 0=Scan, 1=Focus, 2=Perturb
    /// Current scan count in window
    scan_count: usize,
    /// Current perturb count in window
    perturb_count: usize,
}

impl BudgetLimiter {
    /// Create a budget limiter with target rates.
    pub fn new(window_size: usize, scan_target: f64, perturb_target: f64) -> Self {
        Self {
            window_size,
            scan_target,
            perturb_target,
            history: VecDeque::with_capacity(window_size),
            scan_count: 0,
            perturb_count: 0,
        }
    }

    /// Create from calibration rates.
    pub fn from_calibration(window_size: usize, rates: &CalibrationRates) -> Self {
        Self::new(window_size, rates.scan_rate, rates.perturb_rate)
    }

    /// Maximum allowed scans in the window (with small slack for variance).
    fn max_scan(&self) -> usize {
        ((self.window_size as f64) * self.scan_target * 1.05 + 1.0).ceil() as usize
    }

    /// Maximum allowed perturbs in the window.
    fn max_perturb(&self) -> usize {
        ((self.window_size as f64) * self.perturb_target * 1.05 + 1.0).ceil() as usize
    }

    /// Check if an action is allowed under the budget.
    pub fn is_allowed(&self, action: Action) -> bool {
        match action {
            Action::Scan => self.scan_count < self.max_scan(),
            Action::Perturb => self.perturb_count < self.max_perturb(),
            Action::Focus => true, // Always allowed
        }
    }

    /// Apply budget constraint: if preferred action exceeds budget, fall back to Focus.
    pub fn apply(&mut self, preferred: Action) -> Action {
        let action = if self.is_allowed(preferred) {
            preferred
        } else {
            Action::Focus
        };

        self.record(action);
        action
    }

    /// Apply budget constraint given an action index (0=Scan, 1=Focus, 2=Perturb).
    pub fn apply_idx(&mut self, preferred_idx: usize) -> usize {
        let preferred = match preferred_idx {
            0 => Action::Scan,
            2 => Action::Perturb,
            _ => Action::Focus,
        };
        let actual = self.apply(preferred);
        match actual {
            Action::Scan => 0,
            Action::Focus => 1,
            Action::Perturb => 2,
        }
    }

    /// Record an action in the sliding window.
    fn record(&mut self, action: Action) {
        // Remove oldest if window is full
        if self.history.len() >= self.window_size {
            if let Some(oldest) = self.history.pop_front() {
                match oldest {
                    0 => self.scan_count = self.scan_count.saturating_sub(1),
                    2 => self.perturb_count = self.perturb_count.saturating_sub(1),
                    _ => {}
                }
            }
        }

        // Add new action
        let code = match action {
            Action::Scan => 0u8,
            Action::Focus => 1u8,
            Action::Perturb => 2u8,
        };
        self.history.push_back(code);
        match action {
            Action::Scan => self.scan_count += 1,
            Action::Perturb => self.perturb_count += 1,
            Action::Focus => {}
        }
    }

    /// Reset the limiter state.
    pub fn reset(&mut self) {
        self.history.clear();
        self.scan_count = 0;
        self.perturb_count = 0;
    }

    /// Get current scan rate in window.
    pub fn current_scan_rate(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            self.scan_count as f64 / self.history.len() as f64
        }
    }

    /// Get current perturb rate in window.
    pub fn current_perturb_rate(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            self.perturb_count as f64 / self.history.len() as f64
        }
    }
}

/// Evaluation metrics for Demo 12.
#[derive(Clone, Debug, Default)]
pub struct EvalMetrics {
    // Memory performance
    pub coverage_pos: f64,
    pub selective_accuracy: f64,
    pub false_positive_rate: f64,
    pub stable_share: f64,

    // Action counts
    pub scan_count: usize,
    pub focus_count: usize,
    pub perturb_count: usize,

    // Total ticks
    pub total_ticks: usize,
}

impl EvalMetrics {
    pub fn scan_rate(&self) -> f64 {
        let total = self.scan_count + self.focus_count + self.perturb_count;
        if total > 0 {
            self.scan_count as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn focus_rate(&self) -> f64 {
        let total = self.scan_count + self.focus_count + self.perturb_count;
        if total > 0 {
            self.focus_count as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn perturb_rate(&self) -> f64 {
        let total = self.scan_count + self.focus_count + self.perturb_count;
        if total > 0 {
            self.perturb_count as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Acceptance check result for Demo 12.
#[derive(Clone, Debug)]
pub struct AcceptanceResult {
    // Section A: Imitation
    pub imitation_acc: f64,
    pub imitation_ok: bool,

    // Section B: Performance tolerance
    pub coverage_delta: f64,
    pub coverage_ok: bool,
    pub sel_acc_delta: f64,
    pub sel_acc_ok: bool,
    pub fp_rate: f64,
    pub fp_ok: bool,

    // Section C: Budget fairness
    pub scan_delta: f64,
    pub scan_ok: bool,
    pub focus_delta: f64,
    pub focus_ok: bool,
    pub perturb_delta: f64,
    pub perturb_ok: bool,
}

impl AcceptanceResult {
    /// Check if all criteria pass.
    pub fn all_pass(&self) -> bool {
        self.imitation_ok
            && self.coverage_ok
            && self.sel_acc_ok
            && self.fp_ok
            && self.scan_ok
            && self.focus_ok
            && self.perturb_ok
    }

    /// Get which sections failed.
    pub fn failed_sections(&self) -> Vec<&'static str> {
        let mut failed = Vec::new();
        if !self.imitation_ok {
            failed.push("A (imitation)");
        }
        if !self.coverage_ok || !self.sel_acc_ok || !self.fp_ok {
            failed.push("B (performance)");
        }
        if !self.scan_ok || !self.focus_ok || !self.perturb_ok {
            failed.push("C (budget fairness)");
        }
        failed
    }
}
