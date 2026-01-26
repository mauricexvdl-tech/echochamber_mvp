//! Configuration struct for Echo Chamber tunables.
//! Phase 1.4c: Added competitive label binding with abstain.
//! Phase 1.8: VALUE IS CONTROL - Memory lifecycle + self-regulation.
//! Phase 1.9: CONSOLIDATION - Make merges happen + reduce stability flicker.

use std::f64::consts::PI;

/// Configuration for the Echo Chamber network.
#[derive(Clone, Debug)]
pub struct Config {
    // Network topology
    pub num_nodes: usize,
    pub avg_degree: usize,
    pub num_ctx: usize,

    // Dynamics
    pub decay_per_tick: f64,
    pub clamp_max_amp: f64,
    pub eps: f64,

    // Homeostasis
    pub pow_target: f64,
    pub homeostasis_beta: f64,

    // Plasticity
    pub p_min: f64,
    pub align_pos: f64,
    pub align_neg: f64,
    pub phase_learn_rate: f64,
    pub min_edge_power: f64,

    // Affinity tracking
    pub aff_inc: i32,
    pub aff_dec: i32,
    pub aff_margin: i32,
    pub wrong_ctx_penalty: f64,
    pub ctx_suppress: f64,
    pub suppress_min_pow: f64,

    // Training/Eval
    pub train_ticks: usize,
    pub eval_ticks: usize,
    pub top_k: usize,

    // Causes
    pub num_causes: usize,
    pub injectors_per_cause: usize,
    pub inject_amp: f64,
    pub ph_noise: f64,
    pub noise_injects: usize,
    pub noise_amp: f64,
    pub prob_single_cause: f64,

    // RNG seed
    pub seed: u64,

    // =========================================================================
    // Phase 1.4a: Episodic Memory (ctx-based)
    // =========================================================================
    pub memory_max_entries: usize,
    pub memory_max_hamming: u32,
    pub memory_mask_w: i32,
    pub memory_proto_w: i32,
    pub memory_age_w: i32,
    pub memory_min_score: i32,
    pub memory_store_prob: f64,
    pub memory_min_power: f64,
    pub memory_debounce_ticks: u64,

    // =========================================================================
    // Phase 1.4b: One-Shot Label Binding (independent labels)
    // =========================================================================
    pub num_labels: usize,
    pub episode_ticks: usize,
    pub bind_tick: usize,
    pub recall_start_tick: usize,
    pub recall_stride: usize,
    pub num_episodes: usize,
    pub run_capacity_sweep: bool,
    pub label_memory_max_entries: usize,
    pub label_memory_max_hamming: u32,

    // =========================================================================
    // Phase 1.7a: Prototype Vector Config (Anchors as Concept Tokens)
    // =========================================================================
    /// Number of nodes in sparse prototype (PROTO_M).
    pub proto_m: usize,
    /// Learning rate for prototype updates.
    pub proto_eta: f32,
    /// Decay rate per update (applied before update).
    pub proto_decay: f32,
    /// Weight for prototype score in combined scoring.
    pub proto_beta: f32,
    /// Minimum margin to insert a new node into prototype.
    pub proto_insert_margin: f32,

    // =========================================================================
    // Phase 1.7b: Value Learning Config (Credit Assignment)
    // =========================================================================
    /// Learning rate for value updates (TD).
    pub alpha_v: f32,
    /// Discount factor for TD(0).
    pub gamma_v: f32,
    /// Clamp value to [-v_clip, +v_clip].
    pub v_clip: f32,
    /// EMA coefficient for tracking |TD|.
    pub v_td_ema: f32,
    /// Weight for value in tie-breaking score.
    pub v_beta: f32,
    /// Reward weight for power change.
    pub r_w_power: f32,
    /// Reward weight for coherence.
    pub r_w_coh: f32,
    /// Reward weight for prototype alignment.
    pub r_w_proto: f32,
    /// Reward weight for margin penalty.
    pub r_w_margin: f32,
    /// Clamp for power change in reward.
    pub r_p_clip: f32,
    /// Normalization factor for margin (margin/margin_norm maps to 0..1).
    pub margin_norm: f32,
    /// Phase 1.7c: Bootstrap value when abstaining due to margin fail.
    pub v_abstain_margin: f32,

    // =========================================================================
    // Phase 1.7d: Probe-set convergence metrics
    // =========================================================================
    /// Enable probe set for convergence tracking.
    pub probe_enabled: bool,
    /// Maximum number of keys in probe set.
    pub probe_size: usize,
    /// How often (in ticks) to evaluate probe deltas.
    pub probe_eval_stride: u32,
    /// Minimum keys before reporting probe metrics.
    pub probe_min_fill: usize,
    /// Use advantage-centered reward (r - r_ema).
    pub use_advantage_reward: bool,
    /// EMA coefficient for reward baseline (used when use_advantage_reward=true).
    pub reward_ema_beta: f32,

    // =========================================================================
    // Phase 1.4c: Competitive Label Binding with ABSTAIN
    // =========================================================================
    /// Number of labels for competitive experiment (more = harder)
    pub competitive_num_labels: usize,
    /// Max entries in global memory (persistent across episodes)
    pub competitive_max_entries: usize,
    /// Max Hamming distance for recall candidate
    pub competitive_max_hamming: u32,
    /// Minimum margin (dist2 - dist1) to avoid abstaining
    pub competitive_margin_min: u32,
    /// Number of bindings per episode
    pub competitive_binds_per_episode: usize,
    /// Probability of negative query (hard negative)
    pub competitive_p_neg: f64,
    /// Number of bits to flip for negative queries
    pub competitive_neg_flip_bits: u32,
    /// Number of episodes for competitive experiment
    pub competitive_episodes: usize,
    /// Ticks per episode for competitive experiment
    pub competitive_episode_ticks: usize,
    /// Recall stride for competitive experiment
    pub competitive_recall_stride: usize,
    /// Recall start tick within episode
    pub competitive_recall_start: usize,

    // =========================================================================
    // Phase 1.8: VALUE IS CONTROL (Memory lifecycle + self-regulation)
    // =========================================================================
    /// Weight for value component in keep_score() (eviction scoring).
    pub evict_v_weight: f32,
    /// Weight for usage component in keep_score().
    pub evict_use_weight: f32,
    /// Weight for age penalty in keep_score().
    pub evict_age_weight: f32,
    /// Decay constant for age penalty (ticks).
    pub evict_age_tau: f64,
    /// Maximum |v1 - v2| allowed for merging anchors.
    pub merge_v_delta_max: f32,
    /// Multiplier for gate thresholds in explore mode (more permissive).
    pub gate_explore_mult: f64,
    /// Multiplier for gate thresholds in stable mode (stricter).
    pub gate_stable_mult: f64,
    /// Minimum value V to become stable.
    pub stable_v_min: f32,
    /// Minimum wins to become stable.
    pub stable_wins_min: u32,
    /// Fraction of stable anchors needed to switch from explore to stable mode.
    pub stable_mode_threshold: f64,

    // =========================================================================
    // Demo Control Flags
    // =========================================================================
    /// Run baseline 5a (Phase 1.4c Competitive Binding).
    pub run_baseline_5a: bool,
    /// Run keyed 5b (Phase 1.9 Consolidation with Anchor+Mask).
    pub run_keyed_5b: bool,
    /// Run demo 7 (Phase 2.0a Mode Policy).
    pub run_demo_7: bool,
    /// Run demo 8 (Phase 2.0b Ablations + Mode Metrics).
    pub run_demo_8: bool,
    /// Run demo 9 (Phase 2.0c Mode → Action Loop).
    pub run_demo_9: bool,
    /// Run demo 10 (Phase 2.0d Action Ablations + Sweep).
    pub run_demo_10: bool,
    /// Run demo 11 (Phase 2.0e Trigger-Matched Random + Regret Metrics).
    pub run_demo_11: bool,
    /// Run demo 12 (Phase 2.0f-A Action Distillation).
    pub run_demo_12: bool,
    /// Run demo 13 (Phase 2.1 Multi-seed + Lift Metrics).
    pub run_demo_13: bool,

    // =========================================================================
    // Phase 2.1: Multi-seed Evaluation + Lift Metrics Configuration
    // =========================================================================
    /// Number of seeds for Demo 13 multi-seed evaluation.
    pub demo13_num_seeds: usize,
    /// Margin threshold for "bad state" in lift metrics.
    pub lift_bad_margin: f64,
    /// Proto alignment threshold for "bad state" in lift metrics.
    pub lift_bad_proto: f32,
    /// Value threshold for "bad state" in lift metrics.
    pub lift_bad_value: f32,

    // =========================================================================
    // Phase 2.1b: Seed-Robust Policy Stabilization
    // =========================================================================
    /// Minimum ticks to stay in Exploit mode once entered (anti-collapse).
    pub min_exploit_ticks_on: u32,
    /// Explore streak threshold to trigger rescue (force Reset/Perturb).
    pub explore_streak_rescue: u32,
    /// Gate fail streak threshold to trigger rescue.
    pub fail_streak_rescue: u32,
    /// Cooldown ticks after rescue before another rescue can fire.
    pub rescue_cooldown: u32,
    /// Ticks after Reset/Perturb to apply tighter exploit margin.
    pub post_reset_exploit_boost_ticks: u32,
    /// Scale factor for exploit margin during post-reset boost period.
    pub post_reset_exploit_margin_scale: f32,
    /// |TD| threshold to escape exploit lock (catastrophic).
    pub catastrophic_abs_td: f32,
    /// Value drop threshold to escape exploit lock (catastrophic).
    pub catastrophic_value_drop: f32,
    /// Floor for exploit proto threshold (adaptive).
    pub exploit_proto_min_floor: f32,
    /// Floor for exploit margin threshold (adaptive).
    pub exploit_margin_min_floor: f64,
    /// Scale factor for adaptive proto threshold (proto_p50 * scale).
    pub exploit_proto_p50_scale: f32,
    /// Scale factor for adaptive margin threshold (margin_p50 * scale).
    pub exploit_margin_p50_scale: f64,
    /// Enable adaptive thresholds in Demo 13.
    pub demo13_enable_adaptive_thresholds: bool,
    /// Enable rescue mechanism in Demo 13.
    pub demo13_enable_rescue: bool,
    /// Enable minimum perturb rate guard in Demo 13.
    pub demo13_enable_min_perturb_guard: bool,
    /// Minimum perturb rate target for Demo 13 guard.
    pub demo13_min_perturb_rate: f32,
    /// Window size for Demo 13 perturb rate guard.
    pub demo13_perturb_window: usize,

    // =========================================================================
    // Phase 2.1c: Anti-Thrash Post-Rescue Lock
    // =========================================================================
    /// Ticks to lock in Exploit mode after a rescue (anti-thrash).
    pub post_rescue_lock_ticks: u32,
    /// Margin min scale during post-rescue lock (higher = stricter).
    pub lock_margin_min_scale: f32,
    /// Focus bias additive boost during post-rescue lock.
    pub lock_focus_bias: f32,
    /// Require bad_state for rescue to fire (stricter trigger).
    pub rescue_requires_bad_state: bool,
    /// Bad state proto alignment threshold for rescue trigger.
    pub rescue_bad_proto: f32,
    /// Bad state margin threshold for rescue trigger.
    pub rescue_bad_margin: f64,
    /// Bad state value threshold for rescue trigger.
    pub rescue_bad_value: f32,

    // =========================================================================
    // Phase 2.1m: Lock hysteresis + soft exploit relaxation
    // =========================================================================
    /// Consecutive can_exploit failures before dropping exploit lock.
    pub lock_fail_drop_streak: u32,
    /// Minimum proto alignment for soft exploit fallback (relaxed threshold).
    pub proto_soft_min: f32,

    // =========================================================================
    // Phase 2.1n: Quality-aware actions + rescue throttle
    // =========================================================================
    /// Probability of Scan during soft exploit (0.0=always Focus, 1.0=always Scan).
    pub soft_exploit_scan_prob: f32,
    /// Max rescues per 10k ticks before throttle kicks in.
    pub rescue_max_per_10k: u32,
    /// Extended cooldown when rescue throttle is active.
    pub rescue_throttle_cooldown: u32,

    // =========================================================================
    // Phase 2.1o: Soft-exploit quarantine (block writes during low-quality exploit)
    // =========================================================================
    /// Block memory store operations during soft exploit.
    pub soft_exploit_block_store: bool,
    /// Block prototype updates during soft exploit.
    pub soft_exploit_block_proto_update: bool,
    /// Block merge candidate enqueuing during soft exploit.
    pub soft_exploit_block_merge: bool,

    // Phase 2.1o-fix: Rate-limited proto updates in soft exploit
    /// Allow proto update every N ticks in soft exploit (0 = every tick).
    pub soft_proto_update_period: u32,
    /// Minimum margin required for proto update in soft exploit.
    pub soft_proto_update_min_margin: f32,
    /// Require gate pass for proto update in soft exploit.
    pub soft_proto_update_require_gate: bool,
    /// Block proto update when in bad state during soft exploit.
    pub soft_proto_update_block_when_bad: bool,
    /// Maximum recent TD (instability) to allow proto update in soft exploit.
    pub soft_proto_update_td_max: f32,

    // =========================================================================
    // Phase 2.1q: Adaptive soft proto update period (P0 ↔ P12)
    // =========================================================================
    /// Enable adaptive period switching based on bad-regime detection.
    pub soft_proto_adaptive_enabled: bool,
    /// Period to use in "good" regime (default P0 = 0).
    pub soft_proto_period_good: u32,
    /// Period to use in "bad" regime (default P12 = 12).
    pub soft_proto_period_bad: u32,
    /// Enter bad-mode if stable_share < this threshold.
    pub soft_proto_bad_stable_lo: f32,
    /// Enter bad-mode if bad_share > this threshold.
    pub soft_proto_bad_bad_hi: f32,
    /// Enter bad-mode if explore_rate > this threshold.
    pub soft_proto_bad_explore_hi: f32,
    /// Enter bad-mode if rescues_per_tick > this threshold.
    pub soft_proto_bad_rescue_hi: f32,
    /// Ticks to hold in bad-mode before allowing exit.
    pub soft_proto_bad_hold_ticks: u32,
    /// Clear bad-mode early if stable_share > this threshold.
    pub soft_proto_bad_clear_stable: f32,
    /// Clear bad-mode early if bad_share < this threshold.
    pub soft_proto_bad_clear_bad: f32,

    // =========================================================================
    // Phase 2.1v: Bad-Regime Proto Repair (relaxed gating in bad regime)
    // =========================================================================
    /// In bad-regime, max abs_td to allow proto update (more relaxed than normal).
    pub soft_proto_bad_td_max: f32,
    /// In bad-regime, min margin to allow proto update.
    pub soft_proto_bad_margin_min: f32,
    /// In bad-regime, min proto_align to allow proto update.
    pub soft_proto_bad_proto_min: f32,
    /// In bad-regime, require gate pass for proto update.
    pub soft_proto_bad_require_gate: bool,
    /// Ticks after burst trigger to use P0 period (repair window).
    pub soft_proto_repair_after_burst_ticks: u32,

    // =========================================================================
    // Phase 2.1r: Bad-Regime Quality Repair (Targeted Perturb Burst)
    // Phase 2.1s: Episodic Perturb Bursts + Effectiveness Scoring
    // =========================================================================
    /// Enable bad-regime perturb burst repair mechanism.
    pub repair_enabled: bool,
    /// Enter bad-regime if stable_share < this threshold.
    pub repair_bad_stable_lo: f32,
    /// Enter bad-regime if bad_share > this threshold.
    pub repair_bad_share_hi: f32,
    /// Clear bad-regime if stable_share > this threshold.
    pub repair_clear_stable_hi: f32,
    /// Clear bad-regime if bad_share < this threshold.
    pub repair_clear_bad_lo: f32,
    /// Ticks condition must hold before entering bad-regime.
    pub repair_bad_hold_ticks: u32,
    /// Ticks clear conditions must hold before exiting bad-regime.
    pub repair_clear_hold_ticks: u32,
    /// Enter bad-regime if rescue rate exceeds this (per tick in 1000-tick window).
    pub repair_rescue_rate_hi: f32,
    /// Duration of perturb burst in ticks.
    pub repair_burst_ticks: u32,
    /// Probability of choosing Perturb during burst (when eligible).
    pub repair_burst_prob: f32,
    /// Cooldown after burst before another can trigger.
    pub repair_burst_cooldown: u32,
    /// Maximum allowed perturb rate mean (safety cap).
    pub repair_perturb_cap_mean: f32,

    // Phase 2.1s: Episodic Burst Configuration
    /// Minimum gap between burst triggers (prevents rapid re-trigger).
    pub burst_min_gap_ticks: u32,
    /// Maximum bursts per run (cap).
    pub burst_max_per_run: u32,
    /// Base burst probability (starting value).
    pub burst_base_prob: f32,
    /// Maximum burst probability (after escalation).
    pub burst_max_prob: f32,
    /// Base burst duration in ticks.
    pub burst_base_ticks: u32,
    /// Maximum burst duration in ticks (after escalation).
    pub burst_max_ticks: u32,
    /// Pre-burst measurement window (ticks before burst start).
    pub burst_pre_window: u32,
    /// Post-burst measurement window (ticks after burst end).
    pub burst_post_window: u32,
    /// TD reduction ratio threshold for success (post <= pre * ratio).
    pub burst_td_success_ratio: f32,
    /// Bad share absolute improvement threshold for success.
    pub burst_bad_improve_abs: f32,
    /// Stable share absolute gain threshold for success.
    pub burst_stable_gain_abs: f32,
    /// Probability increment per failed burst (escalation).
    pub burst_prob_escalation: f32,
    /// Duration increment per failed burst (escalation).
    pub burst_ticks_escalation: u32,

    // =========================================================================
    // Phase 2.1h: Chronic Instability Clamp v5 (EMA smoothing + hysteresis)
    // =========================================================================
    /// Window size for chronic instability detection (ticks).
    pub chronic_window_ticks: usize,
    /// EMA smoothing alpha for chronic shares (0.0=no smoothing, 1.0=no memory).
    pub chronic_share_ema_alpha: f32,
    /// Bad state share threshold to ENTER clamp.
    pub chronic_bad_share_hi: f32,
    /// Stable share threshold to ENTER clamp (below this = clamp).
    pub chronic_stable_share_lo: f32,
    /// Ticks in enter-hold window before decision.
    pub chronic_enter_hold_ticks: u32,
    /// Fraction of ticks that can fail in enter-hold window.
    pub chronic_enter_hold_tolerance: f32,
    /// Bad state share threshold to EXIT clamp (exit if below this).
    pub chronic_exit_bad_max: f32,
    /// Stable share threshold to EXIT clamp (exit if above this).
    pub chronic_exit_stable_min: f32,
    /// Ticks to hold exit conditions before actually exiting clamp.
    pub chronic_exit_hold_ticks: u32,
    /// Fraction of ticks in hold window that can fail and still count.
    pub chronic_exit_hold_tolerance: f32,
    /// Max Explore rate during clamp (soft cap).
    pub chronic_explore_cap: f32,
    /// Minimum ticks to keep clamp active once triggered.
    pub chronic_lock_ticks: u32,
    /// Margin scale during chronic clamp.
    pub chronic_exploit_margin_scale: f32,
    /// Focus bias during chronic clamp.
    pub chronic_focus_bias: f32,
    /// Minimum ticks before enabling chronic detection.
    pub chronic_min_ticks_before_enable: u32,
    /// Cooldown after chronic lock expires (prevent immediate re-arm).
    pub chronic_rearm_cooldown: u32,
    /// Maximum chronic active share before forced release (watchdog).
    pub chronic_max_share: f32,
    /// Cooldown ticks after watchdog forces release.
    pub chronic_release_cooldown: u32,
    /// Ticks of continuous chronic lock before escape pulse.
    pub chronic_escape_after: u32,
    /// Duration of escape pulse (slightly relaxed).
    pub chronic_escape_ticks: u32,
    /// Disallow Perturb during chronic lock (except Reset mode).
    pub chronic_disallow_perturb: bool,

    // =========================================================================
    // Phase 2.1e: Perturb Budget Cap
    // =========================================================================
    /// Maximum perturb rate (hard cap).
    pub perturb_cap: f32,
    /// Window size for perturb budget calculation.
    pub perturb_budget_window: usize,

    // =========================================================================
    // Phase 2.0e: Regret/Recovery Metrics Configuration
    // =========================================================================
    /// Margin threshold for "bad state" (topk_margin < margin_bad).
    pub regret_margin_bad: f64,
    /// Proto alignment threshold for "bad state".
    pub regret_proto_bad: f32,
    /// Value threshold for "bad state".
    pub regret_v_bad: f32,
    /// TD spike threshold (abs_td > td_spike counts as spike).
    pub regret_td_spike: f32,
    /// Window size for pre-action measurement.
    pub regret_pre_window: usize,
    /// Window size for post-action measurement.
    pub regret_post_window: usize,
    /// Window size for gate pass rate tracking.
    pub regret_post_gate_window: usize,
    /// Minimum improvement ratio to count as "good" recovery.
    pub regret_recovery_good_threshold: f64,

    // =========================================================================
    // Phase 2.0f-A: Action Distillation Configuration
    // =========================================================================
    /// Learning rate for distillation SGD.
    pub distill_lr: f32,
    /// L2 regularization coefficient.
    pub distill_l2: f32,
    /// Softmax temperature.
    pub distill_temperature: f32,
    /// Warmup ticks before training starts.
    pub distill_warmup_ticks: u64,
    /// Replay buffer capacity.
    pub distill_replay_capacity: usize,
    /// Mini-batch size for SGD.
    pub distill_batch_size: usize,
    /// Train every N ticks.
    pub distill_train_every: u64,

    // =========================================================================
    // Phase 2.0f-C: Mode-Conditioned Distillation + Two-Sided Budget Matching
    // =========================================================================
    /// Enable mode-conditioned features (3-dim one-hot for Explore/Exploit/Reset).
    pub distill_mode_cond: bool,
    /// Minimum samples per mode for stratified training.
    pub distill_min_mode_samples: usize,
    /// Sliding window size for target budget limiter.
    pub budget_window: usize,
    /// Tolerance for action rate deviation from target (±tol).
    pub budget_tol: f32,
    /// Weight for student preference (logit) in budget scoring.
    pub budget_lambda_pref: f32,
    /// Weight for deficit (target - current rate) in budget scoring.
    pub budget_lambda_def: f32,

    // =========================================================================
    // Phase 2.0f-E: Natural Exploit Emergence Configuration
    // =========================================================================
    /// Minimum proto_align for Exploit mode (stable + high alignment = exploit).
    pub exploit_proto_min: f32,
    /// Minimum topk_margin for Exploit mode.
    pub exploit_margin_min: f64,
    /// Require anchor to be stable for Exploit mode.
    pub exploit_requires_stable: bool,
    /// Focus bias added in Exploit mode when conditions are met.
    pub focus_bias_exploit: f32,

    // =========================================================================
    // Phase 2.0a: Mode Policy Configuration
    // =========================================================================
    /// Enable mode policy (Explore/Exploit/Reset) in Demo 7.
    pub enable_mode_policy: bool,
    /// Max anchor value to trigger Explore mode (low V = explore).
    pub mode_explore_v_max: f32,
    /// Min anchor value to stay in Exploit mode (high V = exploit).
    pub mode_exploit_v_min: f32,
    /// Min |TD| to consider Reset (high TD = unstable).
    pub mode_reset_td_min: f32,
    /// Value drop threshold to trigger Reset.
    pub mode_reset_value_drop: f32,
    /// Consecutive gate fails to trigger Reset.
    pub mode_reset_fail_streak: u32,
    /// Cooldown ticks after Reset before another Reset can fire.
    pub mode_post_reset_cooldown: u32,
    /// Scale factor for margin_min in Explore mode (< 1.0 = looser).
    pub mode_explore_margin_scale: f32,
    /// Scale factor for margin_min in Exploit mode (> 1.0 = tighter).
    pub mode_exploit_margin_scale: f32,
    /// Dampening factor for Reset (buffer *= factor).
    pub mode_reset_dampen: f32,
    /// Number of top-K nodes to apply dampening to.
    pub mode_reset_dampen_top_k: usize,
    /// Window size for tracking recent values/TD.
    pub mode_window_size: usize,

    // =========================================================================
    // Phase 2.0c: Action Policy Configuration
    // =========================================================================
    /// Enable action policy (Scan/Focus/Perturb) in Demo 9.
    pub enable_action_policy: bool,
    /// Scale factor for Top-K during Scan action (> 1.0 = more nodes).
    pub scan_topk_scale: f32,
    /// Scale factor for Top-K during Focus action (< 1.0 = fewer nodes).
    pub focus_topk_scale: f32,
    /// Scale factor for margin during Scan action (< 1.0 = looser).
    pub scan_margin_scale: f32,
    /// Scale factor for margin during Focus action (> 1.0 = tighter).
    pub focus_margin_scale: f32,
    /// Noise amplitude for Perturb action.
    pub perturb_noise_amp: f32,

    // =========================================================================
    // Phase 2.0c-FIX: Perturb Trigger Reliability
    // =========================================================================
    /// Enable extra perturb triggers (gate fail, low margin, off-proto, value drop).
    pub perturb_extra_triggers: bool,
    /// Enable perturb floor (ensures minimum perturb rate).
    pub perturb_floor_enabled: bool,
    /// Minimum perturb rate target for floor (0.005 = 0.5%).
    pub perturb_floor_min_rate: f32,
    /// Window size for perturb floor rate calculation.
    pub perturb_floor_window: usize,
    /// Gate fail streak threshold to trigger perturb.
    pub perturb_trig_fail_streak: u32,
    /// Minimum margin threshold (below = "low margin").
    pub perturb_trig_margin_min: f32,
    /// Low margin streak threshold to trigger perturb.
    pub perturb_trig_margin_streak: u32,
    /// Minimum proto alignment threshold (below = "off-proto").
    pub perturb_trig_proto_min: f32,
    /// Off-proto streak threshold to trigger perturb.
    pub perturb_trig_offproto_streak: u32,
    /// Value drop threshold over rolling window.
    pub perturb_trig_value_drop: f32,
    /// Value drop streak threshold to trigger perturb.
    pub perturb_trig_value_streak: u32,
    /// Cooldown ticks after perturb before next trigger allowed.
    pub perturb_cooldown_ticks: u32,

    // =========================================================================
    // Phase 1.9b: CONSOLIDATION (Aggressive merge scanning + flicker elimination)
    // =========================================================================
    /// How often (in ticks) to scan for merge candidates (aggressive: 10).
    pub merge_scan_period: u32,
    /// Number of anchors to sample per anchor per scan for merge candidates.
    pub merge_scan_k: u32,
    /// Minimum proto_score(a, b) required for merge candidates.
    pub merge_proto_min_score: f32,
    /// Minimum proto_support for both anchors to be merge candidates.
    pub merge_min_support: u32,
    /// Maximum merges to perform per scan.
    pub merge_max_per_scan: u32,
    /// If true, only merge anchors with the same key.
    pub merge_same_key_only: bool,
    /// Proto score threshold for cross-key merges (higher = stricter).
    pub merge_proto_min_score_cross_key: f32,
    /// Learning rate for prototype merge (weighted average).
    pub merge_proto_eta: f32,
    /// Value epsilon for merging non-stable anchors (looser).
    pub merge_v_eps: f32,
    /// Value epsilon for merging stable anchors (stricter).
    pub merge_v_eps_stable: f32,
    /// Maximum usage_count for an anchor to be eligible for merging.
    /// Protects high-usage anchors with lots of keyed memory history.
    pub merge_max_usage: u32,
    /// Minimum proto_support for non-stable anchors to be merge candidates.
    /// Ensures we don't merge "explore garbage" - only mature anchors.
    pub merge_min_support_nonstable: u32,

    // =========================================================================
    // Phase 1.9d: Matured cross-partition merges
    // =========================================================================
    /// Value epsilon for cross-mode merges (stable anchors only).
    pub mode_merge_v_eps: f32,
    /// Minimum proto similarity for stable cross-mode merges.
    pub proto_min_stable_cross_mode: f32,
    /// Maximum Hamming distance for cross-mask merges.
    pub mask_eps: u32,
    /// Minimum proto similarity for cross-mask merges.
    pub proto_min_cross_mask: f32,
    /// Value epsilon for cross-mask merges.
    pub value_eps_cross_mask: f32,
    /// Allow Mid<->Stable cross-mode merges (both must be stable).
    pub allow_mid_stable_cross_mode: bool,
    /// Maximum cross-partition merges per scan (rate limiting).
    pub max_cross_partition_merges_per_scan: u32,

    // =========================================================================
    // Phase 1.9b: Stability Hysteresis (flicker elimination)
    // =========================================================================
    /// Minimum proto_support to ENTER stable state.
    pub stable_min_support_enter: u32,
    /// Minimum proto_support to stay in stable state (lower = stickier).
    pub stable_min_support_exit: u32,
    /// Entropy threshold to ENTER stable state (must be below).
    pub stable_enter_entropy: f32,
    /// Entropy threshold to EXIT stable state (must exceed, higher = stickier).
    pub stable_exit_entropy: f32,
    /// |TD| threshold to ENTER stable state (must be below).
    pub stable_enter_abs_td: f32,
    /// |TD| threshold to EXIT stable state (must exceed, higher = stickier).
    pub stable_exit_abs_td: f32,
    /// Minimum ticks an anchor must stay stable before it can drop.
    pub stable_min_ticks_on: u32,
    /// Catastrophic V drop threshold (V < stable_v_min - this triggers immediate exit).
    pub stable_catastrophic_v_drop: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            // Network topology
            num_nodes: 32,
            avg_degree: 3,
            num_ctx: 3,

            // Dynamics
            decay_per_tick: 0.08,
            clamp_max_amp: 0.8,
            eps: 1e-12,

            // Homeostasis
            pow_target: 8.0,
            homeostasis_beta: 0.10,

            // Plasticity
            p_min: 0.03,
            align_pos: 0.30,
            align_neg: 0.30,
            phase_learn_rate: 0.008,
            min_edge_power: 1e-6,

            // Affinity tracking
            aff_inc: 3,
            aff_dec: 2,
            aff_margin: 3,
            wrong_ctx_penalty: 1.0,
            ctx_suppress: 0.5,
            suppress_min_pow: 0.05,

            // Training/Eval
            train_ticks: 100_000,
            eval_ticks: 10_000,
            top_k: 5,

            // Causes
            num_causes: 3,
            injectors_per_cause: 6,
            inject_amp: 0.408,
            ph_noise: 0.10,
            noise_injects: 1,
            noise_amp: 0.02,
            prob_single_cause: 0.70,

            // RNG seed
            seed: 0xDEADBEEF,

            // Episodic Memory defaults (Phase 1.4a)
            memory_max_entries: 512,
            memory_max_hamming: 6,
            memory_mask_w: 8,
            memory_proto_w: 1,
            memory_age_w: 0,
            memory_min_score: -999999,
            memory_store_prob: 0.05,
            memory_min_power: 0.5,
            memory_debounce_ticks: 20,

            // One-Shot Label Binding defaults (Phase 1.4b)
            num_labels: 16,
            episode_ticks: 2000,
            bind_tick: 200,
            recall_start_tick: 400,
            recall_stride: 5,
            num_episodes: 200,
            run_capacity_sweep: false,
            label_memory_max_entries: 256,
            label_memory_max_hamming: 8,

            // Phase 1.7a: Prototype Vector defaults
            // Phase 1.9e Option A: Faster prototype sharpening (moderate tuning)
            proto_m: 12,
            proto_eta: 0.11, // Phase 1.9e: 0.10 -> 0.11 (moderate sharpening, not too aggressive)
            proto_decay: 0.009, // Phase 1.9e: 0.01 -> 0.009 (slight reduction for sharper prototypes)
            proto_beta: 0.25,
            proto_insert_margin: 0.02,

            // Phase 1.7b: Value Learning defaults
            // Phase 1.7c: alpha_v 0.05->0.03, v_clip 1.0->0.7 for better calibration
            alpha_v: 0.03,
            gamma_v: 0.95,
            v_clip: 0.7,
            v_td_ema: 0.02,
            v_beta: 0.10,
            r_w_power: 0.15,
            r_w_coh: 0.45,
            r_w_proto: 0.40,
            r_w_margin: 0.15,
            r_p_clip: 1.0,
            margin_norm: 0.10,
            // Phase 1.7c: negative bootstrap when abstaining due to margin fail
            v_abstain_margin: -0.1,

            // Phase 1.7d: Probe-set convergence metrics defaults
            probe_enabled: true,
            probe_size: 256,
            probe_eval_stride: 200,
            probe_min_fill: 64,
            use_advantage_reward: false,
            reward_ema_beta: 0.01,

            // Competitive Label Binding defaults (Phase 1.4c)
            competitive_num_labels: 8,
            competitive_max_entries: 200,
            competitive_max_hamming: 3, // TUNING: stricter (was 6)
            competitive_margin_min: 0,  // TUNING: higher confidence (was 2)
            competitive_binds_per_episode: 1,
            competitive_p_neg: 0.35,
            competitive_neg_flip_bits: 10,
            competitive_episodes: 400,
            competitive_episode_ticks: 500,
            competitive_recall_stride: 5,
            competitive_recall_start: 260,

            // Phase 1.8: VALUE IS CONTROL defaults
            evict_v_weight: 0.5,
            evict_use_weight: 0.3,
            evict_age_weight: 0.2,
            evict_age_tau: 10000.0,     // Age decay over ~10k ticks
            merge_v_delta_max: 0.3,     // Allow merge if |v1-v2| < 0.3
            gate_explore_mult: 0.5,     // Permissive: halve margin threshold
            gate_stable_mult: 1.5,      // Strict: 50% higher margin threshold
            stable_v_min: 0.10, // Need V >= 0.10 to become stable (lowered for aggressive merging)
            stable_wins_min: 2, // Need >= 2 wins to become stable (easier)
            stable_mode_threshold: 0.3, // 30% stable anchors → switch to stable mode

            // Demo Control Flags defaults
            run_baseline_5a: false, // Skip baseline for faster iteration
            run_keyed_5b: true,
            run_demo_7: true,  // Phase 2.0a: Mode Policy demo
            run_demo_8: true,  // Phase 2.0b: Ablations + Mode Metrics demo
            run_demo_9: true,  // Phase 2.0c: Mode → Action Loop demo
            run_demo_10: true, // Phase 2.0d: Action Ablations + Sweep demo
            run_demo_11: true, // Phase 2.0e: Trigger-Matched Random + Regret Metrics demo
            run_demo_12: true, // Phase 2.0f-D: Action Distillation demo
            run_demo_13: true, // Phase 2.1: Multi-seed + Lift Metrics demo

            // Phase 2.1: Multi-seed + Lift Metrics defaults
            demo13_num_seeds: 5,
            lift_bad_margin: 0.02,
            lift_bad_proto: 0.10,
            lift_bad_value: 0.15,

            // Phase 2.1b: Seed-Robust Policy Stabilization defaults
            min_exploit_ticks_on: 100, // Phase 2.1k: stickier exploit
            explore_streak_rescue: 80, // Phase 2.1j: stop rescue spam
            fail_streak_rescue: 4,     // Rescue earlier on gate-fail cascades
            rescue_cooldown: 320,      // Phase 2.1j: stop rescue spam
            post_reset_exploit_boost_ticks: 80, // Phase 2.1j: reduced stickiness
            post_reset_exploit_margin_scale: 1.35, // Strong margin boost post-reset
            catastrophic_abs_td: 0.55, // Moderate threshold
            catastrophic_value_drop: 0.10,
            exploit_proto_min_floor: 0.20, // Much higher floor - require good proto alignment
            exploit_margin_min_floor: 0.045, // Higher margin floor
            exploit_proto_p50_scale: 0.65, // Very conservative - require strong proto
            exploit_margin_p50_scale: 0.60, // Very conservative - require strong margin
            demo13_enable_adaptive_thresholds: true,
            demo13_enable_rescue: true,
            demo13_enable_min_perturb_guard: true,
            demo13_min_perturb_rate: 0.010, // Higher perturb floor
            demo13_perturb_window: 1500,

            // Phase 2.1c: Anti-Thrash Post-Rescue Lock defaults
            post_rescue_lock_ticks: 250, // Lock in Exploit for 250 ticks after rescue (very long stabilization)
            lock_margin_min_scale: 1.40, // Stricter margin during lock
            lock_focus_bias: 4.0,        // Very strong Focus bias during lock
            rescue_requires_bad_state: true, // Require bad state for rescue
            rescue_bad_proto: 0.12,      // Below this = bad proto (stricter)
            rescue_bad_margin: 0.03,     // Below this = bad margin (stricter)
            rescue_bad_value: -0.20,     // Phase 2.1j: correct scale (V often negative)

            // Phase 2.1m: Lock hysteresis + soft exploit relaxation defaults
            lock_fail_drop_streak: 12, // Grace period before dropping lock
            proto_soft_min: 0.08,      // Relaxed proto threshold for soft exploit

            // Phase 2.1n: Quality-aware action defaults
            soft_exploit_scan_prob: 0.0, // Disabled: soft exploit uses Focus (same as hard)
            rescue_max_per_10k: 15,      // Max rescues per 10k ticks before throttle
            rescue_throttle_cooldown: 500, // Extended cooldown when throttle active

            // Phase 2.1o: Soft-exploit quarantine defaults (DISABLED - regresses badly)
            soft_exploit_block_store: false, // Block memory store during soft exploit
            soft_exploit_block_proto_update: false, // Block prototype updates during soft exploit
            soft_exploit_block_merge: false, // Block merge candidates during soft exploit

            // Phase 2.1p: Rate-limited proto updates in soft exploit
            soft_proto_update_period: 0, // Cooldown ticks between soft-exploit proto updates (0 = disabled)
            soft_proto_update_min_margin: 0.03, // Minimum margin to allow proto update
            soft_proto_update_require_gate: true, // Require gate pass for proto update
            soft_proto_update_block_when_bad: true, // Block proto update when in bad state
            soft_proto_update_td_max: 0.27, // Max recent TD to allow proto update (instability gate)

            // Phase 2.1q: Adaptive soft proto update period defaults
            soft_proto_adaptive_enabled: true,
            soft_proto_period_good: 0,         // P0 in good regime
            soft_proto_period_bad: 8,          // P8 in bad regime (Phase 2.1v: reduced from 12)
            soft_proto_bad_stable_lo: 0.55,    // Enter bad if stable < 55%
            soft_proto_bad_bad_hi: 0.28,       // Enter bad if bad > 28%
            soft_proto_bad_explore_hi: 0.25,   // Enter bad if explore > 25%
            soft_proto_bad_rescue_hi: 0.0009,  // Enter bad if rescues/tick > 0.09%
            soft_proto_bad_hold_ticks: 1200,   // Hold bad mode for 1200 ticks
            soft_proto_bad_clear_stable: 0.62, // Early clear if stable > 62%
            soft_proto_bad_clear_bad: 0.24,    // Early clear if bad < 24%

            // Phase 2.1v: Bad-Regime Proto Repair (relaxed gating)
            soft_proto_bad_td_max: 0.30, // Max TD in bad-regime (more relaxed than normal 0.27)
            soft_proto_bad_margin_min: 0.02, // Min margin in bad-regime (relaxed from 0.03)
            soft_proto_bad_proto_min: 0.08, // Min proto_align in bad-regime
            soft_proto_bad_require_gate: true, // Still require gate in bad-regime
            soft_proto_repair_after_burst_ticks: 300, // P0 for 300 ticks after burst

            // Phase 2.1r/2.1s: Bad-Regime Quality Repair defaults (episodic bursts)
            repair_enabled: true,
            repair_bad_stable_lo: 0.52, // Enter if stable < 52% (tighter threshold)
            repair_bad_share_hi: 0.27,  // Enter if bad > 27%
            repair_clear_stable_hi: 0.58, // Clear if stable > 58%
            repair_clear_bad_lo: 0.22,  // Clear if bad < 22%
            repair_bad_hold_ticks: 1500, // Phase 2.1s: Increased from 500 to 1500 (more sustained bad state)
            repair_clear_hold_ticks: 200, // Hold clear condition for 200 ticks
            repair_rescue_rate_hi: 0.025, // Enter if >25 rescues per 1000 ticks (higher threshold)
            repair_burst_ticks: 30,      // Base burst duration (Phase 2.1s: can escalate)
            repair_burst_prob: 0.40,     // Base 40% chance (Phase 2.1s: can escalate)
            repair_burst_cooldown: 1000, // Cooldown between bursts
            repair_perturb_cap_mean: 0.03, // Max 3% perturb rate (stricter cap)

            // Phase 2.1s: Episodic burst parameters
            burst_min_gap_ticks: 4000, // Minimum gap between bursts (prevents rapid re-trigger)
            burst_max_per_run: 30,     // Maximum bursts per run
            burst_base_prob: 0.40,     // Base burst probability
            burst_max_prob: 0.55,      // Maximum burst probability after escalation
            burst_base_ticks: 30,      // Base burst duration
            burst_max_ticks: 60,       // Maximum burst duration after escalation
            burst_pre_window: 200,     // Pre-burst measurement window (ticks)
            burst_post_window: 400,    // Post-burst measurement window (ticks)
            burst_td_success_ratio: 0.93, // Success if post_td <= pre_td * 0.93 (7% reduction)
            burst_bad_improve_abs: 0.03, // Success if bad_share improved by >= 3%
            burst_stable_gain_abs: 0.03, // Success if stable_share gained >= 3%
            burst_prob_escalation: 0.05, // Probability increment per failed burst
            burst_ticks_escalation: 10, // Duration increment per failed burst

            // Phase 2.1h: Chronic Instability Clamp v5 defaults (EMA smoothing + hysteresis)
            chronic_window_ticks: 500,     // Sliding window for raw stats
            chronic_share_ema_alpha: 0.05, // EMA smoothing (slower response)
            chronic_bad_share_hi: 0.35,    // ENTER if bad_ema > 35%
            chronic_stable_share_lo: 0.50, // ENTER if stable_ema < 50%
            chronic_enter_hold_ticks: 400, // Enter-hold window (longer)
            chronic_enter_hold_tolerance: 0.02, // Require 98% pass-rate
            chronic_exit_bad_max: 0.25,    // EXIT if bad_ema < 25% AND stable_ema > 55%
            chronic_exit_stable_min: 0.55, // EXIT requires BOTH
            chronic_exit_hold_ticks: 150,  // Hold exit conditions
            chronic_exit_hold_tolerance: 0.10, // Allow 10% failures
            chronic_explore_cap: 0.10,     // Max 10% Explore while clamped
            chronic_lock_ticks: 190,       // Minimum lock duration
            chronic_exploit_margin_scale: 1.20, // Slightly stricter margin during clamp
            chronic_focus_bias: 2.5,       // Focus bias during clamp
            chronic_min_ticks_before_enable: 20000, // Wait before enabling (reduce early flapping)
            chronic_rearm_cooldown: 6000,  // Cooldown after expiry (further reduce re-entry)
            chronic_max_share: 0.50,       // Watchdog: max 50% chronic time
            chronic_release_cooldown: 200, // Cooldown after watchdog release
            chronic_escape_after: 1000,    // Escape pulse after 1000 continuous ticks
            chronic_escape_ticks: 50,      // Escape pulse duration
            chronic_disallow_perturb: true, // No Perturb during chronic (except Reset)

            // Phase 2.1e: Perturb Budget Cap defaults
            perturb_cap: 0.03,           // Max 3% perturb rate
            perturb_budget_window: 2000, // Budget window size

            // Phase 2.0e: Regret/Recovery Metrics defaults
            regret_margin_bad: 0.02,
            regret_proto_bad: 0.20,
            regret_v_bad: 0.15,
            regret_td_spike: 0.55,
            regret_pre_window: 10,
            regret_post_window: 10,
            regret_post_gate_window: 50,
            regret_recovery_good_threshold: 0.10,

            // Phase 2.0f-A: Action Distillation defaults
            distill_lr: 0.03,
            distill_l2: 1e-4,
            distill_temperature: 1.0,
            distill_warmup_ticks: 5_000,
            distill_replay_capacity: 50_000,
            distill_batch_size: 128,
            distill_train_every: 5,

            // Phase 2.0f-C: Mode-Conditioned Distillation defaults
            distill_mode_cond: true,
            distill_min_mode_samples: 20_000,
            budget_window: 2000,
            budget_tol: 0.01,
            budget_lambda_pref: 1.0,
            budget_lambda_def: 8.0,

            // Phase 2.0f-E: Natural Exploit Emergence defaults
            // Tuned based on observed signal distributions:
            // Phase 2.1b: Stricter exploit requirements for seed robustness
            // - proto_align mean ~0.126, threshold 0.18 requires above-average quality
            // - margin mean ~0.11, threshold 0.06 requires good separation
            // - require stable anchors for reliable exploit
            exploit_proto_min: 0.11, // Phase 2.1l: reverted for more hard exploit
            exploit_margin_min: 0.03, // Phase 2.1l: reverted for more hard exploit
            exploit_requires_stable: true, // Re-enabled: require anchor stability
            focus_bias_exploit: 0.0, // No artificial bias; rely on mode->action mapping

            // Phase 2.0a: Mode Policy defaults
            enable_mode_policy: true,
            mode_explore_v_max: 0.25, // Phase 2.2a: reduce Explore rate for worst-seed improvement
            mode_exploit_v_min: 0.60, // Phase 2.1k: reduce fallback exploit
            mode_reset_td_min: 0.35,  // TD threshold for reset (stable_avg ~0.27)
            mode_reset_value_drop: 1.0, // Disabled
            mode_reset_fail_streak: 2, // Reset after 2 consecutive gate failures
            mode_post_reset_cooldown: 12,
            mode_explore_margin_scale: 0.6,
            mode_exploit_margin_scale: 1.15,
            mode_reset_dampen: 0.52, // Phase 2.1j: less destructive reset
            mode_reset_dampen_top_k: 8,
            mode_window_size: 32,

            // Phase 2.0c: Action Policy defaults
            enable_action_policy: true,
            scan_topk_scale: 1.25,
            focus_topk_scale: 0.80,
            scan_margin_scale: 0.80,
            focus_margin_scale: 1.30,
            perturb_noise_amp: 0.02,

            // Phase 2.0c-FIX: Perturb Trigger Reliability defaults
            perturb_extra_triggers: true, // Enable extra triggers by default
            perturb_floor_enabled: true,  // Enable perturb floor
            perturb_floor_min_rate: 0.005, // Target minimum 0.5%
            perturb_floor_window: 2000,   // Rolling window for floor calculation
            perturb_trig_fail_streak: 5,  // Gate fail streak threshold
            perturb_trig_margin_min: 0.02, // Below this = "low margin"
            perturb_trig_margin_streak: 20, // Low margin streak threshold
            perturb_trig_proto_min: 0.10, // Below this = "off-proto"
            perturb_trig_offproto_streak: 30, // Off-proto streak threshold
            perturb_trig_value_drop: 0.08, // Value drop threshold
            perturb_trig_value_streak: 8, // Value drop streak threshold
            perturb_cooldown_ticks: 100,  // Cooldown after perturb

            // Phase 1.9e: CONSOLIDATION defaults (balanced merge + stable accumulation)
            merge_scan_period: 50, // Phase 1.9e: 40 -> 50 (less frequent for stable accumulation)
            merge_scan_k: 80,      // Sample 80 anchors per anchor per scan (was 64)
            merge_proto_min_score: 0.78, // Require 78% proto similarity (lowered from 0.80)
            merge_min_support: 30, // Anchors need some maturity to merge
            merge_max_per_scan: 20, // Up to 20 merges per scan (was 16)
            merge_same_key_only: false, // Allow cross-key merges if proto similar enough
            merge_proto_min_score_cross_key: 0.83, // Cross-key threshold (lowered from 0.855)
            merge_proto_eta: 0.10, // Proto merge learning rate
            merge_v_eps: 0.38,     // Value epsilon for non-stable anchors
            merge_v_eps_stable: 0.22, // Stricter value epsilon for stable anchors
            merge_max_usage: 2500, // Allow most anchors to merge
            merge_min_support_nonstable: 28, // Phase 1.9e: 40 -> 28 (align with easier stable entry)

            // Phase 1.9d: Matured cross-partition merges (relaxed for stable anchors)
            mode_merge_v_eps: 0.18, // Max |dv| for cross-mode merges (was 0.15)
            proto_min_stable_cross_mode: 0.88, // High proto similarity for cross-mode (was 0.90)
            mask_eps: 2,            // Max Hamming distance for cross-mask (was 1)
            proto_min_cross_mask: 0.88, // High proto similarity for cross-mask (was 0.90)
            value_eps_cross_mask: 0.18, // Max |dv| for cross-mask merges (was 0.15)
            allow_mid_stable_cross_mode: true, // Allow Mid<->Stable (both must be stable)
            max_cross_partition_merges_per_scan: 24, // Phase 1.9e: 48 -> 24 (more conservative to allow stable accumulation)

            // Phase 1.9e Option A: Stability hysteresis (very easy entry, stickier state)
            stable_min_support_enter: 18, // Phase 1.9e: 30 -> 18 (very easy entry)
            stable_min_support_exit: 6, // Phase 1.9e: 8 -> 6 (harder to exit = more stable anchors)
            stable_enter_entropy: 2.75, // Phase 1.9e: 2.65 -> 2.75 (+0.10 relaxation)
            stable_exit_entropy: 2.80,  // Phase 1.9e: 2.75 -> 2.80 (stickier)
            stable_enter_abs_td: 0.55,  // Phase 1.9e: 0.45 -> 0.55 (+0.10 relaxation)
            stable_exit_abs_td: 0.70,   // Phase 1.9e: 0.65 -> 0.70 (stickier)
            stable_min_ticks_on: 120,   // Phase 1.9e: 200 -> 120 (faster cycling)
            stable_catastrophic_v_drop: 0.15, // Catastrophic if V drops 0.15 below stable_v_min
        }
    }
}

impl Config {
    /// Get cause phases: evenly distributed around the circle.
    pub fn cause_phases(&self) -> Vec<f64> {
        (0..self.num_causes)
            .map(|c| 2.0 * PI * c as f64 / self.num_causes as f64)
            .collect()
    }

    /// Get bind ticks for competitive experiment (evenly spaced).
    pub fn competitive_bind_ticks(&self) -> Vec<usize> {
        let n = self.competitive_binds_per_episode;
        let spacing = self.competitive_episode_ticks / (n + 1);
        (1..=n).map(|i| i * spacing).collect()
    }

    /// Convert config to canonical string for hashing.
    /// Includes key parameters in stable order.
    pub fn to_canonical_string(&self) -> String {
        // Include key parameters that affect behavior
        format!(
            "config_v1:\
            num_nodes={},num_ctx={},decay={:.4},clamp={:.2},pow_target={:.1},\
            top_k={},train_ticks={},eval_ticks={},seed={},\
            proto_m={},proto_eta={:.3},alpha_v={:.3},gamma_v={:.2},\
            mode_explore_v_max={:.2},mode_exploit_v_min={:.2},mode_reset_td_min={:.2},\
            scan_topk_scale={:.2},focus_topk_scale={:.2},perturb_noise_amp={:.3},\
            competitive_episodes={},competitive_episode_ticks={},\
            demo13_num_seeds={}",
            self.num_nodes,
            self.num_ctx,
            self.decay_per_tick,
            self.clamp_max_amp,
            self.pow_target,
            self.top_k,
            self.train_ticks,
            self.eval_ticks,
            self.seed,
            self.proto_m,
            self.proto_eta,
            self.alpha_v,
            self.gamma_v,
            self.mode_explore_v_max,
            self.mode_exploit_v_min,
            self.mode_reset_td_min,
            self.scan_topk_scale,
            self.focus_topk_scale,
            self.perturb_noise_amp,
            self.competitive_episodes,
            self.competitive_episode_ticks,
            self.demo13_num_seeds,
        )
    }

    /// Compute SHA256 hash of canonical config string.
    pub fn config_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let canonical = self.to_canonical_string();
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let result = hasher.finalize();
        // Return first 16 hex chars for brevity
        format!("{:x}", result)[..16].to_string()
    }
}
