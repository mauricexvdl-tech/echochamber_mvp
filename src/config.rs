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
            run_demo_7: true, // Phase 2.0a: Mode Policy demo

            // Phase 2.0a: Mode Policy defaults
            enable_mode_policy: true,
            mode_explore_v_max: 0.42,
            mode_exploit_v_min: 0.55,
            mode_reset_td_min: 0.35, // TD threshold for reset (stable_avg ~0.27)
            mode_reset_value_drop: 1.0, // Disabled
            mode_reset_fail_streak: 2, // Reset after 2 consecutive gate failures
            mode_post_reset_cooldown: 12,
            mode_explore_margin_scale: 0.6,
            mode_exploit_margin_scale: 1.15,
            mode_reset_dampen: 0.50, // 50% amplitude reduction on top-K nodes
            mode_reset_dampen_top_k: 8,
            mode_window_size: 32,

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
}
