//! Configuration struct for Echo Chamber tunables.
//! Phase 1.4c: Added competitive label binding with abstain.
//! Phase 1.8: VALUE IS CONTROL - Memory lifecycle + self-regulation.

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
            proto_m: 12,
            proto_eta: 0.10,
            proto_decay: 0.01,
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
            competitive_max_hamming: 3,    // TUNING: stricter (was 6)
            competitive_margin_min: 0,     // TUNING: higher confidence (was 2)
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
            evict_age_tau: 10000.0,    // Age decay over ~10k ticks
            merge_v_delta_max: 0.3,    // Allow merge if |v1-v2| < 0.3
            gate_explore_mult: 0.5,    // Permissive: halve margin threshold
            gate_stable_mult: 1.5,     // Strict: 50% higher margin threshold
            stable_v_min: 0.4,         // Need V >= 0.4 to become stable
            stable_wins_min: 5,        // Need >= 5 wins to become stable
            stable_mode_threshold: 0.3, // 30% stable anchors → switch to stable mode
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
