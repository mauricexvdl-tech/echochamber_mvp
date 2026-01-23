//! AnchorBank: Scalable memory addressing via stable anchor prototypes.
//! Phase 1.5b: Replace expensive similarity search with O(1) key lookup.
//! Phase 1.6: Anchor codebook stabilization with merge, gating, and instrumentation.
//! Phase 1.7a: Prototype vectors - turn anchors into "concept tokens" with online learning.
//! Phase 1.7b: Value learning - self-supervised credit assignment via TD(0).
//! Phase 1.8: VALUE IS CONTROL - Memory lifecycle + self-regulation.
//! Phase 1.9: CONSOLIDATION - Proto-based merging + stability hysteresis.

use crate::config::Config;
use std::collections::HashMap;

// =============================================================================
// Configuration Constants
// =============================================================================

/// Maximum number of anchors to maintain.
pub const MAX_ANCHORS: usize = 256;

/// Maximum Hamming distance to match an existing anchor.
pub const ANCHOR_MAX_HAMMING: u32 = 4;

/// Hamming distance threshold for merging similar anchors.
pub const MERGE_HAMMING: u32 = 2;

/// Minimum usage count before an anchor can be pruned.
pub const ANCHOR_MIN_USE: u32 = 3;

/// How often to attempt anchor merging (in ticks).
pub const MERGE_EVERY_TICKS: u64 = 2000;

/// Minimum margin (top1 - top2 amplitude) to allow anchor creation/refresh.
/// Phase 1.6c: Confidence gating.
pub const ANCHOR_MARGIN_MIN: f64 = 0.02;

/// Minimum total power to allow anchor creation (avoid noisy states).
pub const ANCHOR_MIN_POWER: f64 = 0.1;

/// Maximum total power to allow anchor creation (avoid power explosions).
pub const ANCHOR_MAX_POWER: f64 = 50.0;

/// Default prototype size (number of nodes in sparse prototype).
pub const DEFAULT_PROTO_M: usize = 12;

// =============================================================================
// Anchor Prototype
// =============================================================================

/// A single anchor prototype representing a stable memory address.
/// Phase 1.7a: Extended with prototype vector for "concept token" representation.
/// Phase 1.9b/d: Reasons for blocking a merge candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MergeBlockReason {
    None,
    KeyMismatch,
    CtxMismatch,
    ModeMismatch,
    ModeExplore,
    ProtoScore,
    ValueDelta,
    StabilityMixed,
    UsageTooHigh,
    // Phase 1.9d: Cross-partition specific blocks
    ExploreIsolated,    // Explore bucket must stay isolated
    MaskHammingTooHigh, // Cross-mask Hamming > mask_eps
    CrossModeVDelta,    // Cross-mode |dv| > mode_merge_v_eps
    CrossModeProtoLow,  // Cross-mode proto < proto_min_stable_cross_mode
    CrossMaskProtoLow,  // Cross-mask proto < proto_min_cross_mask
    CrossMaskVDelta,    // Cross-mask |dv| > value_eps_cross_mask
    CrossModeNotStable, // Cross-mode requires both stable
    CrossMaskNotStable, // Cross-mask requires both stable
}

/// Phase 1.9b/d: Merge statistics for reporting.
#[derive(Clone, Debug, Default)]
pub struct MergeStats {
    pub candidates_found: usize,
    pub blocked_proto: usize,
    pub blocked_value: usize,
    pub blocked_stability: usize,
    pub blocked_support: usize,
    pub blocked_key_mismatch: usize,
    pub blocked_ctx_mismatch: usize,
    pub blocked_mode_mismatch: usize,
    pub stable_new: usize,
    pub stable_dropped: usize,
    /// Total anchor pairs checked during all scans.
    pub pairs_checked: usize,
    /// merge_opportunity_rate = candidates_found / pairs_checked
    pub opportunity_rate: f64,
    // Phase 1.9d: Cross-partition merge tracking
    pub cross_mode_merges_done: usize,
    pub cross_mask_merges_done: usize,
    pub cross_partition_merges_done: usize,
    pub blocked_explore_isolated: usize,
    pub blocked_mask_hamming: usize,
    pub blocked_cross_mode_v: usize,
    pub blocked_cross_mode_proto: usize,
    pub blocked_cross_mask_proto: usize,
    pub blocked_cross_mask_v: usize,
    pub blocked_cross_mode_not_stable: usize,
    pub blocked_cross_mask_not_stable: usize,
    /// Sum of |dv| for accepted cross merges (for computing mean).
    pub cross_dv_sum: f64,
    pub cross_dv_count: usize,
    /// Sum of mask Hamming for accepted cross-mask merges.
    pub cross_mask_hamming_sum: u32,
    pub cross_mask_hamming_count: usize,
    /// Cross-partition merges blocked due to rate limiting.
    pub blocked_cross_rate_limited: usize,
}

/// Phase 1.9f: Comprehensive stable anchor statistics.
#[derive(Clone, Debug, Default)]
pub struct StableAnchorStats {
    /// Number of stable anchors at end of run.
    pub count: usize,
    /// Sum of proto_support over stable anchors (stable mass).
    pub mass_sum: u64,
    /// Mean proto_support among stable anchors.
    pub mass_mean: f64,
    /// p50 (median) of proto_support among stable anchors.
    pub support_p50: u32,
    /// p90 of proto_support among stable anchors.
    pub support_p90: u32,
    /// Mean value (v) among stable anchors.
    pub value_mean: f64,
    /// Mean entropy_ema among stable anchors.
    pub entropy_mean: f64,
    /// Mean v_ema_abs_td among stable anchors.
    pub td_mean: f64,
}

/// Phase 1.9c: Top blocked merge candidate for diagnostics.
#[derive(Clone, Debug)]
pub struct BlockedMergeInfo {
    pub id_a: u16,
    pub id_b: u16,
    pub proto_score: f32,
    pub reason: MergeBlockReason,
}

#[derive(Clone, Debug)]
pub struct Anchor {
    /// The prototype windowed signature for this anchor.
    pub proto_signature: u64,
    /// Tick when this anchor was created.
    pub created_at_tick: u64,
    /// Tick when this anchor was last matched.
    pub last_used_tick: u64,
    /// Number of times this anchor has been matched.
    pub usage_count: u32,
    /// Whether this anchor slot is alive (for stable ID management).
    pub alive: bool,

    // Phase 1.7a: Prototype vector fields
    /// Node IDs of the prototype (sparse top nodes).
    pub proto_nodes: [u8; DEFAULT_PROTO_M],
    /// Weights for each node in the prototype (non-negative).
    pub proto_w: [f32; DEFAULT_PROTO_M],
    /// Number of updates to this prototype.
    pub proto_support: u32,
    /// Exponential moving average of prototype entropy (diagnostics).
    pub proto_entropy_ema: f32,

    // Phase 1.7b: Value learning fields
    /// Learned value estimate, clamped to [-1, +1].
    pub v: f32,
    /// Number of TD updates performed.
    pub v_updates: u32,
    /// EMA of |TD| for diagnostics.
    pub v_ema_abs_td: f32,

    // Phase 1.8: VALUE IS CONTROL - Lifecycle fields
    /// Number of times this anchor "won" in recall competition.
    pub wins: u32,
    /// Whether this anchor is considered stable (converged concept).
    pub stable: bool,
    /// Tick when this anchor became stable.
    pub stable_since_tick: u64,

    // Phase 1.9b: Merge compatibility fields
    /// The learned_mask associated with this anchor (for keyed memory partitioning).
    pub learned_mask: u64,
    /// The dominant context/label for this anchor.
    pub ctx_best: u8,
    /// Count of ctx observations for determining ctx_best.
    ctx_counts: [u32; 8],
}

impl Anchor {
    pub fn new(signature: u64, tick: u64) -> Self {
        Anchor {
            proto_signature: signature,
            created_at_tick: tick,
            last_used_tick: tick,
            usage_count: 1,
            alive: true,
            // Phase 1.7a: Initialize empty prototype
            proto_nodes: [0; DEFAULT_PROTO_M],
            proto_w: [0.0; DEFAULT_PROTO_M],
            proto_support: 0,
            proto_entropy_ema: 0.0,
            // Phase 1.7b: Initialize value fields
            v: 0.0,
            v_updates: 0,
            v_ema_abs_td: 0.0,
            // Phase 1.8: Initialize lifecycle fields
            wins: 0,
            stable: false,
            stable_since_tick: 0,
            // Phase 1.9b: Initialize merge compatibility fields
            learned_mask: 0,
            ctx_best: 0,
            ctx_counts: [0; 8],
        }
    }

    /// Compute Hamming distance to another signature.
    pub fn hamming_to(&self, signature: u64) -> u32 {
        (self.proto_signature ^ signature).count_ones()
    }

    // =========================================================================
    // Phase 1.9b: Merge Compatibility Methods
    // =========================================================================

    /// Set the learned_mask for this anchor.
    pub fn set_learned_mask(&mut self, mask: u64) {
        self.learned_mask = mask;
    }

    /// Record a context observation and update ctx_best.
    pub fn record_ctx(&mut self, ctx: u8) {
        let idx = (ctx as usize).min(7);
        self.ctx_counts[idx] = self.ctx_counts[idx].saturating_add(1);
        // Update ctx_best to the most frequent context
        let mut best_idx = 0;
        let mut best_count = 0;
        for (i, &count) in self.ctx_counts.iter().enumerate() {
            if count > best_count {
                best_count = count;
                best_idx = i;
            }
        }
        self.ctx_best = best_idx as u8;
    }

    /// Get the gate mode bucket for merge compatibility.
    /// Returns: 0 = Explore, 1 = Mid, 2 = Stable
    pub fn gate_mode_bucket(&self) -> u8 {
        if self.stable {
            2 // Stable
        } else if self.proto_support >= 20 {
            1 // Mid (has some proto support - lowered threshold)
        } else {
            0 // Explore (very new anchor)
        }
    }

    /// Compute cosine similarity between ctx histograms.
    fn ctx_cosine_similarity(&self, other: &Anchor) -> f32 {
        let mut dot: f32 = 0.0;
        let mut norm_a: f32 = 0.0;
        let mut norm_b: f32 = 0.0;
        for i in 0..8 {
            let a = self.ctx_counts[i] as f32;
            let b = other.ctx_counts[i] as f32;
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        if norm_a < 1.0 || norm_b < 1.0 {
            return 0.0; // Not enough data
        }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Check if this anchor is compatible for merging with another.
    /// Rules:
    /// - Same learned_mask (strict equality)
    /// - Same ctx_best OR high ctx histogram similarity (soft ctx)
    /// - Same gate mode bucket
    /// Returns: (compatible, reason) where reason is for instrumentation
    pub fn merge_compatible_with(&self, other: &Anchor) -> (bool, MergeBlockReason) {
        let self_mode = self.gate_mode_bucket();
        let other_mode = other.gate_mode_bucket();

        // Rule 1: Same gate mode bucket required
        if self_mode != other_mode {
            return (false, MergeBlockReason::ModeMismatch);
        }

        // Rule 2: Same learned_mask
        if self.learned_mask != other.learned_mask {
            return (false, MergeBlockReason::KeyMismatch);
        }

        // Rule 3: Soft ctx compatibility
        // Same ctx_best OR high cosine similarity (>= 0.90)
        let ctx_ok = self.ctx_best == other.ctx_best || self.ctx_cosine_similarity(other) >= 0.90;
        if !ctx_ok {
            return (false, MergeBlockReason::CtxMismatch);
        }

        (true, MergeBlockReason::None)
    }

    /// Phase 1.9d: Check if cross-partition merge is allowed.
    /// Returns: (allowed, reason, is_cross_mode, is_cross_mask, mask_hamming)
    /// Called when standard merge_compatible_with fails.
    pub fn check_cross_partition_merge(
        &self,
        other: &Anchor,
        proto_score: f32,
        config: &Config,
    ) -> (bool, MergeBlockReason, bool, bool, u32) {
        let self_mode = self.gate_mode_bucket();
        let other_mode = other.gate_mode_bucket();
        let both_stable = self.stable && other.stable;
        let v_delta = (self.v - other.v).abs();

        // Check mode differences
        let modes_differ = self_mode != other_mode;
        // Check mask differences
        let masks_differ = self.learned_mask != other.learned_mask;
        let mask_hamming = (self.learned_mask ^ other.learned_mask).count_ones();

        // Check ctx compatibility (same as before)
        let ctx_ok = self.ctx_best == other.ctx_best || self.ctx_cosine_similarity(other) >= 0.90;
        if !ctx_ok {
            return (
                false,
                MergeBlockReason::CtxMismatch,
                modes_differ,
                masks_differ,
                mask_hamming,
            );
        }

        // Phase 1.9d Rule 1: Explore bucket must stay isolated
        // Explore is bucket 0, never allow cross-mode merges involving Explore
        let self_is_explore = self_mode == 0;
        let other_is_explore = other_mode == 0;
        if modes_differ && (self_is_explore || other_is_explore) {
            return (
                false,
                MergeBlockReason::ExploreIsolated,
                true,
                masks_differ,
                mask_hamming,
            );
        }

        // Phase 1.9d Rule 2: Cross-mode merges (Mid<->Stable)
        if modes_differ {
            // Must both be stable for cross-mode merge
            if !both_stable {
                return (
                    false,
                    MergeBlockReason::CrossModeNotStable,
                    true,
                    masks_differ,
                    mask_hamming,
                );
            }
            // Check if Mid<->Stable is allowed
            let is_mid_stable =
                (self_mode == 1 && other_mode == 2) || (self_mode == 2 && other_mode == 1);
            if is_mid_stable && !config.allow_mid_stable_cross_mode {
                return (
                    false,
                    MergeBlockReason::ModeMismatch,
                    true,
                    masks_differ,
                    mask_hamming,
                );
            }
            // Value delta check for cross-mode
            if v_delta > config.mode_merge_v_eps {
                return (
                    false,
                    MergeBlockReason::CrossModeVDelta,
                    true,
                    masks_differ,
                    mask_hamming,
                );
            }
            // Proto score check for cross-mode
            if proto_score < config.proto_min_stable_cross_mode {
                return (
                    false,
                    MergeBlockReason::CrossModeProtoLow,
                    true,
                    masks_differ,
                    mask_hamming,
                );
            }
        }

        // Phase 1.9d Rule 3: Cross-mask merges
        if masks_differ {
            // Must both be stable for cross-mask merge
            if !both_stable {
                return (
                    false,
                    MergeBlockReason::CrossMaskNotStable,
                    modes_differ,
                    true,
                    mask_hamming,
                );
            }
            // Hamming distance check
            if mask_hamming > config.mask_eps {
                return (
                    false,
                    MergeBlockReason::MaskHammingTooHigh,
                    modes_differ,
                    true,
                    mask_hamming,
                );
            }
            // Proto score check for cross-mask
            if proto_score < config.proto_min_cross_mask {
                return (
                    false,
                    MergeBlockReason::CrossMaskProtoLow,
                    modes_differ,
                    true,
                    mask_hamming,
                );
            }
            // Value delta check for cross-mask
            if v_delta > config.value_eps_cross_mask {
                return (
                    false,
                    MergeBlockReason::CrossMaskVDelta,
                    modes_differ,
                    true,
                    mask_hamming,
                );
            }
        }

        // All checks passed - cross-partition merge is allowed
        (
            true,
            MergeBlockReason::None,
            modes_differ,
            masks_differ,
            mask_hamming,
        )
    }

    /// Helper: Check if this anchor is in Explore bucket.
    pub fn is_explore(&self) -> bool {
        self.gate_mode_bucket() == 0
    }

    // =========================================================================
    // Phase 1.7a: Prototype Vector Methods
    // =========================================================================

    /// Update the prototype vector with current TopK observation.
    /// Algorithm:
    /// 1. Apply decay to existing weights
    /// 2. For each TopK entry:
    ///    - If node_id already in proto_nodes: blend weight
    ///    - Else: insert into smallest-weight slot if amp is bigger by margin
    /// 3. Clamp very small weights to 0
    /// 4. Update entropy EMA
    pub fn update_proto(&mut self, topk: &[(usize, f64)], config: &Config) {
        let eta = config.proto_eta;
        let decay = config.proto_decay;
        let insert_margin = config.proto_insert_margin;
        let proto_m = config.proto_m.min(DEFAULT_PROTO_M);

        // Step 1: Apply decay to all weights
        for i in 0..proto_m {
            self.proto_w[i] *= 1.0 - decay;
        }

        // Step 2: Process each TopK entry
        for (idx, &(node_id, amp)) in topk.iter().take(proto_m).enumerate() {
            let node_u8 = node_id as u8;
            let amp_f32 = amp as f32;

            // Check if node already exists in prototype
            let mut found_idx: Option<usize> = None;
            for i in 0..proto_m {
                if self.proto_nodes[i] == node_u8 && self.proto_w[i] > 0.0 {
                    found_idx = Some(i);
                    break;
                }
            }

            if let Some(i) = found_idx {
                // Blend existing weight with new observation
                self.proto_w[i] = (1.0 - eta) * self.proto_w[i] + eta * amp_f32;
            } else {
                // Try to insert into smallest-weight slot
                let mut min_w = f32::MAX;
                let mut min_idx = 0;
                for i in 0..proto_m {
                    if self.proto_w[i] < min_w {
                        min_w = self.proto_w[i];
                        min_idx = i;
                    }
                }

                // Insert if amp is bigger than min weight by margin
                if amp_f32 > min_w + insert_margin {
                    self.proto_nodes[min_idx] = node_u8;
                    self.proto_w[min_idx] = amp_f32;
                } else if idx < proto_m && self.proto_w[idx] == 0.0 {
                    // First fill: put in order
                    self.proto_nodes[idx] = node_u8;
                    self.proto_w[idx] = amp_f32;
                }
            }
        }

        // Step 3: Clamp very small weights to 0
        for i in 0..proto_m {
            if self.proto_w[i] < 1e-6 {
                self.proto_w[i] = 0.0;
            }
        }

        // Step 4: Update entropy EMA
        let entropy = self.compute_entropy(proto_m);
        let ema_alpha = 0.1f32;
        self.proto_entropy_ema = (1.0 - ema_alpha) * self.proto_entropy_ema + ema_alpha * entropy;

        // Increment support counter
        self.proto_support += 1;
    }

    /// Compute prototype similarity score with current TopK.
    /// Returns dot product of prototype weights with current amplitudes.
    /// score = Σ (proto_w[i] * current_amp_of(proto_nodes[i]))
    pub fn proto_score(&self, topk: &[(usize, f64)], proto_m: usize) -> f32 {
        let pm = proto_m.min(DEFAULT_PROTO_M);

        // Build quick lookup for TopK amplitudes
        // (node_id -> amplitude)
        let mut topk_map: [f32; 256] = [0.0; 256]; // Assuming max 256 nodes
        for &(node_id, amp) in topk {
            if node_id < 256 {
                topk_map[node_id] = amp as f32;
            }
        }

        // Compute dot product
        let mut score = 0.0f32;
        let mut weight_sum = 0.0f32;

        for i in 0..pm {
            let w = self.proto_w[i];
            if w > 0.0 {
                let node_id = self.proto_nodes[i] as usize;
                score += w * topk_map[node_id];
                weight_sum += w;
            }
        }

        // Normalize by weight sum for scale invariance
        if weight_sum > 1e-6 {
            score / weight_sum
        } else {
            0.0
        }
    }

    /// Compute entropy of the prototype distribution.
    /// H = -Σ p_i ln p_i, where p_i = w_i / Σw
    fn compute_entropy(&self, proto_m: usize) -> f32 {
        let pm = proto_m.min(DEFAULT_PROTO_M);
        let mut weight_sum = 0.0f32;

        for i in 0..pm {
            weight_sum += self.proto_w[i];
        }

        if weight_sum < 1e-6 {
            return 0.0;
        }

        let mut entropy = 0.0f32;
        for i in 0..pm {
            let w = self.proto_w[i];
            if w > 1e-6 {
                let p = w / weight_sum;
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    /// Get current entropy of this anchor's prototype.
    pub fn entropy(&self, proto_m: usize) -> f32 {
        self.compute_entropy(proto_m)
    }

    // =========================================================================
    // Phase 1.9: Proto-based Similarity
    // =========================================================================

    /// Compute proto similarity score against another anchor.
    /// Returns dot product of normalized weight vectors in [0, 1].
    /// score = Σ (w1[node] * w2[node]) / (|w1| * |w2|)
    pub fn proto_score_against(&self, other: &Anchor, proto_m: usize) -> f32 {
        let pm = proto_m.min(DEFAULT_PROTO_M);

        // Build weight maps for both anchors (node_id -> weight)
        let mut w1_map: [f32; 256] = [0.0; 256];
        let mut w2_map: [f32; 256] = [0.0; 256];
        let mut w1_sum_sq = 0.0f32;
        let mut w2_sum_sq = 0.0f32;

        for i in 0..pm {
            if self.proto_w[i] > 0.0 {
                let node_id = self.proto_nodes[i] as usize;
                w1_map[node_id] = self.proto_w[i];
                w1_sum_sq += self.proto_w[i] * self.proto_w[i];
            }
            if other.proto_w[i] > 0.0 {
                let node_id = other.proto_nodes[i] as usize;
                w2_map[node_id] = other.proto_w[i];
                w2_sum_sq += other.proto_w[i] * other.proto_w[i];
            }
        }

        // Compute norms
        let norm1 = w1_sum_sq.sqrt();
        let norm2 = w2_sum_sq.sqrt();

        if norm1 < 1e-6 || norm2 < 1e-6 {
            return 0.0;
        }

        // Compute dot product
        let mut dot = 0.0f32;
        for i in 0..pm {
            if self.proto_w[i] > 0.0 {
                let node_id = self.proto_nodes[i] as usize;
                dot += self.proto_w[i] * w2_map[node_id];
            }
        }

        // Cosine similarity
        dot / (norm1 * norm2)
    }

    // =========================================================================
    // Phase 1.7b: Value Learning Methods
    // =========================================================================

    /// Update the value estimate using TD error.
    /// v += alpha * td, then clamp to [-v_clip, +v_clip].
    /// Also updates EMA of |TD| for diagnostics.
    pub fn v_update(&mut self, td: f32, config: &Config) {
        let alpha = config.alpha_v;
        let v_clip = config.v_clip;
        let ema_coef = config.v_td_ema;

        // Update value
        self.v += alpha * td;

        // Clamp to allowed range
        self.v = self.v.clamp(-v_clip, v_clip);

        // Update counters and EMA
        self.v_updates += 1;
        self.v_ema_abs_td = (1.0 - ema_coef) * self.v_ema_abs_td + ema_coef * td.abs();
    }

    // =========================================================================
    // Phase 1.8: VALUE IS CONTROL - Lifecycle Methods
    // =========================================================================

    /// Compute keep_score for eviction decisions.
    /// Higher score = more likely to keep (less likely to evict).
    /// score = w_v * (v + 1) / 2 + w_use * log(1 + usage) + w_age * exp(-age/tau)
    /// where v is mapped from [-1,1] to [0,1].
    pub fn keep_score(&self, current_tick: u64, config: &Config) -> f32 {
        // Value component: map v from [-1, 1] to [0, 1]
        let v_norm = (self.v + 1.0) / 2.0;

        // Usage component: log scale for diminishing returns
        let use_term = (1.0 + self.usage_count as f32).ln() / 5.0; // Normalize by ln(~150) ≈ 5

        // Age penalty: exponential decay
        let age = current_tick.saturating_sub(self.last_used_tick) as f64;
        let age_term = (-age / config.evict_age_tau).exp() as f32;

        // Combine components
        config.evict_v_weight * v_norm
            + config.evict_use_weight * use_term.min(1.0)
            + config.evict_age_weight * age_term
    }

    /// Record a "win" when this anchor is successfully recalled.
    pub fn record_win(&mut self) {
        self.wins += 1;
    }

    /// Phase 1.9b: Check stability with two-threshold hysteresis.
    /// Entry thresholds (strict): support >= min_enter, entropy < enter, |TD| < enter
    /// Exit thresholds (loose): support < min_exit, entropy > exit, |TD| > exit
    /// Also respects stable_min_ticks_on and catastrophic drop detection.
    /// Returns (new_stable_state, entered_stable, dropped_stable)
    pub fn check_stability_hysteresis(
        &mut self,
        current_tick: u64,
        config: &Config,
        proto_m: usize,
    ) -> (bool, bool, bool) {
        let entropy = self.entropy(proto_m);
        let abs_td = self.v_ema_abs_td;

        let mut entered = false;
        let mut dropped = false;

        if self.stable {
            // Check for exit conditions
            let ticks_stable = current_tick.saturating_sub(self.stable_since_tick);
            let past_min_ticks = ticks_stable >= config.stable_min_ticks_on as u64;

            // Catastrophic drop: immediate exit if V drops significantly below threshold
            let catastrophic_v_drop =
                self.v < config.stable_v_min - config.stable_catastrophic_v_drop;

            // Check EXIT thresholds (looser than entry)
            let support_exit = self.proto_support < config.stable_min_support_exit;
            let entropy_exit = entropy > config.stable_exit_entropy;
            let td_exit = abs_td > config.stable_exit_abs_td;

            // Normal exit: past min_ticks AND any exit condition met
            let normal_exit = past_min_ticks && (support_exit || entropy_exit || td_exit);

            if catastrophic_v_drop || normal_exit {
                self.stable = false;
                self.stable_since_tick = 0;
                dropped = true;
            }
        } else {
            // Check for ENTRY conditions (all must be met)
            let v_ok = self.v >= config.stable_v_min;
            let wins_ok = self.wins >= config.stable_wins_min;
            let support_ok = self.proto_support >= config.stable_min_support_enter;
            let entropy_ok = entropy < config.stable_enter_entropy;
            let td_ok = abs_td < config.stable_enter_abs_td;

            if v_ok && wins_ok && support_ok && entropy_ok && td_ok {
                self.stable = true;
                self.stable_since_tick = current_tick;
                entered = true;
            }
        }

        (self.stable, entered, dropped)
    }

    /// Legacy check_stability for backward compatibility.
    pub fn check_stability(&mut self, current_tick: u64, config: &Config) -> bool {
        if self.stable {
            return true;
        }
        if self.v >= config.stable_v_min && self.wins >= config.stable_wins_min {
            self.stable = true;
            self.stable_since_tick = current_tick;
            return true;
        }
        false
    }
}

// =============================================================================
// Confidence gate info for anchor creation
// =============================================================================

/// Phase 1.8: Dynamic gate parameters for value-controlled gating.
/// In explore mode (more permissive): multipliers < 1 reduce thresholds.
/// In stable mode (stricter): multipliers > 1 increase thresholds.
#[derive(Clone, Debug)]
pub struct GateParams {
    /// Multiplier for margin threshold.
    pub margin_mult: f64,
    /// Multiplier for min power threshold.
    pub power_min_mult: f64,
}

impl GateParams {
    pub fn new(margin_mult: f64, power_min_mult: f64) -> Self {
        GateParams {
            margin_mult,
            power_min_mult,
        }
    }

    /// Create gate params for explore mode (more permissive).
    pub fn explore(config: &Config) -> Self {
        GateParams {
            margin_mult: config.gate_explore_mult,
            power_min_mult: config.gate_explore_mult,
        }
    }

    /// Create gate params for stable mode (stricter).
    pub fn stable(config: &Config) -> Self {
        GateParams {
            margin_mult: config.gate_stable_mult,
            power_min_mult: 1.0, // Don't change power threshold in stable mode
        }
    }

    /// Default (no modification).
    pub fn default_params() -> Self {
        GateParams {
            margin_mult: 1.0,
            power_min_mult: 1.0,
        }
    }
}

impl Default for GateParams {
    fn default() -> Self {
        GateParams::default_params()
    }
}

/// Information about current signal confidence for gating anchor creation.
#[derive(Clone, Debug)]
pub struct ConfidenceInfo {
    /// Margin between top1 and top2 amplitudes.
    pub topk_margin: f64,
    /// Total power of the network.
    pub total_power: f64,
}

impl ConfidenceInfo {
    pub fn new(topk_margin: f64, total_power: f64) -> Self {
        ConfidenceInfo {
            topk_margin,
            total_power,
        }
    }

    /// Check if confidence passes the gate for anchor creation (default thresholds).
    pub fn passes_gate(&self) -> bool {
        self.passes_gate_with_params(&GateParams::default())
    }

    /// Phase 1.8: Check if confidence passes the gate with dynamic parameters.
    pub fn passes_gate_with_params(&self, params: &GateParams) -> bool {
        let margin_threshold = ANCHOR_MARGIN_MIN * params.margin_mult;
        let power_min_threshold = ANCHOR_MIN_POWER * params.power_min_mult;

        self.topk_margin >= margin_threshold
            && self.total_power >= power_min_threshold
            && self.total_power <= ANCHOR_MAX_POWER
    }
}

// =============================================================================
// AnchorBank with Phase 1.6 enhancements
// =============================================================================

/// Bank of anchor prototypes for stable memory addressing.
/// Provides O(1) lookup after anchor resolution.
/// Phase 1.6: Enhanced with merge, gating, and detailed instrumentation.
pub struct AnchorBank {
    /// Anchors indexed by their ID.
    anchors: HashMap<u16, Anchor>,
    /// ID remapping table: old_id -> current_id (for merged anchors).
    id_remap: HashMap<u16, u16>,
    /// Next anchor ID to assign.
    next_id: u16,

    // Phase 1.6a: Detailed instrumentation
    /// Total anchors created.
    pub anchor_creates: usize,
    /// Total anchors evicted (pruned to make room).
    pub anchor_evictions: usize,
    /// Total anchors merged.
    pub anchor_merges: usize,
    /// Sum of match Hamming distances (for avg calculation).
    match_hamming_sum: u64,
    /// Count of matches (for avg calculation).
    match_count: u64,
    /// Histogram of match Hamming distances for p95 calculation.
    hamming_histogram: [u64; 65], // 0-64 bits possible
    /// Total ticks processed (for thrash rate).
    total_ticks: u64,
    /// Gates passed (anchor created/refreshed).
    gates_passed: usize,
    /// Gates failed (anchor creation blocked).
    gates_failed: usize,
    /// Last tick when merge was attempted.
    last_merge_tick: u64,

    // Phase 1.7a: Prototype metrics
    /// Total prototype updates performed.
    pub proto_updates: usize,
    /// Steps where winning anchor had proto_support > 0.
    proto_active_steps: usize,
    /// Total steps for proto_active calculation.
    proto_total_steps: usize,

    // Phase 1.8: VALUE IS CONTROL - Mode switching
    /// Current operating mode: true = stable (stricter gating), false = explore (permissive).
    pub stable_mode: bool,
    /// Count of anchors marked stable.
    stable_count: usize,
    /// Total stability transitions (explore → stable or stable → explore).
    pub mode_transitions: usize,
    /// Blocked merges due to value inconsistency.
    pub merge_blocked_v: usize,

    // Phase 1.9b: CONSOLIDATION - Merge and stability tracking
    /// Total merge candidates found by proto-based scanning.
    pub merge_candidates_found: usize,
    /// Merges blocked due to proto score too low.
    pub merge_blocked_proto: usize,
    /// Merges blocked due to insufficient support.
    pub merge_blocked_support: usize,
    /// Merges blocked due to value inconsistency (|v1-v2| > eps).
    pub merge_blocked_value: usize,
    /// Merges blocked due to stability constraints (stable + non-stable).
    pub merge_blocked_stability: usize,
    /// Merges blocked due to learned_mask mismatch.
    pub merge_blocked_key_mismatch: usize,
    /// Merges blocked due to ctx_best mismatch.
    pub merge_blocked_ctx_mismatch: usize,
    /// Merges blocked due to gate mode mismatch (including explore never merges).
    pub merge_blocked_mode_mismatch: usize,
    /// Anchors that entered stable state.
    pub stable_new: usize,
    /// Anchors that dropped from stable state.
    pub stable_dropped: usize,
    /// Last tick when merge scan was done.
    last_merge_scan_tick: u64,
    /// Number of proto-based merge scans performed.
    pub merge_scan_runs: usize,
    /// Proto-based merges done (separate from legacy hamming merges).
    pub merges_done_proto: usize,
    /// Sum of proto scores for executed merges (for avg calculation).
    merge_score_sum: f32,
    /// RNG state for sampling-based merge candidate discovery.
    merge_rng_state: u64,
    /// Phase 1.9c: Total pairs checked for merge_opportunity_rate.
    pairs_checked: usize,
    /// Phase 1.9c: Top 5 blocked candidates (best proto scores that didn't merge).
    top_blocked: Vec<BlockedMergeInfo>,

    // Phase 1.9d: Cross-partition merge tracking
    pub cross_mode_merges_done: usize,
    pub cross_mask_merges_done: usize,
    pub cross_partition_merges_done: usize,
    pub blocked_explore_isolated: usize,
    pub blocked_mask_hamming: usize,
    pub blocked_cross_mode_v: usize,
    pub blocked_cross_mode_proto: usize,
    pub blocked_cross_mask_proto: usize,
    pub blocked_cross_mask_v: usize,
    pub blocked_cross_mode_not_stable: usize,
    pub blocked_cross_mask_not_stable: usize,
    /// Sum of |dv| for accepted cross merges.
    cross_dv_sum: f64,
    cross_dv_count: usize,
    /// Sum of mask Hamming for cross-mask merges.
    cross_mask_hamming_sum: u32,
    cross_mask_hamming_count: usize,
    /// Cross-partition merges blocked by rate limiting.
    blocked_cross_rate_limited: usize,
}

impl AnchorBank {
    pub fn new() -> Self {
        AnchorBank {
            anchors: HashMap::new(),
            id_remap: HashMap::new(),
            next_id: 0,
            anchor_creates: 0,
            anchor_evictions: 0,
            anchor_merges: 0,
            match_hamming_sum: 0,
            match_count: 0,
            hamming_histogram: [0; 65],
            total_ticks: 0,
            gates_passed: 0,
            gates_failed: 0,
            last_merge_tick: 0,
            // Phase 1.7a
            proto_updates: 0,
            proto_active_steps: 0,
            proto_total_steps: 0,
            // Phase 1.8
            stable_mode: false, // Start in explore mode
            stable_count: 0,
            mode_transitions: 0,
            merge_blocked_v: 0,
            // Phase 1.9b
            merge_candidates_found: 0,
            merge_blocked_proto: 0,
            merge_blocked_support: 0,
            merge_blocked_value: 0,
            merge_blocked_stability: 0,
            merge_blocked_key_mismatch: 0,
            merge_blocked_ctx_mismatch: 0,
            merge_blocked_mode_mismatch: 0,
            stable_new: 0,
            stable_dropped: 0,
            last_merge_scan_tick: 0,
            merge_scan_runs: 0,
            merges_done_proto: 0,
            merge_score_sum: 0.0,
            merge_rng_state: 0xDEADBEEF_CAFEBABE,
            pairs_checked: 0,
            top_blocked: Vec::new(),
            // Phase 1.9d
            cross_mode_merges_done: 0,
            cross_mask_merges_done: 0,
            cross_partition_merges_done: 0,
            blocked_explore_isolated: 0,
            blocked_mask_hamming: 0,
            blocked_cross_mode_v: 0,
            blocked_cross_mode_proto: 0,
            blocked_cross_mask_proto: 0,
            blocked_cross_mask_v: 0,
            blocked_cross_mode_not_stable: 0,
            blocked_cross_mask_not_stable: 0,
            cross_dv_sum: 0.0,
            cross_dv_count: 0,
            cross_mask_hamming_sum: 0,
            cross_mask_hamming_count: 0,
            blocked_cross_rate_limited: 0,
        }
    }

    /// Get the number of active anchors.
    pub fn len(&self) -> usize {
        self.anchors.values().filter(|a| a.alive).count()
    }

    /// Check if bank is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolve an anchor ID through the remap table.
    fn resolve_id(&self, id: u16) -> u16 {
        let mut current = id;
        // Follow remap chain (should be short, typically 0-1 hops)
        for _ in 0..10 {
            if let Some(&remapped) = self.id_remap.get(&current) {
                current = remapped;
            } else {
                break;
            }
        }
        current
    }

    /// Resolve a signature to an anchor ID (basic version without gating).
    /// Returns (anchor_id, is_new_anchor, match_hamming).
    pub fn resolve(&mut self, signature: u64, current_tick: u64) -> (u16, bool, u32) {
        self.resolve_gated(signature, current_tick, None, None)
    }

    /// Resolve a signature to an anchor ID with optional confidence gating.
    /// If confidence_info is Some and fails gate, will not create new anchors.
    /// Phase 1.8: Added config parameter for value-aware eviction.
    /// Returns (anchor_id, is_new_anchor, match_hamming).
    pub fn resolve_gated(
        &mut self,
        signature: u64,
        current_tick: u64,
        confidence_info: Option<&ConfidenceInfo>,
        config: Option<&Config>,
    ) -> (u16, bool, u32) {
        self.total_ticks = self.total_ticks.max(current_tick);

        // Find nearest existing anchor within ANCHOR_MAX_HAMMING
        let mut best_id: Option<u16> = None;
        let mut best_dist = u32::MAX;

        for (&id, anchor) in &self.anchors {
            if !anchor.alive {
                continue;
            }
            let dist = anchor.hamming_to(signature);
            if dist <= ANCHOR_MAX_HAMMING && dist < best_dist {
                best_dist = dist;
                best_id = Some(id);
            }
        }

        if let Some(id) = best_id {
            // Match found - update anchor stats (only if gate passes or no gate)
            let should_refresh = confidence_info.map(|c| c.passes_gate()).unwrap_or(true);

            if should_refresh {
                let anchor = self.anchors.get_mut(&id).unwrap();
                anchor.last_used_tick = current_tick;
                anchor.usage_count += 1;
                self.gates_passed += 1;
            } else {
                self.gates_failed += 1;
            }

            self.match_hamming_sum += best_dist as u64;
            self.match_count += 1;
            self.hamming_histogram[best_dist as usize] += 1;

            return (self.resolve_id(id), false, best_dist);
        }

        // No match - check if we can create a new anchor
        let can_create = confidence_info.map(|c| c.passes_gate()).unwrap_or(true);

        if !can_create {
            self.gates_failed += 1;
            // Return a "no anchor" signal - use id 0xFFFF as sentinel
            return (0xFFFF, false, u32::MAX);
        }

        self.gates_passed += 1;

        // Create new anchor (with potential pruning)
        if self.len() >= MAX_ANCHORS {
            // Phase 1.8: Use value-aware eviction if config provided
            if let Some(cfg) = config {
                self.prune_one(current_tick, cfg);
            } else {
                // Fallback: use default config for eviction
                let default_cfg = Config::default();
                self.prune_one(current_tick, &default_cfg);
            }
        }

        let new_id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.anchors
            .insert(new_id, Anchor::new(signature, current_tick));
        self.anchor_creates += 1;

        (new_id, true, 0)
    }

    /// Phase 1.8: Prune one anchor using value-aware eviction.
    /// Strategy: Remove anchor with lowest keep_score().
    /// Protected: stable anchors are protected from eviction.
    fn prune_one(&mut self, current_tick: u64, config: &Config) {
        let mut candidate: Option<(u16, f32)> = None; // (id, keep_score)

        for (&id, anchor) in &self.anchors {
            if !anchor.alive {
                continue;
            }
            // Phase 1.8: Stable anchors are protected from eviction
            if anchor.stable {
                continue;
            }

            let score = anchor.keep_score(current_tick, config);

            match candidate {
                None => candidate = Some((id, score)),
                Some((_, best_score)) => {
                    if score < best_score {
                        candidate = Some((id, score));
                    }
                }
            }
        }

        // If all non-stable anchors are exhausted, we have to evict a stable one
        // (fallback to lowest keep_score among stable)
        if candidate.is_none() {
            for (&id, anchor) in &self.anchors {
                if !anchor.alive {
                    continue;
                }
                let score = anchor.keep_score(current_tick, config);
                match candidate {
                    None => candidate = Some((id, score)),
                    Some((_, best_score)) => {
                        if score < best_score {
                            candidate = Some((id, score));
                        }
                    }
                }
            }
        }

        if let Some((id, _)) = candidate {
            if let Some(anchor) = self.anchors.get_mut(&id) {
                anchor.alive = false;
            }
            self.anchor_evictions += 1;
        }
    }

    /// Phase 1.6b: Attempt to merge similar anchors.
    /// Phase 1.8: Added value consistency check (|v1 - v2| < merge_v_delta_max).
    /// Returns Vec of (old_id, new_id) remappings that occurred.
    pub fn merge_similar(&mut self, config: Option<&Config>) -> Vec<(u16, u16)> {
        let mut remaps: Vec<(u16, u16)> = Vec::new();

        // Phase 1.8: Get merge_v_delta_max from config
        let merge_v_delta_max = config.map(|c| c.merge_v_delta_max).unwrap_or(1.0);

        // Collect alive anchor IDs
        let ids: Vec<u16> = self
            .anchors
            .iter()
            .filter(|(_, a)| a.alive)
            .map(|(&id, _)| id)
            .collect();

        // Sort by usage (ascending) so we merge low-use into high-use
        let mut sorted_ids = ids.clone();
        sorted_ids.sort_by_key(|&id| self.anchors.get(&id).map(|a| a.usage_count).unwrap_or(0));

        for i in 0..sorted_ids.len() {
            let id_low = sorted_ids[i];

            // Skip if already merged
            if !self.anchors.get(&id_low).map(|a| a.alive).unwrap_or(false) {
                continue;
            }

            for j in (i + 1)..sorted_ids.len() {
                let id_high = sorted_ids[j];

                // Skip if already merged
                if !self.anchors.get(&id_high).map(|a| a.alive).unwrap_or(false) {
                    continue;
                }

                let (dist, v_delta) = {
                    let a = &self.anchors[&id_low];
                    let b = &self.anchors[&id_high];
                    let dist = (a.proto_signature ^ b.proto_signature).count_ones();
                    let v_delta = (a.v - b.v).abs();
                    (dist, v_delta)
                };

                // Phase 1.8: Check both Hamming distance AND value consistency
                if dist <= MERGE_HAMMING && v_delta < merge_v_delta_max {
                    // Merge id_low into id_high (higher usage keeps ID)
                    let usage_low = self.anchors[&id_low].usage_count;
                    let wins_low = self.anchors[&id_low].wins;

                    // Mark id_low as dead
                    self.anchors.get_mut(&id_low).unwrap().alive = false;

                    // Add usage and wins to id_high (value is preserved from high-usage anchor)
                    let anchor_high = self.anchors.get_mut(&id_high).unwrap();
                    anchor_high.usage_count += usage_low;
                    anchor_high.wins += wins_low;

                    // Record the remap
                    self.id_remap.insert(id_low, id_high);
                    remaps.push((id_low, id_high));

                    self.anchor_merges += 1;
                    break; // id_low is now merged, move to next
                }
            }
        }

        remaps
    }

    /// Check if it's time for a merge attempt.
    pub fn should_merge(&self, current_tick: u64) -> bool {
        current_tick >= self.last_merge_tick + MERGE_EVERY_TICKS
    }

    /// Update the last merge tick.
    pub fn mark_merge_done(&mut self, current_tick: u64) {
        self.last_merge_tick = current_tick;
    }

    // =========================================================================
    // Phase 1.9: Proto-based Merge Scanning
    // =========================================================================

    /// Phase 1.9b: Check if it's time for a proto-based merge scan.
    /// Triggers more aggressively when anchor utilization is high (>90%).
    pub fn should_scan_merges(&self, current_tick: u64, config: &Config) -> bool {
        let period_elapsed =
            current_tick >= self.last_merge_scan_tick + config.merge_scan_period as u64;
        let high_utilization = self.anchor_utilization() > 0.90;

        // Scan if period elapsed, OR if high utilization (encourage consolidation)
        period_elapsed || (high_utilization && current_tick > self.last_merge_scan_tick + 5)
    }

    /// Mark merge scan done.
    pub fn mark_scan_done(&mut self, current_tick: u64) {
        self.last_merge_scan_tick = current_tick;
    }

    /// Phase 1.9b: Find merge candidate pairs using sampling-based proto similarity.
    /// Uses merge_scan_k to sample candidates per anchor for scalability.
    /// Returns Vec of (id1, id2, proto_score) where id1 is the lower-usage anchor.
    /// Uses different value epsilon for stable vs non-stable anchors.
    /// Phase 1.9d: Extended merge candidate with cross-partition tracking.
    /// Returns: (id_low, id_high, proto_score, is_cross_mode, is_cross_mask)
    pub fn find_merge_pairs(&mut self, config: &Config) -> Vec<(u16, u16, f32, bool, bool)> {
        let proto_m = config.proto_m.min(DEFAULT_PROTO_M);
        let min_support = config.merge_min_support;
        let proto_min = config.merge_proto_min_score;
        let proto_min_cross = config.merge_proto_min_score_cross_key;
        let same_key_only = config.merge_same_key_only;
        let scan_k = config.merge_scan_k as usize;

        // Phase 1.9d: Extended candidates with cross-partition info
        // (id_low, id_high, proto_score, is_cross_mode, is_cross_mask, mask_hamming, v_delta)
        let mut extended_candidates: Vec<(u16, u16, f32, bool, bool, u32, f32)> = Vec::new();
        // Phase 1.9c: Collect blocked candidates locally to avoid borrow issues
        let mut blocked_to_track: Vec<BlockedMergeInfo> = Vec::new();

        // Phase 1.9b: Get max usage for merge eligibility (protects high-usage anchors)
        let max_usage = config.merge_max_usage;

        // Collect alive anchors with sufficient support AND not too high usage
        let eligible: Vec<(u16, &Anchor)> = self
            .anchors
            .iter()
            .filter(|(_, a)| {
                a.alive && a.proto_support >= min_support && a.usage_count <= max_usage
            })
            .map(|(&id, a)| (id, a))
            .collect();

        if eligible.len() < 2 {
            return Vec::new();
        }

        // Sort by usage ascending so lower-usage comes first when we output pairs
        let mut sorted = eligible.clone();
        sorted.sort_by_key(|(_, a)| a.usage_count);

        // Use sampling-based approach: for each anchor, sample K others
        // This reduces O(n^2) to O(n*K) when K << n
        let use_sampling = sorted.len() > scan_k * 2;

        for i in 0..sorted.len() {
            let (id1, a1) = sorted[i];

            // Determine which anchors to compare against
            let compare_range: Vec<usize> = if use_sampling {
                // Sample K anchors using deterministic RNG
                let mut samples = Vec::with_capacity(scan_k);
                let mut rng_val = self.merge_rng_state.wrapping_add(id1 as u64);
                for _ in 0..scan_k.min(sorted.len() - 1) {
                    rng_val = rng_val.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let j = (rng_val as usize) % sorted.len();
                    if j != i && !samples.contains(&j) {
                        samples.push(j);
                    }
                }
                samples
            } else {
                // Full O(n^2) scan for small n
                ((i + 1)..sorted.len()).collect()
            };

            for j in compare_range {
                if j == i {
                    continue;
                }
                let (id2, a2) = sorted[j];
                self.pairs_checked += 1;

                // Ensure id1 < id2 for consistent ordering
                let (id_low, id_high, a_low, a_high) = if a1.usage_count <= a2.usage_count {
                    (id1, id2, a1, a2)
                } else {
                    (id2, id1, a2, a1)
                };

                // If same_key_only, check signatures match closely first
                let same_key =
                    (a_low.proto_signature ^ a_high.proto_signature).count_ones() <= MERGE_HAMMING;
                if same_key_only && !same_key {
                    continue;
                }

                // Determine proto threshold (computed early for cross-partition check)
                let threshold = if same_key { proto_min } else { proto_min_cross };

                // Compute proto similarity (needed for cross-partition decisions)
                let proto_score = a_low.proto_score_against(a_high, proto_m);

                // Phase 1.9d: Try standard compatibility first
                let (compatible, block_reason) = a_low.merge_compatible_with(a_high);

                // Track cross-partition info
                let mut is_cross_mode = false;
                let mut is_cross_mask = false;
                let mut mask_hamming: u32 = 0;
                let v_delta = (a_low.v - a_high.v).abs();

                if !compatible {
                    // Phase 1.9d: Try cross-partition merge for stable anchors
                    let (cross_allowed, cross_reason, cross_mode, cross_mask, ham) =
                        a_low.check_cross_partition_merge(a_high, proto_score, config);

                    if !cross_allowed {
                        // Track block reason
                        match cross_reason {
                            MergeBlockReason::KeyMismatch => self.merge_blocked_key_mismatch += 1,
                            MergeBlockReason::CtxMismatch => self.merge_blocked_ctx_mismatch += 1,
                            MergeBlockReason::ModeMismatch | MergeBlockReason::ModeExplore => {
                                self.merge_blocked_mode_mismatch += 1;
                            }
                            MergeBlockReason::ExploreIsolated => self.blocked_explore_isolated += 1,
                            MergeBlockReason::MaskHammingTooHigh => self.blocked_mask_hamming += 1,
                            MergeBlockReason::CrossModeVDelta => self.blocked_cross_mode_v += 1,
                            MergeBlockReason::CrossModeProtoLow => {
                                self.blocked_cross_mode_proto += 1
                            }
                            MergeBlockReason::CrossMaskProtoLow => {
                                self.blocked_cross_mask_proto += 1
                            }
                            MergeBlockReason::CrossMaskVDelta => self.blocked_cross_mask_v += 1,
                            MergeBlockReason::CrossModeNotStable => {
                                self.blocked_cross_mode_not_stable += 1
                            }
                            MergeBlockReason::CrossMaskNotStable => {
                                self.blocked_cross_mask_not_stable += 1
                            }
                            _ => {}
                        }
                        continue;
                    }
                    // Cross-partition merge allowed
                    is_cross_mode = cross_mode;
                    is_cross_mask = cross_mask;
                    mask_hamming = ham;
                }

                // Check proto score threshold (applies to all merges)
                if proto_score < threshold {
                    self.merge_blocked_proto += 1;
                    if proto_score >= 0.70 {
                        blocked_to_track.push(BlockedMergeInfo {
                            id_a: id_low,
                            id_b: id_high,
                            proto_score,
                            reason: MergeBlockReason::ProtoScore,
                        });
                    }
                    continue;
                }

                // Phase 1.9b: Protect stable anchors from non-stable merges
                // Only allow: both stable OR both non-stable
                let both_stable = a_low.stable && a_high.stable;
                let both_non_stable = !a_low.stable && !a_high.stable;
                if !both_stable && !both_non_stable {
                    // One stable, one non-stable - skip to preserve stability
                    self.merge_blocked_stability += 1;
                    if proto_score >= 0.70 {
                        blocked_to_track.push(BlockedMergeInfo {
                            id_a: id_low,
                            id_b: id_high,
                            proto_score,
                            reason: MergeBlockReason::StabilityMixed,
                        });
                    }
                    continue;
                }

                // Phase 1.9c: Non-stable anchors need high support to merge (no explore garbage)
                if both_non_stable {
                    let min_sup = a_low.proto_support.min(a_high.proto_support);
                    if min_sup < config.merge_min_support_nonstable {
                        self.merge_blocked_support += 1;
                        continue;
                    }
                }

                // Phase 1.9b: Use different value epsilon based on stability
                // (Skip for cross-partition merges as they have their own v_eps checks)
                if !is_cross_mode && !is_cross_mask {
                    let v_eps = if both_stable {
                        config.merge_v_eps_stable
                    } else {
                        config.merge_v_eps
                    };

                    if v_delta >= v_eps {
                        self.merge_blocked_value += 1;
                        if proto_score >= 0.70 {
                            blocked_to_track.push(BlockedMergeInfo {
                                id_a: id_low,
                                id_b: id_high,
                                proto_score,
                                reason: MergeBlockReason::ValueDelta,
                            });
                        }
                        continue;
                    }
                }

                // This pair is a merge candidate
                self.merge_candidates_found += 1;
                extended_candidates.push((
                    id_low,
                    id_high,
                    proto_score,
                    is_cross_mode,
                    is_cross_mask,
                    mask_hamming,
                    v_delta,
                ));
            }
        }

        // Phase 1.9c: Update top_blocked from collected candidates
        for info in blocked_to_track {
            self.track_blocked_candidate_from_info(info);
        }

        // Update RNG state for next scan
        self.merge_rng_state = self
            .merge_rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);

        // Phase 1.9d: Sort by (1) proto_score desc, (2) min(v) desc, (3) combined support desc
        extended_candidates.sort_by(|a, b| {
            // First by proto_score descending
            let score_cmp = b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal);
            if score_cmp != std::cmp::Ordering::Equal {
                return score_cmp;
            }
            // Then by v_delta ascending (lower is better)
            a.6.partial_cmp(&b.6).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate pairs (may have duplicates from sampling)
        let mut seen: std::collections::HashSet<(u16, u16)> = std::collections::HashSet::new();
        extended_candidates.retain(|(id1, id2, _, _, _, _, _)| {
            let key = if id1 < id2 {
                (*id1, *id2)
            } else {
                (*id2, *id1)
            };
            seen.insert(key)
        });

        // Phase 1.9d: Apply rate limiting for cross-partition merges
        // Return extended candidates: (id_low, id_high, proto_score, is_cross_mode, is_cross_mask)
        let max_cross = config.max_cross_partition_merges_per_scan as usize;
        let mut cross_count = 0;
        let mut final_candidates: Vec<(u16, u16, f32, bool, bool)> = Vec::new();

        for (id_low, id_high, proto_score, is_cross_mode, is_cross_mask, _mask_hamming, _v_delta) in
            extended_candidates
        {
            let is_cross = is_cross_mode || is_cross_mask;

            if is_cross {
                if cross_count >= max_cross {
                    // Rate limit reached - skip this cross-partition merge
                    self.blocked_cross_rate_limited += 1;
                    continue;
                }
                cross_count += 1;
            }

            final_candidates.push((id_low, id_high, proto_score, is_cross_mode, is_cross_mask));
        }

        final_candidates
    }

    /// Phase 1.9b: Execute value-preserving merge of id_low into id_high.
    /// Uses weighted average for values and blends prototypes.
    /// Stability handling:
    /// - If both stable: preserve stable ONLY if merged anchor meets EXIT thresholds
    /// - If stable + non-stable: merged is non-stable (prevents "stable poisoning")
    pub fn execute_merge(&mut self, id_low: u16, id_high: u16, config: &Config) -> bool {
        let proto_m = config.proto_m.min(DEFAULT_PROTO_M);

        // Verify both anchors are alive and collect state
        let (
            can_merge,
            u_low,
            u_high,
            low_stable,
            high_stable,
            low_support,
            high_support,
            earliest_stable_tick,
        ) = {
            let a_low = match self.anchors.get(&id_low) {
                Some(a) if a.alive => a,
                _ => return false,
            };
            let a_high = match self.anchors.get(&id_high) {
                Some(a) if a.alive => a,
                _ => return false,
            };
            // Use the earlier stable_since_tick (more established stability)
            let earliest = a_low.stable_since_tick.min(a_high.stable_since_tick);
            (
                true,
                a_low.usage_count,
                a_high.usage_count,
                a_low.stable,
                a_high.stable,
                a_low.proto_support,
                a_high.proto_support,
                earliest,
            )
        };

        if !can_merge {
            return false;
        }

        // Extract values from low-usage anchor
        let (v_low, wins_low, proto_w_low, proto_nodes_low, entropy_low, abs_td_low) = {
            let a = &self.anchors[&id_low];
            (
                a.v,
                a.wins,
                a.proto_w.clone(),
                a.proto_nodes.clone(),
                a.entropy(proto_m),
                a.v_ema_abs_td,
            )
        };

        // Compute weighted average factor (low_usage / total_usage)
        let total_u = u_low + u_high;
        let w_low = if total_u > 0 {
            u_low as f32 / total_u as f32
        } else {
            0.5
        };
        let eta = config.merge_proto_eta;

        // Mark id_low as dead
        self.anchors.get_mut(&id_low).unwrap().alive = false;

        // Update id_high with merged values
        {
            let a_high = self.anchors.get_mut(&id_high).unwrap();

            // Merge usage and wins
            a_high.usage_count += u_low;
            a_high.wins += wins_low;

            // Value-preserving merge: weighted average
            a_high.v = (1.0 - w_low) * a_high.v + w_low * v_low;

            // Proto merge: blend weights where nodes overlap
            for i in 0..proto_m {
                if proto_w_low[i] > 0.0 {
                    let node_low = proto_nodes_low[i];
                    // Find if this node exists in high's proto
                    let mut found = false;
                    for j in 0..proto_m {
                        if a_high.proto_nodes[j] == node_low && a_high.proto_w[j] > 0.0 {
                            // Blend weights
                            a_high.proto_w[j] =
                                (1.0 - eta) * a_high.proto_w[j] + eta * proto_w_low[i];
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        // Try to insert into empty or smallest slot
                        let mut min_w = f32::MAX;
                        let mut min_idx = 0;
                        for j in 0..proto_m {
                            if a_high.proto_w[j] < min_w {
                                min_w = a_high.proto_w[j];
                                min_idx = j;
                            }
                        }
                        if proto_w_low[i] > min_w + 0.01 {
                            a_high.proto_nodes[min_idx] = node_low;
                            a_high.proto_w[min_idx] = proto_w_low[i] * eta;
                        }
                    }
                }
            }

            // Merge proto_support (additive)
            a_high.proto_support = a_high.proto_support.saturating_add(low_support);

            // Blend EMA fields
            a_high.v_ema_abs_td = (1.0 - w_low) * a_high.v_ema_abs_td + w_low * abs_td_low;
            a_high.proto_entropy_ema =
                (1.0 - w_low) * a_high.proto_entropy_ema + w_low * entropy_low;

            // Phase 1.9b: Stability handling for merged anchor
            if low_stable && high_stable {
                // Both stable: preserve ONLY if merged anchor still meets EXIT thresholds
                let merged_entropy = a_high.entropy(proto_m);
                let merged_abs_td = a_high.v_ema_abs_td;
                let merged_support = a_high.proto_support;

                let passes_exit = merged_support >= config.stable_min_support_exit
                    && merged_entropy <= config.stable_exit_entropy
                    && merged_abs_td <= config.stable_exit_abs_td
                    && a_high.v >= config.stable_v_min - config.stable_catastrophic_v_drop;

                if passes_exit {
                    a_high.stable = true;
                    a_high.stable_since_tick = earliest_stable_tick;
                } else {
                    // Merged anchor doesn't meet exit thresholds - drop stability
                    a_high.stable = false;
                    a_high.stable_since_tick = 0;
                    self.merge_blocked_stability += 1;
                }
            } else {
                // Stable + non-stable: merged is non-stable (prevents "stable poisoning")
                a_high.stable = false;
                a_high.stable_since_tick = 0;
            }
        }

        // Normalize proto weights to stay bounded
        {
            let a_high = self.anchors.get_mut(&id_high).unwrap();
            let mut weight_sum: f32 = a_high.proto_w.iter().take(proto_m).sum();
            if weight_sum > 2.0 {
                // Normalize to prevent unbounded growth
                for i in 0..proto_m {
                    a_high.proto_w[i] /= weight_sum;
                }
            }
        }

        // Record remap
        self.id_remap.insert(id_low, id_high);
        self.anchor_merges += 1;

        true
    }

    /// Scan for merge candidates and execute merges.
    /// Returns Vec of (old_id, new_id) remappings that occurred.
    pub fn scan_and_merge(&mut self, config: &Config) -> Vec<(u16, u16)> {
        self.merge_scan_runs += 1;
        let candidates = self.find_merge_pairs(config);
        let max_merges = config.merge_max_per_scan as usize;

        let mut remaps: Vec<(u16, u16)> = Vec::new();
        let mut merged_ids: std::collections::HashSet<u16> = std::collections::HashSet::new();

        for (id_low, id_high, score, is_cross_mode, is_cross_mask) in candidates {
            if remaps.len() >= max_merges {
                break;
            }

            // Skip if either anchor was already involved in a merge this scan
            if merged_ids.contains(&id_low) || merged_ids.contains(&id_high) {
                continue;
            }

            // Phase 1.9d: Capture v_delta and mask_hamming before merge for tracking
            let (v_delta, mask_hamming) = {
                let a_low = self.anchors.get(&id_low);
                let a_high = self.anchors.get(&id_high);
                match (a_low, a_high) {
                    (Some(a), Some(b)) => {
                        let vd = (a.v - b.v).abs();
                        let mh = (a.learned_mask ^ b.learned_mask).count_ones();
                        (vd, mh)
                    }
                    _ => (0.0, 0),
                }
            };

            if self.execute_merge(id_low, id_high, config) {
                remaps.push((id_low, id_high));
                merged_ids.insert(id_low);
                merged_ids.insert(id_high);
                // Track proto merge score
                self.merges_done_proto += 1;
                self.merge_score_sum += score;

                // Phase 1.9d: Track cross-partition merge statistics
                let is_cross = is_cross_mode || is_cross_mask;
                if is_cross {
                    self.cross_partition_merges_done += 1;
                    self.cross_dv_sum += v_delta as f64;
                    self.cross_dv_count += 1;
                }
                if is_cross_mode {
                    self.cross_mode_merges_done += 1;
                }
                if is_cross_mask {
                    self.cross_mask_merges_done += 1;
                    self.cross_mask_hamming_sum += mask_hamming;
                    self.cross_mask_hamming_count += 1;
                }
            }
        }

        remaps
    }

    // =========================================================================
    // Phase 1.6a: Metrics
    // =========================================================================

    /// Number of anchors currently in use.
    pub fn anchors_used(&self) -> usize {
        self.len()
    }

    /// Anchor utilization = anchors_used / capacity.
    pub fn anchor_utilization(&self) -> f64 {
        self.len() as f64 / MAX_ANCHORS as f64
    }

    /// Thrash rate = evictions per 10k ticks.
    pub fn thrash_rate(&self) -> f64 {
        if self.total_ticks == 0 {
            0.0
        } else {
            self.anchor_evictions as f64 / self.total_ticks as f64 * 10000.0
        }
    }

    /// Rate of new anchor creation (new / total_resolved).
    pub fn new_anchor_rate(&self) -> f64 {
        let total = self.match_count as usize + self.anchor_creates;
        if total == 0 {
            0.0
        } else {
            self.anchor_creates as f64 / total as f64
        }
    }

    /// Average Hamming distance for matches.
    pub fn avg_match_hamming(&self) -> f64 {
        if self.match_count == 0 {
            0.0
        } else {
            self.match_hamming_sum as f64 / self.match_count as f64
        }
    }

    /// P95 Hamming distance for matches.
    pub fn p95_match_hamming(&self) -> u32 {
        if self.match_count == 0 {
            return 0;
        }

        let target = (self.match_count as f64 * 0.95).ceil() as u64;
        let mut cumsum = 0u64;

        for (dist, &count) in self.hamming_histogram.iter().enumerate() {
            cumsum += count;
            if cumsum >= target {
                return dist as u32;
            }
        }

        64 // max possible
    }

    /// Total anchors evicted.
    pub fn anchors_evicted(&self) -> usize {
        self.anchor_evictions
    }

    /// Gate pass rate.
    pub fn gate_pass_rate(&self) -> f64 {
        let total = self.gates_passed + self.gates_failed;
        if total == 0 {
            1.0
        } else {
            self.gates_passed as f64 / total as f64
        }
    }

    /// Print diagnostics.
    pub fn print_stats(&self) {
        println!("AnchorBank Stats (Phase 1.6):");
        println!(
            "  anchors_used={} / {} ({:.1}% utilization)",
            self.anchors_used(),
            MAX_ANCHORS,
            self.anchor_utilization() * 100.0
        );
        println!("  anchor_creates={}", self.anchor_creates);
        println!("  anchor_evictions={}", self.anchor_evictions);
        println!("  anchor_merges={}", self.anchor_merges);
        println!("  thrash_rate={:.2} per 10k ticks", self.thrash_rate());
        println!("  new_anchor_rate={:.1}%", self.new_anchor_rate() * 100.0);
        println!("  avg_match_hamming={:.2}", self.avg_match_hamming());
        println!("  p95_match_hamming={}", self.p95_match_hamming());
        println!("  gate_pass_rate={:.1}%", self.gate_pass_rate() * 100.0);
    }

    // =========================================================================
    // Phase 1.7a: Prototype Methods
    // =========================================================================

    /// Phase 1.9b: Update anchor's merge compatibility fields (learned_mask, ctx).
    /// Should be called when an anchor is matched to track its meaning partition.
    pub fn update_anchor_partition(&mut self, anchor_id: u16, learned_mask: u64, ctx: Option<u8>) {
        if anchor_id == 0xFFFF {
            return;
        }

        let resolved_id = self.resolve_id(anchor_id);

        if let Some(anchor) = self.anchors.get_mut(&resolved_id) {
            if anchor.alive {
                // Set learned_mask (use the most recent one)
                anchor.set_learned_mask(learned_mask);
                // Record context observation
                if let Some(c) = ctx {
                    anchor.record_ctx(c);
                }
            }
        }
    }

    /// Update the prototype of an anchor with current TopK observation.
    /// Should be called when an anchor is matched and gate passes.
    pub fn update_anchor_proto(&mut self, anchor_id: u16, topk: &[(usize, f64)], config: &Config) {
        // Skip invalid anchor IDs
        if anchor_id == 0xFFFF {
            return;
        }

        // Follow remap chain
        let resolved_id = self.resolve_id(anchor_id);

        if let Some(anchor) = self.anchors.get_mut(&resolved_id) {
            if anchor.alive {
                let had_support = anchor.proto_support > 0;
                anchor.update_proto(topk, config);
                self.proto_updates += 1;

                // Track whether anchor had proto_support after update
                self.proto_total_steps += 1;
                if had_support || anchor.proto_support > 0 {
                    self.proto_active_steps += 1;
                }
            }
        }
    }

    /// Get the prototype score for an anchor with current TopK.
    pub fn get_proto_score(&self, anchor_id: u16, topk: &[(usize, f64)], proto_m: usize) -> f32 {
        if anchor_id == 0xFFFF {
            return 0.0;
        }

        let resolved_id = self.resolve_id(anchor_id);

        if let Some(anchor) = self.anchors.get(&resolved_id) {
            if anchor.alive && anchor.proto_support > 0 {
                return anchor.proto_score(topk, proto_m);
            }
        }

        0.0
    }

    /// Record a step for proto_active tracking (call even when not updating).
    pub fn record_proto_step(&mut self, anchor_id: u16) {
        if anchor_id == 0xFFFF {
            self.proto_total_steps += 1;
            return;
        }

        let resolved_id = self.resolve_id(anchor_id);
        self.proto_total_steps += 1;

        if let Some(anchor) = self.anchors.get(&resolved_id) {
            if anchor.alive && anchor.proto_support > 0 {
                self.proto_active_steps += 1;
            }
        }
    }

    /// Average proto_support across active anchors.
    pub fn avg_proto_support(&self) -> f64 {
        let alive_anchors: Vec<&Anchor> = self.anchors.values().filter(|a| a.alive).collect();

        if alive_anchors.is_empty() {
            return 0.0;
        }

        let total_support: u32 = alive_anchors.iter().map(|a| a.proto_support).sum();

        total_support as f64 / alive_anchors.len() as f64
    }

    /// Rate of steps where winning anchor had proto_support > 0.
    pub fn proto_active_rate(&self) -> f64 {
        if self.proto_total_steps == 0 {
            0.0
        } else {
            self.proto_active_steps as f64 / self.proto_total_steps as f64
        }
    }

    /// Compute entropy of top-N most-used anchors.
    /// Returns (avg_entropy, count_of_anchors_used).
    pub fn proto_entropy_top_n(&self, n: usize, proto_m: usize) -> (f32, usize) {
        // Collect alive anchors with their usage counts
        let mut anchors_by_usage: Vec<(&Anchor, u32)> = self
            .anchors
            .values()
            .filter(|a| a.alive && a.proto_support > 0)
            .map(|a| (a, a.usage_count))
            .collect();

        // Sort by usage count descending
        anchors_by_usage.sort_by(|a, b| b.1.cmp(&a.1));

        // Take top N
        let top_n: Vec<&Anchor> = anchors_by_usage.iter().take(n).map(|(a, _)| *a).collect();

        if top_n.is_empty() {
            return (0.0, 0);
        }

        let total_entropy: f32 = top_n.iter().map(|a| a.entropy(proto_m)).sum();

        (total_entropy / top_n.len() as f32, top_n.len())
    }

    /// Get reference to an anchor by ID (for external proto_score computation).
    pub fn get_anchor(&self, anchor_id: u16) -> Option<&Anchor> {
        if anchor_id == 0xFFFF {
            return None;
        }
        let resolved_id = self.resolve_id(anchor_id);
        self.anchors.get(&resolved_id).filter(|a| a.alive)
    }

    // =========================================================================
    // Phase 1.7b: Value Learning Methods
    // =========================================================================

    /// Get the value estimate for an anchor (returns 0.0 for invalid/abstain).
    pub fn get_value(&self, anchor_id: u16) -> f32 {
        if anchor_id == 0xFFFF {
            return 0.0;
        }
        let resolved_id = self.resolve_id(anchor_id);
        self.anchors
            .get(&resolved_id)
            .filter(|a| a.alive)
            .map(|a| a.v)
            .unwrap_or(0.0)
    }

    /// Update the value of an anchor using TD error.
    pub fn update_anchor_value(&mut self, anchor_id: u16, td: f32, config: &Config) {
        if anchor_id == 0xFFFF {
            return;
        }
        let resolved_id = self.resolve_id(anchor_id);
        if let Some(anchor) = self.anchors.get_mut(&resolved_id) {
            if anchor.alive {
                anchor.v_update(td, config);
            }
        }
    }

    /// Get value metrics: (total_v_updates, avg_v_used, avg_abs_td_used)
    /// Only counts anchors with at least one v_update.
    pub fn value_metrics(&self) -> (usize, f64, f64) {
        let used_anchors: Vec<&Anchor> = self
            .anchors
            .values()
            .filter(|a| a.alive && a.v_updates > 0)
            .collect();

        if used_anchors.is_empty() {
            return (0, 0.0, 0.0);
        }

        let total_updates: usize = used_anchors.iter().map(|a| a.v_updates as usize).sum();

        let avg_v: f64 =
            used_anchors.iter().map(|a| a.v as f64).sum::<f64>() / used_anchors.len() as f64;

        let avg_abs_td: f64 = used_anchors
            .iter()
            .map(|a| a.v_ema_abs_td as f64)
            .sum::<f64>()
            / used_anchors.len() as f64;

        (total_updates, avg_v, avg_abs_td)
    }

    /// Get top-N anchors by value.
    /// Returns Vec of (anchor_id, v, entropy_ema, proto_support, v_updates).
    pub fn top_n_by_value(&self, n: usize, proto_m: usize) -> Vec<(u16, f32, f32, u32, u32)> {
        let mut anchors_with_v: Vec<(u16, &Anchor)> = self
            .anchors
            .iter()
            .filter(|(_, a)| a.alive && a.v_updates > 0)
            .map(|(&id, a)| (id, a))
            .collect();

        // Sort by value descending
        anchors_with_v.sort_by(|a, b| {
            b.1.v
                .partial_cmp(&a.1.v)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        anchors_with_v
            .iter()
            .take(n)
            .map(|(id, a)| (*id, a.v, a.entropy(proto_m), a.proto_support, a.v_updates))
            .collect()
    }

    // =========================================================================
    // Phase 1.8: VALUE IS CONTROL - Lifecycle Methods
    // =========================================================================

    /// Record a "win" for an anchor (successful recall).
    pub fn record_win(&mut self, anchor_id: u16) {
        if anchor_id == 0xFFFF {
            return;
        }
        let resolved_id = self.resolve_id(anchor_id);
        if let Some(anchor) = self.anchors.get_mut(&resolved_id) {
            if anchor.alive {
                anchor.record_win();
            }
        }
    }

    /// Update stability of all anchors and potentially switch modes.
    /// Returns true if mode changed.
    pub fn update_stability(&mut self, current_tick: u64, config: &Config) -> bool {
        let proto_m = config.proto_m.min(DEFAULT_PROTO_M);
        let mut new_stable_count = 0;

        for anchor in self.anchors.values_mut() {
            if anchor.alive {
                let (stable, entered, dropped) =
                    anchor.check_stability_hysteresis(current_tick, config, proto_m);
                if stable {
                    new_stable_count += 1;
                }
                if entered {
                    self.stable_new += 1;
                }
                if dropped {
                    self.stable_dropped += 1;
                }
            }
        }

        self.stable_count = new_stable_count;

        // Check for mode transition
        let alive_count = self.len();
        if alive_count == 0 {
            return false;
        }

        let stable_frac = new_stable_count as f64 / alive_count as f64;
        let should_be_stable = stable_frac >= config.stable_mode_threshold;

        if should_be_stable != self.stable_mode {
            self.stable_mode = should_be_stable;
            self.mode_transitions += 1;
            return true;
        }

        false
    }

    /// Get current fraction of stable anchors.
    pub fn stable_fraction(&self) -> f64 {
        let alive_count = self.len();
        if alive_count == 0 {
            0.0
        } else {
            self.stable_count as f64 / alive_count as f64
        }
    }

    /// Get the gate multiplier based on current mode.
    /// In explore mode: use gate_explore_mult (more permissive).
    /// In stable mode: use gate_stable_mult (stricter).
    pub fn get_gate_mult(&self, config: &Config) -> f64 {
        if self.stable_mode {
            config.gate_stable_mult
        } else {
            config.gate_explore_mult
        }
    }

    /// Get Phase 1.8 lifecycle metrics.
    /// Returns (stable_count, stable_fraction, mode_transitions, merge_blocked_v, current_mode).
    pub fn lifecycle_metrics(&self) -> (usize, f64, usize, usize, bool) {
        (
            self.stable_count,
            self.stable_fraction(),
            self.mode_transitions,
            self.merge_blocked_v,
            self.stable_mode,
        )
    }

    /// Phase 1.9c: Track a blocked merge candidate from info (keep top 5 by proto score).
    fn track_blocked_candidate_from_info(&mut self, info: BlockedMergeInfo) {
        let proto_score = info.proto_score;
        if self.top_blocked.len() < 5 {
            self.top_blocked.push(info);
            self.top_blocked
                .sort_by(|a, b| b.proto_score.partial_cmp(&a.proto_score).unwrap());
        } else if proto_score > self.top_blocked[4].proto_score {
            self.top_blocked[4] = info;
            self.top_blocked
                .sort_by(|a, b| b.proto_score.partial_cmp(&a.proto_score).unwrap());
        }
    }

    /// Get Phase 1.9b/c merge and stability metrics.
    /// Returns a struct with all merge blocking reasons and opportunity rate.
    pub fn consolidation_metrics(&self) -> MergeStats {
        let opportunity_rate = if self.pairs_checked > 0 {
            self.merge_candidates_found as f64 / self.pairs_checked as f64
        } else {
            0.0
        };

        MergeStats {
            candidates_found: self.merge_candidates_found,
            blocked_proto: self.merge_blocked_proto,
            blocked_value: self.merge_blocked_value,
            blocked_stability: self.merge_blocked_stability,
            blocked_support: self.merge_blocked_support,
            blocked_key_mismatch: self.merge_blocked_key_mismatch,
            blocked_ctx_mismatch: self.merge_blocked_ctx_mismatch,
            blocked_mode_mismatch: self.merge_blocked_mode_mismatch,
            stable_new: self.stable_new,
            stable_dropped: self.stable_dropped,
            pairs_checked: self.pairs_checked,
            opportunity_rate,
            // Phase 1.9d: Cross-partition metrics
            cross_mode_merges_done: self.cross_mode_merges_done,
            cross_mask_merges_done: self.cross_mask_merges_done,
            cross_partition_merges_done: self.cross_partition_merges_done,
            blocked_explore_isolated: self.blocked_explore_isolated,
            blocked_mask_hamming: self.blocked_mask_hamming,
            blocked_cross_mode_v: self.blocked_cross_mode_v,
            blocked_cross_mode_proto: self.blocked_cross_mode_proto,
            blocked_cross_mask_proto: self.blocked_cross_mask_proto,
            blocked_cross_mask_v: self.blocked_cross_mask_v,
            blocked_cross_mode_not_stable: self.blocked_cross_mode_not_stable,
            blocked_cross_mask_not_stable: self.blocked_cross_mask_not_stable,
            cross_dv_sum: self.cross_dv_sum,
            cross_dv_count: self.cross_dv_count,
            cross_mask_hamming_sum: self.cross_mask_hamming_sum,
            cross_mask_hamming_count: self.cross_mask_hamming_count,
            blocked_cross_rate_limited: self.blocked_cross_rate_limited,
        }
    }

    /// Phase 1.9c: Get top blocked merge candidates for diagnostics.
    pub fn top_blocked_candidates(&self) -> &[BlockedMergeInfo] {
        &self.top_blocked
    }

    /// Get average proto score of executed merges.
    pub fn avg_merge_score(&self) -> f32 {
        if self.merges_done_proto > 0 {
            self.merge_score_sum / self.merges_done_proto as f32
        } else {
            0.0
        }
    }

    /// Get full Phase 1.9 merge stats.
    /// Returns (merge_scan_runs, merges_done_proto, avg_merge_score, merge_candidates_found).
    pub fn merge_stats(&self) -> (usize, usize, f32, usize) {
        (
            self.merge_scan_runs,
            self.merges_done_proto,
            self.avg_merge_score(),
            self.merge_candidates_found,
        )
    }

    /// Get stable drop ratio (dropped / (dropped + current_stable)).
    /// Used for acceptance criteria: should be <= 0.35.
    pub fn stable_drop_ratio(&self) -> f64 {
        let total = self.stable_dropped + self.stable_count;
        if total == 0 {
            0.0
        } else {
            self.stable_dropped as f64 / total as f64
        }
    }

    /// Phase 1.9e: Get averages for stable anchors (support, entropy, |TD|).
    /// Returns (avg_support, avg_entropy, avg_abs_td, count).
    pub fn stable_anchor_averages(&self) -> (f64, f64, f64, usize) {
        let stable_anchors: Vec<_> = self
            .anchors
            .values()
            .filter(|a| a.alive && a.stable)
            .collect();

        let count = stable_anchors.len();
        if count == 0 {
            return (0.0, 0.0, 0.0, 0);
        }

        let sum_support: u32 = stable_anchors.iter().map(|a| a.proto_support).sum();
        let sum_entropy: f32 = stable_anchors.iter().map(|a| a.proto_entropy_ema).sum();
        let sum_abs_td: f32 = stable_anchors.iter().map(|a| a.v_ema_abs_td).sum();

        (
            sum_support as f64 / count as f64,
            sum_entropy as f64 / count as f64,
            sum_abs_td as f64 / count as f64,
            count,
        )
    }

    /// Phase 1.9f: Check if a specific anchor is stable.
    pub fn is_anchor_stable(&self, anchor_id: u16) -> bool {
        self.anchors
            .get(&anchor_id)
            .map(|a| a.alive && a.stable)
            .unwrap_or(false)
    }

    /// Phase 1.9f: Comprehensive stable anchor statistics.
    /// Returns StableAnchorStats with count, mass, percentiles, and averages.
    pub fn stable_anchor_stats(&self) -> StableAnchorStats {
        let mut supports: Vec<u32> = self
            .anchors
            .values()
            .filter(|a| a.alive && a.stable)
            .map(|a| a.proto_support)
            .collect();

        let count = supports.len();
        if count == 0 {
            return StableAnchorStats::default();
        }

        // Sort for percentiles
        supports.sort_unstable();

        let mass_sum: u64 = supports.iter().map(|&s| s as u64).sum();
        let mass_mean = mass_sum as f64 / count as f64;

        // Percentiles (0-indexed)
        let p50_idx = count / 2;
        let p90_idx = (count * 9) / 10;
        let support_p50 = supports.get(p50_idx).copied().unwrap_or(0);
        let support_p90 = supports.get(p90_idx).copied().unwrap_or(0);

        // Compute means for v, entropy, abs_td
        let stable_anchors: Vec<_> = self
            .anchors
            .values()
            .filter(|a| a.alive && a.stable)
            .collect();

        let value_sum: f32 = stable_anchors.iter().map(|a| a.v).sum();
        let entropy_sum: f32 = stable_anchors.iter().map(|a| a.proto_entropy_ema).sum();
        let td_sum: f32 = stable_anchors.iter().map(|a| a.v_ema_abs_td).sum();

        StableAnchorStats {
            count,
            mass_sum,
            mass_mean,
            support_p50,
            support_p90,
            value_mean: value_sum as f64 / count as f64,
            entropy_mean: entropy_sum as f64 / count as f64,
            td_mean: td_sum as f64 / count as f64,
        }
    }

    /// Get total wins across all anchors.
    pub fn total_wins(&self) -> u32 {
        self.anchors
            .values()
            .filter(|a| a.alive)
            .map(|a| a.wins)
            .sum()
    }

    /// Get top-N anchors by wins.
    /// Returns Vec of (anchor_id, wins, v, stable).
    pub fn top_n_by_wins(&self, n: usize) -> Vec<(u16, u32, f32, bool)> {
        let mut anchors_with_wins: Vec<(u16, &Anchor)> = self
            .anchors
            .iter()
            .filter(|(_, a)| a.alive)
            .map(|(&id, a)| (id, a))
            .collect();

        // Sort by wins descending
        anchors_with_wins.sort_by(|a, b| b.1.wins.cmp(&a.1.wins));

        anchors_with_wins
            .iter()
            .take(n)
            .map(|(id, a)| (*id, a.wins, a.v, a.stable))
            .collect()
    }
}

// =============================================================================
// Anchor+Mask Keyed Memory with remap support
// =============================================================================

/// Memory key combining anchor ID and learned mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryKey {
    pub anchor_id: u16,
    pub learned_mask: u64,
}

impl MemoryKey {
    pub fn new(anchor_id: u16, learned_mask: u64) -> Self {
        MemoryKey {
            anchor_id,
            learned_mask,
        }
    }
}

/// Label statistics for a memory key.
#[derive(Clone, Debug)]
pub struct LabelStats {
    /// Counts per label.
    counts: HashMap<u16, u32>,
    /// Total observations.
    total: u32,
}

impl LabelStats {
    pub fn new() -> Self {
        LabelStats {
            counts: HashMap::new(),
            total: 0,
        }
    }

    /// Record an observation of a label.
    pub fn observe(&mut self, label: u16) {
        *self.counts.entry(label).or_insert(0) += 1;
        self.total += 1;
    }

    /// Merge another LabelStats into this one.
    pub fn merge(&mut self, other: &LabelStats) {
        for (&label, &count) in &other.counts {
            *self.counts.entry(label).or_insert(0) += count;
        }
        self.total += other.total;
    }

    /// Compute probability of a label using Dirichlet smoothing.
    pub fn probability(&self, label: u16, num_labels: usize, alpha: f64) -> f64 {
        let count = *self.counts.get(&label).unwrap_or(&0) as f64;
        (count + alpha) / (self.total as f64 + alpha * num_labels as f64)
    }

    /// Get top-2 labels by probability.
    pub fn top2(&self, num_labels: usize, alpha: f64) -> [(u16, f64); 2] {
        let mut best = (0u16, 0.0f64);
        let mut second = (0u16, 0.0f64);

        for &label in self.counts.keys() {
            let p = self.probability(label, num_labels, alpha);
            if p > best.1 {
                second = best;
                best = (label, p);
            } else if p > second.1 {
                second = (label, p);
            }
        }

        [best, second]
    }

    #[allow(dead_code)]
    pub fn total(&self) -> u32 {
        self.total
    }
}

/// Recall decision from keyed memory.
#[derive(Clone, Debug, PartialEq)]
pub enum KeyedRecallDecision {
    Label(u16, f64),
    Abstain(KeyedAbstainReason),
}

/// Reason for abstaining in keyed recall.
#[derive(Clone, Debug, PartialEq)]
pub enum KeyedAbstainReason {
    NoEntry,
    LowConfidence,
    InsufficientMargin,
    NoAnchor, // Phase 1.6: gate blocked anchor creation
}

/// Configuration for keyed memory recall.
#[derive(Clone, Debug)]
pub struct KeyedMemoryConfig {
    pub label_min_p: f64,
    pub label_margin: f64,
    pub alpha: f64,
    pub num_labels: usize,
}

impl Default for KeyedMemoryConfig {
    fn default() -> Self {
        KeyedMemoryConfig {
            label_min_p: 0.6,
            label_margin: 0.15,
            alpha: 1.0,
            num_labels: 3,
        }
    }
}

/// Keyed memory store using Anchor+Mask addressing.
/// O(1) lookup - no similarity search required.
/// Phase 1.6b: Supports anchor ID remapping after merges.
pub struct KeyedMemoryStore {
    entries: HashMap<MemoryKey, LabelStats>,
    config: KeyedMemoryConfig,
    total_stores: usize,
    total_recalls: usize,
    recalls_with_entry: usize,
    /// Count of keys remapped due to anchor merges.
    keys_remapped: usize,
}

impl KeyedMemoryStore {
    pub fn new(config: KeyedMemoryConfig) -> Self {
        KeyedMemoryStore {
            entries: HashMap::new(),
            config,
            total_stores: 0,
            total_recalls: 0,
            recalls_with_entry: 0,
            keys_remapped: 0,
        }
    }

    /// Store a label observation for a key.
    pub fn store(&mut self, key: MemoryKey, label: u16) {
        // Skip invalid anchor IDs (gate blocked)
        if key.anchor_id == 0xFFFF {
            return;
        }
        self.entries
            .entry(key)
            .or_insert_with(LabelStats::new)
            .observe(label);
        self.total_stores += 1;
    }

    /// Recall a label for a key with abstention logic.
    pub fn recall(&mut self, key: MemoryKey) -> KeyedRecallDecision {
        self.total_recalls += 1;

        // Handle gate-blocked anchor
        if key.anchor_id == 0xFFFF {
            return KeyedRecallDecision::Abstain(KeyedAbstainReason::NoAnchor);
        }

        let stats = match self.entries.get(&key) {
            None => return KeyedRecallDecision::Abstain(KeyedAbstainReason::NoEntry),
            Some(s) => s,
        };

        self.recalls_with_entry += 1;

        let top2 = stats.top2(self.config.num_labels, self.config.alpha);
        let (best_label, best_p) = top2[0];
        let (_, second_p) = top2[1];

        if best_p < self.config.label_min_p {
            return KeyedRecallDecision::Abstain(KeyedAbstainReason::LowConfidence);
        }

        let margin = best_p - second_p;
        if margin < self.config.label_margin {
            return KeyedRecallDecision::Abstain(KeyedAbstainReason::InsufficientMargin);
        }

        KeyedRecallDecision::Label(best_label, best_p)
    }

    /// Phase 1.6b: Remap anchor IDs after a merge.
    /// Moves all entries with old_anchor_id to new_anchor_id.
    pub fn remap_anchor(&mut self, old_id: u16, new_id: u16) {
        // Collect keys to remap
        let keys_to_remap: Vec<MemoryKey> = self
            .entries
            .keys()
            .filter(|k| k.anchor_id == old_id)
            .copied()
            .collect();

        for old_key in keys_to_remap {
            if let Some(old_stats) = self.entries.remove(&old_key) {
                let new_key = MemoryKey::new(new_id, old_key.learned_mask);

                // Merge into existing entry or insert
                self.entries
                    .entry(new_key)
                    .and_modify(|s| s.merge(&old_stats))
                    .or_insert(old_stats);

                self.keys_remapped += 1;
            }
        }
    }

    /// Apply a batch of remappings.
    pub fn apply_remaps(&mut self, remaps: &[(u16, u16)]) {
        for &(old_id, new_id) in remaps {
            self.remap_anchor(old_id, new_id);
        }
    }

    pub fn num_keys(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn total_stores(&self) -> usize {
        self.total_stores
    }

    pub fn entry_hit_rate(&self) -> f64 {
        if self.total_recalls == 0 {
            0.0
        } else {
            self.recalls_with_entry as f64 / self.total_recalls as f64
        }
    }

    pub fn keys_remapped(&self) -> usize {
        self.keys_remapped
    }
}

// =============================================================================
// Metrics for Phase 1.5b/1.6
// =============================================================================

/// Metrics tracker for keyed memory experiments.
pub struct KeyedMemoryMetrics {
    pub pos_queries: usize,
    pub pos_answered: usize,
    pub pos_correct: usize,
    pub pos_wrong: usize,
    pub pos_abstain: usize,

    pub neg_queries: usize,
    pub neg_abstain: usize,
    pub neg_false_positive: usize,

    pub abstain_no_entry: usize,
    pub abstain_low_conf: usize,
    pub abstain_margin: usize,
    pub abstain_no_anchor: usize, // Phase 1.6: gate blocked
}

impl KeyedMemoryMetrics {
    pub fn new() -> Self {
        KeyedMemoryMetrics {
            pos_queries: 0,
            pos_answered: 0,
            pos_correct: 0,
            pos_wrong: 0,
            pos_abstain: 0,
            neg_queries: 0,
            neg_abstain: 0,
            neg_false_positive: 0,
            abstain_no_entry: 0,
            abstain_low_conf: 0,
            abstain_margin: 0,
            abstain_no_anchor: 0,
        }
    }

    pub fn record_positive(&mut self, decision: &KeyedRecallDecision, true_label: u16) {
        self.pos_queries += 1;

        match decision {
            KeyedRecallDecision::Label(label, _) => {
                self.pos_answered += 1;
                if *label == true_label {
                    self.pos_correct += 1;
                } else {
                    self.pos_wrong += 1;
                }
            }
            KeyedRecallDecision::Abstain(reason) => {
                self.pos_abstain += 1;
                self.record_abstain_reason(reason);
            }
        }
    }

    pub fn record_negative(&mut self, decision: &KeyedRecallDecision) {
        self.neg_queries += 1;

        match decision {
            KeyedRecallDecision::Label(_, _) => {
                self.neg_false_positive += 1;
            }
            KeyedRecallDecision::Abstain(reason) => {
                self.neg_abstain += 1;
                self.record_abstain_reason(reason);
            }
        }
    }

    fn record_abstain_reason(&mut self, reason: &KeyedAbstainReason) {
        match reason {
            KeyedAbstainReason::NoEntry => self.abstain_no_entry += 1,
            KeyedAbstainReason::LowConfidence => self.abstain_low_conf += 1,
            KeyedAbstainReason::InsufficientMargin => self.abstain_margin += 1,
            KeyedAbstainReason::NoAnchor => self.abstain_no_anchor += 1,
        }
    }

    pub fn coverage_pos(&self) -> f64 {
        if self.pos_queries == 0 {
            0.0
        } else {
            self.pos_answered as f64 / self.pos_queries as f64
        }
    }

    pub fn accuracy_pos(&self) -> f64 {
        if self.pos_answered == 0 {
            0.0
        } else {
            self.pos_correct as f64 / self.pos_answered as f64
        }
    }

    pub fn abstain_neg_rate(&self) -> f64 {
        if self.neg_queries == 0 {
            0.0
        } else {
            self.neg_abstain as f64 / self.neg_queries as f64
        }
    }

    pub fn false_positive_rate(&self) -> f64 {
        if self.neg_queries == 0 {
            0.0
        } else {
            self.neg_false_positive as f64 / self.neg_queries as f64
        }
    }

    pub fn selective_accuracy(&self) -> f64 {
        let total = self.pos_queries + self.neg_queries;
        if total == 0 {
            return 0.0;
        }
        let correct = self.pos_correct + self.neg_abstain;
        correct as f64 / total as f64
    }

    pub fn print(&self) {
        println!("Keyed Memory Metrics (Phase 1.6):");
        println!();
        println!("  Positive Queries:");
        println!(
            "    total={}, answered={}, correct={}, wrong={}, abstain={}",
            self.pos_queries, self.pos_answered, self.pos_correct, self.pos_wrong, self.pos_abstain
        );
        println!(
            "    coverage_pos={:.1}%, accuracy_pos={:.1}%",
            self.coverage_pos() * 100.0,
            self.accuracy_pos() * 100.0
        );
        println!();
        println!("  Negative Queries:");
        println!(
            "    total={}, abstain={}, false_positive={}",
            self.neg_queries, self.neg_abstain, self.neg_false_positive
        );
        println!(
            "    abstain_neg={:.1}%, false_positive_rate={:.1}%",
            self.abstain_neg_rate() * 100.0,
            self.false_positive_rate() * 100.0
        );
        println!();
        println!("  Overall:");
        println!(
            "    selective_accuracy={:.1}%",
            self.selective_accuracy() * 100.0
        );
        println!();
        println!("  Abstain Breakdown:");
        let total_abstain = self.pos_abstain + self.neg_abstain;
        if total_abstain > 0 {
            println!(
                "    no_entry={} ({:.1}%)",
                self.abstain_no_entry,
                self.abstain_no_entry as f64 / total_abstain as f64 * 100.0
            );
            println!(
                "    low_confidence={} ({:.1}%)",
                self.abstain_low_conf,
                self.abstain_low_conf as f64 / total_abstain as f64 * 100.0
            );
            println!(
                "    insufficient_margin={} ({:.1}%)",
                self.abstain_margin,
                self.abstain_margin as f64 / total_abstain as f64 * 100.0
            );
            println!(
                "    no_anchor (gate blocked)={} ({:.1}%)",
                self.abstain_no_anchor,
                self.abstain_no_anchor as f64 / total_abstain as f64 * 100.0
            );
        }
    }
}
