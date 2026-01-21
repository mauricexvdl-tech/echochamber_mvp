//! AnchorBank: Scalable memory addressing via stable anchor prototypes.
//! Phase 1.5b: Replace expensive similarity search with O(1) key lookup.
//! Phase 1.6: Anchor codebook stabilization with merge, gating, and instrumentation.
//! Phase 1.7a: Prototype vectors - turn anchors into "concept tokens" with online learning.
//! Phase 1.7b: Value learning - self-supervised credit assignment via TD(0).

use std::collections::HashMap;
use crate::config::Config;

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
        }
    }

    /// Compute Hamming distance to another signature.
    pub fn hamming_to(&self, signature: u64) -> u32 {
        (self.proto_signature ^ signature).count_ones()
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
}

// =============================================================================
// Confidence gate info for anchor creation
// =============================================================================

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
        ConfidenceInfo { topk_margin, total_power }
    }

    /// Check if confidence passes the gate for anchor creation.
    pub fn passes_gate(&self) -> bool {
        self.topk_margin >= ANCHOR_MARGIN_MIN
            && self.total_power >= ANCHOR_MIN_POWER
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
        self.resolve_gated(signature, current_tick, None)
    }

    /// Resolve a signature to an anchor ID with optional confidence gating.
    /// If confidence_info is Some and fails gate, will not create new anchors.
    /// Returns (anchor_id, is_new_anchor, match_hamming).
    pub fn resolve_gated(
        &mut self,
        signature: u64,
        current_tick: u64,
        confidence_info: Option<&ConfidenceInfo>,
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
            self.prune_one(current_tick);
        }

        let new_id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.anchors.insert(new_id, Anchor::new(signature, current_tick));
        self.anchor_creates += 1;

        (new_id, true, 0)
    }

    /// Prune one anchor to make room for a new one.
    /// Strategy: Remove least-used anchor with usage < ANCHOR_MIN_USE,
    /// or oldest anchor if all are well-used.
    fn prune_one(&mut self, _current_tick: u64) {
        // First try: find least-used anchor below threshold
        let mut candidate: Option<(u16, u32, u64)> = None; // (id, usage, created_tick)

        for (&id, anchor) in &self.anchors {
            if !anchor.alive {
                continue;
            }
            if anchor.usage_count < ANCHOR_MIN_USE {
                match candidate {
                    None => candidate = Some((id, anchor.usage_count, anchor.created_at_tick)),
                    Some((_, best_usage, best_tick)) => {
                        if anchor.usage_count < best_usage
                            || (anchor.usage_count == best_usage
                                && anchor.created_at_tick < best_tick)
                        {
                            candidate = Some((id, anchor.usage_count, anchor.created_at_tick));
                        }
                    }
                }
            }
        }

        // If no low-usage candidate, find oldest
        if candidate.is_none() {
            for (&id, anchor) in &self.anchors {
                if !anchor.alive {
                    continue;
                }
                match candidate {
                    None => candidate = Some((id, anchor.usage_count, anchor.created_at_tick)),
                    Some((_, _, best_tick)) => {
                        if anchor.created_at_tick < best_tick {
                            candidate = Some((id, anchor.usage_count, anchor.created_at_tick));
                        }
                    }
                }
            }
        }

        if let Some((id, _, _)) = candidate {
            if let Some(anchor) = self.anchors.get_mut(&id) {
                anchor.alive = false;
            }
            self.anchor_evictions += 1;
        }
    }

    /// Phase 1.6b: Attempt to merge similar anchors.
    /// Returns Vec of (old_id, new_id) remappings that occurred.
    pub fn merge_similar(&mut self) -> Vec<(u16, u16)> {
        let mut remaps: Vec<(u16, u16)> = Vec::new();

        // Collect alive anchor IDs
        let ids: Vec<u16> = self.anchors
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

                let dist = {
                    let a = &self.anchors[&id_low];
                    let b = &self.anchors[&id_high];
                    (a.proto_signature ^ b.proto_signature).count_ones()
                };

                if dist <= MERGE_HAMMING {
                    // Merge id_low into id_high (higher usage keeps ID)
                    let usage_low = self.anchors[&id_low].usage_count;

                    // Mark id_low as dead
                    self.anchors.get_mut(&id_low).unwrap().alive = false;

                    // Add usage to id_high
                    self.anchors.get_mut(&id_high).unwrap().usage_count += usage_low;

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
        println!("  anchors_used={} / {} ({:.1}% utilization)",
            self.anchors_used(), MAX_ANCHORS, self.anchor_utilization() * 100.0);
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
        let alive_anchors: Vec<&Anchor> = self.anchors.values()
            .filter(|a| a.alive)
            .collect();

        if alive_anchors.is_empty() {
            return 0.0;
        }

        let total_support: u32 = alive_anchors.iter()
            .map(|a| a.proto_support)
            .sum();

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
        let mut anchors_by_usage: Vec<(&Anchor, u32)> = self.anchors.values()
            .filter(|a| a.alive && a.proto_support > 0)
            .map(|a| (a, a.usage_count))
            .collect();

        // Sort by usage count descending
        anchors_by_usage.sort_by(|a, b| b.1.cmp(&a.1));

        // Take top N
        let top_n: Vec<&Anchor> = anchors_by_usage.iter()
            .take(n)
            .map(|(a, _)| *a)
            .collect();

        if top_n.is_empty() {
            return (0.0, 0);
        }

        let total_entropy: f32 = top_n.iter()
            .map(|a| a.entropy(proto_m))
            .sum();

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
        self.anchors.get(&resolved_id)
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
        let used_anchors: Vec<&Anchor> = self.anchors.values()
            .filter(|a| a.alive && a.v_updates > 0)
            .collect();

        if used_anchors.is_empty() {
            return (0, 0.0, 0.0);
        }

        let total_updates: usize = used_anchors.iter()
            .map(|a| a.v_updates as usize)
            .sum();

        let avg_v: f64 = used_anchors.iter()
            .map(|a| a.v as f64)
            .sum::<f64>() / used_anchors.len() as f64;

        let avg_abs_td: f64 = used_anchors.iter()
            .map(|a| a.v_ema_abs_td as f64)
            .sum::<f64>() / used_anchors.len() as f64;

        (total_updates, avg_v, avg_abs_td)
    }

    /// Get top-N anchors by value.
    /// Returns Vec of (anchor_id, v, entropy_ema, proto_support, v_updates).
    pub fn top_n_by_value(&self, n: usize, proto_m: usize) -> Vec<(u16, f32, f32, u32, u32)> {
        let mut anchors_with_v: Vec<(u16, &Anchor)> = self.anchors.iter()
            .filter(|(_, a)| a.alive && a.v_updates > 0)
            .map(|(&id, a)| (id, a))
            .collect();

        // Sort by value descending
        anchors_with_v.sort_by(|a, b| b.1.v.partial_cmp(&a.1.v).unwrap_or(std::cmp::Ordering::Equal));

        anchors_with_v.iter()
            .take(n)
            .map(|(id, a)| (*id, a.v, a.entropy(proto_m), a.proto_support, a.v_updates))
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
        MemoryKey { anchor_id, learned_mask }
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
        self.entries.entry(key).or_insert_with(LabelStats::new).observe(label);
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
        let keys_to_remap: Vec<MemoryKey> = self.entries
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
        if self.pos_queries == 0 { 0.0 } else { self.pos_answered as f64 / self.pos_queries as f64 }
    }

    pub fn accuracy_pos(&self) -> f64 {
        if self.pos_answered == 0 { 0.0 } else { self.pos_correct as f64 / self.pos_answered as f64 }
    }

    pub fn abstain_neg_rate(&self) -> f64 {
        if self.neg_queries == 0 { 0.0 } else { self.neg_abstain as f64 / self.neg_queries as f64 }
    }

    pub fn false_positive_rate(&self) -> f64 {
        if self.neg_queries == 0 { 0.0 } else { self.neg_false_positive as f64 / self.neg_queries as f64 }
    }

    pub fn selective_accuracy(&self) -> f64 {
        let total = self.pos_queries + self.neg_queries;
        if total == 0 { return 0.0; }
        let correct = self.pos_correct + self.neg_abstain;
        correct as f64 / total as f64
    }

    pub fn print(&self) {
        println!("Keyed Memory Metrics (Phase 1.6):");
        println!();
        println!("  Positive Queries:");
        println!("    total={}, answered={}, correct={}, wrong={}, abstain={}",
            self.pos_queries, self.pos_answered, self.pos_correct, self.pos_wrong, self.pos_abstain);
        println!("    coverage_pos={:.1}%, accuracy_pos={:.1}%",
            self.coverage_pos() * 100.0, self.accuracy_pos() * 100.0);
        println!();
        println!("  Negative Queries:");
        println!("    total={}, abstain={}, false_positive={}",
            self.neg_queries, self.neg_abstain, self.neg_false_positive);
        println!("    abstain_neg={:.1}%, false_positive_rate={:.1}%",
            self.abstain_neg_rate() * 100.0, self.false_positive_rate() * 100.0);
        println!();
        println!("  Overall:");
        println!("    selective_accuracy={:.1}%", self.selective_accuracy() * 100.0);
        println!();
        println!("  Abstain Breakdown:");
        let total_abstain = self.pos_abstain + self.neg_abstain;
        if total_abstain > 0 {
            println!("    no_entry={} ({:.1}%)", self.abstain_no_entry,
                self.abstain_no_entry as f64 / total_abstain as f64 * 100.0);
            println!("    low_confidence={} ({:.1}%)", self.abstain_low_conf,
                self.abstain_low_conf as f64 / total_abstain as f64 * 100.0);
            println!("    insufficient_margin={} ({:.1}%)", self.abstain_margin,
                self.abstain_margin as f64 / total_abstain as f64 * 100.0);
            println!("    no_anchor (gate blocked)={} ({:.1}%)", self.abstain_no_anchor,
                self.abstain_no_anchor as f64 / total_abstain as f64 * 100.0);
        }
    }
}
