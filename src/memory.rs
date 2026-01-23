//! Episodic memory for one-shot label binding.
//! Phase 1.4a: Store label events and recall via similarity matching.
//! Phase 1.4b: Added LabelMemoryStore for independent label binding.
//! Phase 1.4c: Added GlobalLabelMemoryStore with ABSTAIN, competition, and windowed signatures.

use crate::config::Config;

/// Scale factor for converting f64 proto scores to i32
const PROTO_SCALE: i32 = 1000;

/// Maximum candidates to allow before ambiguity gate triggers (Phase 1.4c tuning).
const MAX_CANDIDATES: usize = 16;

/// Window size for rolling signature (Phase 1.4c windowed signatures).
pub const WINDOW_SIZE: usize = 32;

/// Number of top nodes to include in windowed signature mask.
pub const WINDOW_TOP_M: usize = 12;

/// A single memory entry binding a label to a network state.
#[derive(Clone, Debug)]
pub struct MemoryEntry {
    pub label_id: u32,
    pub tick: u64,
    pub topk_mask: u64,
    pub proto_scores: [i32; 3],
}

/// Result of a memory recall operation.
#[derive(Clone, Debug)]
pub struct RecallResult {
    pub label_id: u32,
    pub score: i32,
    pub hamming_dist: u32,
    pub proto_dist: i32,
    pub age: u64,
}

/// Episodic memory store for label binding.
pub struct MemoryStore {
    entries: Vec<MemoryEntry>,
    max_entries: usize,
    max_hamming: u32,
    mask_weight: i32,
    proto_weight: i32,
    age_weight: i32,
    min_score: i32,
    last_store_tick: Vec<u64>,
}

impl MemoryStore {
    pub fn new(config: &Config) -> Self {
        MemoryStore {
            entries: Vec::with_capacity(config.memory_max_entries),
            max_entries: config.memory_max_entries,
            max_hamming: config.memory_max_hamming,
            mask_weight: config.memory_mask_w,
            proto_weight: config.memory_proto_w,
            age_weight: config.memory_age_w,
            min_score: config.memory_min_score,
            last_store_tick: vec![0; config.num_ctx + 1],
        }
    }

    pub fn store(
        &mut self,
        label_id: u32,
        tick: u64,
        topk_mask: u64,
        proto_scores: [i32; 3],
        debounce_ticks: u64,
    ) -> bool {
        let label_idx = label_id as usize;

        if label_idx < self.last_store_tick.len() {
            let last = self.last_store_tick[label_idx];
            if tick > 0 && tick - last < debounce_ticks {
                return false;
            }
        }

        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }

        self.entries.push(MemoryEntry {
            label_id,
            tick,
            topk_mask,
            proto_scores,
        });

        if label_idx < self.last_store_tick.len() {
            self.last_store_tick[label_idx] = tick;
        }

        true
    }

    pub fn recall(
        &self,
        current_tick: u64,
        topk_mask: u64,
        proto_scores: [i32; 3],
    ) -> Option<RecallResult> {
        if self.entries.is_empty() {
            return None;
        }

        let mut best: Option<RecallResult> = None;
        let mut best_score = i32::MIN;

        for entry in &self.entries {
            let hamming_dist = (topk_mask ^ entry.topk_mask).count_ones();

            if hamming_dist > self.max_hamming {
                continue;
            }

            let proto_dist: i32 = (0..3)
                .map(|i| (proto_scores[i] - entry.proto_scores[i]).abs())
                .sum();

            let age = if current_tick >= entry.tick {
                current_tick - entry.tick
            } else {
                0
            };

            let score = -(self.mask_weight * hamming_dist as i32
                + self.proto_weight * proto_dist
                + self.age_weight * (age as i32).min(10000));

            if score > best_score && score >= self.min_score {
                best_score = score;
                best = Some(RecallResult {
                    label_id: entry.label_id,
                    score,
                    hamming_dist,
                    proto_dist,
                    age,
                });
            }
        }

        best
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries_per_label(&self) -> Vec<(u32, usize)> {
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.label_id).or_insert(0) += 1;
        }
        let mut result: Vec<(u32, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

/// Memory recall metrics tracker (Phase 1.4a).
pub struct MemoryMetrics {
    pub attempts: usize,
    pub hits: usize,
    pub correct: usize,
    pub false_recalls: usize,
    pub hamming_sum: u64,
    pub age_sum: u64,
    pub stores: usize,
}

impl MemoryMetrics {
    pub fn new() -> Self {
        MemoryMetrics {
            attempts: 0,
            hits: 0,
            correct: 0,
            false_recalls: 0,
            hamming_sum: 0,
            age_sum: 0,
            stores: 0,
        }
    }

    pub fn record_store(&mut self) {
        self.stores += 1;
    }

    pub fn record_recall(&mut self, result: Option<&RecallResult>, true_label: u32) {
        self.attempts += 1;

        if let Some(r) = result {
            self.hits += 1;
            self.hamming_sum += r.hamming_dist as u64;
            self.age_sum += r.age;

            if r.label_id == true_label {
                self.correct += 1;
            } else {
                self.false_recalls += 1;
            }
        }
    }

    pub fn coverage(&self) -> f64 {
        if self.attempts == 0 {
            0.0
        } else {
            self.hits as f64 / self.attempts as f64
        }
    }

    pub fn accuracy(&self) -> f64 {
        if self.hits == 0 {
            0.0
        } else {
            self.correct as f64 / self.hits as f64
        }
    }

    pub fn false_rate(&self) -> f64 {
        if self.hits == 0 {
            0.0
        } else {
            self.false_recalls as f64 / self.hits as f64
        }
    }

    pub fn avg_hamming(&self) -> f64 {
        if self.hits == 0 {
            0.0
        } else {
            self.hamming_sum as f64 / self.hits as f64
        }
    }

    pub fn avg_age(&self) -> f64 {
        if self.hits == 0 {
            0.0
        } else {
            self.age_sum as f64 / self.hits as f64
        }
    }

    pub fn print(&self) {
        println!("Memory Recall Metrics:");
        println!(
            "  attempts={}, hits={} (coverage={:.1}%)",
            self.attempts,
            self.hits,
            self.coverage() * 100.0
        );
        println!(
            "  acc@1={:.1}%, false_recall={:.1}%",
            self.accuracy() * 100.0,
            self.false_rate() * 100.0
        );
        println!(
            "  avg_hamming={:.2}, avg_age={:.1}",
            self.avg_hamming(),
            self.avg_age()
        );
        println!("  stores={}", self.stores);
    }
}

// =============================================================================
// Phase 1.4b: Label Memory Store (independent labels, not ctx-based)
// =============================================================================

#[derive(Clone, Debug)]
pub struct LabelEntry {
    pub label: u16,
    pub stored_tick: u64,
    pub signature: u64,
}

#[derive(Clone, Debug)]
pub struct LabelRecallResult {
    pub label: u16,
    pub hamming_dist: u32,
    pub age: u64,
}

pub struct LabelMemoryStore {
    entries: Vec<LabelEntry>,
    max_entries: usize,
    max_hamming: u32,
}

impl LabelMemoryStore {
    pub fn new(max_entries: usize, max_hamming: u32) -> Self {
        LabelMemoryStore {
            entries: Vec::with_capacity(max_entries),
            max_entries,
            max_hamming,
        }
    }

    pub fn from_config(config: &Config) -> Self {
        Self::new(
            config.label_memory_max_entries,
            config.label_memory_max_hamming,
        )
    }

    pub fn store(&mut self, label: u16, tick: u64, signature: u64) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(LabelEntry {
            label,
            stored_tick: tick,
            signature,
        });
    }

    pub fn recall(&self, current_tick: u64, signature: u64) -> Option<LabelRecallResult> {
        if self.entries.is_empty() {
            return None;
        }

        let mut best: Option<LabelRecallResult> = None;
        let mut best_hamming = u32::MAX;

        for entry in &self.entries {
            let hamming_dist = (signature ^ entry.signature).count_ones();
            if hamming_dist > self.max_hamming {
                continue;
            }
            if hamming_dist < best_hamming {
                best_hamming = hamming_dist;
                let age = current_tick.saturating_sub(entry.stored_tick);
                best = Some(LabelRecallResult {
                    label: entry.label,
                    hamming_dist,
                    age,
                });
            }
        }
        best
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn entries_per_label(&self, top_n: usize) -> Vec<(u16, usize)> {
        let mut counts: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.label).or_insert(0) += 1;
        }
        let mut result: Vec<(u16, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result.truncate(top_n);
        result
    }
}

pub struct LabelBindingMetrics {
    pub episodes: usize,
    pub binds_done: usize,
    pub recall_queries: usize,
    pub recall_hits: usize,
    pub recall_correct: usize,
    pub hamming_sum: u64,
    pub age_sum: u64,
    label_hits: std::collections::HashMap<u16, usize>,
    label_correct: std::collections::HashMap<u16, usize>,
}

impl LabelBindingMetrics {
    pub fn new() -> Self {
        LabelBindingMetrics {
            episodes: 0,
            binds_done: 0,
            recall_queries: 0,
            recall_hits: 0,
            recall_correct: 0,
            hamming_sum: 0,
            age_sum: 0,
            label_hits: std::collections::HashMap::new(),
            label_correct: std::collections::HashMap::new(),
        }
    }

    pub fn record_episode(&mut self) {
        self.episodes += 1;
    }
    pub fn record_bind(&mut self) {
        self.binds_done += 1;
    }

    pub fn record_recall_query(&mut self, result: Option<&LabelRecallResult>, true_label: u16) {
        self.recall_queries += 1;
        if let Some(r) = result {
            self.recall_hits += 1;
            self.hamming_sum += r.hamming_dist as u64;
            self.age_sum += r.age;
            *self.label_hits.entry(r.label).or_insert(0) += 1;
            if r.label == true_label {
                self.recall_correct += 1;
                *self.label_correct.entry(r.label).or_insert(0) += 1;
            }
        }
    }

    pub fn coverage(&self) -> f64 {
        if self.recall_queries == 0 {
            0.0
        } else {
            self.recall_hits as f64 / self.recall_queries as f64
        }
    }
    pub fn accuracy(&self) -> f64 {
        if self.recall_hits == 0 {
            0.0
        } else {
            self.recall_correct as f64 / self.recall_hits as f64
        }
    }
    pub fn false_rate(&self) -> f64 {
        if self.recall_hits == 0 {
            0.0
        } else {
            (self.recall_hits - self.recall_correct) as f64 / self.recall_hits as f64
        }
    }
    pub fn avg_hamming(&self) -> f64 {
        if self.recall_hits == 0 {
            0.0
        } else {
            self.hamming_sum as f64 / self.recall_hits as f64
        }
    }
    pub fn avg_age(&self) -> f64 {
        if self.recall_hits == 0 {
            0.0
        } else {
            self.age_sum as f64 / self.recall_hits as f64
        }
    }

    pub fn top_labels_by_hits(&self, n: usize) -> Vec<(u16, usize, usize)> {
        let mut result: Vec<(u16, usize, usize)> = self
            .label_hits
            .iter()
            .map(|(&label, &hits)| (label, hits, *self.label_correct.get(&label).unwrap_or(&0)))
            .collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result.truncate(n);
        result
    }

    pub fn print(&self) {
        println!("Label Binding Metrics:");
        println!(
            "  episodes={}, binds_done={}",
            self.episodes, self.binds_done
        );
        println!(
            "  recall_queries={}, recall_hits={} (coverage={:.1}%)",
            self.recall_queries,
            self.recall_hits,
            self.coverage() * 100.0
        );
        println!(
            "  recall_correct={}, accuracy@1={:.1}%, false_recall={:.1}%",
            self.recall_correct,
            self.accuracy() * 100.0,
            self.false_rate() * 100.0
        );
        println!(
            "  avg_hamming={:.2}, avg_age={:.1} ticks",
            self.avg_hamming(),
            self.avg_age()
        );
        let top5 = self.top_labels_by_hits(5);
        if !top5.is_empty() {
            print!("  top 5 labels by hits: ");
            for (label, hits, correct) in &top5 {
                let acc = if *hits > 0 {
                    *correct as f64 / *hits as f64 * 100.0
                } else {
                    0.0
                };
                print!("L{}({},{:.0}%) ", label, hits, acc);
            }
            println!();
        }
    }
}

// =============================================================================
// Phase 1.4c: Rolling Window for Anchored Signatures
// =============================================================================

/// Rolling window for computing stable windowed signatures.
/// Maintains histogram of node frequencies and ctx votes over WINDOW_SIZE ticks.
pub struct RollingWindow {
    /// Ring buffer of TopK node sets (each as Vec<usize>)
    topk_buffer: Vec<Vec<usize>>,
    /// Ring buffer of ctx_hat values
    ctx_buffer: Vec<Option<u8>>,
    /// Current write position in ring buffer
    pos: usize,
    /// Number of valid entries (grows until WINDOW_SIZE)
    count: usize,
    /// Node frequency histogram (indexed by node_id)
    node_counts: Vec<u32>,
    /// Ctx frequency counts (indexed by ctx)
    ctx_counts: Vec<u32>,
    /// Number of nodes in the network
    num_nodes: usize,
    /// Number of ctx values
    num_ctx: usize,
    /// Stability tracking: how often ctx_hat == window mode
    ctx_stable_ticks: usize,
    ctx_total_ticks: usize,
}

impl RollingWindow {
    pub fn new(num_nodes: usize, num_ctx: usize) -> Self {
        RollingWindow {
            topk_buffer: vec![Vec::new(); WINDOW_SIZE],
            ctx_buffer: vec![None; WINDOW_SIZE],
            pos: 0,
            count: 0,
            node_counts: vec![0; num_nodes],
            ctx_counts: vec![0; num_ctx],
            num_nodes,
            num_ctx,
            ctx_stable_ticks: 0,
            ctx_total_ticks: 0,
        }
    }

    /// Push a new tick's TopK and ctx_hat into the window.
    pub fn push(&mut self, topk_ids: &[usize], ctx_hat: Option<u8>) {
        // Remove old entry's contributions
        if self.count == WINDOW_SIZE {
            let old_topk = &self.topk_buffer[self.pos];
            for &node_id in old_topk {
                if node_id < self.num_nodes {
                    self.node_counts[node_id] = self.node_counts[node_id].saturating_sub(1);
                }
            }
            if let Some(old_ctx) = self.ctx_buffer[self.pos] {
                if (old_ctx as usize) < self.num_ctx {
                    self.ctx_counts[old_ctx as usize] =
                        self.ctx_counts[old_ctx as usize].saturating_sub(1);
                }
            }
        }

        // Add new entry's contributions
        self.topk_buffer[self.pos] = topk_ids.to_vec();
        self.ctx_buffer[self.pos] = ctx_hat;

        for &node_id in topk_ids {
            if node_id < self.num_nodes {
                self.node_counts[node_id] += 1;
            }
        }
        if let Some(ctx) = ctx_hat {
            if (ctx as usize) < self.num_ctx {
                self.ctx_counts[ctx as usize] += 1;
            }
        }

        // Track stability
        if let Some(ctx) = ctx_hat {
            self.ctx_total_ticks += 1;
            if Some(ctx) == self.ctx_mode() {
                self.ctx_stable_ticks += 1;
            }
        }

        // Advance position
        self.pos = (self.pos + 1) % WINDOW_SIZE;
        if self.count < WINDOW_SIZE {
            self.count += 1;
        }
    }

    /// Get the window mode ctx (most frequent ctx_hat).
    pub fn ctx_mode(&self) -> Option<u8> {
        if self.count == 0 {
            return None;
        }
        let mut best_ctx = 0u8;
        let mut best_count = 0u32;
        for (ctx, &count) in self.ctx_counts.iter().enumerate() {
            if count > best_count {
                best_count = count;
                best_ctx = ctx as u8;
            }
        }
        if best_count > 0 {
            Some(best_ctx)
        } else {
            None
        }
    }

    /// Compute the windowed signature: top M most frequent nodes as bitmask.
    pub fn signature_mask(&self) -> u64 {
        if self.count == 0 {
            return 0;
        }

        // Get top M nodes by frequency
        let mut node_freqs: Vec<(usize, u32)> = self
            .node_counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(id, &c)| (id, c))
            .collect();

        node_freqs.sort_by(|a, b| b.1.cmp(&a.1)); // descending by count
        node_freqs.truncate(WINDOW_TOP_M);

        // Convert to bitmask
        let mut mask = 0u64;
        for (node_id, _) in node_freqs {
            if node_id < 64 {
                mask |= 1u64 << node_id;
            }
        }
        mask
    }

    /// Get the full competitive signature (ctx + mask).
    pub fn competitive_sig(&self) -> CompetitiveSig {
        CompetitiveSig {
            ctx_hat: self.ctx_mode(),
            mask: self.signature_mask(),
        }
    }

    /// Get stability ratio: fraction of ticks where ctx_hat matched window mode.
    pub fn stability(&self) -> f64 {
        if self.ctx_total_ticks == 0 {
            0.0
        } else {
            self.ctx_stable_ticks as f64 / self.ctx_total_ticks as f64
        }
    }

    /// Check if window is fully populated.
    pub fn is_ready(&self) -> bool {
        self.count == WINDOW_SIZE
    }

    /// Reset the window (e.g., at episode boundaries).
    pub fn reset(&mut self) {
        self.topk_buffer = vec![Vec::new(); WINDOW_SIZE];
        self.ctx_buffer = vec![None; WINDOW_SIZE];
        self.pos = 0;
        self.count = 0;
        self.node_counts = vec![0; self.num_nodes];
        self.ctx_counts = vec![0; self.num_ctx];
        // Don't reset stability counters - keep cumulative
    }
}

/// Competitive signature combining ctx anchor and windowed mask.
#[derive(Clone, Debug, PartialEq)]
pub struct CompetitiveSig {
    pub ctx_hat: Option<u8>,
    pub mask: u64,
}

impl CompetitiveSig {
    /// Compute Hamming distance on mask (only if ctx matches).
    /// Returns None if ctx mismatch.
    pub fn distance(&self, other: &CompetitiveSig) -> Option<u32> {
        match (self.ctx_hat, other.ctx_hat) {
            (Some(a), Some(b)) if a == b => Some((self.mask ^ other.mask).count_ones()),
            (None, _) | (_, None) => Some((self.mask ^ other.mask).count_ones()), // allow if either unknown
            _ => None,                                                            // ctx mismatch
        }
    }
}

// =============================================================================
// Phase 1.4c: Global Label Memory with ABSTAIN, Competition, and Anchored Signatures
// =============================================================================

/// Entry in global label memory with LRU tracking and competitive signature.
#[derive(Clone, Debug)]
pub struct GlobalLabelEntry {
    pub label: u16,
    pub signature: CompetitiveSig,
    pub created_tick: u64,
    pub last_hit_tick: u64,
}

/// Recall decision: either a label or abstain (unknown).
#[derive(Clone, Debug, PartialEq)]
pub enum RecallDecision {
    Label(u16, u32), // (label, hamming_dist)
    Unknown,
}

/// Reason for abstaining.
#[derive(Clone, Debug, PartialEq)]
pub enum AbstainReason {
    NoCandidates,
    InsufficientMargin,
    TooManyCandidates,
    CtxMismatch,
    WindowNotReady,
}

/// Detailed recall result with collision info.
#[derive(Clone, Debug)]
pub struct GlobalRecallResult {
    pub decision: RecallDecision,
    pub candidates_in_radius: usize,
    pub candidates_after_ctx_filter: usize,
    pub best_dist: u32,
    pub second_dist: Option<u32>,
    pub margin: Option<u32>,
    pub abstain_reason: Option<AbstainReason>,
}

/// Global label memory store with FIFO eviction, ABSTAIN support, and ctx filtering.
pub struct GlobalLabelMemoryStore {
    entries: Vec<GlobalLabelEntry>,
    max_entries: usize,
    max_hamming: u32,
    margin_min: u32,
    evictions: usize,
    hit_updates: usize,
}

impl GlobalLabelMemoryStore {
    pub fn new(max_entries: usize, max_hamming: u32, margin_min: u32) -> Self {
        GlobalLabelMemoryStore {
            entries: Vec::with_capacity(max_entries),
            max_entries,
            max_hamming,
            margin_min,
            evictions: 0,
            hit_updates: 0,
        }
    }

    pub fn from_config(config: &Config) -> Self {
        Self::new(
            config.competitive_max_entries,
            config.competitive_max_hamming,
            config.competitive_margin_min,
        )
    }

    /// Store a new label binding with competitive signature and FIFO eviction.
    pub fn store_competitive(&mut self, label: u16, tick: u64, signature: CompetitiveSig) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
            self.evictions += 1;
        }
        self.entries.push(GlobalLabelEntry {
            label,
            signature,
            created_tick: tick,
            last_hit_tick: tick,
        });
    }

    /// Store with simple u64 mask (backward compatibility).
    pub fn store(&mut self, label: u16, tick: u64, signature: u64) {
        self.store_competitive(
            label,
            tick,
            CompetitiveSig {
                ctx_hat: None,
                mask: signature,
            },
        );
    }

    /// Recall with ABSTAIN based on margin, ambiguity gating, and ctx filtering.
    pub fn recall_competitive(
        &mut self,
        current_tick: u64,
        query_sig: &CompetitiveSig,
    ) -> GlobalRecallResult {
        if self.entries.is_empty() {
            return GlobalRecallResult {
                decision: RecallDecision::Unknown,
                candidates_in_radius: 0,
                candidates_after_ctx_filter: 0,
                best_dist: u32::MAX,
                second_dist: None,
                margin: None,
                abstain_reason: Some(AbstainReason::NoCandidates),
            };
        }

        // Collect all candidates: first filter by ctx, then by hamming distance
        let mut candidates: Vec<(usize, u32, u16)> = Vec::new(); // (index, dist, label)
        let mut total_ctx_matched = 0usize;

        for (idx, entry) in self.entries.iter().enumerate() {
            // Check ctx match and compute distance
            if let Some(dist) = query_sig.distance(&entry.signature) {
                total_ctx_matched += 1;
                if dist <= self.max_hamming {
                    candidates.push((idx, dist, entry.label));
                }
            }
        }

        let candidates_in_radius = candidates.len();

        // Gate 1: No candidates after ctx filter
        if candidates.is_empty() {
            let reason = if total_ctx_matched == 0
                && self.entries.iter().any(|e| e.signature.ctx_hat.is_some())
            {
                AbstainReason::CtxMismatch
            } else {
                AbstainReason::NoCandidates
            };
            return GlobalRecallResult {
                decision: RecallDecision::Unknown,
                candidates_in_radius: 0,
                candidates_after_ctx_filter: total_ctx_matched,
                best_dist: u32::MAX,
                second_dist: None,
                margin: None,
                abstain_reason: Some(reason),
            };
        }

        // Gate 2: Too many candidates (ambiguity)
        if candidates_in_radius > MAX_CANDIDATES {
            candidates.sort_by_key(|c| c.1);
            let best_dist = candidates[0].1;
            let second_dist = if candidates.len() > 1 {
                Some(candidates[1].1)
            } else {
                None
            };
            let margin = second_dist.map(|sd| sd.saturating_sub(best_dist));

            return GlobalRecallResult {
                decision: RecallDecision::Unknown,
                candidates_in_radius,
                candidates_after_ctx_filter: total_ctx_matched,
                best_dist,
                second_dist,
                margin,
                abstain_reason: Some(AbstainReason::TooManyCandidates),
            };
        }

        // Sort by distance (ascending)
        candidates.sort_by_key(|c| c.1);

        let (best_idx, best_dist, best_label) = candidates[0];
        let second_dist = if candidates.len() > 1 {
            Some(candidates[1].1)
        } else {
            None
        };

        // Compute margin
        let margin = second_dist.map(|sd| sd.saturating_sub(best_dist));

        // Gate 3: Margin check (if there's a second candidate)
        if candidates.len() > 1 && margin.unwrap_or(u32::MAX) < self.margin_min {
            return GlobalRecallResult {
                decision: RecallDecision::Unknown,
                candidates_in_radius,
                candidates_after_ctx_filter: total_ctx_matched,
                best_dist,
                second_dist,
                margin,
                abstain_reason: Some(AbstainReason::InsufficientMargin),
            };
        }

        // Passed all gates: predict the label
        self.entries[best_idx].last_hit_tick = current_tick;
        self.hit_updates += 1;

        GlobalRecallResult {
            decision: RecallDecision::Label(best_label, best_dist),
            candidates_in_radius,
            candidates_after_ctx_filter: total_ctx_matched,
            best_dist,
            second_dist,
            margin,
            abstain_reason: None,
        }
    }

    /// Backward compatible recall (without competitive sig).
    pub fn recall_or_abstain(&mut self, current_tick: u64, signature: u64) -> GlobalRecallResult {
        self.recall_competitive(
            current_tick,
            &CompetitiveSig {
                ctx_hat: None,
                mask: signature,
            },
        )
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn evictions(&self) -> usize {
        self.evictions
    }
    pub fn hit_updates(&self) -> usize {
        self.hit_updates
    }

    pub fn entries_per_label(&self, top_n: usize) -> Vec<(u16, usize)> {
        let mut counts: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.label).or_insert(0) += 1;
        }
        let mut result: Vec<(u16, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result.truncate(top_n);
        result
    }
}

/// Metrics for Phase 1.4c competitive label binding with windowed signatures.
pub struct GlobalLabelMetrics {
    // Positive queries
    pub pos_queries: usize,
    pub pos_non_abstain: usize,
    pub pos_correct: usize,
    pub pos_wrong: usize,
    pub pos_abstain: usize,

    // Negative queries
    pub neg_queries: usize,
    pub neg_abstain: usize,
    pub neg_false_positive: usize,

    // Abstain reason breakdown
    pub abstain_no_candidates: usize,
    pub abstain_margin: usize,
    pub abstain_too_many: usize,
    pub abstain_ctx_mismatch: usize,
    pub abstain_window_not_ready: usize,

    // Candidate distribution buckets: [0, 1-2, 3-5, 6-8, >8]
    pub bucket_0: usize,
    pub bucket_1_2: usize,
    pub bucket_3_5: usize,
    pub bucket_6_8: usize,
    pub bucket_gt8: usize,

    // Collision metrics
    pub total_candidates_sum: usize,
    pub total_ctx_matched_sum: usize,
    pub queries_with_2plus_candidates: usize,
    pub margin_sum: u64,
    pub margin_count: usize,
    pub margin_min_seen: u32,
    pub margin_max_seen: u32,

    // Hamming stats
    pub hamming_sum: u64,
    pub hamming_count: usize,

    // Window/stability stats
    pub window_size: usize,
    pub window_top_m: usize,
}

impl GlobalLabelMetrics {
    pub fn new() -> Self {
        GlobalLabelMetrics {
            pos_queries: 0,
            pos_non_abstain: 0,
            pos_correct: 0,
            pos_wrong: 0,
            pos_abstain: 0,
            neg_queries: 0,
            neg_abstain: 0,
            neg_false_positive: 0,
            abstain_no_candidates: 0,
            abstain_margin: 0,
            abstain_too_many: 0,
            abstain_ctx_mismatch: 0,
            abstain_window_not_ready: 0,
            bucket_0: 0,
            bucket_1_2: 0,
            bucket_3_5: 0,
            bucket_6_8: 0,
            bucket_gt8: 0,
            total_candidates_sum: 0,
            total_ctx_matched_sum: 0,
            queries_with_2plus_candidates: 0,
            margin_sum: 0,
            margin_count: 0,
            margin_min_seen: u32::MAX,
            margin_max_seen: 0,
            hamming_sum: 0,
            hamming_count: 0,
            window_size: WINDOW_SIZE,
            window_top_m: WINDOW_TOP_M,
        }
    }

    fn record_candidate_bucket(&mut self, n: usize) {
        match n {
            0 => self.bucket_0 += 1,
            1..=2 => self.bucket_1_2 += 1,
            3..=5 => self.bucket_3_5 += 1,
            6..=8 => self.bucket_6_8 += 1,
            _ => self.bucket_gt8 += 1,
        }
    }

    fn record_abstain_reason(&mut self, reason: &Option<AbstainReason>) {
        if let Some(r) = reason {
            match r {
                AbstainReason::NoCandidates => self.abstain_no_candidates += 1,
                AbstainReason::InsufficientMargin => self.abstain_margin += 1,
                AbstainReason::TooManyCandidates => self.abstain_too_many += 1,
                AbstainReason::CtxMismatch => self.abstain_ctx_mismatch += 1,
                AbstainReason::WindowNotReady => self.abstain_window_not_ready += 1,
            }
        }
    }

    /// Record a positive query result.
    pub fn record_positive(&mut self, result: &GlobalRecallResult, true_label: u16) {
        self.pos_queries += 1;
        self.total_candidates_sum += result.candidates_in_radius;
        self.total_ctx_matched_sum += result.candidates_after_ctx_filter;
        self.record_candidate_bucket(result.candidates_in_radius);

        if result.candidates_in_radius >= 2 {
            self.queries_with_2plus_candidates += 1;
        }

        if let Some(m) = result.margin {
            self.margin_sum += m as u64;
            self.margin_count += 1;
            self.margin_min_seen = self.margin_min_seen.min(m);
            self.margin_max_seen = self.margin_max_seen.max(m);
        }

        match &result.decision {
            RecallDecision::Label(label, dist) => {
                self.pos_non_abstain += 1;
                self.hamming_sum += *dist as u64;
                self.hamming_count += 1;
                if *label == true_label {
                    self.pos_correct += 1;
                } else {
                    self.pos_wrong += 1;
                }
            }
            RecallDecision::Unknown => {
                self.pos_abstain += 1;
                self.record_abstain_reason(&result.abstain_reason);
            }
        }
    }

    /// Record a negative query result (ground truth = unknown).
    pub fn record_negative(&mut self, result: &GlobalRecallResult) {
        self.neg_queries += 1;
        self.total_candidates_sum += result.candidates_in_radius;
        self.total_ctx_matched_sum += result.candidates_after_ctx_filter;
        self.record_candidate_bucket(result.candidates_in_radius);

        if result.candidates_in_radius >= 2 {
            self.queries_with_2plus_candidates += 1;
        }

        match &result.decision {
            RecallDecision::Label(_, _) => {
                self.neg_false_positive += 1;
            }
            RecallDecision::Unknown => {
                self.neg_abstain += 1;
                self.record_abstain_reason(&result.abstain_reason);
            }
        }
    }

    // Positive metrics
    pub fn coverage_pos(&self) -> f64 {
        if self.pos_queries == 0 {
            0.0
        } else {
            self.pos_non_abstain as f64 / self.pos_queries as f64
        }
    }

    pub fn accuracy_pos(&self) -> f64 {
        if self.pos_non_abstain == 0 {
            0.0
        } else {
            self.pos_correct as f64 / self.pos_non_abstain as f64
        }
    }

    pub fn abstain_pos_rate(&self) -> f64 {
        if self.pos_queries == 0 {
            0.0
        } else {
            self.pos_abstain as f64 / self.pos_queries as f64
        }
    }

    // Negative metrics
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

    // Overall selective accuracy
    pub fn selective_accuracy(&self) -> f64 {
        let total = self.pos_queries + self.neg_queries;
        if total == 0 {
            return 0.0;
        }
        let correct = self.pos_correct + self.neg_abstain;
        correct as f64 / total as f64
    }

    // Collision metrics
    pub fn avg_candidates(&self) -> f64 {
        let total = self.pos_queries + self.neg_queries;
        if total == 0 {
            0.0
        } else {
            self.total_candidates_sum as f64 / total as f64
        }
    }

    pub fn avg_ctx_matched(&self) -> f64 {
        let total = self.pos_queries + self.neg_queries;
        if total == 0 {
            0.0
        } else {
            self.total_ctx_matched_sum as f64 / total as f64
        }
    }

    pub fn pct_2plus_candidates(&self) -> f64 {
        let total = self.pos_queries + self.neg_queries;
        if total == 0 {
            0.0
        } else {
            self.queries_with_2plus_candidates as f64 / total as f64
        }
    }

    pub fn avg_margin(&self) -> f64 {
        if self.margin_count == 0 {
            0.0
        } else {
            self.margin_sum as f64 / self.margin_count as f64
        }
    }

    pub fn avg_hamming(&self) -> f64 {
        if self.hamming_count == 0 {
            0.0
        } else {
            self.hamming_sum as f64 / self.hamming_count as f64
        }
    }

    // Abstain breakdown percentages
    pub fn pct_abstain_margin(&self) -> f64 {
        let total_abstain = self.pos_abstain + self.neg_abstain;
        if total_abstain == 0 {
            0.0
        } else {
            self.abstain_margin as f64 / total_abstain as f64
        }
    }

    pub fn pct_abstain_too_many(&self) -> f64 {
        let total_abstain = self.pos_abstain + self.neg_abstain;
        if total_abstain == 0 {
            0.0
        } else {
            self.abstain_too_many as f64 / total_abstain as f64
        }
    }

    pub fn pct_abstain_no_candidates(&self) -> f64 {
        let total_abstain = self.pos_abstain + self.neg_abstain;
        if total_abstain == 0 {
            0.0
        } else {
            self.abstain_no_candidates as f64 / total_abstain as f64
        }
    }

    pub fn pct_abstain_ctx_mismatch(&self) -> f64 {
        let total_abstain = self.pos_abstain + self.neg_abstain;
        if total_abstain == 0 {
            0.0
        } else {
            self.abstain_ctx_mismatch as f64 / total_abstain as f64
        }
    }

    pub fn print(&self) {
        println!("Competitive Label Binding Metrics (Phase 1.4c):");
        println!();
        println!(
            "  Window Params: W={}, M={}",
            self.window_size, self.window_top_m
        );
        println!();
        println!("  Positive Queries:");
        println!(
            "    total={}, non_abstain={}, correct={}, wrong={}, abstain={}",
            self.pos_queries,
            self.pos_non_abstain,
            self.pos_correct,
            self.pos_wrong,
            self.pos_abstain
        );
        println!(
            "    coverage_pos={:.1}%, accuracy_pos={:.1}%, abstain_pos={:.1}%",
            self.coverage_pos() * 100.0,
            self.accuracy_pos() * 100.0,
            self.abstain_pos_rate() * 100.0
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
        println!(
            "  Abstain Breakdown (of total {} abstentions):",
            self.pos_abstain + self.neg_abstain
        );
        println!(
            "    no_candidates={} ({:.1}%)",
            self.abstain_no_candidates,
            self.pct_abstain_no_candidates() * 100.0
        );
        println!(
            "    insufficient_margin={} ({:.1}%)",
            self.abstain_margin,
            self.pct_abstain_margin() * 100.0
        );
        println!(
            "    too_many_candidates={} ({:.1}%)",
            self.abstain_too_many,
            self.pct_abstain_too_many() * 100.0
        );
        println!(
            "    ctx_mismatch={} ({:.1}%)",
            self.abstain_ctx_mismatch,
            self.pct_abstain_ctx_mismatch() * 100.0
        );
        println!();
        println!("  Candidate Distribution (after ctx filter):");
        let total_q = self.pos_queries + self.neg_queries;
        println!(
            "    [0]: {} ({:.1}%)",
            self.bucket_0,
            if total_q > 0 {
                self.bucket_0 as f64 / total_q as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "    [1-2]: {} ({:.1}%)",
            self.bucket_1_2,
            if total_q > 0 {
                self.bucket_1_2 as f64 / total_q as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "    [3-5]: {} ({:.1}%)",
            self.bucket_3_5,
            if total_q > 0 {
                self.bucket_3_5 as f64 / total_q as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "    [6-8]: {} ({:.1}%)",
            self.bucket_6_8,
            if total_q > 0 {
                self.bucket_6_8 as f64 / total_q as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "    [>8]: {} ({:.1}%)",
            self.bucket_gt8,
            if total_q > 0 {
                self.bucket_gt8 as f64 / total_q as f64 * 100.0
            } else {
                0.0
            }
        );
        println!();
        println!("  Collision Metrics:");
        println!("    avg_candidates_in_radius={:.2}", self.avg_candidates());
        println!("    avg_ctx_matched={:.2}", self.avg_ctx_matched());
        println!(
            "    pct_queries_with_2+_candidates={:.1}%",
            self.pct_2plus_candidates() * 100.0
        );
        if self.margin_count > 0 {
            println!(
                "    avg_margin={:.2}, min={}, max={}",
                self.avg_margin(),
                self.margin_min_seen,
                self.margin_max_seen
            );
        }
        println!("    avg_hamming={:.2}", self.avg_hamming());
    }

    pub fn print_with_stability(&self, stability: f64) {
        self.print();
        println!();
        println!("  Stability:");
        println!(
            "    ctx_stability={:.1}% (pct ticks where ctx_hat == window mode)",
            stability * 100.0
        );
    }
}

/// Convert Top-K node IDs to a bitmask (assumes N <= 64).
pub fn topk_to_mask(topk_ids: &[usize]) -> u64 {
    let mut mask = 0u64;
    for &id in topk_ids {
        if id < 64 {
            mask |= 1u64 << id;
        }
    }
    mask
}

/// Convert prototype scores (f64) to scaled i32 array.
pub fn proto_scores_to_int(scores: &[f64]) -> [i32; 3] {
    let mut result = [0i32; 3];
    for (i, &s) in scores.iter().take(3).enumerate() {
        result[i] = (s * PROTO_SCALE as f64) as i32;
    }
    result
}

/// Flip N random bits in a signature to create a negative query.
pub fn flip_bits(signature: u64, n_bits: u32, rng_val: u64) -> u64 {
    let mut result = signature;
    let mut val = rng_val;
    for _ in 0..n_bits {
        let bit = (val % 64) as u32;
        result ^= 1u64 << bit;
        val = val.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    result
}

/// Flip bits in a CompetitiveSig mask (keep ctx).
pub fn flip_competitive_sig(sig: &CompetitiveSig, n_bits: u32, rng_val: u64) -> CompetitiveSig {
    CompetitiveSig {
        ctx_hat: sig.ctx_hat,
        mask: flip_bits(sig.mask, n_bits, rng_val),
    }
}
