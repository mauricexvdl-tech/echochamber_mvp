//! Echo Chamber: signal propagation with complex-valued interference.
//! Phase 1.3: Concept Readout + Config-based parameters.

use crate::complex::{kuramoto_order_parameter, kuramoto_weighted, Complex};
use crate::config::Config;
use crate::rng::Rng;
use std::f64::consts::PI;

/// Wrap angle to [-π, +π]
pub fn wrap_to_pi(angle: f64) -> f64 {
    let mut a = angle % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Quantize phase angle to context bin.
pub fn quantize_phase_to_ctx(phi: f64, num_ctx: usize) -> usize {
    // phi in [-π, +π] -> bin in [0, num_ctx-1]
    let normalized = (phi + PI) / (2.0 * PI); // [0, 1)
    let bin = (normalized * num_ctx as f64).floor() as usize;
    bin.min(num_ctx - 1)
}

/// A directed edge with context-conditioned phase shifts.
#[derive(Clone, Debug)]
pub struct Edge {
    pub to: usize,
    /// Phase shift per context channel (dynamic size based on num_ctx).
    pub phase_by_ctx: Vec<f64>,
}

impl Edge {
    /// Create a new edge with random initial phases per context channel.
    pub fn new_random(to: usize, num_ctx: usize, rng: &mut Rng) -> Self {
        let phase_by_ctx: Vec<f64> = (0..num_ctx).map(|_| rng.next_range(-PI, PI)).collect();
        Edge { to, phase_by_ctx }
    }

    /// Create edge with fixed phases (for Lie Triangle demo).
    pub fn new_fixed(to: usize, num_ctx: usize, phase: f64) -> Self {
        Edge {
            to,
            phase_by_ctx: vec![phase; num_ctx],
        }
    }

    /// Get phase for a specific context.
    pub fn phase(&self, ctx: usize) -> f64 {
        self.phase_by_ctx.get(ctx).copied().unwrap_or(0.0)
    }

    /// Adjust phase for a specific context by a signed delta.
    pub fn adjust_phase(&mut self, ctx: usize, delta: f64) {
        if let Some(p) = self.phase_by_ctx.get_mut(ctx) {
            *p = wrap_to_pi(*p + delta);
        }
    }
}

/// A node in the echo chamber network.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub edges: Vec<Edge>,
    pub buffer: Complex,
    /// Context affinity tracking: positive = prefers this context.
    pub ctx_affinity: Vec<i32>,
}

impl Node {
    pub fn new(id: usize, num_ctx: usize) -> Self {
        Node {
            id,
            edges: Vec::new(),
            buffer: Complex::ZERO,
            ctx_affinity: vec![0; num_ctx],
        }
    }

    /// Add incoming signal to buffer (local addition only).
    pub fn receive(&mut self, sig: Complex) {
        self.buffer += sig;
    }

    /// Get the node's preferred context (highest affinity), or None if not specialized.
    pub fn preferred_ctx(&self, aff_margin: i32) -> Option<usize> {
        if self.ctx_affinity.is_empty() {
            return None;
        }

        let mut best_ctx = 0;
        let mut best_val = self.ctx_affinity[0];
        for (c, &aff) in self.ctx_affinity.iter().enumerate().skip(1) {
            if aff > best_val {
                best_val = aff;
                best_ctx = c;
            }
        }

        // Check if clearly specialized (margin over second best)
        let mut second_best = i32::MIN;
        for (c, &aff) in self.ctx_affinity.iter().enumerate() {
            if c != best_ctx && aff > second_best {
                second_best = aff;
            }
        }

        if best_val >= aff_margin && (best_val - second_best) >= aff_margin {
            Some(best_ctx)
        } else {
            None // Not yet specialized
        }
    }

    /// Update affinity: increment for current ctx, decrement others.
    pub fn update_affinity(&mut self, ctx: usize, aff_inc: i32, aff_dec: i32) {
        for (c, aff) in self.ctx_affinity.iter_mut().enumerate() {
            if c == ctx {
                *aff = aff.saturating_add(aff_inc);
            } else {
                *aff = aff.saturating_sub(aff_dec);
            }
        }
    }

    /// Process buffer with context-specific phase shifts.
    /// Returns packets (local_edge_idx, target_id, signal) to deliver.
    pub fn process(&mut self, ctx: usize, eps: f64) -> Vec<(usize, usize, Complex)> {
        let norm = self.buffer.norm();
        if norm < eps {
            self.buffer = Complex::ZERO;
            return Vec::new();
        }

        let out_degree = self.edges.len();
        if out_degree == 0 {
            return Vec::new();
        }

        let split_factor = 1.0 / (out_degree as f64).sqrt();

        let mut packets = Vec::with_capacity(out_degree);
        for (edge_idx, edge) in self.edges.iter().enumerate() {
            let rotator = Complex::from_polar(1.0, edge.phase(ctx));
            let outgoing = self.buffer.scale(split_factor) * rotator;
            packets.push((edge_idx, edge.to, outgoing));
        }

        self.buffer = Complex::ZERO;
        packets
    }
}

/// Per-tick metrics for destruction ratio analysis.
#[derive(Clone, Debug)]
pub struct TickMetrics {
    pub inflow_power: Vec<f64>,
    pub self_power: Vec<f64>,
    pub destruction: Vec<f64>,
    pub ctx: Option<usize>,
    pub tot_pow_pre: f64,
    pub tot_pow_post: f64,
    pub homeostasis_scale: f64,
    /// Kuramoto order parameter R ∈ [0,1]: phase synchronization across all nodes
    pub kuramoto_r: f64,
    /// Weighted Kuramoto: amplitude-weighted phase coherence
    pub kuramoto_r_weighted: f64,
}

impl TickMetrics {
    pub fn new(n: usize) -> Self {
        TickMetrics {
            inflow_power: vec![0.0; n],
            self_power: vec![0.0; n],
            destruction: vec![0.0; n],
            ctx: None,
            tot_pow_pre: 0.0,
            tot_pow_post: 0.0,
            homeostasis_scale: 1.0,
            kuramoto_r: 0.0,
            kuramoto_r_weighted: 0.0,
        }
    }

    pub fn compute_destruction(&mut self, eps: f64) {
        for i in 0..self.inflow_power.len() {
            let inflow = self.inflow_power[i];
            let self_p = self.self_power[i];
            if inflow > eps {
                let ratio = 1.0 - self_p / (inflow + eps);
                self.destruction[i] = ratio.clamp(0.0, 1.0);
            } else {
                self.destruction[i] = 0.0;
            }
        }
    }
}

/// Edge contribution tracking for plasticity.
#[derive(Clone)]
pub struct EdgeContrib {
    pub from: usize,
    pub edge_idx: usize,
}

/// The echo chamber network.
pub struct EchoChamber {
    pub nodes: Vec<Node>,
    pub tick_counter: usize,
    pub config: Config,
    /// Spectral scale factor for unitary-like signal propagation.
    /// Applied to outgoing signals to bound spectral norm.
    spectral_scale: f64,
}

impl EchoChamber {
    pub fn new(config: Config) -> Self {
        let nodes = (0..config.num_nodes)
            .map(|id| Node::new(id, config.num_ctx))
            .collect();
        EchoChamber {
            nodes,
            tick_counter: 0,
            spectral_scale: 1.0, // Default: no scaling
            config,
        }
    }

    pub fn add_edge_random(&mut self, from: usize, to: usize, rng: &mut Rng) {
        let num_ctx = self.config.num_ctx;
        self.nodes[from]
            .edges
            .push(Edge::new_random(to, num_ctx, rng));
    }

    pub fn inject(&mut self, node_id: usize, signal: Complex) {
        self.nodes[node_id].receive(signal);
    }

    /// Apply per-node decay and amplitude clamp.
    pub fn apply_decay_clamp(&mut self) {
        let decay_factor = 1.0 - self.config.decay_per_tick;
        let clamp = self.config.clamp_max_amp;

        for node in &mut self.nodes {
            node.buffer *= decay_factor;
            let amp = node.buffer.norm();
            if amp > clamp {
                let scale = clamp / amp;
                node.buffer *= scale;
            }
        }
    }

    /// Apply global power homeostasis (soft renormalization).
    pub fn apply_homeostasis(&mut self) -> (f64, f64, f64) {
        let tot_pow_pre: f64 = self.nodes.iter().map(|n| n.buffer.power()).sum();

        if tot_pow_pre < self.config.eps {
            return (tot_pow_pre, tot_pow_pre, 1.0);
        }

        let ideal_scale = (self.config.pow_target / tot_pow_pre.max(1e-12)).sqrt();
        let scale = 1.0 + self.config.homeostasis_beta * (ideal_scale - 1.0);

        for node in &mut self.nodes {
            node.buffer *= scale;
        }

        let tot_pow_post: f64 = self.nodes.iter().map(|n| n.buffer.power()).sum();

        (tot_pow_pre, tot_pow_post, scale)
    }

    /// Run one tick without metrics (for Demo 1 - uses ctx=0).
    pub fn tick(&mut self) {
        let scale = self.spectral_scale;
        let mut all_packets: Vec<(usize, Complex)> = Vec::new();
        for node in &mut self.nodes {
            let packets = node.process(0, self.config.eps);
            for (_, target, sig) in packets {
                // Apply spectral scale for unitary-like propagation
                all_packets.push((target, sig.scale(scale)));
            }
        }
        for (target_id, signal) in all_packets {
            self.nodes[target_id].receive(signal);
        }
        self.tick_counter += 1;
    }

    /// Compute alignment between node buffer and reference signal.
    fn compute_alignment(z_node: Complex, z_ref: Complex, eps: f64) -> f64 {
        let norm_node = z_node.norm();
        let norm_ref = z_ref.norm();
        if norm_node < eps || norm_ref < eps {
            return 0.0;
        }
        let product = z_node * z_ref.conj();
        product.re / (norm_node * norm_ref)
    }

    /// Run one tick with context-conditioned edges, homeostasis, and plasticity.
    pub fn tick_with_context_plasticity(
        &mut self,
        z_inj: Complex,
        topk_nodes: &[usize],
        learn: bool,
    ) -> TickMetrics {
        let n = self.nodes.len();
        let mut metrics = TickMetrics::new(n);

        // Apply decay BEFORE processing
        self.apply_decay_clamp();

        // Compute context from PRE-MIX injection signal
        let ctx = if z_inj.norm() > 1e-9 {
            Some(quantize_phase_to_ctx(z_inj.arg(), self.config.num_ctx))
        } else {
            None
        };
        metrics.ctx = ctx;

        let ctx_for_process = ctx.unwrap_or(0);
        let spectral_scale = self.spectral_scale;

        // Collect packets with edge info
        let mut all_packets: Vec<(usize, usize, usize, Complex)> = Vec::new();
        for node in &mut self.nodes {
            let from = node.id;
            let packets = node.process(ctx_for_process, self.config.eps);
            for (edge_idx, target, sig) in packets {
                // Apply spectral scale for unitary-like propagation
                all_packets.push((from, edge_idx, target, sig.scale(spectral_scale)));
            }
        }

        // Track edge contributions per target node
        let mut incoming_edges: Vec<Vec<EdgeContrib>> = vec![Vec::new(); n];

        // Deliver packets
        for (from, edge_idx, target, signal) in all_packets {
            let power = signal.power();
            metrics.inflow_power[target] += power;

            if power >= self.config.min_edge_power {
                incoming_edges[target].push(EdgeContrib { from, edge_idx });
            }

            self.nodes[target].receive(signal);
        }

        // Apply context suppression
        if let Some(ctx_bin) = ctx {
            for node in &mut self.nodes {
                let power = node.buffer.power();
                if power >= self.config.suppress_min_pow {
                    if let Some(pref) = node.preferred_ctx(self.config.aff_margin) {
                        if pref != ctx_bin {
                            node.buffer *= self.config.ctx_suppress;
                        }
                    }
                }
            }
        }

        // Apply homeostasis
        let (tot_pow_pre, tot_pow_post, scale) = self.apply_homeostasis();
        metrics.tot_pow_pre = tot_pow_pre;
        metrics.tot_pow_post = tot_pow_post;
        metrics.homeostasis_scale = scale;

        // Compute self_power after homeostasis
        for node in &self.nodes {
            metrics.self_power[node.id] = node.buffer.power();
        }

        metrics.compute_destruction(self.config.eps);

        // Compute Kuramoto coherence
        let (r, r_weighted) = self.compute_kuramoto_coherence();
        metrics.kuramoto_r = r;
        metrics.kuramoto_r_weighted = r_weighted;

        // Plasticity updates
        if learn {
            if let Some(ctx_bin) = ctx {
                let phi_ref = z_inj.arg();

                for &node_j in topk_nodes {
                    let z_j = self.nodes[node_j].buffer;
                    let power_j = z_j.power();

                    if power_j < self.config.p_min {
                        continue;
                    }

                    let alignment = Self::compute_alignment(z_j, z_inj, self.config.eps);
                    let phi_j = z_j.arg();
                    let phase_err = wrap_to_pi(phi_j - phi_ref);

                    let pref_ctx = self.nodes[node_j].preferred_ctx(self.config.aff_margin);
                    let ctx_match = match pref_ctx {
                        Some(pref) => pref == ctx_bin,
                        None => true,
                    };

                    if alignment >= self.config.align_pos {
                        self.nodes[node_j].update_affinity(
                            ctx_bin,
                            self.config.aff_inc,
                            self.config.aff_dec,
                        );

                        if ctx_match {
                            for contrib in &incoming_edges[node_j] {
                                let edge = &mut self.nodes[contrib.from].edges[contrib.edge_idx];
                                edge.adjust_phase(
                                    ctx_bin,
                                    -self.config.phase_learn_rate * phase_err,
                                );
                            }
                        } else {
                            for contrib in &incoming_edges[node_j] {
                                let edge = &mut self.nodes[contrib.from].edges[contrib.edge_idx];
                                edge.adjust_phase(
                                    ctx_bin,
                                    self.config.phase_learn_rate
                                        * self.config.wrong_ctx_penalty
                                        * phase_err,
                                );
                            }
                        }
                    } else if alignment <= -self.config.align_neg {
                        if ctx_match {
                            for contrib in &incoming_edges[node_j] {
                                let edge = &mut self.nodes[contrib.from].edges[contrib.edge_idx];
                                edge.adjust_phase(
                                    ctx_bin,
                                    self.config.phase_learn_rate * phase_err,
                                );
                            }
                        }
                    }
                }
            }
        }

        self.tick_counter += 1;
        metrics
    }

    /// Compute Kuramoto coherence metrics for the current network state.
    /// Returns (R, R_weighted) where:
    ///   - R: standard Kuramoto order parameter (all nodes equal weight)
    ///   - R_weighted: amplitude-weighted coherence (stronger signals matter more)
    pub fn compute_kuramoto_coherence(&self) -> (f64, f64) {
        let eps = self.config.eps;

        // Collect phases and amplitudes from all nodes with non-negligible signal
        let mut phases: Vec<f64> = Vec::with_capacity(self.nodes.len());
        let mut amplitudes: Vec<f64> = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
            let amp = node.buffer.norm();
            if amp > eps {
                phases.push(node.buffer.arg());
                amplitudes.push(amp);
            }
        }

        if phases.is_empty() {
            return (0.0, 0.0);
        }

        let r = kuramoto_order_parameter(&phases);
        let r_weighted = kuramoto_weighted(&amplitudes, &phases);

        (r, r_weighted)
    }

    pub fn top_nodes(&self, n: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .nodes
            .iter()
            .map(|node| (node.id, node.buffer.norm()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().take(n).collect()
    }

    /// Get nodes with their specialization info.
    pub fn get_specialization_info(&self) -> Vec<(usize, Option<usize>, i32)> {
        self.nodes
            .iter()
            .map(|n| {
                let pref = n.preferred_ctx(self.config.aff_margin);
                let max_aff = *n.ctx_affinity.iter().max().unwrap_or(&0);
                let min_aff = *n.ctx_affinity.iter().min().unwrap_or(&0);
                (n.id, pref, max_aff - min_aff)
            })
            .collect()
    }

    pub fn lie_triangle(config: Config) -> Self {
        let mut chamber = EchoChamber::new(Config {
            num_nodes: 4,
            num_ctx: config.num_ctx,
            ..config
        });
        // Rebuild nodes for the 4-node topology
        chamber.nodes = (0..4)
            .map(|id| Node::new(id, chamber.config.num_ctx))
            .collect();

        let num_ctx = chamber.config.num_ctx;
        chamber.nodes[0]
            .edges
            .push(Edge::new_fixed(1, num_ctx, 0.0));
        chamber.nodes[1]
            .edges
            .push(Edge::new_fixed(2, num_ctx, 0.0));
        chamber.nodes[0]
            .edges
            .push(Edge::new_fixed(3, num_ctx, 0.0));
        chamber.nodes[3].edges.push(Edge::new_fixed(2, num_ctx, PI));
        chamber
    }

    pub fn random_graph(config: Config, rng: &mut Rng) -> Self {
        let mut chamber = EchoChamber::new(config.clone());
        let total_edges = config.num_nodes * config.avg_degree;

        for _ in 0..total_edges {
            let from = rng.next_usize(config.num_nodes);
            let to = rng.next_usize(config.num_nodes);
            if from != to {
                chamber.add_edge_random(from, to, rng);
            }
        }

        // Apply spectral normalization if configured
        if config.spectral_normalize {
            chamber.compute_spectral_scale();
        }

        chamber
    }

    /// Compute spectral scale factor to bound the graph's spectral norm.
    ///
    /// This builds an adjacency matrix from the current graph structure,
    /// computes its spectral norm (largest singular value), and sets a
    /// scaling factor to ensure σ_max ≤ target.
    ///
    /// The scaling factor is applied during signal propagation in process().
    pub fn compute_spectral_scale(&mut self) -> f64 {
        use crate::spectral::spectral_norm;

        let n = self.nodes.len();
        if n == 0 {
            self.spectral_scale = 1.0;
            return 1.0;
        }

        // Build adjacency matrix with 1/sqrt(out_degree) weights
        // This matches how signals are actually split in process()
        let mut matrix = vec![vec![0.0; n]; n];

        for node in &self.nodes {
            let out_degree = node.edges.len();
            if out_degree > 0 {
                let weight = 1.0 / (out_degree as f64).sqrt();
                for edge in &node.edges {
                    matrix[node.id][edge.to] = weight;
                }
            }
        }

        // Compute spectral norm
        let sigma = spectral_norm(&matrix, self.config.spectral_iterations);

        // Compute scale factor to achieve target
        if sigma > self.config.spectral_target && sigma > 1e-12 {
            self.spectral_scale = self.config.spectral_target / sigma;
        } else {
            self.spectral_scale = 1.0;
        }

        sigma
    }

    /// Get the current spectral scale factor.
    pub fn spectral_scale(&self) -> f64 {
        self.spectral_scale
    }

    /// Dampen node buffers by a factor in (0, 1].
    /// Used by Reset mode to reduce dominant node amplitudes.
    pub fn dampen_nodes(&mut self, node_ids: &[usize], factor: f32) {
        let factor = factor.clamp(0.0, 1.0) as f64;
        for &node_id in node_ids {
            if node_id < self.nodes.len() {
                self.nodes[node_id].buffer *= factor;
            }
        }
    }

    /// Apply small noise perturbation to node buffers.
    /// Used by Perturb action for controlled disruption.
    /// Adds random phase noise scaled by amplitude.
    pub fn apply_noise(&mut self, node_ids: &[usize], noise_amp: f32, rng: &mut crate::rng::Rng) {
        use crate::complex::Complex;

        let noise_amp = noise_amp.clamp(0.0, 0.1) as f64; // Cap noise amplitude
        for &node_id in node_ids {
            if node_id < self.nodes.len() {
                let current = self.nodes[node_id].buffer;
                let current_amp = current.norm();
                if current_amp > 1e-6 {
                    // Add small random phase perturbation
                    let phase_noise = (rng.next_f64() - 0.5) * std::f64::consts::PI * noise_amp;
                    let amp_noise = 1.0 + (rng.next_f64() - 0.5) * noise_amp;
                    let new_phase = current.arg() + phase_noise;
                    let new_amp = current_amp * amp_noise;
                    self.nodes[node_id].buffer = Complex::from_polar(new_amp, new_phase);
                }
            }
        }
    }
}
