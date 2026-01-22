//! Echo Chamber MVP: Emergent cancellation via complex signal interference.
//! Phase 1.9: CONSOLIDATION - Make merges happen + reduce stability flicker.

mod causes;
mod complex;
mod concepts;
mod config;
mod echo;
mod memory;
mod anchor;
mod rng;

use causes::{get_top_k, Causes, TopKStats};
use complex::Complex;
use concepts::{ConceptBank, ConfusionMatrix};
use config::Config;
use echo::EchoChamber;
use memory::{
    flip_competitive_sig, CompetitiveSig, GlobalLabelMemoryStore, GlobalLabelMetrics,
    LabelBindingMetrics, LabelMemoryStore, MemoryMetrics, MemoryStore, RollingWindow,
    proto_scores_to_int, topk_to_mask, WINDOW_SIZE, WINDOW_TOP_M,
};
use rng::Rng;
use anchor::{AnchorBank, KeyedMemoryStore, KeyedMemoryConfig, KeyedMemoryMetrics, MemoryKey, ConfidenceInfo, GateParams, MAX_ANCHORS};

const EPS_PRINT: f64 = 1e-9;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  ECHO CHAMBER MVP - Phase 1.9: CONSOLIDATION                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let config = Config::default();

    demo_lie_triangle(&config);
    println!();
    demo_latent_causes(&config);
    println!();
    demo_label_binding(&config);
    println!();
    demo_competitive_binding(&config);
    println!();
    demo_phase_1_5b_comparison(&config);

    if config.run_capacity_sweep {
        println!();
        run_capacity_sweep(&config);
    }
}

fn fmt_amp_phase(z: &Complex) -> String {
    let amp = z.norm();
    if amp < EPS_PRINT {
        format!("amp={:.6} φ=undef  ", amp)
    } else {
        format!("amp={:.6} φ={:+.4}", amp, z.arg())
    }
}

fn demo_lie_triangle(config: &Config) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 1: Lie Triangle Cancellation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Network topology:");
    println!("  Node 0 (start) ─┬─ phase=0 ──► Node 1 ─ phase=0 ──► Node 2 (target)");
    println!("                  └─ phase=0 ──► Node 3 ─ phase=π ──►");
    println!();
    println!("Injecting Complex(1, 0) into node 0...");
    println!();

    let mut chamber = EchoChamber::lie_triangle(config.clone());
    chamber.inject(0, Complex::new(1.0, 0.0));

    println!("{:<6} │ {:^24} │ {:^24} │ {:^24}", "Tick", "Node 1 (path A)", "Node 3 (path B)", "Node 2 (target)");
    println!("───────┼──────────────────────────┼──────────────────────────┼──────────────────────────");

    for tick in 0..=4 {
        let n1 = &chamber.nodes[1].buffer;
        let n3 = &chamber.nodes[3].buffer;
        let n2 = &chamber.nodes[2].buffer;
        println!("{:<6} │ {} │ {} │ {}", tick, fmt_amp_phase(n1), fmt_amp_phase(n3), fmt_amp_phase(n2));
        if tick < 4 { chamber.tick(); }
    }

    println!();
    let final_amp = chamber.nodes[2].buffer.norm();
    if final_amp < 1e-6 {
        println!("✓ CANCELLATION ACHIEVED: Target amplitude = {:.2e}", final_amp);
        println!("  Signals arrived out of phase (π) and destructively interfered.");
    } else {
        println!("✗ Unexpected: Target amplitude = {:.6}", final_amp);
    }
}

struct EvalStats {
    ctx_hit_counts: Vec<usize>,
    tot_pow_samples: Vec<f64>,
    scale_sum: f64,
    scale_count: usize,
}

impl EvalStats {
    fn new(num_ctx: usize) -> Self {
        EvalStats { ctx_hit_counts: vec![0; num_ctx], tot_pow_samples: Vec::new(), scale_sum: 0.0, scale_count: 0 }
    }
    fn record(&mut self, ctx: Option<usize>, tot_pow: f64, scale: f64) {
        if let Some(c) = ctx { if c < self.ctx_hit_counts.len() { self.ctx_hit_counts[c] += 1; } }
        self.tot_pow_samples.push(tot_pow);
        self.scale_sum += scale;
        self.scale_count += 1;
    }
    fn mean_tot_pow(&self) -> f64 {
        if self.tot_pow_samples.is_empty() { 0.0 } else { self.tot_pow_samples.iter().sum::<f64>() / self.tot_pow_samples.len() as f64 }
    }
    fn _p95_tot_pow(&self) -> f64 {
        if self.tot_pow_samples.is_empty() { return 0.0; }
        let mut sorted = self.tot_pow_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (sorted.len() as f64 * 0.95).floor() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    fn _avg_scale(&self) -> f64 { if self.scale_count > 0 { self.scale_sum / self.scale_count as f64 } else { 1.0 } }
}

fn demo_latent_causes(config: &Config) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 2: Concept Readout + Episodic Memory (N={}, K={})", config.num_nodes, config.top_k);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Configuration:");
    println!("  Dynamics: decay={:.0}%/tick, clamp_max={:.1}", config.decay_per_tick * 100.0, config.clamp_max_amp);
    println!("  Training: {} ticks, Eval: {} ticks", config.train_ticks, config.eval_ticks);
    println!();

    let _cause_phases = config.cause_phases();
    let mut rng = Rng::new(config.seed);
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Training
    println!("Training {} ticks...", config.train_ticks);
    for tick in 0..config.train_ticks {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
        if tick % 50000 == 0 { print!("."); }
    }
    println!(" done.");
    println!();

    // Eval Phase 1
    let mut stats = TopKStats::new(config.num_nodes, config.num_causes);
    let mut eval_stats = EvalStats::new(config.num_ctx);
    let mut concept_bank = ConceptBank::new(config.num_ctx, config.num_nodes);

    for _tick in 0..config.eval_ticks {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        stats.record_topk(&topk, active_mask);
        eval_stats.record(metrics.ctx, metrics.tot_pow_post, metrics.homeostasis_scale);
        if let Some(ctx) = metrics.ctx { concept_bank.record(ctx, &topk_ids); }
    }
    concept_bank.normalize_all();

    // Eval Phase 2
    let mut confusion = ConfusionMatrix::new();
    let mut memory_store = MemoryStore::new(&config);
    let mut memory_metrics = MemoryMetrics::new();

    for tick in 0..config.eval_ticks {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        let topk_mask = topk_to_mask(&topk_ids);
        let proto_scores_f64: Vec<f64> = (0..config.num_ctx).map(|ctx| concept_bank.score(ctx, &topk_ids)).collect();
        let proto_scores = proto_scores_to_int(&proto_scores_f64);

        if let Some(true_ctx) = metrics.ctx {
            let predicted_ctx = concept_bank.classify(&topk_ids);
            confusion.record(true_ctx, predicted_ctx);
            if metrics.tot_pow_post >= config.memory_min_power {
                let store_roll: f64 = rng.next_f64();
                if store_roll < config.memory_store_prob {
                    if memory_store.store(true_ctx as u32, tick as u64, topk_mask, proto_scores, config.memory_debounce_ticks) {
                        memory_metrics.record_store();
                    }
                }
            }
            let recall_result = memory_store.recall(tick as u64, topk_mask, proto_scores);
            memory_metrics.record_recall(recall_result.as_ref(), true_ctx as u32);
        }
    }

    println!("Classification accuracy: {:.1}%", confusion.accuracy() * 100.0);
    println!("Memory: coverage={:.1}%, accuracy={:.1}%", memory_metrics.coverage() * 100.0, memory_metrics.accuracy() * 100.0);

    let coverage = stats.coverage();
    let top15 = stats.top_by_hits(15, config.eval_ticks);
    let avg_purity: f64 = if top15.is_empty() { 0.0 } else { top15.iter().map(|(_, _, _, p, _, _)| p).sum::<f64>() / top15.len() as f64 };

    println!();
    println!("PHASE 1.4a: [✓] TotPow={:.2}, Purity={:.2}, Coverage={}, Class={:.1}%, Mem={:.1}%/{:.1}%",
        eval_stats.mean_tot_pow(), avg_purity, coverage,
        confusion.accuracy() * 100.0, memory_metrics.coverage() * 100.0, memory_metrics.accuracy() * 100.0);
}

fn demo_label_binding(config: &Config) {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 3: One-Shot Label Binding (L={}, episodes={})", config.num_labels, config.num_episodes);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let metrics = run_label_binding_experiment(config, config.num_labels);

    println!("Label Binding: episodes={}, binds={}", metrics.episodes, metrics.binds_done);
    println!("  coverage={:.1}%, accuracy={:.1}%, false_rate={:.1}%",
        metrics.coverage() * 100.0, metrics.accuracy() * 100.0, metrics.false_rate() * 100.0);
    println!();

    let binds_ok = metrics.binds_done == metrics.episodes;
    let coverage_ok = metrics.coverage() >= 0.10;
    let accuracy_ok = metrics.accuracy() >= 0.60;
    let false_rate_ok = metrics.false_rate() <= 0.40;

    if binds_ok && coverage_ok && accuracy_ok && false_rate_ok {
        println!("PHASE 1.4b: ALL CRITERIA MET!");
    } else {
        println!("PHASE 1.4b: Some criteria not met.");
    }
}

fn run_label_binding_experiment(config: &Config, num_labels: usize) -> LabelBindingMetrics {
    let mut rng = Rng::new(config.seed.wrapping_add(0x1234_5678));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

    let mut label_memory = LabelMemoryStore::from_config(config);
    let mut metrics = LabelBindingMetrics::new();
    let mut global_tick: u64 = 0;

    for _ep in 0..config.num_episodes {
        metrics.record_episode();
        label_memory.clear();
        let episode_label = (rng.next_u64() % num_labels as u64) as u16;

        for t in 0..config.episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
            let topk = get_top_k(&chamber, config.top_k);
            let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);
            let signature = topk_to_mask(&topk_ids);

            if t == config.bind_tick {
                label_memory.store(episode_label, global_tick, signature);
                metrics.record_bind();
            }
            if t >= config.recall_start_tick && t % config.recall_stride == 0 {
                let recall_result = label_memory.recall(global_tick, signature);
                metrics.record_recall_query(recall_result.as_ref(), episode_label);
            }
            global_tick += 1;
        }
    }
    metrics
}

fn run_capacity_sweep(config: &Config) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CAPACITY SWEEP");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    let label_counts = [4, 8, 16, 32];
    println!("{:>8} │ {:>10} │ {:>10} │ {:>12}", "Labels", "Coverage", "Accuracy", "False Rate");
    println!("─────────┼────────────┼────────────┼──────────────");
    for &num_labels in &label_counts {
        let metrics = run_label_binding_experiment(config, num_labels);
        println!("{:>8} │ {:>9.1}% │ {:>9.1}% │ {:>11.1}%",
            num_labels, metrics.coverage() * 100.0, metrics.accuracy() * 100.0, metrics.false_rate() * 100.0);
    }
    println!();
}

// =============================================================================
// DEMO 4: Competitive Label Binding with ABSTAIN and Windowed Signatures
// =============================================================================

fn demo_competitive_binding(config: &Config) {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 4: Competitive Label Binding with ABSTAIN (Phase 1.4c)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("Configuration:");
    println!("  num_labels={}, episodes={}, ticks/ep={}, binds/ep={}",
        config.competitive_num_labels, config.competitive_episodes,
        config.competitive_episode_ticks, config.competitive_binds_per_episode);
    println!("  max_entries={}, max_hamming={}, margin_min={}",
        config.competitive_max_entries, config.competitive_max_hamming, config.competitive_margin_min);
    println!("  p_neg={:.0}%, neg_flip_bits={}, recall_stride={}, recall_start={}",
        config.competitive_p_neg * 100.0, config.competitive_neg_flip_bits,
        config.competitive_recall_stride, config.competitive_recall_start);
    println!("  Window: W={}, M={}", WINDOW_SIZE, WINDOW_TOP_M);
    println!();

    let bind_ticks = config.competitive_bind_ticks();
    println!("  bind_ticks={:?}", bind_ticks);
    println!();

    let (metrics, mem_entries, mem_evictions, mem_hit_updates, stability) =
        run_competitive_experiment(config);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("COMPETITIVE BINDING RESULTS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    println!("Memory Status:");
    println!("  entries={}, evictions={}, hit_updates={}", mem_entries, mem_evictions, mem_hit_updates);
    println!();

    metrics.print_with_stability(stability);
    println!();

    // Acceptance check
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 1.4c ACCEPTANCE:");
    println!("═══════════════════════════════════════════════════════════════════");

    let entries_ok = mem_entries > 0 && mem_evictions > 0;
    let coverage_pos_ok = metrics.coverage_pos() >= 0.25;
    let accuracy_pos_ok = metrics.accuracy_pos() >= 0.70;
    let abstain_neg_ok = metrics.abstain_neg_rate() >= 0.80;
    let false_pos_ok = metrics.false_positive_rate() <= 0.15;
    let selective_ok = metrics.selective_accuracy() >= 0.75;

    println!("  [{}] Memory filled (entries={}, evictions={})",
        if entries_ok { "✓" } else { "✗" }, mem_entries, mem_evictions);
    println!("  [{}] coverage_pos >= 25%: {:.1}%",
        if coverage_pos_ok { "✓" } else { "✗" }, metrics.coverage_pos() * 100.0);
    println!("  [{}] accuracy_pos >= 70%: {:.1}%",
        if accuracy_pos_ok { "✓" } else { "✗" }, metrics.accuracy_pos() * 100.0);
    println!("  [{}] abstain_neg >= 80%: {:.1}%",
        if abstain_neg_ok { "✓" } else { "✗" }, metrics.abstain_neg_rate() * 100.0);
    println!("  [{}] false_positive <= 15%: {:.1}%",
        if false_pos_ok { "✓" } else { "✗" }, metrics.false_positive_rate() * 100.0);
    println!("  [{}] selective_accuracy >= 75%: {:.1}%",
        if selective_ok { "✓" } else { "✗" }, metrics.selective_accuracy() * 100.0);

    let all_ok = entries_ok && coverage_pos_ok && accuracy_pos_ok && abstain_neg_ok && false_pos_ok && selective_ok;
    if all_ok {
        println!();
        println!("  → Phase 1.4c: ALL CRITERIA MET!");
    } else {
        println!();
        println!("  → Some criteria not met. Tuning may be needed.");
    }
}

fn run_competitive_experiment(config: &Config) -> (GlobalLabelMetrics, usize, usize, usize, f64) {
    let mut rng = Rng::new(config.seed.wrapping_add(0xABCD_EF01));

    // Create and pre-train chamber
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Build concept bank during pre-training for ctx classification
    let mut concept_bank = ConceptBank::new(config.num_ctx, config.num_nodes);

    println!("Pre-training chamber (10000 ticks)...");
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        let metrics = chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);

        // Record for concept bank
        if let Some(ctx) = metrics.ctx {
            concept_bank.record(ctx, &topk_ids);
        }
    }
    concept_bank.normalize_all();
    println!("Pre-training done.");
    println!();

    // Initialize global memory (persists across episodes) and metrics
    let mut memory = GlobalLabelMemoryStore::from_config(config);
    let mut metrics = GlobalLabelMetrics::new();

    // Rolling window for windowed signatures
    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);

    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;
    let mut total_binds = 0;

    println!("Running {} episodes...", config.competitive_episodes);

    for ep in 0..config.competitive_episodes {
        // Track labels bound in this episode: (tick_in_ep, label, signature)
        let mut episode_bindings: Vec<(usize, u16, CompetitiveSig)> = Vec::new();

        // Reset window at episode start for fresh signatures
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            // Run chamber dynamics
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
            let topk = get_top_k(&chamber, config.top_k);
            let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            // Classify ctx using concept bank
            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);

            // Update rolling window
            window.push(&topk_ids, ctx_hat);

            // Get current windowed signature
            let current_sig = window.competitive_sig();

            // Multiple binds per episode (only if window is ready)
            if bind_ticks.contains(&t) && window.is_ready() {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                memory.store_competitive(label, global_tick, current_sig.clone());
                episode_bindings.push((t, label, current_sig.clone()));
                total_binds += 1;
            }

            // Query phase (positive and negative queries)
            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 && window.is_ready() {
                // Determine if this is a negative query
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    // Negative query: flip bits in mask to create hard negative (keep ctx)
                    let neg_sig = flip_competitive_sig(&current_sig, config.competitive_neg_flip_bits, rng.next_u64());
                    let result = memory.recall_competitive(global_tick, &neg_sig);
                    metrics.record_negative(&result);
                } else {
                    // Positive query: use current windowed signature
                    // Ground truth = most recent label bound in this episode (if any)
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16; // ctx-based label
                    let result = memory.recall_competitive(global_tick, &current_sig);
                    metrics.record_positive(&result, true_label);
                }
            }

            global_tick += 1;
        }

        if (ep + 1) % 50 == 0 {
            print!(".");
        }
    }
    println!(" done.");
    println!();

    println!("Total binds performed: {}", total_binds);

    let stability = window.stability();
    (metrics, memory.len(), memory.evictions(), memory.hit_updates(), stability)
}

// =============================================================================
// DEMO 5: Phase 1.5b Comparison - Baseline vs Anchor+Mask Memory
// =============================================================================


// =============================================================================
// DEMO 5: Phase 1.6 Comparison - Baseline vs Anchor+Mask Memory with Stabilization
// =============================================================================

fn demo_phase_1_5b_comparison(config: &Config) {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 5/6: Phase 1.9 - CONSOLIDATION (Merges + Stability)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Run both experiments with same seed for fair comparison
    let (metrics_5a, stability_5a) = run_demo_5a_baseline(config);
    println!();
    let (metrics_5b, anchor_stats, keyed_stats, probe_stats, lifecycle_stats) = run_demo_5b_keyed(config);

    // Print comparison summary
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 1.8 COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("{:<25} {:>12} {:>12}", "Metric", "5a (1.4c)", "5b (Keyed)");
    println!("─────────────────────────────────────────────────────────────────");
    println!("{:<25} {:>11.1}% {:>11.1}%", "coverage_pos",
        metrics_5a.coverage_pos() * 100.0, metrics_5b.coverage_pos() * 100.0);
    println!("{:<25} {:>11.1}% {:>11.1}%", "accuracy_pos",
        metrics_5a.accuracy_pos() * 100.0, metrics_5b.accuracy_pos() * 100.0);
    println!("{:<25} {:>11.1}% {:>11.1}%", "abstain_neg",
        metrics_5a.abstain_neg_rate() * 100.0, metrics_5b.abstain_neg_rate() * 100.0);
    println!("{:<25} {:>11.1}% {:>11.1}%", "false_positive",
        metrics_5a.false_positive_rate() * 100.0, metrics_5b.false_positive_rate() * 100.0);
    println!("{:<25} {:>11.1}% {:>11.1}%", "selective_accuracy",
        metrics_5a.selective_accuracy() * 100.0, metrics_5b.selective_accuracy() * 100.0);
    println!();

    // Phase 1.6 specific metrics
    println!("5a stability: {:.1}%", stability_5a * 100.0);
    println!();
    println!("5b Anchor Codebook (Phase 1.6):");
    println!("  anchors_used: {} / {} ({:.1}% utilization)",
        anchor_stats.anchors_used, MAX_ANCHORS, anchor_stats.utilization * 100.0);
    println!("  anchor_creates: {}", anchor_stats.creates);
    println!("  anchor_evictions: {}", anchor_stats.evictions);
    println!("  anchor_merges: {}", anchor_stats.merges);
    println!("  thrash_rate: {:.2} per 10k ticks", anchor_stats.thrash_rate);
    println!("  new_anchor_rate: {:.1}%", anchor_stats.new_rate * 100.0);
    println!("  avg_anchor_match_hamming: {:.2}", anchor_stats.avg_hamming);
    println!("  p95_anchor_match_hamming: {}", anchor_stats.p95_hamming);
    println!("  gate_pass_rate: {:.1}%", anchor_stats.gate_pass_rate * 100.0);
    println!();
    println!("5b Keyed Memory:");
    println!("  unique_keys: {}", keyed_stats.0);
    println!("  entry_hit_rate: {:.1}%", keyed_stats.1 * 100.0);
    println!("  keys_remapped: {}", keyed_stats.2);
    println!();
    println!("5b Prototype Vectors (Phase 1.7a):");
    println!("  proto_updates: {}", anchor_stats.proto_updates);
    println!("  avg_proto_support: {:.1}", anchor_stats.avg_proto_support);
    println!("  proto_active_rate: {:.1}%", anchor_stats.proto_active_rate * 100.0);
    println!("  proto_entropy_early (top10): {:.3}", anchor_stats.proto_entropy_early);
    println!("  proto_entropy_late (top10): {:.3}", anchor_stats.proto_entropy_late);
    let entropy_decreased = anchor_stats.proto_entropy_late < anchor_stats.proto_entropy_early;
    println!("  entropy_decreased: {}", if entropy_decreased { "yes (concepts sharpening)" } else { "no" });
    println!();
    println!("5b Value Learning (Phase 1.7b):");
    println!("  value_updates_total: {}", anchor_stats.value_stats.updates_total);
    println!("  avg_v_used: {:.4}", anchor_stats.value_stats.avg_v_used);
    println!("  avg_abs_td_used: {:.4}", anchor_stats.value_stats.avg_abs_td_used);
    println!("  mean_v_when_r_pos: {:.4} (n={})", anchor_stats.value_stats.mean_v_when_r_pos, anchor_stats.value_stats.r_pos_count);
    println!("  mean_v_when_r_neg: {:.4} (n={})", anchor_stats.value_stats.mean_v_when_r_neg, anchor_stats.value_stats.r_neg_count);
    println!("  delta_v (pos - neg): {:.4}", anchor_stats.value_stats.delta_v());
    println!("  avg_abs_td_early: {:.4}", anchor_stats.value_stats.avg_abs_td_early);
    println!("  avg_abs_td_late: {:.4}", anchor_stats.value_stats.avg_abs_td_late);
    let td_ratio = if anchor_stats.value_stats.avg_abs_td_early > 0.0 {
        anchor_stats.value_stats.avg_abs_td_late / anchor_stats.value_stats.avg_abs_td_early
    } else {
        1.0
    };
    println!("  td_late/early ratio: {:.3}", td_ratio);
    println!("  top5_by_v: (id, v, entropy, support, updates)");
    for (id, v, ent, sup, upd) in &anchor_stats.value_stats.top5_by_v {
        println!("    anchor {}: v={:.3}, entropy={:.2}, support={}, updates={}", id, v, ent, sup, upd);
    }

    // Phase 1.7c diagnostics
    println!();
    println!("Phase 1.7c Diagnostics:");
    let clip_rate = anchor_stats.value_stats.clip_rate();
    println!("  clip_rate: {:.1}% ({} / {} updates clipped)",
        clip_rate,
        anchor_stats.value_stats.clip_count,
        anchor_stats.value_stats.clip_total);
    let hist = anchor_stats.value_stats.reward_hist_pct();
    println!("  reward histogram:");
    println!("    [-1.0, -0.5): {:5.1}%", hist[0]);
    println!("    [-0.5,  0.0): {:5.1}%", hist[1]);
    println!("    [ 0.0,  0.5): {:5.1}%", hist[2]);
    println!("    [ 0.5,  1.0]: {:5.1}%", hist[3]);
    if hist[2] + hist[3] > 90.0 {
        println!("  ⚠ Reward is heavily biased positive (>90%) - may want to tune centering");
    } else {
        println!("  ✓ Reward distribution looks balanced");
    }

    // Phase 1.7d: Probe convergence metrics
    if let Some((probe_size, eval_count, last_mean, last_p95, last_missing, early_mean, late_mean)) = probe_stats {
        println!();
        println!("Phase 1.7d Probe Convergence:");
        let filled = probe_size >= config.probe_min_fill;
        println!("  probe_size: {} (filled={})", probe_size, filled);
        println!("  eval_stride: {} ticks", config.probe_eval_stride);
        println!("  eval_count: {}", eval_count);
        println!("  last_mean_abs_delta_v: {:.6}", last_mean);
        println!("  last_p95_abs_delta_v: {:.6}", last_p95);
        println!("  last_missing: {}", last_missing);
        if eval_count >= 10 {
            println!("  mean_early (first 10 evals): {:.6}", early_mean);
            println!("  mean_late (last 10 evals): {:.6}", late_mean);
            let ratio = if early_mean > 0.0 { late_mean / early_mean } else { 1.0 };
            println!("  probe_late/early ratio: {:.3}", ratio);
            if last_mean < 0.01 || ratio < 1.0 {
                println!("  ✓ Probe ΔV is low (<0.01) — values appear stable");
            } else {
                println!("  ⚠ Probe ΔV not decreasing — values may still be drifting");
            }
        }
    }

    // Phase 1.8: Lifecycle metrics
    println!();
    println!("Phase 1.8 Lifecycle (VALUE IS CONTROL):");
    println!("  stable_count: {}", lifecycle_stats.stable_count);
    println!("  stable_fraction: {:.1}%", lifecycle_stats.stable_fraction * 100.0);
    println!("  mode_transitions: {}", lifecycle_stats.mode_transitions);
    println!("  total_wins: {}", lifecycle_stats.total_wins);
    println!("  final_mode: {}", if lifecycle_stats.final_mode { "STABLE" } else { "EXPLORE" });

    // Phase 1.9: Consolidation metrics
    println!();
    println!("Phase 1.9 Consolidation (MERGES + STABILITY):");
    println!("  merge_scan_runs: {}", lifecycle_stats.merge_scan_runs);
    println!("  merges_done_proto: {}", lifecycle_stats.merges_done_proto);
    println!("  avg_merge_score: {:.3}", lifecycle_stats.avg_merge_score);
    println!("  merge_candidates_found: {}", lifecycle_stats.merge_candidates_found);
    println!("  stable_new: {}", lifecycle_stats.stable_new);
    println!("  stable_dropped: {}", lifecycle_stats.stable_dropped);
    println!("  stable_drop_ratio: {:.1}%", lifecycle_stats.stable_drop_ratio * 100.0);

    // Acceptance check for Phase 1.9
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 1.9 ACCEPTANCE:");
    println!("═══════════════════════════════════════════════════════════════════");

    // Phase 1.6 criteria (must not regress) - slightly relaxed from 1.7a
    let cov_ok = metrics_5b.coverage_pos() >= 0.72; // Phase 1.7b: 74.7% - 2%
    let sel_ok = metrics_5b.selective_accuracy() >= 0.815; // Phase 1.7b: 83.5% - 2%
    let fp_ok = metrics_5b.false_positive_rate() == 0.0; // Must be 0%
    let abs_ok = metrics_5b.abstain_neg_rate() >= 0.99; // Must remain ~100%
    let anc_ok = anchor_stats.anchors_used <= MAX_ANCHORS;
    let thrash_ok = anchor_stats.thrash_rate < 100.0;

    // Phase 1.7a specific criteria
    let proto_updates_ok = anchor_stats.proto_updates > 0;
    let proto_support_ok = anchor_stats.avg_proto_support > 0.0;

    // Phase 1.7b specific criteria
    let v_updates_ok = anchor_stats.value_stats.updates_total > 10000;
    let delta_v_ok = anchor_stats.value_stats.delta_v() >= 0.10;
    let td_decreasing_ok = td_ratio <= 0.95; // At least 5% reduction

    println!("Phase 1.6/1.7a (no regression):");
    println!("  [{}] coverage_pos >= 72%: {:.1}%",
        if cov_ok { "✓" } else { "✗" }, metrics_5b.coverage_pos() * 100.0);
    println!("  [{}] selective_accuracy >= 81.5%: {:.1}%",
        if sel_ok { "✓" } else { "✗" }, metrics_5b.selective_accuracy() * 100.0);
    println!("  [{}] false_positive == 0%: {:.1}%",
        if fp_ok { "✓" } else { "✗" }, metrics_5b.false_positive_rate() * 100.0);
    println!("  [{}] abstain_neg >= 99%: {:.1}%",
        if abs_ok { "✓" } else { "✗" }, metrics_5b.abstain_neg_rate() * 100.0);
    println!("  [{}] anchors_used <= {}: {}",
        if anc_ok { "✓" } else { "✗" }, MAX_ANCHORS, anchor_stats.anchors_used);
    println!("  [{}] thrash_rate < 100 per 10k: {:.2}",
        if thrash_ok { "✓" } else { "✗" }, anchor_stats.thrash_rate);
    println!("  [{}] proto_updates > 0: {}",
        if proto_updates_ok { "✓" } else { "✗" }, anchor_stats.proto_updates);
    println!("  [{}] avg_proto_support > 0: {:.1}",
        if proto_support_ok { "✓" } else { "✗" }, anchor_stats.avg_proto_support);
    println!();
    println!("Phase 1.7b (value learning):");
    println!("  [{}] value_updates > 10000: {}",
        if v_updates_ok { "✓" } else { "✗" }, anchor_stats.value_stats.updates_total);
    println!("  [{}] delta_v (pos - neg) >= 0.10: {:.4}",
        if delta_v_ok { "✓" } else { "✗" }, anchor_stats.value_stats.delta_v());
    println!("  [{}] td_late/early <= 0.95: {:.3}",
        if td_decreasing_ok { "✓" } else { "~" }, td_ratio);

    // Phase 1.7d acceptance criteria
    let (probe_filled_ok, probe_evals_ok, probe_converging_ok, probe_ratio) = if let Some((probe_size, eval_count, last_mean, _, _, early_mean, late_mean)) = probe_stats {
        let filled = probe_size >= config.probe_min_fill;
        let evals = eval_count >= 10;
        let ratio = if early_mean > 0.0 { late_mean / early_mean } else { 1.0 };
        let converging = last_mean < 0.01 || ratio < 1.0;
        (filled, evals, converging, ratio)
    } else {
        (false, false, false, 1.0)
    };
    let probe_last_mean = probe_stats.map(|(_, _, m, _, _, _, _)| m).unwrap_or(0.0);

    println!();
    println!("Phase 1.7d (probe convergence):");
    println!("  [{}] probe_size >= {}: {}",
        if probe_filled_ok { "✓" } else { "✗" }, config.probe_min_fill,
        probe_stats.map(|(s, _, _, _, _, _, _)| s).unwrap_or(0));
    println!("  [{}] probe_evals >= 10: {}",
        if probe_evals_ok { "✓" } else { "✗" },
        probe_stats.map(|(_, e, _, _, _, _, _)| e).unwrap_or(0));
    println!("  [{}] probe_converging (ΔV<0.01 or ratio<1): mean={:.6}, ratio={:.3}",
        if probe_converging_ok { "✓" } else { "✗" }, probe_last_mean, probe_ratio);

    let phase16_ok = cov_ok && sel_ok && fp_ok && abs_ok && anc_ok && thrash_ok;
    let phase17a_ok = proto_updates_ok && proto_support_ok;
    let phase17b_ok = v_updates_ok && delta_v_ok;
    let phase17d_ok = probe_filled_ok && probe_evals_ok && probe_converging_ok;

    // Phase 1.8 acceptance criteria
    let has_wins = lifecycle_stats.total_wins > 0;
    let has_stable_anchors = lifecycle_stats.stable_count > 0;
    let lifecycle_active = has_wins || has_stable_anchors || lifecycle_stats.mode_transitions > 0;

    println!();
    println!("Phase 1.8 (VALUE IS CONTROL):");
    println!("  [{}] has_wins (total > 0): {}",
        if has_wins { "✓" } else { "✗" }, lifecycle_stats.total_wins);
    println!("  [{}] has_stable_anchors (stable_count > 0): {}",
        if has_stable_anchors { "✓" } else { "~" }, lifecycle_stats.stable_count);
    println!("  [{}] lifecycle_active (wins or stable or transitions): {}",
        if lifecycle_active { "✓" } else { "✗" }, lifecycle_active);

    let phase18_ok = has_wins && lifecycle_active;

    // Phase 1.9 acceptance criteria
    let merges_ok = lifecycle_stats.merges_done_proto > 0;
    let drop_ratio_ok = lifecycle_stats.stable_drop_ratio <= 0.35;

    println!();
    println!("Phase 1.9 (CONSOLIDATION):");
    println!("  [{}] merges_done_proto > 0: {}",
        if merges_ok { "✓" } else { "✗" }, lifecycle_stats.merges_done_proto);
    println!("  [{}] stable_drop_ratio <= 35%: {:.1}%",
        if drop_ratio_ok { "✓" } else { "✗" }, lifecycle_stats.stable_drop_ratio * 100.0);
    println!("  [i] avg_merge_score: {:.3}", lifecycle_stats.avg_merge_score);
    println!("  [i] merge_candidates_found: {}", lifecycle_stats.merge_candidates_found);
    println!("  [i] stable_new: {}, stable_dropped: {}",
        lifecycle_stats.stable_new, lifecycle_stats.stable_dropped);

    let phase19_ok = merges_ok && drop_ratio_ok;

    if phase16_ok && phase17a_ok && phase17b_ok && phase17d_ok && phase18_ok && phase19_ok {
        println!();
        println!("  → Phase 1.9: ALL CRITERIA MET!");
    } else if phase16_ok && phase17a_ok && phase17b_ok && phase17d_ok && phase18_ok {
        println!();
        println!("  → Phase 1.8 OK, Phase 1.9 consolidation needs tuning.");
    } else if phase16_ok && phase17a_ok && phase17b_ok && phase17d_ok {
        println!();
        println!("  → Phase 1.7d OK, Phase 1.8 lifecycle needs more data.");
    } else if phase16_ok && phase17a_ok && phase17d_ok {
        println!();
        println!("  → Phase 1.7d OK (probe instrumentation working), value learning needs tuning.");
    } else if phase16_ok && phase17a_ok {
        println!();
        println!("  → Phase 1.7a OK, Phase 1.7b value learning needs tuning.");
    } else if phase16_ok {
        println!();
        println!("  → Phase 1.6 OK, prototype/value criteria need attention.");
    } else {
        println!();
        println!("  → Some criteria not met. Tuning may be needed.");
    }
}

/// Anchor stats for reporting.
struct AnchorStats {
    anchors_used: usize,
    creates: usize,
    evictions: usize,
    merges: usize,
    thrash_rate: f64,
    new_rate: f64,
    avg_hamming: f64,
    p95_hamming: u32,
    utilization: f64,
    gate_pass_rate: f64,
    // Phase 1.7a: Prototype metrics
    proto_updates: usize,
    avg_proto_support: f64,
    proto_active_rate: f64,
    proto_entropy_early: f32,
    proto_entropy_late: f32,
    // Phase 1.7b: Value learning metrics
    value_stats: ValueStats,
}

/// Phase 1.7b: Value learning statistics.
#[derive(Clone, Debug, Default)]
struct ValueStats {
    /// Total value updates across all anchors.
    updates_total: usize,
    /// Average value for anchors with updates.
    avg_v_used: f64,
    /// Average |TD| EMA for anchors with updates.
    avg_abs_td_used: f64,
    /// Mean value when reward was positive (r > 0.05).
    mean_v_when_r_pos: f64,
    /// Mean value when reward was negative (r < -0.05).
    mean_v_when_r_neg: f64,
    /// Count of positive reward events.
    r_pos_count: usize,
    /// Count of negative reward events.
    r_neg_count: usize,
    /// Sum of v for positive reward events.
    sum_v_pos: f64,
    /// Sum of v for negative reward events.
    sum_v_neg: f64,
    /// Average |TD| in early phase (first 20% of ticks).
    avg_abs_td_early: f64,
    /// Average |TD| in late phase (last 20% of ticks).
    avg_abs_td_late: f64,
    /// Top 5 anchors by value.
    top5_by_v: Vec<(u16, f32, f32, u32, u32)>,
    // Phase 1.7c diagnostics
    /// Count of value updates that hit v_clip bounds.
    clip_count: usize,
    /// Total value updates for clip rate calculation.
    clip_total: usize,
    /// Reward histogram: buckets [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]
    reward_hist: [usize; 4],
}

impl ValueStats {
    fn new() -> Self {
        Self::default()
    }

    fn record_reward(&mut self, v: f32, reward: f32) {
        if reward > 0.05 {
            self.r_pos_count += 1;
            self.sum_v_pos += v as f64;
        } else if reward < -0.05 {
            self.r_neg_count += 1;
            self.sum_v_neg += v as f64;
        }
        // Phase 1.7c: Record reward histogram
        let bucket = if reward < -0.5 {
            0 // [-1, -0.5)
        } else if reward < 0.0 {
            1 // [-0.5, 0)
        } else if reward < 0.5 {
            2 // [0, 0.5)
        } else {
            3 // [0.5, 1]
        };
        self.reward_hist[bucket] += 1;
    }

    /// Phase 1.7c: Record a value update and track if it clipped.
    fn record_v_update(&mut self, v_before: f32, v_after: f32, v_clip: f32) {
        self.clip_total += 1;
        if v_after.abs() >= v_clip - 0.001 {
            self.clip_count += 1;
        }
    }

    fn finalize(&mut self) {
        if self.r_pos_count > 0 {
            self.mean_v_when_r_pos = self.sum_v_pos / self.r_pos_count as f64;
        }
        if self.r_neg_count > 0 {
            self.mean_v_when_r_neg = self.sum_v_neg / self.r_neg_count as f64;
        }
    }

    fn delta_v(&self) -> f64 {
        self.mean_v_when_r_pos - self.mean_v_when_r_neg
    }

    /// Phase 1.7c: Get clip rate as percentage.
    fn clip_rate(&self) -> f64 {
        if self.clip_total > 0 {
            100.0 * self.clip_count as f64 / self.clip_total as f64
        } else {
            0.0
        }
    }

    /// Phase 1.7c: Get reward histogram percentages.
    fn reward_hist_pct(&self) -> [f64; 4] {
        let total: usize = self.reward_hist.iter().sum();
        if total > 0 {
            [
                100.0 * self.reward_hist[0] as f64 / total as f64,
                100.0 * self.reward_hist[1] as f64 / total as f64,
                100.0 * self.reward_hist[2] as f64 / total as f64,
                100.0 * self.reward_hist[3] as f64 / total as f64,
            ]
        } else {
            [0.0; 4]
        }
    }
}

// =============================================================================
// Phase 1.7d: Probe Set for Convergence Tracking
// =============================================================================

use std::collections::HashSet;

/// Probe set for tracking value convergence on fixed keys.
struct ProbeSet {
    /// Unique anchor IDs in the probe set.
    keys: Vec<u16>,
    /// Set for quick uniqueness check.
    seen: HashSet<u16>,
    /// Previous values for delta computation.
    prev_vals: Vec<f32>,
    /// Last tick when we evaluated.
    last_eval_tick: u64,
    /// Count of missing keys in last eval.
    missing: usize,
    /// Number of evaluations done.
    eval_count: usize,
    /// Early evaluation means (first 10).
    early_means: Vec<f64>,
    /// Late evaluation means (last 10).
    late_means: Vec<f64>,
    /// Last computed mean_abs_delta.
    last_mean_abs: f64,
    /// Last computed p95_abs_delta.
    last_p95_abs: f64,
}

/// Stats from a probe evaluation.
#[derive(Clone, Debug)]
struct ProbeStats {
    filled: usize,
    used: usize,
    missing_frac: f64,
    mean_abs_delta: f64,
    p95_abs_delta: f64,
}

impl ProbeSet {
    fn new() -> Self {
        ProbeSet {
            keys: Vec::new(),
            seen: HashSet::new(),
            prev_vals: Vec::new(),
            last_eval_tick: 0,
            missing: 0,
            eval_count: 0,
            early_means: Vec::new(),
            late_means: Vec::new(),
            last_mean_abs: 0.0,
            last_p95_abs: 0.0,
        }
    }

    /// Try to add an anchor_id to the probe set (only if not already present and not full).
    fn maybe_add(&mut self, anchor_id: u16, max_size: usize) {
        if anchor_id == 0xFFFF {
            return;
        }
        if self.keys.len() >= max_size {
            return;
        }
        if self.seen.contains(&anchor_id) {
            return;
        }
        self.seen.insert(anchor_id);
        self.keys.push(anchor_id);
    }

    /// Check if we should evaluate this tick.
    fn should_eval(&self, current_tick: u64, stride: u32, min_fill: usize) -> bool {
        if self.keys.len() < min_fill {
            return false;
        }
        current_tick >= self.last_eval_tick + stride as u64
    }

    /// Evaluate probe set and compute delta statistics.
    /// value_fn: closure that returns Option<f32> for an anchor_id.
    fn eval<F>(&mut self, current_tick: u64, value_fn: F) -> Option<ProbeStats>
    where
        F: Fn(u16) -> Option<f32>,
    {
        if self.keys.is_empty() {
            return None;
        }

        // Get current values
        let mut current_vals: Vec<f32> = Vec::with_capacity(self.keys.len());
        let mut valid_indices: Vec<usize> = Vec::new();
        self.missing = 0;

        for (i, &anchor_id) in self.keys.iter().enumerate() {
            if let Some(v) = value_fn(anchor_id) {
                current_vals.push(v);
                valid_indices.push(i);
            } else {
                self.missing += 1;
            }
        }

        let used = current_vals.len();
        if used == 0 {
            return None;
        }

        // First eval: just store values, no delta yet
        if self.prev_vals.is_empty() {
            self.prev_vals = current_vals;
            self.last_eval_tick = current_tick;
            self.eval_count += 1;
            return Some(ProbeStats {
                filled: self.keys.len(),
                used,
                missing_frac: self.missing as f64 / self.keys.len() as f64,
                mean_abs_delta: 0.0,
                p95_abs_delta: 0.0,
            });
        }

        // Compute deltas
        let mut deltas: Vec<f64> = Vec::with_capacity(used);
        for (new_idx, &old_idx) in valid_indices.iter().enumerate() {
            if old_idx < self.prev_vals.len() {
                let delta = (current_vals[new_idx] - self.prev_vals[old_idx]).abs() as f64;
                deltas.push(delta);
            }
        }

        if deltas.is_empty() {
            // Update prev_vals for next time
            self.prev_vals = current_vals;
            self.last_eval_tick = current_tick;
            return None;
        }

        // Compute stats
        let mean_abs = deltas.iter().sum::<f64>() / deltas.len() as f64;

        // p95
        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_idx = (deltas.len() as f64 * 0.95).ceil() as usize;
        let p95_abs = deltas.get(p95_idx.saturating_sub(1)).copied().unwrap_or(0.0);

        // Track early/late means
        self.eval_count += 1;
        if self.eval_count <= 10 {
            self.early_means.push(mean_abs);
        }
        // Always update late_means (keep last 10)
        self.late_means.push(mean_abs);
        if self.late_means.len() > 10 {
            self.late_means.remove(0);
        }

        self.last_mean_abs = mean_abs;
        self.last_p95_abs = p95_abs;

        // Update prev_vals for next eval
        self.prev_vals = current_vals;
        self.last_eval_tick = current_tick;

        Some(ProbeStats {
            filled: self.keys.len(),
            used,
            missing_frac: self.missing as f64 / self.keys.len() as f64,
            mean_abs_delta: mean_abs,
            p95_abs_delta: p95_abs,
        })
    }

    /// Get early mean (average of first 10 evals).
    fn early_mean(&self) -> f64 {
        if self.early_means.is_empty() {
            0.0
        } else {
            self.early_means.iter().sum::<f64>() / self.early_means.len() as f64
        }
    }

    /// Get late mean (average of last 10 evals).
    fn late_mean(&self) -> f64 {
        if self.late_means.is_empty() {
            0.0
        } else {
            self.late_means.iter().sum::<f64>() / self.late_means.len() as f64
        }
    }

    /// Get convergence ratio (late/early).
    fn convergence_ratio(&self) -> f64 {
        let early = self.early_mean();
        if early > 0.0 {
            self.late_mean() / early
        } else {
            1.0
        }
    }
}

/// Compute self-supervised reward signal r_t.
/// Based on power change, coherence (margin), prototype alignment.
fn compute_reward(
    delta_power: f64,
    topk_margin: f64,
    proto_align: f32,
    config: &Config,
) -> f32 {
    // Power component (clamped)
    let power_term = (delta_power as f32).clamp(-config.r_p_clip, config.r_p_clip);

    // Coherence proxy: margin normalized to [0,1]
    let coherence = (topk_margin as f32 / config.margin_norm).clamp(0.0, 1.0);
    // Phase 1.7c: Center to [-1, +1] for zero-mean reward
    let coh_z = 2.0 * coherence - 1.0;

    // Prototype alignment (in [0,1])
    let proto_term = proto_align.clamp(0.0, 1.0);
    // Phase 1.7c: Center to [-1, +1] for zero-mean reward
    let proto_z = 2.0 * proto_term - 1.0;

    // Margin penalty: penalize if margin is below gate threshold
    let gate_margin = ANCHOR_MARGIN_MIN as f32;
    let margin_penalty = ((gate_margin - topk_margin as f32) / gate_margin).clamp(0.0, 1.0);

    // Combine components (using centered terms)
    let reward = config.r_w_power * power_term
        + config.r_w_coh * coh_z
        + config.r_w_proto * proto_z
        - config.r_w_margin * margin_penalty;

    // Clamp final reward to [-1, +1]
    reward.clamp(-1.0, 1.0)
}

use anchor::ANCHOR_MARGIN_MIN;

/// DEMO 5a: Baseline Phase 1.4c (unchanged competitive binding)
fn run_demo_5a_baseline(config: &Config) -> (GlobalLabelMetrics, f64) {
    println!("DEMO 5a: Baseline (Phase 1.4c Competitive Binding)");
    println!("─────────────────────────────────────────────────────");

    let mut rng = Rng::new(config.seed.wrapping_add(0x5A5A_5A5A));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

    let mut memory = GlobalLabelMemoryStore::from_config(config);
    let mut metrics = GlobalLabelMetrics::new();
    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);

    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;

    print!("  Running {} episodes... ", config.competitive_episodes);

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
            let topk = get_top_k(&chamber, config.top_k);
            let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            window.push(&topk_ids, ctx_hat);
            let current_sig = window.competitive_sig();

            if bind_ticks.contains(&t) && window.is_ready() {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                memory.store_competitive(label, global_tick, current_sig.clone());
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 && window.is_ready() {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig = flip_competitive_sig(&current_sig, config.competitive_neg_flip_bits, rng.next_u64());
                    let result = memory.recall_competitive(global_tick, &neg_sig);
                    metrics.record_negative(&result);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let result = memory.recall_competitive(global_tick, &current_sig);
                    metrics.record_positive(&result, true_label);
                }
            }

            global_tick += 1;
        }
    }
    println!("done.");

    println!("  coverage_pos={:.1}%, accuracy_pos={:.1}%, selective_acc={:.1}%",
        metrics.coverage_pos() * 100.0, metrics.accuracy_pos() * 100.0, metrics.selective_accuracy() * 100.0);

    let stability = window.stability();
    (metrics, stability)
}

/// Phase 1.8/1.9 lifecycle stats for reporting.
struct LifecycleStats {
    stable_count: usize,
    stable_fraction: f64,
    mode_transitions: usize,
    total_wins: u32,
    final_mode: bool,
    // Phase 1.9: Consolidation metrics
    merges_done: usize,
    merge_candidates_found: usize,
    stable_new: usize,
    stable_dropped: usize,
    stable_drop_ratio: f64,
    // Phase 1.9: Additional merge stats
    merge_scan_runs: usize,
    merges_done_proto: usize,
    avg_merge_score: f32,
}

/// DEMO 5b: Anchor Concept Tokens with Phase 1.9 CONSOLIDATION
fn run_demo_5b_keyed(config: &Config) -> (KeyedMemoryMetrics, AnchorStats, (usize, f64, usize), Option<(usize, usize, f64, f64, usize, f64, f64)>, LifecycleStats) {
    println!("DEMO 5b: Phase 1.9 CONSOLIDATION (Merges + Stability Hysteresis)");
    println!("─────────────────────────────────────────────────────────────────");

    // Use same seed as 5a for fair comparison
    let mut rng = Rng::new(config.seed.wrapping_add(0x5A5A_5A5A));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train (same as 5a)
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

    // Phase 1.6 components
    let mut anchor_bank = AnchorBank::new();
    let keyed_config = KeyedMemoryConfig {
        label_min_p: 0.50,
        label_margin: 0.10,
        alpha: 0.5,
        num_labels: config.num_ctx,
    };
    let mut keyed_memory = KeyedMemoryStore::new(keyed_config);
    let mut metrics = KeyedMemoryMetrics::new();

    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);
    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;

    // Phase 1.7a: Track entropy at early and late stages
    let early_checkpoint = config.competitive_episodes / 4; // 25%
    let mut proto_entropy_early: f32 = 0.0;

    // Phase 1.7b: Value learning state
    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut value_stats = ValueStats::new();

    // Phase 1.7d: Probe set for convergence tracking
    let mut probe_set = ProbeSet::new();
    let mut reward_ema: f32 = 0.0;

    // Phase 1.7b: Early/late TD tracking
    let total_ticks = (config.competitive_episodes * config.competitive_episode_ticks) as u64;
    let early_end_tick = total_ticks / 5; // First 20%
    let late_start_tick = total_ticks * 4 / 5; // Last 20%
    let mut early_td_sum: f64 = 0.0;
    let mut early_td_count: usize = 0;
    let mut late_td_sum: f64 = 0.0;
    let mut late_td_count: usize = 0;

    print!("  Running {} episodes... ", config.competitive_episodes);

    for _ep in 0..config.competitive_episodes {
        // Phase 1.7a: Capture early entropy
        if _ep == early_checkpoint {
            let (entropy, _) = anchor_bank.proto_entropy_top_n(10, config.proto_m);
            proto_entropy_early = entropy;
        }
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
            let topk = get_top_k(&chamber, config.top_k);
            let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            // Phase 1.6c: Compute confidence info for gating
            // Use topk amplitudes to compute margin
            let topk_margin = if topk.len() >= 2 {
                topk[0].1 - topk[1].1
            } else if !topk.is_empty() {
                topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            // Phase 1.6b: Periodic Hamming-based merge (legacy)
            // Phase 1.8: Pass config for value consistency check
            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            // Phase 1.9: Proto-based merge scanning (new)
            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            // Phase 1.8/1.9: Update stability with hysteresis
            anchor_bank.update_stability(global_tick, config);

            // Phase 1.8: Get dynamic gate params based on current mode
            let gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            // Resolve signature to anchor with confidence gating
            // Phase 1.8: Pass config for value-aware eviction
            let (anchor_id, _is_new, _match_dist) = anchor_bank.resolve_gated(
                sig_mask,
                global_tick,
                Some(&confidence),
                Some(config),
            );

            // Phase 1.7a: Update anchor prototype when gate passes
            // Phase 1.8: Use dynamic gate params
            if confidence.passes_gate_with_params(&gate_params) && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &topk, config);
                // Phase 1.7d: Add to probe set
                if config.probe_enabled {
                    probe_set.maybe_add(anchor_id, config.probe_size);
                }
            }

            // Phase 1.7b: Compute reward and TD update for previous anchor
            if prev_anchor_id != 0xFFFF {
                // Compute current V for TD target
                // Phase 1.7c: Use v_abstain_margin for margin-fail cases
                // Phase 1.8: Use dynamic gate params
                let gate_passed = confidence.passes_gate_with_params(&gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * gate_params.margin_mult {
                    // Margin fail: use negative bootstrap to penalize uncertain states
                    config.v_abstain_margin
                } else {
                    0.0 // Other abstain reasons -> V_next = 0
                };

                // Get previous anchor's current value (before update)
                let v_prev = anchor_bank.get_value(prev_anchor_id);

                // Compute prototype alignment for reward
                let proto_align_for_reward = prev_proto_align;

                // Compute reward based on previous tick's state
                let delta_power = total_power - prev_power;
                let mut reward = compute_reward(delta_power, prev_topk_margin, proto_align_for_reward, config);

                // Phase 1.7d: Optional advantage reward centering
                reward_ema = (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }

                // TD(0) error
                let td = reward + config.gamma_v * v_next - v_prev;

                // Update previous anchor's value
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);

                // Phase 1.7c: Track clip rate
                let v_after = anchor_bank.get_value(prev_anchor_id);
                value_stats.record_v_update(v_prev, v_after, config.v_clip);

                // Track reward-value correlation
                value_stats.record_reward(v_prev, reward);

                // Track early/late TD
                if global_tick < early_end_tick {
                    early_td_sum += td.abs() as f64;
                    early_td_count += 1;
                } else if global_tick >= late_start_tick {
                    late_td_sum += td.abs() as f64;
                    late_td_count += 1;
                }
            }

            // Phase 1.7b: Update previous state for next tick
            // Phase 1.8: Use dynamic gate params
            if confidence.passes_gate_with_params(&gate_params) && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                // Compute prototype alignment for current anchor
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                // Gate failed - reset previous state
                prev_anchor_id = 0xFFFF;
            }

            // Use ctx_hat as learned_mask
            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            // Store at bind ticks
            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            // Query phase
            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    // For negative query: perturb the signature
                    let neg_sig_mask = flip_bits_simple(sig_mask, config.competitive_neg_flip_bits, rng.next_u64());
                    // Don't gate negative queries - just resolve
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    // Phase 1.8: Record wins for correct positive recalls
                    if let anchor::KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;

            // Phase 1.7d: Periodic probe evaluation
            if config.probe_enabled && probe_set.should_eval(global_tick, config.probe_eval_stride, config.probe_min_fill) {
                let _stats = probe_set.eval(global_tick, |id| {
                    Some(anchor_bank.get_value(id))
                });
            }
        }
    }
    println!("done.");

    println!("  coverage_pos={:.1}%, accuracy_pos={:.1}%, selective_acc={:.1}%",
        metrics.coverage_pos() * 100.0, metrics.accuracy_pos() * 100.0, metrics.selective_accuracy() * 100.0);

    // Phase 1.7a: Capture late entropy
    let (proto_entropy_late, _) = anchor_bank.proto_entropy_top_n(10, config.proto_m);

    // Phase 1.7b: Finalize value stats
    let (v_updates_total, avg_v, avg_abs_td) = anchor_bank.value_metrics();
    value_stats.updates_total = v_updates_total;
    value_stats.avg_v_used = avg_v;
    value_stats.avg_abs_td_used = avg_abs_td;
    value_stats.avg_abs_td_early = if early_td_count > 0 { early_td_sum / early_td_count as f64 } else { 0.0 };
    value_stats.avg_abs_td_late = if late_td_count > 0 { late_td_sum / late_td_count as f64 } else { 0.0 };
    value_stats.top5_by_v = anchor_bank.top_n_by_value(5, config.proto_m);
    value_stats.finalize();

    let anchor_stats = AnchorStats {
        anchors_used: anchor_bank.anchors_used(),
        creates: anchor_bank.anchor_creates,
        evictions: anchor_bank.anchor_evictions,
        merges: anchor_bank.anchor_merges,
        thrash_rate: anchor_bank.thrash_rate(),
        new_rate: anchor_bank.new_anchor_rate(),
        avg_hamming: anchor_bank.avg_match_hamming(),
        p95_hamming: anchor_bank.p95_match_hamming(),
        utilization: anchor_bank.anchor_utilization(),
        gate_pass_rate: anchor_bank.gate_pass_rate(),
        // Phase 1.7a
        proto_updates: anchor_bank.proto_updates,
        avg_proto_support: anchor_bank.avg_proto_support(),
        proto_active_rate: anchor_bank.proto_active_rate(),
        proto_entropy_early,
        proto_entropy_late,
        // Phase 1.7b
        value_stats,
    };

    let keyed_stats = (
        keyed_memory.num_keys(),
        keyed_memory.entry_hit_rate(),
        keyed_memory.keys_remapped(),
    );

    // Phase 1.7d: Store probe stats for reporting
    let probe_stats = if config.probe_enabled {
        Some((
            probe_set.keys.len(),
            probe_set.eval_count,
            probe_set.last_mean_abs,
            probe_set.last_p95_abs,
            probe_set.missing,
            probe_set.early_mean(),
            probe_set.late_mean(),
        ))
    } else {
        None
    };

    // Phase 1.8: Lifecycle stats
    let (stable_count, stable_fraction, mode_transitions, _, final_mode) = anchor_bank.lifecycle_metrics();
    // Phase 1.9: Consolidation metrics
    let (merge_candidates_found, _, _, stable_new, stable_dropped) = anchor_bank.consolidation_metrics();
    let (merge_scan_runs, merges_done_proto, avg_merge_score, _) = anchor_bank.merge_stats();
    let lifecycle_stats = LifecycleStats {
        stable_count,
        stable_fraction,
        mode_transitions,
        total_wins: anchor_bank.total_wins(),
        final_mode,
        // Phase 1.9
        merges_done: anchor_bank.anchor_merges,
        merge_candidates_found,
        stable_new,
        stable_dropped,
        stable_drop_ratio: anchor_bank.stable_drop_ratio(),
        // Phase 1.9: Additional merge stats
        merge_scan_runs,
        merges_done_proto,
        avg_merge_score,
    };

    (metrics, anchor_stats, keyed_stats, probe_stats, lifecycle_stats)
}

/// Simple bit flip helper for negative queries
fn flip_bits_simple(signature: u64, n_bits: u32, rng_val: u64) -> u64 {
    let mut result = signature;
    let mut val = rng_val;
    for _ in 0..n_bits {
        let bit = (val % 64) as u32;
        result ^= 1u64 << bit;
        val = val.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    result
}
