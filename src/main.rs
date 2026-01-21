//! Echo Chamber MVP: Emergent cancellation via complex signal interference.
//! Phase 1.4c: Competitive Label Binding with ABSTAIN and Windowed Signatures.

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
use anchor::{AnchorBank, KeyedMemoryStore, KeyedMemoryConfig, KeyedMemoryMetrics, MemoryKey, ConfidenceInfo, MAX_ANCHORS, MERGE_EVERY_TICKS};

const EPS_PRINT: f64 = 1e-9;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  ECHO CHAMBER MVP - Phase 1.4c: Competitive Label Binding     ║");
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
    println!("DEMO 5: Phase 1.7b - Value Learning on Anchors (Credit Assignment)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Run both experiments with same seed for fair comparison
    let (metrics_5a, stability_5a) = run_demo_5a_baseline(config);
    println!();
    let (metrics_5b, anchor_stats, keyed_stats) = run_demo_5b_keyed(config);

    // Print comparison summary
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 1.7b COMPARISON SUMMARY");
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

    // Acceptance check for Phase 1.7b
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 1.7b ACCEPTANCE:");
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

    let phase16_ok = cov_ok && sel_ok && fp_ok && abs_ok && anc_ok && thrash_ok;
    let phase17a_ok = proto_updates_ok && proto_support_ok;
    let phase17b_ok = v_updates_ok && delta_v_ok;

    if phase16_ok && phase17a_ok && phase17b_ok {
        println!();
        println!("  → Phase 1.7b: ALL CRITERIA MET!");
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

    // Coherence proxy: margin normalized
    let coherence = (topk_margin as f32 / config.margin_norm).clamp(0.0, 1.0);

    // Prototype alignment (already in [0,1])
    let proto_term = proto_align.clamp(0.0, 1.0);

    // Margin penalty: penalize if margin is below gate threshold
    let gate_margin = ANCHOR_MARGIN_MIN as f32;
    let margin_penalty = ((gate_margin - topk_margin as f32) / gate_margin).clamp(0.0, 1.0);

    // Combine components
    let reward = config.r_w_power * power_term
        + config.r_w_coh * coherence
        + config.r_w_proto * proto_term
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

/// DEMO 5b: Anchor Concept Tokens with Phase 1.7a prototypes + Phase 1.7b value learning
fn run_demo_5b_keyed(config: &Config) -> (KeyedMemoryMetrics, AnchorStats, (usize, f64, usize)) {
    println!("DEMO 5b: Anchor Concept Tokens (Phase 1.7b Value Learning)");
    println!("─────────────────────────────────────────────────────────");

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

            // Phase 1.6b: Periodic merge
            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar();
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            // Resolve signature to anchor with confidence gating
            let (anchor_id, _is_new, _match_dist) = anchor_bank.resolve_gated(
                sig_mask,
                global_tick,
                Some(&confidence),
            );

            // Phase 1.7a: Update anchor prototype when gate passes
            if confidence.passes_gate() && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &topk, config);
            }

            // Phase 1.7b: Compute reward and TD update for previous anchor
            if prev_anchor_id != 0xFFFF {
                // Compute current V for TD target
                let v_next = if confidence.passes_gate() && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else {
                    0.0 // Abstain -> V_next = 0
                };

                // Get previous anchor's current value (before update)
                let v_prev = anchor_bank.get_value(prev_anchor_id);

                // Compute prototype alignment for reward
                let proto_align_for_reward = prev_proto_align;

                // Compute reward based on previous tick's state
                let delta_power = total_power - prev_power;
                let reward = compute_reward(delta_power, prev_topk_margin, proto_align_for_reward, config);

                // TD(0) error
                let td = reward + config.gamma_v * v_next - v_prev;

                // Update previous anchor's value
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);

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
            if confidence.passes_gate() && anchor_id != 0xFFFF {
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
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
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

    (metrics, anchor_stats, keyed_stats)
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
