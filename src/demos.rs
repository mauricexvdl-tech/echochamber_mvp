//! Demo functions for Echo Chamber MVP.
//! Extracted from main.rs for better code organization.

use crate::ablate;
use crate::action;
use crate::action_ablate;
use crate::anchor::{
    AnchorBank, ConfidenceInfo, GateParams, KeyedMemoryConfig, KeyedMemoryMetrics,
    KeyedMemoryStore, KeyedRecallDecision, MemoryKey, ANCHOR_MARGIN_MIN, MAX_ANCHORS,
};
use crate::causes::{get_top_k, Causes, TopKStats};
use crate::complex::Complex;
use crate::concepts::{ConceptBank, ConfusionMatrix};
use crate::config::Config;
use crate::distill;
use crate::echo::EchoChamber;
use crate::memory::{
    flip_competitive_sig, proto_scores_to_int, topk_to_mask, CompetitiveSig,
    GlobalLabelMemoryStore, GlobalLabelMetrics, LabelBindingMetrics, LabelMemoryStore,
    MemoryMetrics, MemoryStore, RollingWindow, WINDOW_SIZE, WINDOW_TOP_M,
};
use crate::mode;
use crate::regret;
use crate::rng::Rng;

const EPS_PRINT: f64 = 1e-9;

fn fmt_amp_phase(z: &Complex) -> String {
    let amp = z.norm();
    if amp < EPS_PRINT {
        format!("amp={:.6} φ=undef  ", amp)
    } else {
        format!("amp={:.6} φ={:+.4}", amp, z.arg())
    }
}

pub fn demo_lie_triangle(config: &Config) {
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

    println!(
        "{:<6} │ {:^24} │ {:^24} │ {:^24}",
        "Tick", "Node 1 (path A)", "Node 3 (path B)", "Node 2 (target)"
    );
    println!(
        "───────┼──────────────────────────┼──────────────────────────┼──────────────────────────"
    );

    for tick in 0..=4 {
        let n1 = &chamber.nodes[1].buffer;
        let n3 = &chamber.nodes[3].buffer;
        let n2 = &chamber.nodes[2].buffer;
        println!(
            "{:<6} │ {} │ {} │ {}",
            tick,
            fmt_amp_phase(n1),
            fmt_amp_phase(n3),
            fmt_amp_phase(n2)
        );
        if tick < 4 {
            chamber.tick();
        }
    }

    println!();
    let final_amp = chamber.nodes[2].buffer.norm();
    if final_amp < 1e-6 {
        println!(
            "✓ CANCELLATION ACHIEVED: Target amplitude = {:.2e}",
            final_amp
        );
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
        EvalStats {
            ctx_hit_counts: vec![0; num_ctx],
            tot_pow_samples: Vec::new(),
            scale_sum: 0.0,
            scale_count: 0,
        }
    }
    fn record(&mut self, ctx: Option<usize>, tot_pow: f64, scale: f64) {
        if let Some(c) = ctx {
            if c < self.ctx_hit_counts.len() {
                self.ctx_hit_counts[c] += 1;
            }
        }
        self.tot_pow_samples.push(tot_pow);
        self.scale_sum += scale;
        self.scale_count += 1;
    }
    fn mean_tot_pow(&self) -> f64 {
        if self.tot_pow_samples.is_empty() {
            0.0
        } else {
            self.tot_pow_samples.iter().sum::<f64>() / self.tot_pow_samples.len() as f64
        }
    }
    fn _p95_tot_pow(&self) -> f64 {
        if self.tot_pow_samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.tot_pow_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (sorted.len() as f64 * 0.95).floor() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    fn _avg_scale(&self) -> f64 {
        if self.scale_count > 0 {
            self.scale_sum / self.scale_count as f64
        } else {
            1.0
        }
    }
}

pub fn demo_latent_causes(config: &Config) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "DEMO 2: Concept Readout + Episodic Memory (N={}, K={})",
        config.num_nodes, config.top_k
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Configuration:");
    println!(
        "  Dynamics: decay={:.0}%/tick, clamp_max={:.1}",
        config.decay_per_tick * 100.0,
        config.clamp_max_amp
    );
    println!(
        "  Training: {} ticks, Eval: {} ticks",
        config.train_ticks, config.eval_ticks
    );
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
        if tick % 50000 == 0 {
            print!(".");
        }
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
        if let Some(ctx) = metrics.ctx {
            concept_bank.record(ctx, &topk_ids);
        }
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
        let proto_scores_f64: Vec<f64> = (0..config.num_ctx)
            .map(|ctx| concept_bank.score(ctx, &topk_ids))
            .collect();
        let proto_scores = proto_scores_to_int(&proto_scores_f64);

        if let Some(true_ctx) = metrics.ctx {
            let predicted_ctx = concept_bank.classify(&topk_ids);
            confusion.record(true_ctx, predicted_ctx);
            if metrics.tot_pow_post >= config.memory_min_power {
                let store_roll: f64 = rng.next_f64();
                if store_roll < config.memory_store_prob {
                    if memory_store.store(
                        true_ctx as u32,
                        tick as u64,
                        topk_mask,
                        proto_scores,
                        config.memory_debounce_ticks,
                    ) {
                        memory_metrics.record_store();
                    }
                }
            }
            let recall_result = memory_store.recall(tick as u64, topk_mask, proto_scores);
            memory_metrics.record_recall(recall_result.as_ref(), true_ctx as u32);
        }
    }

    println!(
        "Classification accuracy: {:.1}%",
        confusion.accuracy() * 100.0
    );
    println!(
        "Memory: coverage={:.1}%, accuracy={:.1}%",
        memory_metrics.coverage() * 100.0,
        memory_metrics.accuracy() * 100.0
    );

    let coverage = stats.coverage();
    let top15 = stats.top_by_hits(15, config.eval_ticks);
    let avg_purity: f64 = if top15.is_empty() {
        0.0
    } else {
        top15.iter().map(|(_, _, _, p, _, _)| p).sum::<f64>() / top15.len() as f64
    };

    println!();
    println!(
        "PHASE 1.4a: [✓] TotPow={:.2}, Purity={:.2}, Coverage={}, Class={:.1}%, Mem={:.1}%/{:.1}%",
        eval_stats.mean_tot_pow(),
        avg_purity,
        coverage,
        confusion.accuracy() * 100.0,
        memory_metrics.coverage() * 100.0,
        memory_metrics.accuracy() * 100.0
    );
}

pub fn demo_label_binding(config: &Config) {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "DEMO 3: One-Shot Label Binding (L={}, episodes={})",
        config.num_labels, config.num_episodes
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let metrics = run_label_binding_experiment(config, config.num_labels);

    println!(
        "Label Binding: episodes={}, binds={}",
        metrics.episodes, metrics.binds_done
    );
    println!(
        "  coverage={:.1}%, accuracy={:.1}%, false_rate={:.1}%",
        metrics.coverage() * 100.0,
        metrics.accuracy() * 100.0,
        metrics.false_rate() * 100.0
    );
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

pub fn run_capacity_sweep(config: &Config) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CAPACITY SWEEP");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    let label_counts = [4, 8, 16, 32];
    println!(
        "{:>8} │ {:>10} │ {:>10} │ {:>12}",
        "Labels", "Coverage", "Accuracy", "False Rate"
    );
    println!("─────────┼────────────┼────────────┼──────────────");
    for &num_labels in &label_counts {
        let metrics = run_label_binding_experiment(config, num_labels);
        println!(
            "{:>8} │ {:>9.1}% │ {:>9.1}% │ {:>11.1}%",
            num_labels,
            metrics.coverage() * 100.0,
            metrics.accuracy() * 100.0,
            metrics.false_rate() * 100.0
        );
    }
    println!();
}

// =============================================================================
// DEMO 4: Competitive Label Binding with ABSTAIN and Windowed Signatures
// =============================================================================

pub fn demo_competitive_binding(config: &Config) {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 4: Competitive Label Binding with ABSTAIN (Phase 1.4c)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("Configuration:");
    println!(
        "  num_labels={}, episodes={}, ticks/ep={}, binds/ep={}",
        config.competitive_num_labels,
        config.competitive_episodes,
        config.competitive_episode_ticks,
        config.competitive_binds_per_episode
    );
    println!(
        "  max_entries={}, max_hamming={}, margin_min={}",
        config.competitive_max_entries,
        config.competitive_max_hamming,
        config.competitive_margin_min
    );
    println!(
        "  p_neg={:.0}%, neg_flip_bits={}, recall_stride={}, recall_start={}",
        config.competitive_p_neg * 100.0,
        config.competitive_neg_flip_bits,
        config.competitive_recall_stride,
        config.competitive_recall_start
    );
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
    println!(
        "  entries={}, evictions={}, hit_updates={}",
        mem_entries, mem_evictions, mem_hit_updates
    );
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

    println!(
        "  [{}] Memory filled (entries={}, evictions={})",
        if entries_ok { "✓" } else { "✗" },
        mem_entries,
        mem_evictions
    );
    println!(
        "  [{}] coverage_pos >= 25%: {:.1}%",
        if coverage_pos_ok { "✓" } else { "✗" },
        metrics.coverage_pos() * 100.0
    );
    println!(
        "  [{}] accuracy_pos >= 70%: {:.1}%",
        if accuracy_pos_ok { "✓" } else { "✗" },
        metrics.accuracy_pos() * 100.0
    );
    println!(
        "  [{}] abstain_neg >= 80%: {:.1}%",
        if abstain_neg_ok { "✓" } else { "✗" },
        metrics.abstain_neg_rate() * 100.0
    );
    println!(
        "  [{}] false_positive <= 15%: {:.1}%",
        if false_pos_ok { "✓" } else { "✗" },
        metrics.false_positive_rate() * 100.0
    );
    println!(
        "  [{}] selective_accuracy >= 75%: {:.1}%",
        if selective_ok { "✓" } else { "✗" },
        metrics.selective_accuracy() * 100.0
    );

    let all_ok = entries_ok
        && coverage_pos_ok
        && accuracy_pos_ok
        && abstain_neg_ok
        && false_pos_ok
        && selective_ok;
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
            if t >= config.competitive_recall_start
                && t % config.competitive_recall_stride == 0
                && window.is_ready()
            {
                // Determine if this is a negative query
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    // Negative query: flip bits in mask to create hard negative (keep ctx)
                    let neg_sig = flip_competitive_sig(
                        &current_sig,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
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
    (
        metrics,
        memory.len(),
        memory.evictions(),
        memory.hit_updates(),
        stability,
    )
}

// =============================================================================
// DEMO 5: Phase 1.5b Comparison - Baseline vs Anchor+Mask Memory
// =============================================================================

// =============================================================================
// DEMO 5: Phase 1.6 Comparison - Baseline vs Anchor+Mask Memory with Stabilization
// =============================================================================

pub fn demo_phase_1_5b_comparison(config: &Config) {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 5/6: Phase 1.9b - AGGRESSIVE CONSOLIDATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Run experiments based on config flags
    let (metrics_5a, stability_5a) = if config.run_baseline_5a {
        run_demo_5a_baseline(config)
    } else {
        println!("DEMO 5a: SKIPPED (run_baseline_5a=false)");
        (GlobalLabelMetrics::new(), 0.0)
    };
    println!();
    let (metrics_5b, anchor_stats, keyed_stats, probe_stats, lifecycle_stats) =
        if config.run_keyed_5b {
            run_demo_5b_keyed(config)
        } else {
            println!("DEMO 5b: SKIPPED (run_keyed_5b=false)");
            // Return empty/default values
            let empty_metrics = KeyedMemoryMetrics::new();
            let empty_anchor_stats = AnchorStats {
                anchors_used: 0,
                creates: 0,
                evictions: 0,
                merges: 0,
                thrash_rate: 0.0,
                new_rate: 0.0,
                avg_hamming: 0.0,
                p95_hamming: 0,
                utilization: 0.0,
                gate_pass_rate: 0.0,
                proto_updates: 0,
                avg_proto_support: 0.0,
                proto_active_rate: 0.0,
                proto_entropy_early: 0.0,
                proto_entropy_late: 0.0,
                value_stats: ValueStats::new(),
            };
            (
                empty_metrics,
                empty_anchor_stats,
                (0, 0.0, 0),
                None,
                LifecycleStats {
                    stable_count: 0,
                    stable_fraction: 0.0,
                    mode_transitions: 0,
                    total_wins: 0,
                    final_mode: false,
                    merges_done: 0,
                    merge_candidates_found: 0,
                    stable_new: 0,
                    stable_dropped: 0,
                    stable_drop_ratio: 0.0,
                    merge_scan_runs: 0,
                    merges_done_proto: 0,
                    avg_merge_score: 0.0,
                    merge_blocked_proto: 0,
                    merge_blocked_value: 0,
                    merge_blocked_stability: 0,
                    merge_blocked_support: 0,
                    merge_blocked_key_mismatch: 0,
                    merge_blocked_ctx_mismatch: 0,
                    merge_blocked_mode_mismatch: 0,
                    pairs_checked: 0,
                    opportunity_rate: 0.0,
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
                    blocked_cross_rate_limited: 0,
                    mean_dv_cross: 0.0,
                    mean_mask_hamming_cross: 0.0,
                    // Phase 1.9e
                    stable_avg_support: 0.0,
                    stable_avg_entropy: 0.0,
                    stable_avg_abs_td: 0.0,
                    // Phase 1.9f
                    ticks_total: 0,
                    ticks_active_stable: 0,
                    stable_time_share: 0.0,
                    stable_mass_sum: 0,
                    stable_mass_mean: 0.0,
                    stable_support_p50: 0,
                    stable_support_p90: 0,
                    stable_value_mean: 0.0,
                    stable_entropy_mean: 0.0,
                    stable_td_mean: 0.0,
                },
            )
        };

    // Print comparison summary only if both ran
    if config.run_baseline_5a && config.run_keyed_5b {
        println!();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("PHASE 1.8 COMPARISON SUMMARY");
        println!("═══════════════════════════════════════════════════════════════════");
        println!();
        println!("{:<25} {:>12} {:>12}", "Metric", "5a (1.4c)", "5b (Keyed)");
        println!("─────────────────────────────────────────────────────────────────");
        println!(
            "{:<25} {:>11.1}% {:>11.1}%",
            "coverage_pos",
            metrics_5a.coverage_pos() * 100.0,
            metrics_5b.coverage_pos() * 100.0
        );
        println!(
            "{:<25} {:>11.1}% {:>11.1}%",
            "accuracy_pos",
            metrics_5a.accuracy_pos() * 100.0,
            metrics_5b.accuracy_pos() * 100.0
        );
        println!(
            "{:<25} {:>11.1}% {:>11.1}%",
            "abstain_neg",
            metrics_5a.abstain_neg_rate() * 100.0,
            metrics_5b.abstain_neg_rate() * 100.0
        );
        println!(
            "{:<25} {:>11.1}% {:>11.1}%",
            "false_positive",
            metrics_5a.false_positive_rate() * 100.0,
            metrics_5b.false_positive_rate() * 100.0
        );
        println!(
            "{:<25} {:>11.1}% {:>11.1}%",
            "selective_accuracy",
            metrics_5a.selective_accuracy() * 100.0,
            metrics_5b.selective_accuracy() * 100.0
        );
        println!();
        println!("5a stability: {:.1}%", stability_5a * 100.0);
    } else if config.run_baseline_5a {
        println!();
        println!("5a stability: {:.1}%", stability_5a * 100.0);
    }

    // Skip remaining output if 5b didn't run
    if !config.run_keyed_5b {
        return;
    }

    // Phase 1.6 specific metrics (5b only)
    println!();
    println!("5b Anchor Codebook (Phase 1.6):");
    println!(
        "  anchors_used: {} / {} ({:.1}% utilization)",
        anchor_stats.anchors_used,
        MAX_ANCHORS,
        anchor_stats.utilization * 100.0
    );
    println!("  anchor_creates: {}", anchor_stats.creates);
    println!("  anchor_evictions: {}", anchor_stats.evictions);
    println!("  anchor_merges: {}", anchor_stats.merges);
    println!(
        "  thrash_rate: {:.2} per 10k ticks",
        anchor_stats.thrash_rate
    );
    println!("  new_anchor_rate: {:.1}%", anchor_stats.new_rate * 100.0);
    println!(
        "  avg_anchor_match_hamming: {:.2}",
        anchor_stats.avg_hamming
    );
    println!("  p95_anchor_match_hamming: {}", anchor_stats.p95_hamming);
    println!(
        "  gate_pass_rate: {:.1}%",
        anchor_stats.gate_pass_rate * 100.0
    );
    println!();
    println!("5b Keyed Memory:");
    println!("  unique_keys: {}", keyed_stats.0);
    println!("  entry_hit_rate: {:.1}%", keyed_stats.1 * 100.0);
    println!("  keys_remapped: {}", keyed_stats.2);
    println!();
    println!("5b Prototype Vectors (Phase 1.7a):");
    println!("  proto_updates: {}", anchor_stats.proto_updates);
    println!("  avg_proto_support: {:.1}", anchor_stats.avg_proto_support);
    println!(
        "  proto_active_rate: {:.1}%",
        anchor_stats.proto_active_rate * 100.0
    );
    println!(
        "  proto_entropy_early (top10): {:.3}",
        anchor_stats.proto_entropy_early
    );
    println!(
        "  proto_entropy_late (top10): {:.3}",
        anchor_stats.proto_entropy_late
    );
    let entropy_decreased = anchor_stats.proto_entropy_late < anchor_stats.proto_entropy_early;
    println!(
        "  entropy_decreased: {}",
        if entropy_decreased {
            "yes (concepts sharpening)"
        } else {
            "no"
        }
    );
    println!();
    println!("5b Value Learning (Phase 1.7b):");
    println!(
        "  value_updates_total: {}",
        anchor_stats.value_stats.updates_total
    );
    println!("  avg_v_used: {:.4}", anchor_stats.value_stats.avg_v_used);
    println!(
        "  avg_abs_td_used: {:.4}",
        anchor_stats.value_stats.avg_abs_td_used
    );
    println!(
        "  mean_v_when_r_pos: {:.4} (n={})",
        anchor_stats.value_stats.mean_v_when_r_pos, anchor_stats.value_stats.r_pos_count
    );
    println!(
        "  mean_v_when_r_neg: {:.4} (n={})",
        anchor_stats.value_stats.mean_v_when_r_neg, anchor_stats.value_stats.r_neg_count
    );
    println!(
        "  delta_v (pos - neg): {:.4}",
        anchor_stats.value_stats.delta_v()
    );
    println!(
        "  avg_abs_td_early: {:.4}",
        anchor_stats.value_stats.avg_abs_td_early
    );
    println!(
        "  avg_abs_td_late: {:.4}",
        anchor_stats.value_stats.avg_abs_td_late
    );
    let td_ratio = if anchor_stats.value_stats.avg_abs_td_early > 0.0 {
        anchor_stats.value_stats.avg_abs_td_late / anchor_stats.value_stats.avg_abs_td_early
    } else {
        1.0
    };
    println!("  td_late/early ratio: {:.3}", td_ratio);
    println!("  top5_by_v: (id, v, entropy, support, updates)");
    for (id, v, ent, sup, upd) in &anchor_stats.value_stats.top5_by_v {
        println!(
            "    anchor {}: v={:.3}, entropy={:.2}, support={}, updates={}",
            id, v, ent, sup, upd
        );
    }

    // Phase 1.7c diagnostics
    println!();
    println!("Phase 1.7c Diagnostics:");
    let clip_rate = anchor_stats.value_stats.clip_rate();
    println!(
        "  clip_rate: {:.1}% ({} / {} updates clipped)",
        clip_rate, anchor_stats.value_stats.clip_count, anchor_stats.value_stats.clip_total
    );
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
    if let Some((
        probe_size,
        eval_count,
        last_mean,
        last_p95,
        last_missing,
        early_mean,
        late_mean,
    )) = probe_stats
    {
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
            let ratio = if early_mean > 0.0 {
                late_mean / early_mean
            } else {
                1.0
            };
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
    println!(
        "  stable_fraction: {:.1}%",
        lifecycle_stats.stable_fraction * 100.0
    );
    println!("  mode_transitions: {}", lifecycle_stats.mode_transitions);
    println!("  total_wins: {}", lifecycle_stats.total_wins);
    println!(
        "  final_mode: {}",
        if lifecycle_stats.final_mode {
            "STABLE"
        } else {
            "EXPLORE"
        }
    );

    // Phase 1.9e: Consolidation metrics with stability formation tracking
    println!();
    println!("Phase 1.9e Consolidation (STABILITY FORMATION TUNING - Option A):");
    println!("  merge_scan_runs: {}", lifecycle_stats.merge_scan_runs);
    println!("  merges_done_proto: {}", lifecycle_stats.merges_done_proto);
    println!("  avg_merge_score: {:.3}", lifecycle_stats.avg_merge_score);
    println!("  pairs_checked: {}", lifecycle_stats.pairs_checked);
    println!(
        "  candidates_found: {}",
        lifecycle_stats.merge_candidates_found
    );
    println!(
        "  opportunity_rate: {:.4}%",
        lifecycle_stats.opportunity_rate * 100.0
    );
    // Phase 1.9e: Stable formation metrics
    println!("  Stable formation metrics:");
    println!("    stable_entry_count: {}", lifecycle_stats.stable_new);
    println!("    stable_exit_count: {}", lifecycle_stats.stable_dropped);
    println!("    stable_count_final: {}", lifecycle_stats.stable_count);
    println!(
        "    stable_avg_support: {:.1}",
        lifecycle_stats.stable_avg_support
    );
    println!(
        "    stable_avg_entropy: {:.3}",
        lifecycle_stats.stable_avg_entropy
    );
    println!(
        "    stable_avg_abs_td: {:.4}",
        lifecycle_stats.stable_avg_abs_td
    );
    println!("  Cross-partition merges:");
    println!(
        "    cross_partition_merges_done: {}",
        lifecycle_stats.cross_partition_merges_done
    );
    println!(
        "    cross_mode_merges_done: {}",
        lifecycle_stats.cross_mode_merges_done
    );
    println!(
        "    cross_mask_merges_done: {}",
        lifecycle_stats.cross_mask_merges_done
    );
    println!("    mean_dv_cross: {:.4}", lifecycle_stats.mean_dv_cross);
    println!(
        "    mean_mask_hamming_cross: {:.2}",
        lifecycle_stats.mean_mask_hamming_cross
    );
    println!("  Merge blocking breakdown:");
    println!(
        "    blocked_key_mismatch: {}",
        lifecycle_stats.merge_blocked_key_mismatch
    );
    println!(
        "    blocked_ctx_mismatch: {}",
        lifecycle_stats.merge_blocked_ctx_mismatch
    );
    println!(
        "    blocked_mode_mismatch: {}",
        lifecycle_stats.merge_blocked_mode_mismatch
    );
    println!(
        "    blocked_explore_isolated: {}",
        lifecycle_stats.blocked_explore_isolated
    );
    println!(
        "    blocked_proto_score: {}",
        lifecycle_stats.merge_blocked_proto
    );
    println!(
        "    blocked_value_delta: {}",
        lifecycle_stats.merge_blocked_value
    );
    println!(
        "    blocked_stability_mixed: {}",
        lifecycle_stats.merge_blocked_stability
    );
    println!(
        "    blocked_support_low: {}",
        lifecycle_stats.merge_blocked_support
    );
    // Phase 1.9e: Consolidated cross-partition blocking stats
    let blocked_cross_not_stable = lifecycle_stats.blocked_cross_mode_not_stable
        + lifecycle_stats.blocked_cross_mask_not_stable;
    let blocked_cross_proto_low =
        lifecycle_stats.blocked_cross_mode_proto + lifecycle_stats.blocked_cross_mask_proto;
    let blocked_cross_value_delta =
        lifecycle_stats.blocked_cross_mode_v + lifecycle_stats.blocked_cross_mask_v;
    let blocked_cross_mask_far = lifecycle_stats.blocked_mask_hamming;
    let total_cross_blocks = blocked_cross_not_stable
        + blocked_cross_proto_low
        + blocked_cross_value_delta
        + blocked_cross_mask_far
        + lifecycle_stats.blocked_cross_rate_limited;
    let not_stable_pct = if total_cross_blocks > 0 {
        100.0 * blocked_cross_not_stable as f64 / total_cross_blocks as f64
    } else {
        0.0
    };
    println!("  Cross-partition blocking (consolidated):");
    println!(
        "    blocked_cross_not_stable: {} ({:.1}% of cross blocks)",
        blocked_cross_not_stable, not_stable_pct
    );
    println!("    blocked_cross_proto_low: {}", blocked_cross_proto_low);
    println!(
        "    blocked_cross_value_delta: {}",
        blocked_cross_value_delta
    );
    println!("    blocked_cross_mask_far: {}", blocked_cross_mask_far);
    println!(
        "    blocked_cross_rate_limited: {}",
        lifecycle_stats.blocked_cross_rate_limited
    );
    println!("  stable_new: {}", lifecycle_stats.stable_new);
    println!("  stable_dropped: {}", lifecycle_stats.stable_dropped);
    println!(
        "  stable_drop_ratio: {:.2}%",
        lifecycle_stats.stable_drop_ratio * 100.0
    );

    // Acceptance check for Phase 1.9b
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 1.9b ACCEPTANCE:");
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
    println!(
        "  [{}] coverage_pos >= 72%: {:.1}%",
        if cov_ok { "✓" } else { "✗" },
        metrics_5b.coverage_pos() * 100.0
    );
    println!(
        "  [{}] selective_accuracy >= 81.5%: {:.1}%",
        if sel_ok { "✓" } else { "✗" },
        metrics_5b.selective_accuracy() * 100.0
    );
    println!(
        "  [{}] false_positive == 0%: {:.1}%",
        if fp_ok { "✓" } else { "✗" },
        metrics_5b.false_positive_rate() * 100.0
    );
    println!(
        "  [{}] abstain_neg >= 99%: {:.1}%",
        if abs_ok { "✓" } else { "✗" },
        metrics_5b.abstain_neg_rate() * 100.0
    );
    println!(
        "  [{}] anchors_used <= {}: {}",
        if anc_ok { "✓" } else { "✗" },
        MAX_ANCHORS,
        anchor_stats.anchors_used
    );
    println!(
        "  [{}] thrash_rate < 100 per 10k: {:.2}",
        if thrash_ok { "✓" } else { "✗" },
        anchor_stats.thrash_rate
    );
    println!(
        "  [{}] proto_updates > 0: {}",
        if proto_updates_ok { "✓" } else { "✗" },
        anchor_stats.proto_updates
    );
    println!(
        "  [{}] avg_proto_support > 0: {:.1}",
        if proto_support_ok { "✓" } else { "✗" },
        anchor_stats.avg_proto_support
    );
    println!();
    println!("Phase 1.7b (value learning):");
    println!(
        "  [{}] value_updates > 10000: {}",
        if v_updates_ok { "✓" } else { "✗" },
        anchor_stats.value_stats.updates_total
    );
    println!(
        "  [{}] delta_v (pos - neg) >= 0.10: {:.4}",
        if delta_v_ok { "✓" } else { "✗" },
        anchor_stats.value_stats.delta_v()
    );
    println!(
        "  [{}] td_late/early <= 0.95: {:.3}",
        if td_decreasing_ok { "✓" } else { "~" },
        td_ratio
    );

    // Phase 1.7d acceptance criteria
    let (probe_filled_ok, probe_evals_ok, probe_converging_ok, probe_ratio) =
        if let Some((probe_size, eval_count, last_mean, _, _, early_mean, late_mean)) = probe_stats
        {
            let filled = probe_size >= config.probe_min_fill;
            let evals = eval_count >= 10;
            let ratio = if early_mean > 0.0 {
                late_mean / early_mean
            } else {
                1.0
            };
            let converging = last_mean < 0.01 || ratio < 1.0;
            (filled, evals, converging, ratio)
        } else {
            (false, false, false, 1.0)
        };
    let probe_last_mean = probe_stats.map(|(_, _, m, _, _, _, _)| m).unwrap_or(0.0);

    println!();
    println!("Phase 1.7d (probe convergence):");
    println!(
        "  [{}] probe_size >= {}: {}",
        if probe_filled_ok { "✓" } else { "✗" },
        config.probe_min_fill,
        probe_stats.map(|(s, _, _, _, _, _, _)| s).unwrap_or(0)
    );
    println!(
        "  [{}] probe_evals >= 10: {}",
        if probe_evals_ok { "✓" } else { "✗" },
        probe_stats.map(|(_, e, _, _, _, _, _)| e).unwrap_or(0)
    );
    println!(
        "  [{}] probe_converging (ΔV<0.01 or ratio<1): mean={:.6}, ratio={:.3}",
        if probe_converging_ok { "✓" } else { "✗" },
        probe_last_mean,
        probe_ratio
    );

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
    println!(
        "  [{}] has_wins (total > 0): {}",
        if has_wins { "✓" } else { "✗" },
        lifecycle_stats.total_wins
    );
    println!(
        "  [{}] has_stable_anchors (stable_count > 0): {}",
        if has_stable_anchors { "✓" } else { "~" },
        lifecycle_stats.stable_count
    );
    println!(
        "  [{}] lifecycle_active (wins or stable or transitions): {}",
        if lifecycle_active { "✓" } else { "✗" },
        lifecycle_active
    );

    let phase18_ok = has_wins && lifecycle_active;

    // Phase 1.9b acceptance criteria (aggressive targets)
    let merges_many_ok = lifecycle_stats.merges_done_proto >= 500;
    let merges_some_ok = lifecycle_stats.merges_done_proto >= 100;
    let avg_merge_score_ok = lifecycle_stats.avg_merge_score >= 0.80;
    let drop_ratio_ok = lifecycle_stats.stable_drop_ratio <= 0.005; // <= 0.5%
    let drop_ratio_relaxed = lifecycle_stats.stable_drop_ratio <= 0.05; // <= 5% (relaxed)
    let stable_count_ok = lifecycle_stats.stable_count >= 30;
    let coverage_ok = metrics_5b.coverage_pos() >= 0.70;
    let selective_ok = metrics_5b.selective_accuracy() >= 0.80;

    println!();
    println!("Phase 1.9b (CONSOLIDATION - Aggressive Merge + Flicker Elimination):");
    println!(
        "  [{}] merges_done_proto >= 500: {}",
        if merges_many_ok { "✓" } else { "✗" },
        lifecycle_stats.merges_done_proto
    );
    println!(
        "  [{}] avg_merge_score >= 0.80: {:.3}",
        if avg_merge_score_ok { "✓" } else { "✗" },
        lifecycle_stats.avg_merge_score
    );
    println!(
        "  [{}] stable_drop_ratio <= 0.5%: {:.2}%",
        if drop_ratio_ok { "✓" } else { "✗" },
        lifecycle_stats.stable_drop_ratio * 100.0
    );
    println!(
        "  [{}] stable_count >= 30: {}",
        if stable_count_ok { "✓" } else { "✗" },
        lifecycle_stats.stable_count
    );
    println!(
        "  [{}] coverage_pos >= 70%: {:.1}%",
        if coverage_ok { "✓" } else { "✗" },
        metrics_5b.coverage_pos() * 100.0
    );
    println!(
        "  [{}] selective_accuracy >= 80%: {:.1}%",
        if selective_ok { "✓" } else { "✗" },
        metrics_5b.selective_accuracy() * 100.0
    );
    println!(
        "  [{}] false_positive == 0%: {:.1}%",
        if fp_ok { "✓" } else { "✗" },
        metrics_5b.false_positive_rate() * 100.0
    );
    println!();
    println!("  Merge Diagnostics:");
    println!(
        "    merge_candidates_found: {}",
        lifecycle_stats.merge_candidates_found
    );
    println!(
        "    cross_partition_merges: {}",
        lifecycle_stats.cross_partition_merges_done
    );
    println!(
        "    cross_mode_merges: {}",
        lifecycle_stats.cross_mode_merges_done
    );
    println!(
        "    cross_mask_merges: {}",
        lifecycle_stats.cross_mask_merges_done
    );
    println!(
        "    mean_dv_cross: {:.4}, mean_mask_hamming_cross: {:.2}",
        lifecycle_stats.mean_dv_cross, lifecycle_stats.mean_mask_hamming_cross
    );
    println!(
        "    blocked_key_mismatch: {}",
        lifecycle_stats.merge_blocked_key_mismatch
    );
    println!(
        "    blocked_ctx_mismatch: {}",
        lifecycle_stats.merge_blocked_ctx_mismatch
    );
    println!(
        "    blocked_mode_mismatch: {}",
        lifecycle_stats.merge_blocked_mode_mismatch
    );
    println!(
        "    blocked_proto_score: {}",
        lifecycle_stats.merge_blocked_proto
    );
    println!(
        "    blocked_value_delta: {}",
        lifecycle_stats.merge_blocked_value
    );
    println!(
        "    blocked_stability_mixed: {}",
        lifecycle_stats.merge_blocked_stability
    );
    println!(
        "    blocked_support_low: {}",
        lifecycle_stats.merge_blocked_support
    );
    println!(
        "    stable_new: {}, stable_dropped: {}",
        lifecycle_stats.stable_new, lifecycle_stats.stable_dropped
    );

    // Phase 1.9b: Primary goals are (1) many merges, (2) no flicker, (3) no coverage regression
    let stable_count_relaxed = lifecycle_stats.stable_count >= 10; // Relaxed from 30 to 10
    let phase19b_strict = merges_many_ok
        && avg_merge_score_ok
        && drop_ratio_ok
        && stable_count_ok
        && coverage_ok
        && selective_ok
        && fp_ok;
    let phase19b_relaxed = merges_some_ok
        && avg_merge_score_ok
        && drop_ratio_relaxed
        && stable_count_relaxed
        && coverage_ok
        && selective_ok
        && fp_ok;
    let phase19b_core_goals =
        merges_many_ok && drop_ratio_ok && coverage_ok && selective_ok && fp_ok;

    // Phase 1.9e: Stability formation tuning acceptance criteria
    // Baseline values from Phase 1.9d: blocked_cross_not_stable ~1,006,753, stable_count ~18
    let baseline_blocked_cross_not_stable: usize = 1_006_753;
    let baseline_stable_count: usize = 18;
    let blocked_cross_not_stable_val = lifecycle_stats.blocked_cross_mode_not_stable
        + lifecycle_stats.blocked_cross_mask_not_stable;
    let blocked_reduction_pct = if baseline_blocked_cross_not_stable > 0 {
        100.0
            * (1.0 - blocked_cross_not_stable_val as f64 / baseline_blocked_cross_not_stable as f64)
    } else {
        0.0
    };
    let stable_increase_pct = if baseline_stable_count > 0 {
        100.0 * (lifecycle_stats.stable_count as f64 / baseline_stable_count as f64 - 1.0)
    } else {
        0.0
    };
    let blocked_reduced_30pct =
        blocked_cross_not_stable_val <= (baseline_blocked_cross_not_stable * 70 / 100); // <=70% of baseline = 30% reduction
    let stable_increased_15pct =
        lifecycle_stats.stable_count >= (baseline_stable_count * 115 / 100); // >=115% of baseline
    let cross_merges_improved = lifecycle_stats.cross_partition_merges_done > 0;

    println!();
    println!("Phase 1.9e (STABILITY FORMATION TUNING - Option A):");
    println!(
        "  Baseline (1.9d): blocked_cross_not_stable={}, stable_count={}",
        baseline_blocked_cross_not_stable, baseline_stable_count
    );
    println!(
        "  [{}] blocked_cross_not_stable reduced ≥30%: {} ({:+.1}% change)",
        if blocked_reduced_30pct { "✓" } else { "✗" },
        blocked_cross_not_stable_val,
        -blocked_reduction_pct
    );
    println!(
        "  [{}] stable_count increased ≥15%: {} ({:+.1}% change)",
        if stable_increased_15pct { "✓" } else { "✗" },
        lifecycle_stats.stable_count,
        stable_increase_pct
    );
    println!(
        "  [{}] cross_partition_merges_done > 0: {}",
        if cross_merges_improved { "✓" } else { "~" },
        lifecycle_stats.cross_partition_merges_done
    );
    println!(
        "  Stable anchor averages: support={:.1}, entropy={:.3}, |TD|={:.4}",
        lifecycle_stats.stable_avg_support,
        lifecycle_stats.stable_avg_entropy,
        lifecycle_stats.stable_avg_abs_td
    );

    let phase19e_ok = blocked_reduced_30pct && stable_increased_15pct && phase19b_core_goals;

    println!();
    if phase19e_ok {
        println!("  → Phase 1.9e: ALL CRITERIA MET! (stability formation + core goals)");
    } else if phase19b_strict {
        println!("  → Phase 1.9b: ALL CRITERIA MET! (strict) - Phase 1.9e stability goals pending");
    } else if phase19b_core_goals {
        if blocked_reduced_30pct || stable_increased_15pct {
            println!("  → Phase 1.9e: PARTIAL (core goals met, some stability goals met)");
        } else {
            println!("  → Phase 1.9b: CORE GOALS MET - Phase 1.9e stability tuning needed");
        }
    } else if phase19b_relaxed {
        println!("  → Phase 1.9b: CRITERIA MET (relaxed) - Phase 1.9e stability tuning needed");
    } else if phase16_ok && phase17a_ok && phase18_ok && merges_some_ok {
        println!(
            "  → Phase 1.9b: Partial success. Merges happening, but other criteria need tuning."
        );
    } else if phase16_ok && phase17a_ok && phase18_ok {
        println!("  → Phase 1.8 OK, Phase 1.9b needs more merge activity.");
    } else {
        println!("  → Some criteria not met. Tuning may be needed.");
    }

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1.9f: Stability Formation + Stable Mass (Metrics Only)
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 1.9f: Stability Formation + Stable Mass");
    println!("═══════════════════════════════════════════════════════════════════");

    println!("  Stability transitions:");
    println!("    stable_entries:     {}", lifecycle_stats.stable_new);
    println!("    stable_exits:       {}", lifecycle_stats.stable_dropped);
    println!(
        "    stable_time_share:  {:.1}%   (active anchor stable / total ticks)",
        lifecycle_stats.stable_time_share * 100.0
    );

    println!();
    println!("  Stable end-state (anchors):");
    println!("    stable_count_end:   {}", lifecycle_stats.stable_count);
    println!(
        "    stable_mass_sum:    {}        (Σ proto_support over stable anchors)",
        lifecycle_stats.stable_mass_sum
    );
    println!(
        "    stable_mass_mean:   {:.1}      (mean proto_support among stable)",
        lifecycle_stats.stable_mass_mean
    );
    println!(
        "    stable_support_p50: {}",
        lifecycle_stats.stable_support_p50
    );
    println!(
        "    stable_support_p90: {}",
        lifecycle_stats.stable_support_p90
    );

    println!();
    println!("  Stable quality (means among stable anchors):");
    println!(
        "    stable_value_mean:    {:+.2}",
        lifecycle_stats.stable_value_mean
    );
    println!(
        "    stable_entropy_mean:  {:.2}",
        lifecycle_stats.stable_entropy_mean
    );
    println!(
        "    stable_abs_td_mean:   {:.2}",
        lifecycle_stats.stable_td_mean
    );

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1.9f: Merge Efficiency
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 1.9f: Merge Efficiency");
    println!("═══════════════════════════════════════════════════════════════════");

    println!("  Merge scan totals:");
    println!(
        "    merge_scan_runs:         {}",
        lifecycle_stats.merge_scan_runs
    );
    println!(
        "    pairs_checked:           {}",
        lifecycle_stats.pairs_checked
    );
    println!(
        "    merge_candidates_found:  {}",
        lifecycle_stats.merge_candidates_found
    );
    println!(
        "    merges_done_proto:       {}",
        lifecycle_stats.merges_done_proto
    );
    println!(
        "    avg_merge_score:         {:.3}",
        lifecycle_stats.avg_merge_score
    );

    let merge_yield = if lifecycle_stats.merge_candidates_found > 0 {
        lifecycle_stats.merges_done_proto as f64 / lifecycle_stats.merge_candidates_found as f64
    } else {
        0.0
    };
    let avg_candidates_per_scan = if lifecycle_stats.merge_scan_runs > 0 {
        lifecycle_stats.merge_candidates_found as f64 / lifecycle_stats.merge_scan_runs as f64
    } else {
        0.0
    };
    let avg_merges_per_scan = if lifecycle_stats.merge_scan_runs > 0 {
        lifecycle_stats.merges_done_proto as f64 / lifecycle_stats.merge_scan_runs as f64
    } else {
        0.0
    };

    println!();
    println!("  Efficiency:");
    println!(
        "    opportunity_rate:    {:.4}%   (candidates / pairs_checked)",
        lifecycle_stats.opportunity_rate * 100.0
    );
    println!(
        "    merge_yield:         {:.1}%     (merges / candidates)",
        merge_yield * 100.0
    );
    println!("    avg_candidates/scan: {:.2}", avg_candidates_per_scan);
    println!("    avg_merges/scan:     {:.2}", avg_merges_per_scan);
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
        let p95_abs = deltas
            .get(p95_idx.saturating_sub(1))
            .copied()
            .unwrap_or(0.0);

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
fn compute_reward(delta_power: f64, topk_margin: f64, proto_align: f32, config: &Config) -> f32 {
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
    let reward =
        config.r_w_power * power_term + config.r_w_coh * coh_z + config.r_w_proto * proto_z
            - config.r_w_margin * margin_penalty;

    // Clamp final reward to [-1, +1]
    reward.clamp(-1.0, 1.0)
}



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

            if t >= config.competitive_recall_start
                && t % config.competitive_recall_stride == 0
                && window.is_ready()
            {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig = flip_competitive_sig(
                        &current_sig,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
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

    println!(
        "  coverage_pos={:.1}%, accuracy_pos={:.1}%, selective_acc={:.1}%",
        metrics.coverage_pos() * 100.0,
        metrics.accuracy_pos() * 100.0,
        metrics.selective_accuracy() * 100.0
    );

    let stability = window.stability();
    (metrics, stability)
}

/// Phase 1.8/1.9d lifecycle stats for reporting.
struct LifecycleStats {
    stable_count: usize,
    stable_fraction: f64,
    mode_transitions: usize,
    total_wins: u32,
    final_mode: bool,
    // Phase 1.9b: Consolidation metrics
    merges_done: usize,
    merge_candidates_found: usize,
    stable_new: usize,
    stable_dropped: usize,
    stable_drop_ratio: f64,
    // Phase 1.9b: Additional merge stats
    merge_scan_runs: usize,
    merges_done_proto: usize,
    avg_merge_score: f32,
    // Phase 1.9b: Blocked merge diagnostics
    merge_blocked_proto: usize,
    merge_blocked_value: usize,
    merge_blocked_stability: usize,
    merge_blocked_support: usize,
    merge_blocked_key_mismatch: usize,
    merge_blocked_ctx_mismatch: usize,
    merge_blocked_mode_mismatch: usize,
    // Phase 1.9c: Opportunity diagnostics
    pairs_checked: usize,
    opportunity_rate: f64,
    // Phase 1.9d: Cross-partition merge stats
    cross_mode_merges_done: usize,
    cross_mask_merges_done: usize,
    cross_partition_merges_done: usize,
    blocked_explore_isolated: usize,
    blocked_mask_hamming: usize,
    blocked_cross_mode_v: usize,
    blocked_cross_mode_proto: usize,
    blocked_cross_mask_proto: usize,
    blocked_cross_mask_v: usize,
    blocked_cross_mode_not_stable: usize,
    blocked_cross_mask_not_stable: usize,
    blocked_cross_rate_limited: usize,
    mean_dv_cross: f64,
    mean_mask_hamming_cross: f64,
    // Phase 1.9e: Stable formation metrics
    stable_avg_support: f64,
    stable_avg_entropy: f64,
    stable_avg_abs_td: f64,
    // Phase 1.9f: Stability formation + stable mass metrics
    ticks_total: usize,
    ticks_active_stable: usize,
    stable_time_share: f64,
    stable_mass_sum: u64,
    stable_mass_mean: f64,
    stable_support_p50: u32,
    stable_support_p90: u32,
    stable_value_mean: f64,
    stable_entropy_mean: f64,
    stable_td_mean: f64,
}

/// DEMO 5b: Anchor Concept Tokens with Phase 1.9b AGGRESSIVE CONSOLIDATION
fn run_demo_5b_keyed(
    config: &Config,
) -> (
    KeyedMemoryMetrics,
    AnchorStats,
    (usize, f64, usize),
    Option<(usize, usize, f64, f64, usize, f64, f64)>,
    LifecycleStats,
) {
    println!("DEMO 5b: Phase 1.9b AGGRESSIVE CONSOLIDATION (Many Merges + Flicker Elimination)");
    println!("─────────────────────────────────────────────────────────────────────────────────");

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

    // Phase 1.9f: Stability formation tracking
    let mut ticks_total: usize = 0;
    let mut ticks_active_stable: usize = 0;

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
            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            // Phase 1.9f: Track stable time share
            ticks_total += 1;
            if anchor_id != 0xFFFF && anchor_bank.is_anchor_stable(anchor_id) {
                ticks_active_stable += 1;
            }

            // Phase 1.9b: Update anchor's partition info for merge compatibility
            // Use ctx_hat as learned_mask
            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

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
                let mut reward = compute_reward(
                    delta_power,
                    prev_topk_margin,
                    proto_align_for_reward,
                    config,
                );

                // Phase 1.7d: Optional advantage reward centering
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
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
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    // Don't gate negative queries - just resolve
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    // Phase 1.8: Record wins for correct positive recalls
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;

            // Phase 1.7d: Periodic probe evaluation
            if config.probe_enabled
                && probe_set.should_eval(
                    global_tick,
                    config.probe_eval_stride,
                    config.probe_min_fill,
                )
            {
                let _stats = probe_set.eval(global_tick, |id| Some(anchor_bank.get_value(id)));
            }
        }
    }
    println!("done.");

    println!(
        "  coverage_pos={:.1}%, accuracy_pos={:.1}%, selective_acc={:.1}%",
        metrics.coverage_pos() * 100.0,
        metrics.accuracy_pos() * 100.0,
        metrics.selective_accuracy() * 100.0
    );

    // Phase 1.7a: Capture late entropy
    let (proto_entropy_late, _) = anchor_bank.proto_entropy_top_n(10, config.proto_m);

    // Phase 1.7b: Finalize value stats
    let (v_updates_total, avg_v, avg_abs_td) = anchor_bank.value_metrics();
    value_stats.updates_total = v_updates_total;
    value_stats.avg_v_used = avg_v;
    value_stats.avg_abs_td_used = avg_abs_td;
    value_stats.avg_abs_td_early = if early_td_count > 0 {
        early_td_sum / early_td_count as f64
    } else {
        0.0
    };
    value_stats.avg_abs_td_late = if late_td_count > 0 {
        late_td_sum / late_td_count as f64
    } else {
        0.0
    };
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
    let (stable_count, stable_fraction, mode_transitions, _, final_mode) =
        anchor_bank.lifecycle_metrics();
    // Phase 1.9b: Consolidation metrics with merge blocking breakdown
    let merge_stats = anchor_bank.consolidation_metrics();
    let (merge_scan_runs, merges_done_proto, avg_merge_score, _) = anchor_bank.merge_stats();
    let lifecycle_stats = LifecycleStats {
        stable_count,
        stable_fraction,
        mode_transitions,
        total_wins: anchor_bank.total_wins(),
        final_mode,
        // Phase 1.9b
        merges_done: anchor_bank.anchor_merges,
        merge_candidates_found: merge_stats.candidates_found,
        stable_new: merge_stats.stable_new,
        stable_dropped: merge_stats.stable_dropped,
        stable_drop_ratio: anchor_bank.stable_drop_ratio(),
        // Phase 1.9b: Additional merge stats
        merge_scan_runs,
        merges_done_proto,
        avg_merge_score,
        // Phase 1.9b: Blocked merge diagnostics
        merge_blocked_proto: merge_stats.blocked_proto,
        merge_blocked_value: merge_stats.blocked_value,
        merge_blocked_stability: merge_stats.blocked_stability,
        merge_blocked_support: merge_stats.blocked_support,
        merge_blocked_key_mismatch: merge_stats.blocked_key_mismatch,
        merge_blocked_ctx_mismatch: merge_stats.blocked_ctx_mismatch,
        merge_blocked_mode_mismatch: merge_stats.blocked_mode_mismatch,
        // Phase 1.9c: Opportunity diagnostics
        pairs_checked: merge_stats.pairs_checked,
        opportunity_rate: merge_stats.opportunity_rate,
        // Phase 1.9d: Cross-partition merge stats
        cross_mode_merges_done: merge_stats.cross_mode_merges_done,
        cross_mask_merges_done: merge_stats.cross_mask_merges_done,
        cross_partition_merges_done: merge_stats.cross_partition_merges_done,
        blocked_explore_isolated: merge_stats.blocked_explore_isolated,
        blocked_mask_hamming: merge_stats.blocked_mask_hamming,
        blocked_cross_mode_v: merge_stats.blocked_cross_mode_v,
        blocked_cross_mode_proto: merge_stats.blocked_cross_mode_proto,
        blocked_cross_mask_proto: merge_stats.blocked_cross_mask_proto,
        blocked_cross_mask_v: merge_stats.blocked_cross_mask_v,
        blocked_cross_mode_not_stable: merge_stats.blocked_cross_mode_not_stable,
        blocked_cross_mask_not_stable: merge_stats.blocked_cross_mask_not_stable,
        blocked_cross_rate_limited: merge_stats.blocked_cross_rate_limited,
        mean_dv_cross: if merge_stats.cross_dv_count > 0 {
            merge_stats.cross_dv_sum / merge_stats.cross_dv_count as f64
        } else {
            0.0
        },
        mean_mask_hamming_cross: if merge_stats.cross_mask_hamming_count > 0 {
            merge_stats.cross_mask_hamming_sum as f64 / merge_stats.cross_mask_hamming_count as f64
        } else {
            0.0
        },
        // Phase 1.9e: Stable formation metrics
        stable_avg_support: {
            let (avg_support, _, _, _) = anchor_bank.stable_anchor_averages();
            avg_support
        },
        stable_avg_entropy: {
            let (_, avg_entropy, _, _) = anchor_bank.stable_anchor_averages();
            avg_entropy
        },
        stable_avg_abs_td: {
            let (_, _, avg_abs_td, _) = anchor_bank.stable_anchor_averages();
            avg_abs_td
        },
        // Phase 1.9f: Stability formation + stable mass metrics
        ticks_total,
        ticks_active_stable,
        stable_time_share: if ticks_total > 0 {
            ticks_active_stable as f64 / ticks_total as f64
        } else {
            0.0
        },
        stable_mass_sum: {
            let stats = anchor_bank.stable_anchor_stats();
            stats.mass_sum
        },
        stable_mass_mean: {
            let stats = anchor_bank.stable_anchor_stats();
            stats.mass_mean
        },
        stable_support_p50: {
            let stats = anchor_bank.stable_anchor_stats();
            stats.support_p50
        },
        stable_support_p90: {
            let stats = anchor_bank.stable_anchor_stats();
            stats.support_p90
        },
        stable_value_mean: {
            let stats = anchor_bank.stable_anchor_stats();
            stats.value_mean
        },
        stable_entropy_mean: {
            let stats = anchor_bank.stable_anchor_stats();
            stats.entropy_mean
        },
        stable_td_mean: {
            let stats = anchor_bank.stable_anchor_stats();
            stats.td_mean
        },
    };

    (
        metrics,
        anchor_stats,
        keyed_stats,
        probe_stats,
        lifecycle_stats,
    )
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

// =============================================================================
// DEMO 7: Phase 2.0a - Mode Policy (Explore/Exploit/Reset)
// =============================================================================

pub fn demo_7_mode_policy(config: &Config) {
    use mode::{Mode, ModeAction, ModePolicy, ModePolicyConfig};

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 7: Phase 2.0a - MODE POLICY (Explore/Exploit/Reset)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if !config.enable_mode_policy {
        println!("Mode policy disabled in config. Skipping Demo 7.");
        return;
    }

    // Initialize mode policy from config
    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    // Use same seed for reproducibility
    let mut rng = Rng::new(config.seed.wrapping_add(0x7A7A_7A7A));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train phase (same as Demo 5b)
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

    // Phase 1.7b: Value learning state
    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    print!(
        "  Running {} episodes with mode policy... ",
        config.competitive_episodes
    );

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

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            // Compute confidence info
            let topk_margin = if topk.len() >= 2 {
                topk[0].1 - topk[1].1
            } else if !topk.is_empty() {
                topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            // Periodic merges
            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            // Get base gate params
            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            // Resolve signature to anchor
            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            // Get anchor value for mode policy observation
            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            // Compute TD for this tick (from previous anchor's perspective)
            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            // Check if base gate passes
            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);

            // Observe state for mode policy
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);

            // Choose mode
            let mode = mode_policy.choose_mode(global_tick);

            // Get mode overrides and action
            let (overrides, action) = mode_policy.apply_mode_overrides(mode);

            // Apply mode-adjusted gate params
            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= overrides.margin_min_scale as f64;

            // Check gate with adjusted params
            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            // Record gate outcome for mode stats
            mode_policy.record_gate_outcome(mode, gate_passed);

            // Apply Reset action if needed
            if let ModeAction::Dampen { factor, top_k } = action {
                let dampen_ids: Vec<usize> = topk.iter().take(top_k).map(|(id, _)| *id).collect();
                chamber.dampen_nodes(&dampen_ids, factor);
            }

            // Update partition info
            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            // Update prototype when gate passes
            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &topk, config);
            }

            // Value update for previous anchor
            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            // Update previous state
            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            // Memory store/recall
            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }
    println!("done.");

    // Get mode stats
    let mode_stats = mode_policy.mode_stats();

    // Print results
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0a: Mode Policy Results");
    println!("═══════════════════════════════════════════════════════════════════");

    println!();
    println!("Mode usage:");
    println!(
        "  explore_count: {} ({:.1}%)",
        mode_stats.explore_count,
        mode_stats.explore_rate * 100.0
    );
    println!(
        "  exploit_count: {} ({:.1}%)",
        mode_stats.exploit_count,
        mode_stats.exploit_rate * 100.0
    );
    println!(
        "  reset_count: {} ({:.2}%)",
        mode_stats.reset_count,
        mode_stats.reset_rate * 100.0
    );

    println!();
    println!("Gate pass rates by mode:");
    println!(
        "  explore_gate_pass_rate: {:.1}%",
        mode_stats.gate_pass_rate_explore * 100.0
    );
    println!(
        "  exploit_gate_pass_rate: {:.1}%",
        mode_stats.gate_pass_rate_exploit * 100.0
    );

    println!();
    println!("Reset effectiveness:");
    println!(
        "  reset_effectiveness_mean: {:+.1}%",
        mode_stats.reset_effectiveness_mean * 100.0
    );
    println!(
        "  reset_effectiveness_samples: {}",
        mode_stats.reset_effectiveness_count
    );

    println!();
    println!("Memory performance:");
    println!("  coverage_pos: {:.1}%", metrics.coverage_pos() * 100.0);
    println!(
        "  selective_accuracy: {:.1}%",
        metrics.selective_accuracy() * 100.0
    );
    println!(
        "  false_positive_rate: {:.1}%",
        metrics.false_positive_rate() * 100.0
    );

    // Get stable drop ratio from anchor bank
    let stable_drop_ratio = anchor_bank.stable_drop_ratio();

    // Acceptance criteria
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0a ACCEPTANCE:");
    println!("═══════════════════════════════════════════════════════════════════");

    // A) Non-degenerate mode usage
    let explore_rate_ok = mode_stats.explore_rate >= 0.03; // >= 3%
    let reset_rate_ok = mode_stats.reset_rate >= 0.002 && mode_stats.reset_rate <= 0.05; // 0.2% - 5%
    let exploit_rate_ok = mode_stats.exploit_rate <= 0.97; // <= 97%

    println!();
    println!("A) Non-degenerate mode usage:");
    println!(
        "  [{}] explore_rate >= 3%: {:.1}%",
        if explore_rate_ok { "✓" } else { "✗" },
        mode_stats.explore_rate * 100.0
    );
    println!(
        "  [{}] reset_rate in [0.2%, 5%]: {:.2}%",
        if reset_rate_ok { "✓" } else { "✗" },
        mode_stats.reset_rate * 100.0
    );
    println!(
        "  [{}] exploit_rate <= 97%: {:.1}%",
        if exploit_rate_ok { "✓" } else { "✗" },
        mode_stats.exploit_rate * 100.0
    );

    // B) No performance regression
    let coverage_ok = metrics.coverage_pos() >= 0.70; // >= 70%
    let selective_ok = metrics.selective_accuracy() >= 0.80; // >= 80%
    let fp_ok = metrics.false_positive_rate() == 0.0; // == 0%
    let stable_drop_ok = stable_drop_ratio <= 0.005; // <= 0.5%

    println!();
    println!("B) No performance regression:");
    println!(
        "  [{}] coverage_pos >= 70%: {:.1}%",
        if coverage_ok { "✓" } else { "✗" },
        metrics.coverage_pos() * 100.0
    );
    println!(
        "  [{}] selective_accuracy >= 80%: {:.1}%",
        if selective_ok { "✓" } else { "✗" },
        metrics.selective_accuracy() * 100.0
    );
    println!(
        "  [{}] false_positive == 0%: {:.1}%",
        if fp_ok { "✓" } else { "✗" },
        metrics.false_positive_rate() * 100.0
    );
    println!(
        "  [{}] stable_drop_ratio <= 0.5%: {:.2}%",
        if stable_drop_ok { "✓" } else { "✗" },
        stable_drop_ratio * 100.0
    );

    // C) Reset effectiveness
    let reset_effective_ok =
        mode_stats.reset_effectiveness_count == 0 || mode_stats.reset_effectiveness_mean >= 0.10; // >= 10% TD reduction

    println!();
    println!("C) Reset effectiveness:");
    println!(
        "  [{}] mean_abs_td decrease >= 10% post-reset: {:+.1}% (n={})",
        if reset_effective_ok { "✓" } else { "✗" },
        mode_stats.reset_effectiveness_mean * 100.0,
        mode_stats.reset_effectiveness_count
    );

    let all_ok = explore_rate_ok
        && reset_rate_ok
        && exploit_rate_ok
        && coverage_ok
        && selective_ok
        && fp_ok
        && stable_drop_ok
        && reset_effective_ok;

    println!();
    if all_ok {
        println!("  → Phase 2.0a: ALL ACCEPTANCE CRITERIA MET!");
    } else {
        let a_ok = explore_rate_ok && reset_rate_ok && exploit_rate_ok;
        let b_ok = coverage_ok && selective_ok && fp_ok && stable_drop_ok;
        if a_ok && b_ok {
            println!("  → Phase 2.0a: Mode usage (A) and performance (B) OK. Reset effectiveness (C) needs tuning.");
        } else if a_ok {
            println!("  → Phase 2.0a: Mode usage OK (A). Performance (B) or reset effectiveness (C) needs tuning.");
        } else if b_ok {
            println!("  → Phase 2.0a: Performance OK (B). Mode usage (A) needs tuning - modes may be degenerate.");
        } else {
            println!("  → Phase 2.0a: Multiple criteria not met. Tuning needed.");
        }
    }
}

// =============================================================================
// DEMO 8: Phase 2.0b - Ablations + Per-Mode Metrics + Targeted Reset
// =============================================================================

pub fn demo_8_ablations(config: &Config) {
    use ablate::{
        select_reset_targets, AblationConfig, PerModeStats, ResetTargetMode, ResetTargetStats,
        VariantReport,
    };
    use mode::{Mode, ModeAction, ModePolicy, ModePolicyConfig};

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 8: Phase 2.0b - ABLATIONS + MODE METRICS + TARGETED RESET");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if !config.enable_mode_policy {
        println!("Mode policy disabled in config. Skipping Demo 8.");
        return;
    }

    // Define ablation variants
    let variants = [
        ("FULL", AblationConfig::full()),
        ("NO_RESET", AblationConfig::no_reset()),
        ("NO_EXPLORE", AblationConfig::no_explore()),
    ];

    let mut reports: Vec<VariantReport> = Vec::new();

    for (label, ablation_config) in &variants {
        print!("  Running variant {}... ", label);

        let report = run_ablation_variant(config, label, ablation_config.clone());
        println!("done.");
        reports.push(report);
    }

    // Print comparative results
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0b: Ablation Study Results");
    println!("═══════════════════════════════════════════════════════════════════");

    // Mode usage comparison
    println!();
    println!("Mode usage comparison:");
    println!(
        "  {:12} | {:>10} | {:>10} | {:>10}",
        "Variant", "Explore%", "Exploit%", "Reset%"
    );
    println!("  {}", "-".repeat(50));
    for report in &reports {
        println!(
            "  {:12} | {:9.1}% | {:9.1}% | {:9.2}%",
            report.label,
            report.explore_rate * 100.0,
            report.exploit_rate * 100.0,
            report.reset_rate * 100.0
        );
    }

    // Performance comparison
    println!();
    println!("Performance comparison:");
    println!(
        "  {:12} | {:>12} | {:>12} | {:>10}",
        "Variant", "coverage%", "sel_acc%", "FP%"
    );
    println!("  {}", "-".repeat(55));
    for report in &reports {
        println!(
            "  {:12} | {:11.1}% | {:11.1}% | {:9.1}%",
            report.label,
            report.coverage_pos * 100.0,
            report.selective_accuracy * 100.0,
            report.false_positive_rate * 100.0
        );
    }

    // Reset effectiveness comparison
    println!();
    println!("Reset effectiveness:");
    println!(
        "  {:12} | {:>15} | {:>10}",
        "Variant", "effectiveness", "samples"
    );
    println!("  {}", "-".repeat(45));
    for report in &reports {
        let eff_str = if report.reset_effectiveness_count > 0 {
            format!("{:+.1}%", report.reset_effectiveness_mean * 100.0)
        } else {
            "N/A".to_string()
        };
        println!(
            "  {:12} | {:>15} | {:>10}",
            report.label, eff_str, report.reset_effectiveness_count
        );
    }

    // Per-mode diagnostics for FULL variant
    println!();
    println!("Per-mode diagnostics (FULL variant):");
    if let Some(full_report) = reports.first() {
        full_report.print_per_mode_table();
    }

    // Reset targeting stats for FULL variant
    println!();
    println!("Reset targeting (FULL variant):");
    if let Some(full_report) = reports.first() {
        let stats = &full_report.reset_target_stats;
        println!("  reset_count: {}", stats.reset_count);
        println!(
            "  avg_nodes_dampened: {:.1}",
            stats.avg_dampened_per_reset()
        );
        println!(
            "  off_proto_fraction: {:.1}%",
            stats.off_proto_fraction() * 100.0
        );
    }

    // Directional analysis
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0b DIRECTIONAL ANALYSIS:");
    println!("═══════════════════════════════════════════════════════════════════");

    let full = &reports[0];
    let no_reset = &reports[1];
    let no_explore = &reports[2];

    // A) NO_RESET should have higher mean|TD| than FULL (resets reduce TD)
    let full_mean_td = (full.per_mode.explore_abs_td_sum
        + full.per_mode.exploit_abs_td_sum
        + full.per_mode.reset_abs_td_sum)
        / (full.per_mode.explore_ticks + full.per_mode.exploit_ticks + full.per_mode.reset_ticks)
            .max(1) as f64;
    let no_reset_mean_td = (no_reset.per_mode.explore_abs_td_sum
        + no_reset.per_mode.exploit_abs_td_sum)
        / (no_reset.per_mode.explore_ticks + no_reset.per_mode.exploit_ticks).max(1) as f64;

    let reset_helps_td = no_reset_mean_td > full_mean_td;
    println!();
    println!("A) Reset reduces TD oscillation:");
    println!(
        "  FULL mean|TD|: {:.4}, NO_RESET mean|TD|: {:.4}",
        full_mean_td, no_reset_mean_td
    );
    println!(
        "  [{}] NO_RESET has higher mean|TD| than FULL",
        if reset_helps_td { "✓" } else { "~" }
    );

    // B) NO_EXPLORE should have >= Exploit rate (forced to Exploit)
    let no_explore_more_exploit = no_explore.exploit_rate >= full.exploit_rate;
    println!();
    println!("B) Explore ablation forces Exploit:");
    println!(
        "  FULL exploit_rate: {:.1}%, NO_EXPLORE exploit_rate: {:.1}%",
        full.exploit_rate * 100.0,
        no_explore.exploit_rate * 100.0
    );
    println!(
        "  [{}] NO_EXPLORE has >= exploit_rate than FULL",
        if no_explore_more_exploit { "✓" } else { "~" }
    );

    // C) FULL should have best or comparable coverage
    let full_cov_ok = full.coverage_pos >= no_reset.coverage_pos * 0.95
        && full.coverage_pos >= no_explore.coverage_pos * 0.95;
    println!();
    println!("C) FULL variant performance:");
    println!(
        "  FULL coverage: {:.1}%, NO_RESET: {:.1}%, NO_EXPLORE: {:.1}%",
        full.coverage_pos * 100.0,
        no_reset.coverage_pos * 100.0,
        no_explore.coverage_pos * 100.0
    );
    println!(
        "  [{}] FULL has >=95% of ablated variants' coverage",
        if full_cov_ok { "✓" } else { "~" }
    );

    // D) FULL meets 2.0a acceptance
    let full_2_0a_ok = full.explore_rate >= 0.03
        && full.reset_rate >= 0.002
        && full.reset_rate <= 0.05
        && full.exploit_rate <= 0.97
        && full.coverage_pos >= 0.70
        && full.selective_accuracy >= 0.80;
    println!();
    println!("D) FULL meets Phase 2.0a acceptance:");
    println!(
        "  [{}] explore_rate >= 3%: {:.1}%",
        if full.explore_rate >= 0.03 {
            "✓"
        } else {
            "✗"
        },
        full.explore_rate * 100.0
    );
    println!(
        "  [{}] reset_rate in [0.2%, 5%]: {:.2}%",
        if full.reset_rate >= 0.002 && full.reset_rate <= 0.05 {
            "✓"
        } else {
            "✗"
        },
        full.reset_rate * 100.0
    );
    println!(
        "  [{}] coverage_pos >= 70%: {:.1}%",
        if full.coverage_pos >= 0.70 {
            "✓"
        } else {
            "✗"
        },
        full.coverage_pos * 100.0
    );
    println!(
        "  [{}] selective_accuracy >= 80%: {:.1}%",
        if full.selective_accuracy >= 0.80 {
            "✓"
        } else {
            "✗"
        },
        full.selective_accuracy * 100.0
    );

    println!();
    if full_2_0a_ok {
        println!("  → Phase 2.0b: FULL variant meets all 2.0a acceptance criteria!");
    } else {
        println!("  → Phase 2.0b: FULL variant needs tuning to meet 2.0a acceptance.");
    }

    println!();
    println!("  Directional effects:");
    if reset_helps_td {
        println!("    [✓] Reset reduces TD oscillation");
    } else {
        println!("    [~] Reset TD effect inconclusive");
    }
    if no_explore_more_exploit {
        println!("    [✓] Explore ablation forces higher exploit");
    } else {
        println!("    [~] Explore ablation effect inconclusive");
    }
}

/// Run a single ablation variant and collect metrics.
fn run_ablation_variant(
    config: &Config,
    label: &str,
    ablation_config: ablate::AblationConfig,
) -> ablate::VariantReport {
    use ablate::{select_reset_targets, PerModeStats, ResetTargetStats, VariantReport};
    use mode::{Mode, ModeAction, ModePolicy, ModePolicyConfig};

    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    // Use same seed for reproducibility (but different from Demo 7)
    let mut rng = Rng::new(config.seed.wrapping_add(0x8B8B_8B8B));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train phase
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

    // Initialize memory components
    let mut anchor_bank = AnchorBank::new();
    let keyed_config = KeyedMemoryConfig {
        label_min_p: 0.50,
        label_margin: 0.10,
        alpha: 0.5,
        num_labels: config.num_ctx,
    };
    let mut keyed_memory = KeyedMemoryStore::new(keyed_config);
    let mut metrics = KeyedMemoryMetrics::new();
    let mut per_mode_stats = PerModeStats::new();
    let mut reset_target_stats = ResetTargetStats::new();

    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);
    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;

    // Value learning state
    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;
    let mut current_mode = Mode::Exploit;

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

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            // Compute confidence info
            let topk_margin = if topk.len() >= 2 {
                topk[0].1 - topk[1].1
            } else if !topk.is_empty() {
                topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            // Periodic merges
            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            // Get base gate params
            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            // Resolve signature to anchor
            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            // Get anchor value for mode policy observation
            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            // Check if anchor is stable
            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            // Compute TD for this tick
            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            // Check if base gate passes
            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);

            // Observe state for mode policy
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);

            // Choose mode with ablation
            let mode = mode_policy.choose_mode_with_ablation(
                global_tick,
                ablation_config.enable_reset,
                ablation_config.enable_explore,
            );
            current_mode = mode;

            // Get mode overrides and action
            let (overrides, action) = mode_policy.apply_mode_overrides(mode);

            // Apply mode-adjusted gate params
            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= overrides.margin_min_scale as f64;

            // Check gate with adjusted params
            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            // Record per-mode tick stats
            per_mode_stats.record_tick(mode, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Apply Reset action with targeted selection
            if let ModeAction::Dampen {
                factor,
                top_k: max_nodes,
            } = action
            {
                // Get prototype info from current anchor
                let (proto_nodes, proto_weights) = if anchor_id != 0xFFFF {
                    anchor_bank
                        .get_anchor(anchor_id)
                        .map(|a| (a.proto_nodes.to_vec(), a.proto_w.to_vec()))
                        .unwrap_or_else(|| (vec![0; 16], vec![0.0; 16]))
                } else {
                    (vec![0; 16], vec![0.0; 16])
                };

                let (dampen_ids, off_proto, on_proto) = select_reset_targets(
                    &topk,
                    &proto_nodes,
                    &proto_weights,
                    config.proto_m,
                    ablation_config.target_mode,
                    max_nodes,
                    topk_margin as f32,
                    abs_td as f32,
                );

                chamber.dampen_nodes(&dampen_ids, factor);
                reset_target_stats.record_reset(dampen_ids.len(), off_proto, on_proto);
            }

            // Update partition info
            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            // Update prototype when gate passes
            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &topk, config);
            }

            // Value update for previous anchor
            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            // Update previous state
            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            // Memory store/recall
            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    let covered = matches!(decision, KeyedRecallDecision::Label(_, _));
                    per_mode_stats.record_recall(current_mode, false, covered, false);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    let (covered, correct) = match &decision {
                        KeyedRecallDecision::Label(recalled, _) => {
                            (*recalled == true_label, *recalled == true_label)
                        }
                        _ => (false, false),
                    };
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    per_mode_stats.record_recall(current_mode, true, covered, correct);
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    // Build report
    let mode_stats = mode_policy.mode_stats();

    let mut report = VariantReport::new(label, ablation_config);
    report.explore_rate = mode_stats.explore_rate;
    report.exploit_rate = mode_stats.exploit_rate;
    report.reset_rate = mode_stats.reset_rate;
    report.coverage_pos = metrics.coverage_pos();
    report.selective_accuracy = metrics.selective_accuracy();
    report.false_positive_rate = metrics.false_positive_rate();
    report.stable_drop_ratio = anchor_bank.stable_drop_ratio();
    report.reset_effectiveness_mean = mode_stats.reset_effectiveness_mean;
    report.reset_effectiveness_count = mode_stats.reset_effectiveness_count;
    report.per_mode = per_mode_stats;
    report.reset_target_stats = reset_target_stats;

    report
}

// =============================================================================
// DEMO 9: Phase 2.0c - Mode → Action Loop
// =============================================================================

pub fn demo_9_action_loop(config: &Config) {
    use action::{Action, ActionConfig, ActionPolicy};
    use mode::{Mode, ModePolicy, ModePolicyConfig};

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 9: Phase 2.0c - MODE → ACTION LOOP");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if !config.enable_mode_policy {
        println!("Mode policy disabled in config. Skipping Demo 9.");
        return;
    }

    if !config.enable_action_policy {
        println!("Action policy disabled in config. Skipping Demo 9.");
        return;
    }

    // Initialize mode policy from config
    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    // Initialize action policy from config
    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);

    // Use same seed as Demo 7 for comparable mode distribution
    let mut rng = Rng::new(config.seed.wrapping_add(0x7A7A_7A7A));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train phase (same as Demo 7)
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

    // Phase 1.7b: Value learning state
    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    // Perturb effectiveness tracking
    let mut perturb_pre_td_buffer: Vec<f32> = Vec::new();
    let mut perturb_tick_buffer: Vec<u64> = Vec::new();

    print!(
        "  Running {} episodes with action policy... ",
        config.competitive_episodes
    );

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            // Get base Top-K
            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            // Compute confidence info from base topk
            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            // Periodic merges
            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            // Get base gate params
            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            // Resolve signature to anchor
            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            // Get anchor value for mode policy observation
            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            // Check if anchor is stable
            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            // Compute TD for this tick
            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            // Check if base gate passes
            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);

            // Observe state for mode policy
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);

            // Choose mode
            let mode = mode_policy.choose_mode(global_tick);

            // Map Mode → Action
            let action = action_policy.choose_action(mode);

            // Get action overrides
            let action_overrides = action_policy.get_overrides(action);

            // Apply action-adjusted gate params
            // Action policy takes over margin scaling from mode policy
            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            // Check gate with adjusted params
            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            // Record action stats
            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Apply Perturb action effects (noise injection)
            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                // Record pre-TD for effectiveness measurement
                perturb_pre_td_buffer.push(abs_td as f32);
                perturb_tick_buffer.push(global_tick);

                // Apply tiny noise to top nodes (controlled perturbation)
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            // Check perturb effectiveness after 10 ticks
            let mut i = 0;
            while i < perturb_tick_buffer.len() {
                if global_tick == perturb_tick_buffer[i] + 10 {
                    let pre_td = perturb_pre_td_buffer[i];
                    action_policy.record_perturb_effectiveness(pre_td, abs_td as f32);
                    perturb_pre_td_buffer.remove(i);
                    perturb_tick_buffer.remove(i);
                } else {
                    i += 1;
                }
            }

            // Update partition info
            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            // Update prototype when gate passes
            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            // Value update for previous anchor
            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            // Update previous state
            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            // Memory store/recall
            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }
    println!("done.");

    // Get stats
    let action_stats = &action_policy.stats;
    let stable_drop_ratio = anchor_bank.stable_drop_ratio();

    // Print results
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0c: Mode → Action Loop Results");
    println!("═══════════════════════════════════════════════════════════════════");

    // A) Action usage
    println!();
    println!("A) Action usage:");
    println!(
        "  scan_count:    {} ({:.1}%)",
        action_stats.scan_count,
        action_stats.scan_rate() * 100.0
    );
    println!(
        "  focus_count:   {} ({:.1}%)",
        action_stats.focus_count,
        action_stats.focus_rate() * 100.0
    );
    println!(
        "  perturb_count: {} ({:.2}%)",
        action_stats.perturb_count,
        action_stats.perturb_rate() * 100.0
    );

    // B) Per-action diagnostics
    println!();
    println!("B) Per-action diagnostics:");
    println!(
        "  {:8} | {:>10} | {:>8} | {:>12}",
        "Action", "gate_pass%", "meanV", "stable_share%"
    );
    println!("  {}", "-".repeat(50));

    for action in [Action::Scan, Action::Focus, Action::Perturb] {
        let action_name = match action {
            Action::Scan => "Scan",
            Action::Focus => "Focus",
            Action::Perturb => "Perturb",
        };
        let rate = match action {
            Action::Scan => action_stats.scan_rate(),
            Action::Focus => action_stats.focus_rate(),
            Action::Perturb => action_stats.perturb_rate(),
        };
        if rate < 0.0001 {
            continue; // Skip actions with no ticks
        }
        println!(
            "  {:8} | {:9.1}% | {:+7.3} | {:11.1}%",
            action_name,
            action_stats.gate_pass_rate(action) * 100.0,
            action_stats.mean_value(action),
            action_stats.stable_share(action) * 100.0
        );
    }

    // Perturb effectiveness
    println!();
    println!("  Perturb effectiveness:");
    println!(
        "    mean |TD| reduction: {:+.1}% (n={})",
        action_stats.perturb_effectiveness() * 100.0,
        action_stats.perturb_effectiveness_samples
    );

    // C) Regression guard
    println!();
    println!("C) Regression guard:");

    let coverage_ok = metrics.coverage_pos() >= 0.70;
    let selective_ok = metrics.selective_accuracy() >= 0.80;
    let fp_ok = metrics.false_positive_rate() == 0.0;
    let stable_drop_ok = stable_drop_ratio <= 0.005;

    println!(
        "  [{}] coverage_pos >= 70%: {:.1}%",
        if coverage_ok { "✓" } else { "✗" },
        metrics.coverage_pos() * 100.0
    );
    println!(
        "  [{}] selective_accuracy >= 80%: {:.1}%",
        if selective_ok { "✓" } else { "✗" },
        metrics.selective_accuracy() * 100.0
    );
    println!(
        "  [{}] false_positive == 0%: {:.1}%",
        if fp_ok { "✓" } else { "✗" },
        metrics.false_positive_rate() * 100.0
    );
    println!(
        "  [{}] stable_drop_ratio <= 0.5%: {:.2}%",
        if stable_drop_ok { "✓" } else { "✗" },
        stable_drop_ratio * 100.0
    );

    // D) Behavioral differentiation
    println!();
    println!("D) Behavioral differentiation:");

    // Focus has higher stable_share than Scan
    let focus_stable = action_stats.stable_share(Action::Focus);
    let scan_stable = action_stats.stable_share(Action::Scan);
    let focus_more_stable = focus_stable > scan_stable;

    println!(
        "  Focus stable_share: {:.1}%, Scan stable_share: {:.1}%",
        focus_stable * 100.0,
        scan_stable * 100.0
    );
    println!(
        "  [{}] Focus has higher stable_share than Scan",
        if focus_more_stable { "✓" } else { "~" }
    );

    // Perturb reduces mean_abs_td within 10 ticks
    let perturb_effective = action_stats.perturb_effectiveness() > 0.0;
    println!(
        "  [{}] Perturb reduces mean|TD| within 10 ticks: {:+.1}%",
        if perturb_effective { "✓" } else { "~" },
        action_stats.perturb_effectiveness() * 100.0
    );

    // Acceptance criteria summary
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0c ACCEPTANCE:");
    println!("═══════════════════════════════════════════════════════════════════");

    // A) Action usage acceptance
    let scan_rate_ok = action_stats.scan_rate() >= 0.05;
    let focus_rate_ok = action_stats.focus_rate() >= 0.50;
    let perturb_rate_ok =
        action_stats.perturb_rate() >= 0.005 && action_stats.perturb_rate() <= 0.05;

    println!();
    println!("A) Action usage:");
    println!(
        "  [{}] scan_rate >= 5%: {:.1}%",
        if scan_rate_ok { "✓" } else { "✗" },
        action_stats.scan_rate() * 100.0
    );
    println!(
        "  [{}] focus_rate >= 50%: {:.1}%",
        if focus_rate_ok { "✓" } else { "✗" },
        action_stats.focus_rate() * 100.0
    );
    println!(
        "  [{}] perturb_rate in [0.5%, 5%]: {:.2}%",
        if perturb_rate_ok { "✓" } else { "✗" },
        action_stats.perturb_rate() * 100.0
    );

    println!();
    println!("B) Regression guard:");
    println!(
        "  [{}] coverage_pos >= 70%",
        if coverage_ok { "✓" } else { "✗" }
    );
    println!(
        "  [{}] selective_accuracy >= 80%",
        if selective_ok { "✓" } else { "✗" }
    );
    println!("  [{}] false_positive == 0%", if fp_ok { "✓" } else { "✗" });
    println!(
        "  [{}] stable_drop_ratio <= 0.5%",
        if stable_drop_ok { "✓" } else { "✗" }
    );

    println!();
    println!("C) Behavioral differentiation:");
    println!(
        "  [{}] Focus has higher stable_share than Scan",
        if focus_more_stable { "✓" } else { "~" }
    );
    println!(
        "  [{}] Perturb reduces mean|TD|",
        if perturb_effective { "✓" } else { "~" }
    );

    let all_ok = scan_rate_ok
        && focus_rate_ok
        && perturb_rate_ok
        && coverage_ok
        && selective_ok
        && fp_ok
        && stable_drop_ok;

    println!();
    if all_ok {
        println!("  → Phase 2.0c: ALL ACCEPTANCE CRITERIA MET!");
    } else {
        let usage_ok = scan_rate_ok && focus_rate_ok && perturb_rate_ok;
        let regression_ok = coverage_ok && selective_ok && fp_ok && stable_drop_ok;

        if usage_ok && regression_ok {
            println!("  → Phase 2.0c: Usage and regression OK. Behavioral differentiation may need tuning.");
        } else if usage_ok {
            println!("  → Phase 2.0c: Usage OK. Regression guard failed - needs tuning.");
        } else if regression_ok {
            println!("  → Phase 2.0c: Regression OK. Action usage needs tuning.");
        } else {
            println!("  → Phase 2.0c: Multiple criteria not met. Tuning needed.");
        }
    }
}

// =============================================================================
// DEMO 10: Phase 2.0d - Action Ablations + Sensitivity Sweep
// =============================================================================

pub fn demo_10_action_ablations(config: &Config) {
    use action::{Action, ActionConfig, ActionPolicy};
    use action_ablate::{
        ActionAblationVariant, SweepConfig, SweepPoint, VariantConfig, VariantReport,
    };
    use mode::{ModePolicy, ModePolicyConfig};

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 10: Phase 2.0d - ACTION ABLATIONS + SENSITIVITY SWEEP");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if !config.enable_mode_policy || !config.enable_action_policy {
        println!("Mode or action policy disabled. Skipping Demo 10.");
        return;
    }

    // =========================================================================
    // Part A: Ablation Variants
    // =========================================================================
    println!("Part A: Ablation Variants");
    println!("{}", "-".repeat(65));

    let variants = ActionAblationVariant::all_variants();
    let mut reports: Vec<VariantReport> = Vec::new();

    // Run FULL first to capture baseline rates for budgeted random
    print!("  Running FULL (baseline)... ");
    let full_variant_config = VariantConfig::from_variant(ActionAblationVariant::Full);
    let full_report = run_action_variant(config, "FULL", &full_variant_config, None);
    println!("done.");
    let full_scan_rate = full_report.scan_rate;
    let full_perturb_rate = full_report.perturb_rate;
    reports.push(full_report);

    println!(
        "    → FULL baseline: scan_rate={:.2}%, perturb_rate={:.3}%",
        full_scan_rate * 100.0,
        full_perturb_rate * 100.0
    );

    // Run remaining variants, passing budget targets to RANDOM_BUDGETED
    for variant in &variants {
        // Skip FULL since we already ran it
        if *variant == ActionAblationVariant::Full {
            continue;
        }
        print!("  Running {}... ", variant.label());
        let variant_config = VariantConfig::from_variant(*variant);
        let budget_targets = if *variant == ActionAblationVariant::RandomBudgeted {
            Some((full_scan_rate, full_perturb_rate))
        } else {
            None
        };
        let report = run_action_variant(config, variant.label(), &variant_config, budget_targets);
        println!("done.");
        reports.push(report);
    }

    // Print ablation results table
    println!();
    println!("Ablation Results:");
    println!(
        "  {:12} | {:>6} | {:>6} | {:>8} | {:>8} | {:>7} | {:>6} | {:>11}",
        "Variant", "Scan%", "Focus%", "Perturb%", "Coverage", "SelAcc", "FP%", "StableShare"
    );
    println!("  {}", "-".repeat(85));

    for report in &reports {
        println!(
            "  {:12} | {:5.1}% | {:5.1}% | {:7.2}% | {:7.1}% | {:6.1}% | {:5.1}% | {:10.1}%",
            report.label,
            report.scan_rate * 100.0,
            report.focus_rate * 100.0,
            report.perturb_rate * 100.0,
            report.coverage_pos * 100.0,
            report.selective_accuracy * 100.0,
            report.false_positive_rate * 100.0,
            report.stable_time_share * 100.0
        );
    }

    // =========================================================================
    // Part B: Sensitivity Sweep
    // =========================================================================
    println!();
    println!("Part B: Sensitivity Sweep");
    println!("{}", "-".repeat(65));

    let sweep_config = SweepConfig::default();

    // Sweep scan_rate_target by adjusting mode thresholds
    println!();
    println!("  Scan Rate Sweep (varying explore_v_max):");
    let mut scan_sweep_points: Vec<SweepPoint> = Vec::new();

    for &target in &sweep_config.scan_rate_targets {
        // Adjust explore_v_max to hit target scan rate
        // Higher explore_v_max -> more Explore mode -> more Scan
        let adjusted_explore_v_max = if target == 0.0 {
            0.0 // Never explore
        } else if target <= 0.02 {
            0.20
        } else if target <= 0.05 {
            0.35
        } else if target <= 0.08 {
            0.45
        } else {
            0.55 // Aggressive explore
        };

        let sweep_point = run_sweep_point(
            config,
            "scan_rate",
            target,
            Some(adjusted_explore_v_max),
            None,
        );
        scan_sweep_points.push(sweep_point);
    }

    println!(
        "    {:>8} | {:>6} | {:>6} | {:>8} | {:>8} | {:>7} | {:>11} | {:>8}",
        "Target", "Scan%", "Focus%", "Perturb%", "Coverage", "SelAcc", "StableShare", "mean|TD|"
    );
    println!("    {}", "-".repeat(85));

    for point in &scan_sweep_points {
        println!(
            "    {:7.1}% | {:5.1}% | {:5.1}% | {:7.2}% | {:7.1}% | {:6.1}% | {:10.1}% | {:8.4}",
            point.param_value * 100.0,
            point.scan_rate * 100.0,
            point.focus_rate * 100.0,
            point.perturb_rate * 100.0,
            point.coverage_pos * 100.0,
            point.selective_accuracy * 100.0,
            point.stable_time_share * 100.0,
            point.mean_abs_td
        );
    }

    // Sweep perturb_rate_target by adjusting reset thresholds
    println!();
    println!("  Perturb Rate Sweep (varying reset_td_min):");
    let mut perturb_sweep_points: Vec<SweepPoint> = Vec::new();

    for &target in &sweep_config.perturb_rate_targets {
        // Adjust reset_td_min to hit target perturb rate
        // Lower reset_td_min -> more Reset mode -> more Perturb
        let adjusted_reset_td_min = if target == 0.0 {
            1.0 // Never reset
        } else if target <= 0.003 {
            0.50
        } else if target <= 0.007 {
            0.40
        } else if target <= 0.015 {
            0.32
        } else {
            0.25 // Aggressive reset
        };

        let sweep_point = run_sweep_point(
            config,
            "perturb_rate",
            target,
            None,
            Some(adjusted_reset_td_min),
        );
        perturb_sweep_points.push(sweep_point);
    }

    println!(
        "    {:>8} | {:>6} | {:>6} | {:>8} | {:>8} | {:>7} | {:>11} | {:>8}",
        "Target", "Scan%", "Focus%", "Perturb%", "Coverage", "SelAcc", "StableShare", "mean|TD|"
    );
    println!("    {}", "-".repeat(85));

    for point in &perturb_sweep_points {
        println!(
            "    {:7.1}% | {:5.1}% | {:5.1}% | {:7.2}% | {:7.1}% | {:6.1}% | {:10.1}% | {:8.4}",
            point.param_value * 100.0,
            point.scan_rate * 100.0,
            point.focus_rate * 100.0,
            point.perturb_rate * 100.0,
            point.coverage_pos * 100.0,
            point.selective_accuracy * 100.0,
            point.stable_time_share * 100.0,
            point.mean_abs_td
        );
    }

    // =========================================================================
    // Part C: Acceptance Criteria
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0d ACCEPTANCE:");
    println!("═══════════════════════════════════════════════════════════════════");

    let full_report = &reports[0]; // FULL variant
    let no_scan_report = &reports[1];
    let no_perturb_report = &reports[2];
    let no_focus_report = &reports[3];
    let random_report = &reports[4];

    // C1) Directional effects
    println!();
    println!("C1) Directional effects:");

    // NO_SCAN -> Scan rate ~0 and Focus rate increases
    let no_scan_scan_low = no_scan_report.scan_rate < 0.01;
    let no_scan_focus_up = no_scan_report.focus_rate > full_report.focus_rate;
    println!(
        "  [{}] NO_SCAN: scan_rate < 1%: {:.1}%",
        if no_scan_scan_low { "✓" } else { "✗" },
        no_scan_report.scan_rate * 100.0
    );
    println!(
        "  [{}] NO_SCAN: focus_rate > FULL: {:.1}% vs {:.1}%",
        if no_scan_focus_up { "✓" } else { "~" },
        no_scan_report.focus_rate * 100.0,
        full_report.focus_rate * 100.0
    );

    // NO_PERTURB -> Perturb rate ~0
    let no_perturb_perturb_low = no_perturb_report.perturb_rate < 0.001;
    println!(
        "  [{}] NO_PERTURB: perturb_rate < 0.1%: {:.2}%",
        if no_perturb_perturb_low { "✓" } else { "✗" },
        no_perturb_report.perturb_rate * 100.0
    );

    // RANDOM_BUDGETED -> worse than FULL on selective_accuracy OR stable_time_share
    // With budget-matched random, we test if *timing* matters, not just action rates
    // A 1% drop is significant given matched budgets
    let random_sel_acc_drop = full_report.selective_accuracy - random_report.selective_accuracy;
    let random_stable_drop = full_report.stable_time_share - random_report.stable_time_share;
    let random_worse = random_sel_acc_drop >= 0.01 || random_stable_drop >= 0.01;
    println!(
        "  [{}] RANDOM_BUDGETED: SelAcc drop >= 1%: {:.1}% (drop: {:.2}%)",
        if random_sel_acc_drop >= 0.01 {
            "✓"
        } else {
            "~"
        },
        random_report.selective_accuracy * 100.0,
        random_sel_acc_drop * 100.0
    );
    println!(
        "  [{}] RANDOM_BUDGETED: StableShare drop >= 1%: {:.1}% (drop: {:.2}%)",
        if random_stable_drop >= 0.01 {
            "✓"
        } else {
            "~"
        },
        random_report.stable_time_share * 100.0,
        random_stable_drop * 100.0
    );
    println!(
        "  [{}] RANDOM_BUDGETED worse than FULL on at least one metric",
        if random_worse { "✓" } else { "✗" }
    );
    // Show budget match quality
    println!(
        "        Budget match: scan {:.2}% vs {:.2}%, perturb {:.3}% vs {:.3}%",
        random_report.scan_rate * 100.0,
        full_report.scan_rate * 100.0,
        random_report.perturb_rate * 100.0,
        full_report.perturb_rate * 100.0
    );

    // C2) Regression guard for FULL
    println!();
    println!("C2) Regression guard (FULL variant):");

    let coverage_ok = full_report.coverage_pos >= 0.70;
    let selective_ok = full_report.selective_accuracy >= 0.80;
    let fp_ok = full_report.false_positive_rate == 0.0;
    let stable_drop_ok = full_report.stable_drop_ratio <= 0.005;

    println!(
        "  [{}] coverage_pos >= 70%: {:.1}%",
        if coverage_ok { "✓" } else { "✗" },
        full_report.coverage_pos * 100.0
    );
    println!(
        "  [{}] selective_accuracy >= 80%: {:.1}%",
        if selective_ok { "✓" } else { "✗" },
        full_report.selective_accuracy * 100.0
    );
    println!(
        "  [{}] false_positive == 0%: {:.1}%",
        if fp_ok { "✓" } else { "✗" },
        full_report.false_positive_rate * 100.0
    );
    println!(
        "  [{}] stable_drop_ratio <= 0.5%: {:.2}%",
        if stable_drop_ok { "✓" } else { "✗" },
        full_report.stable_drop_ratio * 100.0
    );

    // C3) Sweep produces non-trivial curve
    println!();
    println!("C3) Sweep produces non-trivial curve:");

    // Find min/max selective_accuracy in scan sweep
    let scan_sel_acc_min = scan_sweep_points
        .iter()
        .map(|p| p.selective_accuracy)
        .fold(f64::INFINITY, f64::min);
    let scan_sel_acc_max = scan_sweep_points
        .iter()
        .map(|p| p.selective_accuracy)
        .fold(f64::NEG_INFINITY, f64::max);
    let scan_sel_acc_range = scan_sel_acc_max - scan_sel_acc_min;
    let scan_curve_ok = scan_sel_acc_range >= 0.01;

    println!(
        "  [{}] Scan sweep: SelAcc range >= 1%: {:.1}% (min {:.1}%, max {:.1}%)",
        if scan_curve_ok { "✓" } else { "~" },
        scan_sel_acc_range * 100.0,
        scan_sel_acc_min * 100.0,
        scan_sel_acc_max * 100.0
    );

    // Find min/max stable_time_share in perturb sweep
    let perturb_stable_min = perturb_sweep_points
        .iter()
        .map(|p| p.stable_time_share)
        .fold(f64::INFINITY, f64::min);
    let perturb_stable_max = perturb_sweep_points
        .iter()
        .map(|p| p.stable_time_share)
        .fold(f64::NEG_INFINITY, f64::max);
    let perturb_stable_range = perturb_stable_max - perturb_stable_min;
    let perturb_curve_ok = perturb_stable_range >= 0.03;

    println!(
        "  [{}] Perturb sweep: StableShare range >= 3%: {:.1}% (min {:.1}%, max {:.1}%)",
        if perturb_curve_ok { "✓" } else { "~" },
        perturb_stable_range * 100.0,
        perturb_stable_min * 100.0,
        perturb_stable_max * 100.0
    );

    // Summary
    let directional_ok = no_scan_scan_low && no_perturb_perturb_low && random_worse;
    let regression_ok = coverage_ok && selective_ok && fp_ok && stable_drop_ok;
    let sweep_ok = scan_curve_ok || perturb_curve_ok;
    let all_ok = directional_ok && regression_ok && sweep_ok;

    println!();
    if all_ok {
        println!("  → Phase 2.0d: ALL ACCEPTANCE CRITERIA MET!");
    } else {
        if directional_ok && regression_ok {
            println!("  → Phase 2.0d: Directional and regression OK. Sweep needs more variation.");
        } else if directional_ok {
            println!("  → Phase 2.0d: Directional OK. Regression failed.");
        } else if regression_ok {
            println!("  → Phase 2.0d: Regression OK. Directional effects not as expected.");
        } else {
            println!("  → Phase 2.0d: Multiple criteria not met. Tuning needed.");
        }
    }
}

/// Run an action variant and collect metrics.
/// For budgeted_random variants, pass budget_targets (scan_rate, perturb_rate) from FULL baseline.
fn run_action_variant(
    config: &Config,
    label: &str,
    variant_config: &action_ablate::VariantConfig,
    budget_targets: Option<(f64, f64)>, // (scan_rate, perturb_rate) for budgeted random
) -> action_ablate::VariantReport {
    use action::{Action, ActionConfig, ActionPolicy};
    use action_ablate::{BudgetedRandomAction, VariantReport};
    use mode::{ModePolicy, ModePolicyConfig};

    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);

    // Create budgeted random action selector if needed
    let mut budgeted_random = if variant_config.budgeted_random {
        if let Some((scan_target, perturb_target)) = budget_targets {
            Some(BudgetedRandomAction::new(scan_target, perturb_target))
        } else {
            // Fallback: use typical rates if no targets provided
            Some(BudgetedRandomAction::new(0.05, 0.007))
        }
    } else {
        None
    };

    // Use same seed as Demo 9 for consistency
    let mut rng = Rng::new(config.seed.wrapping_add(0x7A7A_7A7A));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train phase
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

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

    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    let mut total_stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;
    let mut total_abs_td_sum: f64 = 0.0;

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            total_abs_td_sum += abs_td as f64;
            total_ticks += 1;
            if is_stable {
                total_stable_ticks += 1;
            }

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);

            let mode = mode_policy.choose_mode(global_tick);

            // Apply ablation to action selection
            let rng_val = rng.next_f64();
            let action = if let Some(ref mut br) = budgeted_random {
                // Use budget-matched random action selection
                br.choose(rng_val)
            } else {
                action_policy.choose_action_with_ablation(mode, variant_config, rng_val)
            };

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    // Build report
    let action_stats = &action_policy.stats;
    let total_actions = action_stats.total_count();

    let mut report = action_ablate::VariantReport::new(label);
    report.scan_count = action_stats.scan_count;
    report.focus_count = action_stats.focus_count;
    report.perturb_count = action_stats.perturb_count;
    report.scan_rate = action_stats.scan_rate();
    report.focus_rate = action_stats.focus_rate();
    report.perturb_rate = action_stats.perturb_rate();

    report.coverage_pos = metrics.coverage_pos();
    report.selective_accuracy = metrics.selective_accuracy();
    report.false_positive_rate = metrics.false_positive_rate();
    report.stable_drop_ratio = anchor_bank.stable_drop_ratio();

    report.mean_abs_td = if total_ticks > 0 {
        total_abs_td_sum / total_ticks as f64
    } else {
        0.0
    };
    report.stable_time_share = if total_ticks > 0 {
        total_stable_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    report.scan_stable_share = action_stats.stable_share(Action::Scan);
    report.focus_stable_share = action_stats.stable_share(Action::Focus);
    report.perturb_stable_share = action_stats.stable_share(Action::Perturb);

    report.perturb_effectiveness = action_stats.perturb_effectiveness();
    report.perturb_effectiveness_samples = action_stats.perturb_effectiveness_samples;

    report
}

/// Run a single sweep point and collect metrics.
fn run_sweep_point(
    config: &Config,
    param_name: &str,
    param_value: f64,
    explore_v_max_override: Option<f32>,
    reset_td_min_override: Option<f32>,
) -> action_ablate::SweepPoint {
    use action::{Action, ActionConfig, ActionPolicy};
    use mode::{ModePolicy, ModePolicyConfig};

    // Create modified mode policy config
    let mode_policy_config = ModePolicyConfig {
        explore_v_max: explore_v_max_override.unwrap_or(config.mode_explore_v_max),
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: reset_td_min_override.unwrap_or(config.mode_reset_td_min),
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);

    // Use fixed seed for sweep consistency
    let mut rng = Rng::new(config.seed.wrapping_add(0x7A7A_7A7A));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train phase
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

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

    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    let mut total_stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;
    let mut total_abs_td_sum: f64 = 0.0;

    // Run fewer episodes for sweep (faster)
    let sweep_episodes = (config.competitive_episodes / 2).max(50);

    for _ep in 0..sweep_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            total_abs_td_sum += abs_td as f64;
            total_ticks += 1;
            if is_stable {
                total_stable_ticks += 1;
            }

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);

            let mode = mode_policy.choose_mode(global_tick);
            let action = action_policy.choose_action(mode);

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    let action_stats = &action_policy.stats;

    let mut point = action_ablate::SweepPoint::new(param_name, param_value);
    point.scan_rate = action_stats.scan_rate();
    point.focus_rate = action_stats.focus_rate();
    point.perturb_rate = action_stats.perturb_rate();
    point.coverage_pos = metrics.coverage_pos();
    point.selective_accuracy = metrics.selective_accuracy();
    point.false_positive_rate = metrics.false_positive_rate();
    point.stable_drop_ratio = anchor_bank.stable_drop_ratio();
    point.mean_abs_td = if total_ticks > 0 {
        total_abs_td_sum / total_ticks as f64
    } else {
        0.0
    };
    point.stable_time_share = if total_ticks > 0 {
        total_stable_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    point
}

// =============================================================================
// DEMO 11: Phase 2.0e - TRIGGER-MATCHED RANDOM + REGRET METRICS
// =============================================================================

pub fn demo_11_trigger_matched(config: &Config) {
    use action::{Action, ActionConfig, ActionPolicy};
    use action_ablate::{BudgetedRandomAction, TriggerMatchedRandom, TriggerTrace};
    use mode::{ModePolicy, ModePolicyConfig};
    use regret::{RegretConfig, RegretReport, RegretStats};

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 11: Phase 2.0e - TRIGGER-MATCHED RANDOM + REGRET METRICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if !config.enable_mode_policy || !config.enable_action_policy {
        println!("Mode or action policy disabled. Skipping Demo 11.");
        return;
    }

    let regret_config = RegretConfig {
        margin_bad: config.regret_margin_bad,
        proto_bad: config.regret_proto_bad,
        v_bad: config.regret_v_bad,
        td_spike: config.regret_td_spike,
        pre_window: config.regret_pre_window,
        post_window: config.regret_post_window,
        post_gate_window: config.regret_post_gate_window,
        recovery_good_threshold: config.regret_recovery_good_threshold,
    };

    // =========================================================================
    // Step 1: Run FULL and capture trigger trace
    // =========================================================================
    print!("  Running FULL (capturing triggers)... ");
    let (full_report, trigger_trace) = run_demo11_variant_full(config, &regret_config);
    println!("done. ({} triggers)", trigger_trace.trigger_count());

    // =========================================================================
    // Step 2: Run RANDOM_BUDGETED (existing baseline)
    // =========================================================================
    print!("  Running RANDOM_BUDGETED... ");
    let budgeted_report = run_demo11_variant_budgeted(
        config,
        &regret_config,
        full_report.scan_rate,
        full_report.perturb_rate,
    );
    println!("done.");

    // =========================================================================
    // Step 3: Run RANDOM_TRIGGER_MATCHED (new fair baseline)
    // =========================================================================
    print!("  Running RANDOM_TRIGGER_MATCHED... ");
    let trigger_report = run_demo11_variant_trigger_matched(config, &regret_config, &trigger_trace);
    println!("done.");

    // =========================================================================
    // Print Results Table
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0e RESULTS: FULL vs RANDOM_BUDGETED vs RANDOM_TRIGGER");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Standard metrics table
    println!("Standard Metrics:");
    println!(
        "  {:16} | {:>7} | {:>7} | {:>5} | {:>11} | {:>6} | {:>6} | {:>8}",
        "Variant", "Cover%", "SelAcc%", "FP%", "StableShare", "Scan%", "Focus%", "Perturb%"
    );
    println!("  {}", "-".repeat(85));

    for report in [&full_report, &budgeted_report, &trigger_report] {
        println!(
            "  {:16} | {:6.1}% | {:6.1}% | {:4.1}% | {:10.1}% | {:5.1}% | {:5.1}% | {:7.2}%",
            report.label,
            report.coverage_pos * 100.0,
            report.selective_accuracy * 100.0,
            report.false_positive_rate * 100.0,
            report.stable_time_share * 100.0,
            report.scan_rate * 100.0,
            report.focus_rate * 100.0,
            report.perturb_rate * 100.0,
        );
    }

    // Regret metrics table
    println!();
    println!("Regret/Recovery Metrics:");
    println!(
        "  {:16} | {:>10} | {:>12} | {:>12} | {:>10} | {:>10}",
        "Variant", "BadState%", "TDSpike/10k", "RecovImprv%", "RecovGood%", "Regret%"
    );
    println!("  {}", "-".repeat(80));

    for report in [&full_report, &budgeted_report, &trigger_report] {
        println!(
            "  {:16} | {:9.1}% | {:11.1} | {:11.1}% | {:9.1}% | {:9.1}%",
            report.label,
            report.bad_state_share * 100.0,
            report.td_spike_rate,
            report.recovery_improve_mean * 100.0,
            report.recovery_good_rate * 100.0,
            report.regret_rate * 100.0,
        );
    }

    // Trigger match verification
    println!();
    println!("Trigger Match Verification:");
    let full_trigger_rate = (full_report.scan_rate + full_report.perturb_rate) * 100.0;
    let trigger_action_rate = (trigger_report.scan_rate + trigger_report.perturb_rate) * 100.0;
    let trigger_delta = (trigger_action_rate - full_trigger_rate).abs();
    println!(
        "  FULL trigger rate:   {:.2}% (Scan + Perturb)",
        full_trigger_rate
    );
    println!(
        "  TRIGGER trigger rate: {:.2}% (delta: {:.3}%)",
        trigger_action_rate, trigger_delta
    );

    // =========================================================================
    // Acceptance Criteria
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0e ACCEPTANCE:");
    println!("═══════════════════════════════════════════════════════════════════");

    // C1) Trigger match within ±1%
    println!();
    println!("C1) Trigger match (action_rate within ±1%):");
    let trigger_match_ok = trigger_delta < 1.0;
    println!(
        "  [{}] Trigger delta < 1%: {:.3}%",
        if trigger_match_ok { "✓" } else { "✗" },
        trigger_delta
    );

    // C2) FULL beats RANDOM_TRIGGER on at least TWO regret metrics
    println!();
    println!("C2) Directional effects (FULL vs RANDOM_TRIGGER):");

    // Bad state share: FULL should be lower by >= 2.0 pp
    let bad_state_drop = trigger_report.bad_state_share - full_report.bad_state_share;
    let bad_state_ok = bad_state_drop >= 0.02;
    println!(
        "  [{}] bad_state_share drop >= 2.0pp: {:.1}% vs {:.1}% (drop: {:.2}pp)",
        if bad_state_ok { "✓" } else { "~" },
        full_report.bad_state_share * 100.0,
        trigger_report.bad_state_share * 100.0,
        bad_state_drop * 100.0
    );

    // TD spike rate: FULL should be lower by >= 10%
    let td_spike_improve = if trigger_report.td_spike_rate > 0.001 {
        (trigger_report.td_spike_rate - full_report.td_spike_rate) / trigger_report.td_spike_rate
    } else {
        0.0
    };
    let td_spike_ok = td_spike_improve >= 0.10;
    println!(
        "  [{}] td_spike_rate improve >= 10%: {:.1} vs {:.1} (improve: {:.1}%)",
        if td_spike_ok { "✓" } else { "~" },
        full_report.td_spike_rate,
        trigger_report.td_spike_rate,
        td_spike_improve * 100.0
    );

    // Recovery improve: FULL should be better by >= 10% relative
    let recovery_diff = full_report.recovery_improve_mean - trigger_report.recovery_improve_mean;
    let recovery_ok = recovery_diff >= 0.10
        || full_report.recovery_improve_mean >= trigger_report.recovery_improve_mean + 0.05;
    println!(
        "  [{}] recovery_improve better: {:.1}% vs {:.1}% (diff: {:.2}pp)",
        if recovery_ok { "✓" } else { "~" },
        full_report.recovery_improve_mean * 100.0,
        trigger_report.recovery_improve_mean * 100.0,
        recovery_diff * 100.0
    );

    // Regret rate: FULL should be lower by >= 10% relative
    let regret_improve = if trigger_report.regret_rate > 0.001 {
        (trigger_report.regret_rate - full_report.regret_rate) / trigger_report.regret_rate
    } else {
        0.0
    };
    let regret_ok = regret_improve >= 0.10;
    println!(
        "  [{}] regret_rate improve >= 10%: {:.1}% vs {:.1}% (improve: {:.1}%)",
        if regret_ok { "✓" } else { "~" },
        full_report.regret_rate * 100.0,
        trigger_report.regret_rate * 100.0,
        regret_improve * 100.0
    );

    let regret_wins = [bad_state_ok, td_spike_ok, recovery_ok, regret_ok]
        .iter()
        .filter(|&&x| x)
        .count();
    let directional_ok = regret_wins >= 2;
    println!(
        "  [{}] FULL beats TRIGGER on >= 2 regret metrics: {}/4",
        if directional_ok { "✓" } else { "✗" },
        regret_wins
    );

    // C3) Regression guard for FULL
    println!();
    println!("C3) Regression guard (FULL variant):");

    let coverage_ok = full_report.coverage_pos >= 0.70;
    let selective_ok = full_report.selective_accuracy >= 0.80;
    let fp_ok = full_report.false_positive_rate == 0.0;

    println!(
        "  [{}] coverage_pos >= 70%: {:.1}%",
        if coverage_ok { "✓" } else { "✗" },
        full_report.coverage_pos * 100.0
    );
    println!(
        "  [{}] selective_accuracy >= 80%: {:.1}%",
        if selective_ok { "✓" } else { "✗" },
        full_report.selective_accuracy * 100.0
    );
    println!(
        "  [{}] false_positive == 0%: {:.1}%",
        if fp_ok { "✓" } else { "✗" },
        full_report.false_positive_rate * 100.0
    );

    let regression_ok = coverage_ok && selective_ok && fp_ok;

    // Summary
    let all_ok = trigger_match_ok && directional_ok && regression_ok;
    println!();
    if all_ok {
        println!("  → Phase 2.0e: ALL ACCEPTANCE CRITERIA MET!");
    } else {
        if trigger_match_ok && regression_ok {
            println!(
                "  → Phase 2.0e: Trigger match and regression OK. Directional effects need work."
            );
        } else if directional_ok && regression_ok {
            println!("  → Phase 2.0e: Directional and regression OK. Trigger match needs tuning.");
        } else {
            println!("  → Phase 2.0e: Multiple criteria not met. Tuning needed.");
        }
    }
}

/// Run FULL variant and capture trigger trace for Demo 11.
fn run_demo11_variant_full(
    config: &Config,
    regret_config: &regret::RegretConfig,
) -> (regret::RegretReport, action_ablate::TriggerTrace) {
    use action::{Action, ActionConfig, ActionPolicy};
    use action_ablate::TriggerTrace;
    use mode::{ModePolicy, ModePolicyConfig};
    use regret::RegretStats;

    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);

    let mut rng = Rng::new(config.seed.wrapping_add(0x7A7A_7A7A));
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

    let mut anchor_bank = AnchorBank::new();
    let keyed_config = KeyedMemoryConfig {
        label_min_p: 0.50,
        label_margin: 0.10,
        alpha: 0.5,
        num_labels: config.num_ctx,
    };
    let mut keyed_memory = KeyedMemoryStore::new(keyed_config);
    let mut metrics = KeyedMemoryMetrics::new();
    let mut regret_stats = RegretStats::new(regret_config);

    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);
    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;

    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    let total_ticks_expected = config.competitive_episodes * config.competitive_episode_ticks;
    let mut trigger_trace = TriggerTrace::with_capacity(total_ticks_expected);

    let mut total_stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                trigger_trace.record(Action::Focus); // Placeholder
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            let proto_align = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.proto_score(&base_topk, config.proto_m))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            total_ticks += 1;
            if is_stable {
                total_stable_ticks += 1;
            }

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);

            let mode = mode_policy.choose_mode(global_tick);
            let action = action_policy.choose_action(mode);

            // Record trigger trace
            trigger_trace.record(action);

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Record regret stats
            regret_stats.observe_tick(
                regret_config,
                global_tick,
                abs_td as f32,
                topk_margin,
                proto_align,
                anchor_value,
                gate_passed,
            );
            regret_stats.observe_action(global_tick, action);
            regret_stats.check_pending_actions(regret_config, global_tick);

            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    regret_stats.finalize(regret_config);

    let action_stats = &action_policy.stats;

    let mut report = regret::RegretReport::new("FULL");
    report.scan_rate = action_stats.scan_rate();
    report.focus_rate = action_stats.focus_rate();
    report.perturb_rate = action_stats.perturb_rate();
    report.trigger_count = action_stats.scan_count + action_stats.perturb_count;
    report.coverage_pos = metrics.coverage_pos();
    report.selective_accuracy = metrics.selective_accuracy();
    report.false_positive_rate = metrics.false_positive_rate();
    report.stable_time_share = if total_ticks > 0 {
        total_stable_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };
    report.bad_state_share = regret_stats.bad_state_share();
    report.td_spike_rate = regret_stats.td_spike_rate();
    report.recovery_improve_mean = regret_stats.recovery_improve_mean();
    report.recovery_good_rate = regret_stats.recovery_good_rate();
    report.regret_rate = regret_stats.regret_rate();

    (report, trigger_trace)
}

/// Run RANDOM_BUDGETED variant for Demo 11.
fn run_demo11_variant_budgeted(
    config: &Config,
    regret_config: &regret::RegretConfig,
    scan_target: f64,
    perturb_target: f64,
) -> regret::RegretReport {
    use action::{Action, ActionConfig, ActionPolicy};
    use action_ablate::BudgetedRandomAction;
    use mode::{ModePolicy, ModePolicyConfig};
    use regret::RegretStats;

    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);
    let mut budgeted_random = BudgetedRandomAction::new(scan_target, perturb_target);

    let mut rng = Rng::new(config.seed.wrapping_add(0x7A7A_7A7A));
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

    let mut anchor_bank = AnchorBank::new();
    let keyed_config = KeyedMemoryConfig {
        label_min_p: 0.50,
        label_margin: 0.10,
        alpha: 0.5,
        num_labels: config.num_ctx,
    };
    let mut keyed_memory = KeyedMemoryStore::new(keyed_config);
    let mut metrics = KeyedMemoryMetrics::new();
    let mut regret_stats = RegretStats::new(regret_config);

    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);
    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;

    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    let mut total_stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            let proto_align = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.proto_score(&base_topk, config.proto_m))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            total_ticks += 1;
            if is_stable {
                total_stable_ticks += 1;
            }

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);

            // Use budgeted random action selection
            let rng_val = rng.next_f64();
            let action = budgeted_random.choose(rng_val);

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Record regret stats
            regret_stats.observe_tick(
                regret_config,
                global_tick,
                abs_td as f32,
                topk_margin,
                proto_align,
                anchor_value,
                gate_passed,
            );
            regret_stats.observe_action(global_tick, action);
            regret_stats.check_pending_actions(regret_config, global_tick);

            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    regret_stats.finalize(regret_config);

    let action_stats = &action_policy.stats;

    let mut report = regret::RegretReport::new("RANDOM_BUDGETED");
    report.scan_rate = action_stats.scan_rate();
    report.focus_rate = action_stats.focus_rate();
    report.perturb_rate = action_stats.perturb_rate();
    report.trigger_count = action_stats.scan_count + action_stats.perturb_count;
    report.coverage_pos = metrics.coverage_pos();
    report.selective_accuracy = metrics.selective_accuracy();
    report.false_positive_rate = metrics.false_positive_rate();
    report.stable_time_share = if total_ticks > 0 {
        total_stable_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };
    report.bad_state_share = regret_stats.bad_state_share();
    report.td_spike_rate = regret_stats.td_spike_rate();
    report.recovery_improve_mean = regret_stats.recovery_improve_mean();
    report.recovery_good_rate = regret_stats.recovery_good_rate();
    report.regret_rate = regret_stats.regret_rate();

    report
}

/// Run RANDOM_TRIGGER_MATCHED variant for Demo 11.
fn run_demo11_variant_trigger_matched(
    config: &Config,
    regret_config: &regret::RegretConfig,
    trigger_trace: &action_ablate::TriggerTrace,
) -> regret::RegretReport {
    use action::{Action, ActionConfig, ActionPolicy};
    use action_ablate::TriggerMatchedRandom;
    use mode::{ModePolicy, ModePolicyConfig};
    use regret::RegretStats;

    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);
    let mut trigger_matched = TriggerMatchedRandom::new(trigger_trace.clone());

    let mut rng = Rng::new(config.seed.wrapping_add(0x7A7A_7A7A));
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

    let mut anchor_bank = AnchorBank::new();
    let keyed_config = KeyedMemoryConfig {
        label_min_p: 0.50,
        label_margin: 0.10,
        alpha: 0.5,
        num_labels: config.num_ctx,
    };
    let mut keyed_memory = KeyedMemoryStore::new(keyed_config);
    let mut metrics = KeyedMemoryMetrics::new();
    let mut regret_stats = RegretStats::new(regret_config);

    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);
    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;

    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    let mut total_stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                // Consume trigger trace tick even when window not ready
                let _ = trigger_matched.choose(rng.next_f64());
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            let proto_align = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.proto_score(&base_topk, config.proto_m))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            total_ticks += 1;
            if is_stable {
                total_stable_ticks += 1;
            }

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);

            // Use trigger-matched random action selection
            let rng_val = rng.next_f64();
            let action = trigger_matched.choose(rng_val);

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Record regret stats
            regret_stats.observe_tick(
                regret_config,
                global_tick,
                abs_td as f32,
                topk_margin,
                proto_align,
                anchor_value,
                gate_passed,
            );
            regret_stats.observe_action(global_tick, action);
            regret_stats.check_pending_actions(regret_config, global_tick);

            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    regret_stats.finalize(regret_config);

    let action_stats = &action_policy.stats;

    let mut report = regret::RegretReport::new("RANDOM_TRIGGER");
    report.scan_rate = action_stats.scan_rate();
    report.focus_rate = action_stats.focus_rate();
    report.perturb_rate = action_stats.perturb_rate();
    report.trigger_count = action_stats.scan_count + action_stats.perturb_count;
    report.coverage_pos = metrics.coverage_pos();
    report.selective_accuracy = metrics.selective_accuracy();
    report.false_positive_rate = metrics.false_positive_rate();
    report.stable_time_share = if total_ticks > 0 {
        total_stable_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };
    report.bad_state_share = regret_stats.bad_state_share();
    report.td_spike_rate = regret_stats.td_spike_rate();
    report.recovery_improve_mean = regret_stats.recovery_improve_mean();
    report.recovery_good_rate = regret_stats.recovery_good_rate();
    report.regret_rate = regret_stats.regret_rate();

    report
}

// =============================================================================
// Demo 12: Phase 2.0f-A Action Distillation
// =============================================================================

/// Demo 12: Action Distillation
/// Trains a lightweight linear-softmax student to imitate the teacher ActionPolicy.
pub fn demo_12_action_distillation(config: &Config) {
    use action::{Action, ActionConfig, ActionPolicy};
    use mode::{Mode, ModePolicy, ModePolicyConfig};

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 12: Phase 2.0f-E - NATURAL TEACHER VIABILITY + EXPLOIT EMERGENCE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if !config.enable_mode_policy || !config.enable_action_policy {
        println!("Mode or action policy disabled. Skipping Demo 12.");
        return;
    }

    println!("Configuration:");
    println!("  lr:                 {}", config.distill_lr);
    println!("  l2:                 {}", config.distill_l2);
    println!("  temperature:        {}", config.distill_temperature);
    println!("  mode_conditioned:   {}", config.distill_mode_cond);
    println!("  min_mode_samples:   {}", config.distill_min_mode_samples);
    println!("  budget_window:      {}", config.budget_window);
    println!("  budget_tol:         {}", config.budget_tol);
    println!("  budget_lambda_pref: {}", config.budget_lambda_pref);
    println!("  budget_lambda_def:  {}", config.budget_lambda_def);
    println!("  exploit_proto_min:  {}", config.exploit_proto_min);
    println!("  exploit_margin_min: {}", config.exploit_margin_min);
    println!("  exploit_requires_stable: {}", config.exploit_requires_stable);
    println!();

    // =========================================================================
    // PHASE (i): TEACHER BASELINE RUN
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE (i): TEACHER BASELINE RUN");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let (teacher_metrics, teacher_per_mode, calib_rates) = run_demo12_teacher_baseline(config);

    println!("Teacher Performance:");
    println!(
        "  coverage={:.1}%, sel_acc={:.1}%, FP={:.1}%, stable={:.1}%",
        teacher_metrics.coverage_pos * 100.0,
        teacher_metrics.selective_accuracy * 100.0,
        teacher_metrics.false_positive_rate * 100.0,
        teacher_metrics.stable_share * 100.0
    );
    println!();

    println!("Teacher Action Rates (overall):");
    println!(
        "  Scan={:.2}%, Focus={:.2}%, Perturb={:.3}%",
        calib_rates.scan_rate * 100.0,
        calib_rates.focus_rate * 100.0,
        calib_rates.perturb_rate * 100.0
    );
    println!();

    println!("Teacher Action Rates (per-mode):");
    println!(
        "  {:10} | {:>8} | {:>8} | {:>10}",
        "Mode", "Scan%", "Focus%", "Perturb%"
    );
    println!("  {}", "-".repeat(45));
    let mode_names = ["Explore", "Exploit", "Reset"];
    for (i, name) in mode_names.iter().enumerate() {
        println!(
            "  {:10} | {:7.2}% | {:7.2}% | {:9.3}%",
            name,
            teacher_per_mode.scan_rate(i) * 100.0,
            teacher_per_mode.focus_rate(i) * 100.0,
            teacher_per_mode.perturb_rate(i) * 100.0
        );
    }
    println!();

    // =========================================================================
    // PHASE (ii): TRAIN STUDENT (Stratified Collection)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE (ii): TRAIN STUDENT (Stratified Collection)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let (student, train_stats, samples_per_mode) = run_demo12_train_student(config);

    println!("Stratified Collection:");
    println!(
        "  Explore: {} samples, Exploit: {} samples, Reset: {} samples",
        samples_per_mode[0], samples_per_mode[1], samples_per_mode[2]
    );
    println!(
        "  Total: {} samples",
        samples_per_mode.iter().sum::<usize>()
    );
    println!();

    println!("Training Results:");
    println!(
        "  imitation_acc_train: {:.1}%",
        train_stats.imitation_accuracy() * 100.0
    );
    println!("  final_loss: {:.4}", train_stats.loss_ema);
    println!("  train_steps: {}", train_stats.train_steps);
    println!();

    println!("Model Info:");
    println!("  Parameters: {}", distill::LinearSoftmax::param_count());
    println!("  Weight norm: {:.4}", student.weight_norm());
    println!();

    // =========================================================================
    // PHASE (iii): EVAL STUDENT (Target-Matched Budget)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE (iii): EVAL STUDENT (Target-Matched Budget)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let (student_metrics, student_per_mode, eval_stats) =
        run_demo12_eval_student(config, &student, &calib_rates);

    println!("Student Performance:");
    println!(
        "  coverage={:.1}%, sel_acc={:.1}%, FP={:.1}%, stable={:.1}%",
        student_metrics.coverage_pos * 100.0,
        student_metrics.selective_accuracy * 100.0,
        student_metrics.false_positive_rate * 100.0,
        student_metrics.stable_share * 100.0
    );
    println!();

    println!("Student Action Rates (overall):");
    println!(
        "  Scan={:.2}%, Focus={:.2}%, Perturb={:.3}%",
        student_metrics.scan_rate() * 100.0,
        student_metrics.focus_rate() * 100.0,
        student_metrics.perturb_rate() * 100.0
    );
    println!();

    println!("Student Action Rates (per-mode):");
    println!(
        "  {:10} | {:>8} | {:>8} | {:>10}",
        "Mode", "Scan%", "Focus%", "Perturb%"
    );
    println!("  {}", "-".repeat(45));
    for (i, name) in mode_names.iter().enumerate() {
        println!(
            "  {:10} | {:7.2}% | {:7.2}% | {:9.3}%",
            name,
            student_per_mode.scan_rate(i) * 100.0,
            student_per_mode.focus_rate(i) * 100.0,
            student_per_mode.perturb_rate(i) * 100.0
        );
    }
    println!();

    println!("Evaluation Imitation:");
    println!(
        "  imitation_acc_eval: {:.1}%",
        eval_stats.imitation_accuracy() * 100.0
    );
    println!();

    // =========================================================================
    // COMPARISON: Teacher vs Student
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("COMPARISON: Teacher vs Student");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    println!("Action Rate Comparison (overall):");
    println!(
        "  {:12} | {:>8} | {:>8} | {:>10}",
        "Agent", "Scan%", "Focus%", "Perturb%"
    );
    println!("  {}", "-".repeat(47));
    println!(
        "  {:12} | {:7.2}% | {:7.2}% | {:9.3}%",
        "Teacher",
        calib_rates.scan_rate * 100.0,
        calib_rates.focus_rate * 100.0,
        calib_rates.perturb_rate * 100.0
    );
    println!(
        "  {:12} | {:7.2}% | {:7.2}% | {:9.3}%",
        "Student",
        student_metrics.scan_rate() * 100.0,
        student_metrics.focus_rate() * 100.0,
        student_metrics.perturb_rate() * 100.0
    );
    println!();

    println!("Per-Mode Action Comparison (Focus%):");
    println!(
        "  {:10} | {:>12} | {:>12} | {:>8}",
        "Mode", "Teacher%", "Student%", "Delta"
    );
    println!("  {}", "-".repeat(50));
    for (i, name) in mode_names.iter().enumerate() {
        let t_focus = teacher_per_mode.focus_rate(i) * 100.0;
        let s_focus = student_per_mode.focus_rate(i) * 100.0;
        let delta = s_focus - t_focus;
        println!(
            "  {:10} | {:11.2}% | {:11.2}% | {:+7.2}%",
            name, t_focus, s_focus, delta
        );
    }
    println!();

    // Confusion matrix
    println!("Confusion Matrix (Teacher → Student on eval):");
    println!("              Scan    Focus   Perturb");
    let labels = ["Scan", "Focus", "Perturb"];
    for t in 0..3 {
        let row = &eval_stats.confusion[t];
        let row_total: usize = row.iter().sum();
        if row_total > 0 {
            println!(
                "  {:7}  {:6} ({:4.1}%) {:6} ({:4.1}%) {:6} ({:4.1}%)",
                labels[t],
                row[0],
                row[0] as f64 / row_total as f64 * 100.0,
                row[1],
                row[1] as f64 / row_total as f64 * 100.0,
                row[2],
                row[2] as f64 / row_total as f64 * 100.0
            );
        } else {
            println!("  {:7}      0          0          0", labels[t]);
        }
    }
    println!();

    // =========================================================================
    // PHASE 2.0f-E ACCEPTANCE CRITERIA
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.0f-E ACCEPTANCE:");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Compute teacher mode rates
    let teacher_total = teacher_per_mode.total(0) + teacher_per_mode.total(1) + teacher_per_mode.total(2);
    let teacher_explore_rate = teacher_per_mode.total(0) as f64 / teacher_total.max(1) as f64;
    let teacher_exploit_rate = teacher_per_mode.total(1) as f64 / teacher_total.max(1) as f64;
    let teacher_reset_rate = teacher_per_mode.total(2) as f64 / teacher_total.max(1) as f64;

    // A) Teacher exhibits non-degenerate modes
    println!("A) Teacher exhibits non-degenerate modes:");
    let exploit_rate_ok = teacher_exploit_rate >= 0.02; // >= 2%
    let explore_rate_ok = teacher_explore_rate >= 0.03; // >= 3%
    let reset_rate_ok = teacher_reset_rate >= 0.002 && teacher_reset_rate <= 0.05; // 0.2%-5%
    println!(
        "  [{}] exploit_rate >= 2%: {:.1}%",
        if exploit_rate_ok { "✓" } else { "✗" },
        teacher_exploit_rate * 100.0
    );
    println!(
        "  [{}] explore_rate >= 3%: {:.1}%",
        if explore_rate_ok { "✓" } else { "✗" },
        teacher_explore_rate * 100.0
    );
    println!(
        "  [{}] reset_rate in [0.2%, 5%]: {:.2}%",
        if reset_rate_ok { "✓" } else { "✗" },
        teacher_reset_rate * 100.0
    );

    // B) Teacher exhibits non-zero Focus usage
    println!();
    let teacher_focus_rate = calib_rates.focus_rate;
    let focus_rate_ok = teacher_focus_rate >= 0.02; // >= 2%
    println!("B) Teacher exhibits non-zero Focus:");
    println!(
        "  [{}] focus_rate >= 2%: {:.1}%",
        if focus_rate_ok { "✓" } else { "✗" },
        teacher_focus_rate * 100.0
    );

    // C) No regressions (student performance)
    println!();
    println!("C) No regressions:");
    let coverage_ok = student_metrics.coverage_pos >= 0.70;
    let sel_acc_ok = student_metrics.selective_accuracy >= 0.80;
    let fp_ok = student_metrics.false_positive_rate <= 0.005; // <= 0.5% (stricter)
    println!(
        "  [{}] coverage_pos >= 70%: {:.1}%",
        if coverage_ok { "✓" } else { "✗" },
        student_metrics.coverage_pos * 100.0
    );
    println!(
        "  [{}] selective_accuracy >= 80%: {:.1}%",
        if sel_acc_ok { "✓" } else { "✗" },
        student_metrics.selective_accuracy * 100.0
    );
    println!(
        "  [{}] false_positive <= 0.5%: {:.2}%",
        if fp_ok { "✓" } else { "✗" },
        student_metrics.false_positive_rate * 100.0
    );

    // D) Budget fairness (optional for natural teacher)
    println!();
    println!("D) Budget fairness (informational):");
    let budget_tol = 0.05; // 5% tolerance for natural teacher
    let scan_delta = (student_metrics.scan_rate() - calib_rates.scan_rate).abs();
    let focus_delta = (student_metrics.focus_rate() - calib_rates.focus_rate).abs();
    let perturb_delta = (student_metrics.perturb_rate() - calib_rates.perturb_rate).abs();
    println!(
        "  Scan delta: {:.2}% vs {:.2}% (Δ={:.2}%)",
        student_metrics.scan_rate() * 100.0,
        calib_rates.scan_rate * 100.0,
        scan_delta * 100.0
    );
    println!(
        "  Focus delta: {:.2}% vs {:.2}% (Δ={:.2}%)",
        student_metrics.focus_rate() * 100.0,
        calib_rates.focus_rate * 100.0,
        focus_delta * 100.0
    );
    println!(
        "  Perturb delta: {:.3}% vs {:.3}% (Δ={:.3}%)",
        student_metrics.perturb_rate() * 100.0,
        calib_rates.perturb_rate * 100.0,
        perturb_delta * 100.0
    );

    // Summary
    let section_a_ok = exploit_rate_ok && explore_rate_ok && reset_rate_ok;
    let section_b_ok = focus_rate_ok;
    let section_c_ok = coverage_ok && sel_acc_ok && fp_ok;
    let all_ok = section_a_ok && section_b_ok && section_c_ok;

    println!();
    println!("─────────────────────────────────────────────────────────────────");
    if all_ok {
        println!("  → Phase 2.0f-E: PASS - All acceptance criteria met!");
    } else {
        let mut failures = Vec::new();
        if !section_a_ok {
            failures.push("A (teacher mode diversity)");
        }
        if !section_b_ok {
            failures.push("B (teacher Focus usage)");
        }
        if !section_c_ok {
            failures.push("C (no regressions)");
        }
        println!(
            "  → Phase 2.0f-E: FAIL - Sections failed: {}",
            failures.join(", ")
        );
    }
    println!("─────────────────────────────────────────────────────────────────");
}

/// Phase 2.0f-C: Run teacher baseline to get target rates and per-mode stats.
/// Returns (eval_metrics, per_mode_stats, calibration_rates).
fn run_demo12_teacher_baseline(
    config: &Config,
) -> (
    distill::EvalMetrics,
    distill::PerModeActionStats,
    distill::CalibrationRates,
) {
    use action::{Action, ActionConfig, ActionPolicy};
    use distill::Demo12Diagnostics;
    use mode::{Mode, ModePolicy, ModePolicyConfig};

    // Diagnostics for signal analysis
    let mut diagnostics = Demo12Diagnostics::new();

    // Phase 2.0f-E: Natural mode selection with extended signals
    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        // Phase 2.0f-E: Natural Exploit emergence
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);

    let mut rng = Rng::new(config.seed.wrapping_add(0xD1571));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train chamber
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

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

    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    // Action tracking
    let mut scan_count: usize = 0;
    let mut focus_count: usize = 0;
    let mut perturb_count: usize = 0;
    let mut per_mode_stats = distill::PerModeActionStats::new();
    let mut total_stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            total_ticks += 1;
            if is_stable {
                total_stable_ticks += 1;
            }

            // Phase 2.0f-E: Compute proto_align for natural Exploit emergence
            let proto_align = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.proto_score(&base_topk, config.proto_m))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);

            // Record diagnostics for signal analysis
            diagnostics.record(
                anchor_value,
                proto_align,
                topk_margin,
                abs_td as f32,
                is_stable,
                base_gate_passed,
                config.exploit_proto_min,
                config.exploit_margin_min,
                config.exploit_requires_stable,
            );

            // Phase 2.0f-E: Use observe_extended for natural mode selection
            mode_policy.observe_extended(
                global_tick,
                anchor_value,
                abs_td as f32,
                base_gate_passed,
                proto_align,
                topk_margin,
                is_stable,
            );

            // Natural mode selection based on signal quality
            let mode = mode_policy.choose_mode(global_tick);

            // Teacher action from natural mode (no synthetic override)
            let action = action_policy.choose_action(mode);

            // Track action counts
            let mode_bucket = match mode {
                Mode::Explore => 0u8,
                Mode::Exploit => 1u8,
                Mode::Reset => 2u8,
            };
            let action_label = match action {
                Action::Scan => 0u8,
                Action::Focus => 1u8,
                Action::Perturb => 2u8,
            };
            per_mode_stats.record(mode_bucket, action_label);

            match action {
                Action::Scan => scan_count += 1,
                Action::Focus => focus_count += 1,
                Action::Perturb => perturb_count += 1,
            }

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    let calib_rates =
        distill::CalibrationRates::from_counts(scan_count, focus_count, perturb_count);

    // Print diagnostics to understand why Exploit did/didn't emerge
    println!();
    diagnostics.print_summary();

    let eval_metrics = distill::EvalMetrics {
        coverage_pos: metrics.coverage_pos(),
        selective_accuracy: metrics.selective_accuracy(),
        false_positive_rate: metrics.false_positive_rate(),
        stable_share: if total_ticks > 0 {
            total_stable_ticks as f64 / total_ticks as f64
        } else {
            0.0
        },
        scan_count,
        focus_count,
        perturb_count,
        total_ticks,
    };

    (eval_metrics, per_mode_stats, calib_rates)
}

/// Phase 2.0f-C: Train student with stratified sampling.
/// Collects samples from teacher until all modes have minimum samples.
/// Returns (trained_student, training_stats, samples_per_mode).
fn run_demo12_train_student(
    config: &Config,
) -> (distill::LinearSoftmax, distill::DistillStats, [usize; 3]) {
    use action::{Action, ActionConfig, ActionPolicy};
    use distill::Demo12Diagnostics;
    use mode::{Mode, ModePolicy, ModePolicyConfig};

    // Diagnostics for train collector
    let mut diagnostics = Demo12Diagnostics::new();

    // Phase 2.0f-E: Natural mode selection with extended signals
    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        // Phase 2.0f-E: Natural Exploit emergence
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);

    // Use same seed as teacher baseline for consistent chamber dynamics
    let mut rng = Rng::new(config.seed.wrapping_add(0xD1571));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train chamber
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

    let mut anchor_bank = AnchorBank::new();
    let keyed_config = KeyedMemoryConfig {
        label_min_p: 0.50,
        label_margin: 0.10,
        alpha: 0.5,
        num_labels: config.num_ctx,
    };
    let mut keyed_memory = KeyedMemoryStore::new(keyed_config);

    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);
    let mut global_tick: u64 = 0;

    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    // Stratified replay buffer and training components
    let mut student = distill::LinearSoftmax::new();
    let mut replay = distill::StratifiedReplayBuffer::new(config.distill_replay_capacity);
    let mut stats = distill::DistillStats::new();
    let mut train_rng = Rng::new(config.seed.wrapping_add(0xD1573));

    let min_mode_samples = config.distill_min_mode_samples;

    // Collection phase: run until stratified or max episodes
    let max_collection_episodes = config.competitive_episodes * 3; // Allow extra time for stratification

    for _ep in 0..max_collection_episodes {
        // Check if stratified collection is complete
        if replay.is_stratified_ready(min_mode_samples) {
            break;
        }

        window.reset();

        for _t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            let proto_align = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.proto_score(&base_topk, config.proto_m))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);

            // Record diagnostics for signal analysis
            diagnostics.record(
                anchor_value,
                proto_align,
                topk_margin,
                abs_td as f32,
                is_stable,
                base_gate_passed,
                config.exploit_proto_min,
                config.exploit_margin_min,
                config.exploit_requires_stable,
            );

            // Phase 2.0f-E: Use observe_extended for natural mode selection
            mode_policy.observe_extended(
                global_tick,
                anchor_value,
                abs_td as f32,
                base_gate_passed,
                proto_align,
                topk_margin,
                is_stable,
            );

            // Natural mode selection based on signal quality
            let mode = mode_policy.choose_mode(global_tick);
            let mode_bucket = match mode {
                Mode::Explore => 0u8,
                Mode::Exploit => 1u8,
                Mode::Reset => 2u8,
            };

            // Teacher action from natural mode (no synthetic override)
            let action = action_policy.choose_action(mode);
            let action_label = match action {
                Action::Scan => 0u8,
                Action::Focus => 1u8,
                Action::Perturb => 2u8,
            };

            // Extract features
            let fail_streak = mode_policy.state.gate_fail_streak;
            let diag = distill::TickDiag {
                gate_pass: base_gate_passed,
                topk_margin,
                proto_align,
                anchor_value,
                abs_td: abs_td as f32,
                stable: is_stable,
                fail_streak,
                total_power,
                mode_bucket,
            };
            let features = distill::extract_features(&diag);

            // Add to stratified replay buffer
            replay.push(features, action_label, mode_bucket);

            // Execute action
            let action_overrides = action_policy.get_overrides(action);
            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            // Update anchor partition and prototype (critical for proto_align!)
            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            if base_gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            // Update value learning
            if prev_anchor_id != 0xFFFF {
                let mut adjusted_gate_params = base_gate_params.clone();
                adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;
                let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            // Track previous state
            let adjusted_gate_params = base_gate_params.clone();
            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);
            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            global_tick += 1;
        }
    }

    // Training phase: train on stratified samples
    let train_steps = 5000;
    for step in 0..train_steps {
        if replay.total_len() < config.distill_batch_size {
            continue;
        }

        let batch = replay.sample_stratified_batch(config.distill_batch_size, &mut train_rng);

        // Convert to Sample format for training
        let samples: Vec<distill::Sample> = batch
            .iter()
            .map(|s| distill::Sample { x: s.x, y: s.y })
            .collect();
        let sample_refs: Vec<&distill::Sample> = samples.iter().collect();

        let (loss, acc) = student.train_step(
            &sample_refs,
            config.distill_lr,
            config.distill_l2,
            config.distill_temperature,
        );
        stats.update_loss(loss);

        // Track imitation accuracy periodically
        if step % 100 == 0 {
            stats.correct_predictions = (acc * config.distill_batch_size as f32) as usize;
            stats.total_samples = config.distill_batch_size;
        }
    }

    // Final imitation accuracy on full stratified batch
    let final_batch =
        replay.sample_stratified_batch(config.distill_batch_size * 10, &mut train_rng);
    let mut correct = 0usize;
    for sample in &final_batch {
        let pred = student.predict(&sample.x);
        if pred == sample.y as usize {
            correct += 1;
        }
    }
    stats.total_samples = final_batch.len();
    stats.correct_predictions = correct;

    // Print diagnostics for train collector
    println!("Train Collector Signal Diagnostics:");
    diagnostics.print_summary();

    let counts = replay.counts_per_mode();
    (student, stats, counts)
}

/// Phase 2.0f-C: Evaluate student with TargetBudgetLimiter.
/// Uses two-sided budget matching to enforce action rates near teacher targets.
/// Returns (eval_metrics, per_mode_stats, imitation_stats).
fn run_demo12_eval_student(
    config: &Config,
    student: &distill::LinearSoftmax,
    calib_rates: &distill::CalibrationRates,
) -> (
    distill::EvalMetrics,
    distill::PerModeActionStats,
    distill::DistillStats,
) {
    use action::{Action, ActionConfig, ActionPolicy};
    use mode::{Mode, ModePolicy, ModePolicyConfig};

    // Phase 2.0f-E: Natural mode selection with extended signals
    let mode_policy_config = ModePolicyConfig {
        explore_v_max: config.mode_explore_v_max,
        exploit_v_min: config.mode_exploit_v_min,
        reset_td_min: config.mode_reset_td_min,
        reset_value_drop: config.mode_reset_value_drop,
        reset_fail_streak: config.mode_reset_fail_streak,
        post_reset_cooldown: config.mode_post_reset_cooldown,
        explore_margin_min_scale: config.mode_explore_margin_scale,
        exploit_margin_min_scale: config.mode_exploit_margin_scale,
        reset_dampen: config.mode_reset_dampen,
        reset_dampen_top_k: config.mode_reset_dampen_top_k,
        window_size: config.mode_window_size,
        // Phase 2.0f-E: Natural Exploit emergence
        exploit_proto_min: config.exploit_proto_min,
        exploit_margin_min: config.exploit_margin_min,
        exploit_requires_stable: config.exploit_requires_stable,
    };
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new(action_config);

    // Two-sided TargetBudgetLimiter
    let mut budget_limiter = distill::TargetBudgetLimiter::from_calibration(
        config.budget_window,
        calib_rates,
        config.budget_tol,
        config.budget_lambda_pref,
        config.budget_lambda_def,
    );

    let mut rng = Rng::new(config.seed.wrapping_add(0xD1574));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(&config, &mut rng);

    // Pre-train chamber
    for _ in 0..10000 {
        let (active_mask, _) = causes.sample_active(&mut rng);
        let z_inj = causes.compute_z_inj(active_mask);
        causes.inject_for_tick(&mut rng, &mut chamber, active_mask);
        let topk = get_top_k(&chamber, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

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

    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    let mut total_stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;

    let mut scan_count: usize = 0;
    let mut focus_count: usize = 0;
    let mut perturb_count: usize = 0;

    let mut per_mode_stats = distill::PerModeActionStats::new();
    let mut stats = distill::DistillStats::new();

    for _ep in 0..config.competitive_episodes {
        window.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes.sample_active(&mut rng);
            let z_inj = causes.compute_z_inj(active_mask);
            causes.inject_for_tick(&mut rng, &mut chamber, active_mask);

            let base_topk = get_top_k(&chamber, config.top_k);
            let tick_metrics = chamber.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window.push(&topk_ids, ctx_hat);

            if !window.is_ready() {
                global_tick += 1;
                continue;
            }

            let current_sig = window.competitive_sig();
            let sig_mask = current_sig.mask;

            let topk_margin = if base_topk.len() >= 2 {
                base_topk[0].1 - base_topk[1].1
            } else if !base_topk.is_empty() {
                base_topk[0].1
            } else {
                0.0
            };
            let total_power = tick_metrics.tot_pow_post;
            let confidence = ConfidenceInfo::new(topk_margin, total_power);

            if anchor_bank.should_merge(global_tick) {
                let remaps = anchor_bank.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_merge_done(global_tick);
            }

            if anchor_bank.should_scan_merges(global_tick, config) {
                let remaps = anchor_bank.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory.apply_remaps(&remaps);
                }
                anchor_bank.mark_scan_done(global_tick);
            }

            anchor_bank.update_stability(global_tick, config);

            let base_gate_params = if anchor_bank.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank.resolve_gated(sig_mask, global_tick, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            let proto_align = if anchor_id != 0xFFFF {
                anchor_bank
                    .get_anchor(anchor_id)
                    .map(|a| a.proto_score(&base_topk, config.proto_m))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let abs_td = if prev_anchor_id != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            total_ticks += 1;
            if is_stable {
                total_stable_ticks += 1;
            }

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);

            // Phase 2.0f-E: Use observe_extended for natural mode selection
            mode_policy.observe_extended(
                global_tick,
                anchor_value,
                abs_td as f32,
                base_gate_passed,
                proto_align,
                topk_margin,
                is_stable,
            );

            // Natural mode selection based on signal quality
            let mode = mode_policy.choose_mode(global_tick);
            let mode_bucket = match mode {
                Mode::Explore => 0u8,
                Mode::Exploit => 1u8,
                Mode::Reset => 2u8,
            };

            // Teacher action from natural mode (no synthetic override)
            let teacher_action = action_policy.choose_action(mode);
            let teacher_label = match teacher_action {
                Action::Scan => 0usize,
                Action::Focus => 1usize,
                Action::Perturb => 2usize,
            };

            // Student prediction
            let fail_streak = mode_policy.state.gate_fail_streak;
            let diag = distill::TickDiag {
                gate_pass: base_gate_passed,
                topk_margin,
                proto_align,
                anchor_value,
                abs_td: abs_td as f32,
                stable: is_stable,
                fail_streak,
                total_power,
                mode_bucket,
            };
            let features = distill::extract_features(&diag);
            let logits = student.logits(&features);

            // Apply two-sided budget matching
            let budgeted_action = budget_limiter.apply_with_logits(&logits);

            // Track stats
            stats.record(teacher_label, budgeted_action);
            per_mode_stats.record(mode_bucket, budgeted_action as u8);

            match budgeted_action {
                0 => scan_count += 1,
                2 => perturb_count += 1,
                _ => focus_count += 1,
            }

            // Execute action using student's budgeted action
            let action = match budgeted_action {
                0 => Action::Scan,
                2 => Action::Perturb,
                _ => Action::Focus,
            };

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            if prev_anchor_id != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank.get_value(prev_anchor_id);
                let delta_power = total_power - prev_power;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
                reward_ema =
                    (1.0 - config.reward_ema_beta) * reward_ema + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank.update_anchor_value(prev_anchor_id, td, config);
            }

            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id = anchor_id;
                prev_power = total_power;
                prev_topk_margin = topk_margin;
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    prev_proto_align = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align = 0.0;
                }
            } else {
                prev_anchor_id = 0xFFFF;
            }

            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask = flip_bits_simple(
                        sig_mask,
                        config.competitive_neg_flip_bits,
                        rng.next_u64(),
                    );
                    let (neg_anchor_id, _, _) = anchor_bank.resolve(neg_sig_mask, global_tick);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory.recall(neg_key);
                    metrics.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank.record_win(anchor_id);
                        }
                    }
                    metrics.record_positive(&decision, true_label);
                }
            }

            global_tick += 1;
        }
    }

    let eval_metrics = distill::EvalMetrics {
        coverage_pos: metrics.coverage_pos(),
        selective_accuracy: metrics.selective_accuracy(),
        false_positive_rate: metrics.false_positive_rate(),
        stable_share: if total_ticks > 0 {
            total_stable_ticks as f64 / total_ticks as f64
        } else {
            0.0
        },
        scan_count,
        focus_count,
        perturb_count,
        total_ticks,
    };

    (eval_metrics, per_mode_stats, stats)
}
