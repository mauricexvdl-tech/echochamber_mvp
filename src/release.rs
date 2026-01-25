//! Phase 2.2a: Release Harness
//!
//! A single entrypoint that runs MVP-critical demos (9, 11, 13),
//! prints a compact PASS/FAIL summary, and returns non-zero exit code on failure.

use crate::config::Config;
use crate::demo13::SeedDiagnostics;
use crate::lift::{self, LiftConfig};
use crate::multiseed::{self, SeedRun};
use serde::{Deserialize, Serialize};

/// Fixed seeds for deterministic release harness (matches Demo 13 spec).
pub const RELEASE_SEEDS: [u64; 5] = [
    0xDEADBEEF,
    0xEEADBEEF,
    0xFEADBEEF,
    0x10EADBEEF,
    0x11EADBEEF,
];

/// Per-demo row in the release table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseRow {
    pub demo_id: u32,
    pub pass: bool,
    pub coverage: Option<f64>,
    pub sel_acc: Option<f64>,
    pub fp: Option<f64>,
    pub notes: String,
}

/// Overall release result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseResult {
    pub rows: Vec<ReleaseRow>,
    pub pass: bool,
}

/// Demo 9 metrics returned by run_demo9_metrics.
#[derive(Clone, Debug)]
pub struct Demo9Metrics {
    pub coverage_pos: f64,
    pub selective_accuracy: f64,
    pub false_positive_rate: f64,
    pub perturb_rate: f64,
    pub stable_drop_ratio: f64,
    pub scan_rate: f64,
    pub focus_rate: f64,
    pub pass: bool,
}

/// Demo 11 metrics returned by run_demo11_metrics.
#[derive(Clone, Debug)]
pub struct Demo11Metrics {
    pub full_coverage_pos: f64,
    pub full_selective_accuracy: f64,
    pub full_false_positive_rate: f64,
    pub regret_wins: usize,
    pub trigger_match_ok: bool,
    pub pass: bool,
}

/// Demo 13 metrics returned by run_demo13_metrics.
#[derive(Clone, Debug)]
pub struct Demo13Metrics {
    pub coverage_pos_mean: f64,
    pub selective_accuracy_mean: f64,
    pub false_positive_mean: f64,
    pub worst_coverage: f64,
    pub worst_sel_acc: f64,
    pub lift_wins: usize,
    pub max_rescues: usize,
    pub perturb_rate_mean: f64,
    pub chronic_mean: f64,
    pub pass: bool,
}

/// Run the release suite: Demo 9, Demo 11, Demo 13.
/// Returns ReleaseResult with per-demo outcomes and overall pass/fail.
pub fn run_release_suite(config: &Config, quick: bool) -> ReleaseResult {
    let mut rows = Vec::new();

    // ==========================================================================
    // Demo 9: Mode→Action loop sanity
    // ==========================================================================
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RELEASE HARNESS: Running Demo 9 (Mode→Action loop)...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let demo9 = run_demo9_metrics(config);
    let demo9_row = ReleaseRow {
        demo_id: 9,
        pass: demo9.pass,
        coverage: Some(demo9.coverage_pos),
        sel_acc: Some(demo9.selective_accuracy),
        fp: Some(demo9.false_positive_rate),
        notes: format!("perturb={:.2}%", demo9.perturb_rate * 100.0),
    };
    rows.push(demo9_row);

    // ==========================================================================
    // Demo 11: Trigger-matched random + regret metrics
    // ==========================================================================
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RELEASE HARNESS: Running Demo 11 (Causality proof)...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let demo11 = run_demo11_metrics(config);
    let demo11_row = ReleaseRow {
        demo_id: 11,
        pass: demo11.pass,
        coverage: Some(demo11.full_coverage_pos),
        sel_acc: Some(demo11.full_selective_accuracy),
        fp: Some(demo11.full_false_positive_rate),
        notes: format!(
            "regret: {} wins",
            if demo11.regret_wins >= 2 {
                "FULL better"
            } else {
                "needs work"
            }
        ),
    };
    rows.push(demo11_row);

    // ==========================================================================
    // Demo 13: Multi-seed evaluation + lift metrics
    // ==========================================================================
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RELEASE HARNESS: Running Demo 13 (Multi-seed + lift)...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let demo13 = run_demo13_metrics(config, quick);
    let mut notes_parts = Vec::new();
    if demo13.coverage_pos_mean < 0.70 {
        notes_parts.push("cov_mean<70%".to_string());
    }
    if demo13.worst_coverage < 0.65 {
        notes_parts.push("worst_cov<65%".to_string());
    }
    if demo13.lift_wins < 2 {
        notes_parts.push(format!("lift={}/3", demo13.lift_wins));
    }
    let notes = if notes_parts.is_empty() {
        format!("lift={}/3", demo13.lift_wins)
    } else {
        notes_parts.join(", ")
    };

    let demo13_row = ReleaseRow {
        demo_id: 13,
        pass: demo13.pass,
        coverage: Some(demo13.coverage_pos_mean),
        sel_acc: Some(demo13.selective_accuracy_mean),
        fp: Some(demo13.false_positive_mean),
        notes,
    };
    rows.push(demo13_row);

    // ==========================================================================
    // Overall pass/fail
    // ==========================================================================
    let overall_pass = rows.iter().all(|r| r.pass);

    ReleaseResult {
        rows,
        pass: overall_pass,
    }
}

/// Print the release table.
pub fn print_release_table(result: &ReleaseResult) {
    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              RELEASE SUITE (Phase 2.2a)                        ║");
    println!("╠══════╦══════╦══════════╦══════════╦══════╦═════════════════════╣");
    println!("║ Demo ║ Pass ║ Coverage ║  SelAcc  ║  FP  ║ Notes               ║");
    println!("╠══════╬══════╬══════════╬══════════╬══════╬═════════════════════╣");

    for row in &result.rows {
        let pass_str = if row.pass { "✓" } else { "✗" };
        let cov_str = row
            .coverage
            .map(|c| format!("{:5.1}%", c * 100.0))
            .unwrap_or_else(|| "  N/A ".to_string());
        let sel_str = row
            .sel_acc
            .map(|s| format!("{:5.1}%", s * 100.0))
            .unwrap_or_else(|| "  N/A ".to_string());
        let fp_str = row
            .fp
            .map(|f| format!("{:3.1}%", f * 100.0))
            .unwrap_or_else(|| "N/A".to_string());

        // Truncate notes to 19 chars
        let notes = if row.notes.len() > 19 {
            format!("{}...", &row.notes[..16])
        } else {
            row.notes.clone()
        };

        println!(
            "║  {:2}  ║  {}   ║ {:>8} ║ {:>8} ║{:>5} ║ {:19} ║",
            row.demo_id, pass_str, cov_str, sel_str, fp_str, notes
        );
    }

    println!("╚══════╩══════╩══════════╩══════════╩══════╩═════════════════════╝");
    println!();

    if result.pass {
        println!("Overall: PASS");
    } else {
        println!("Overall: FAIL");
        let failed: Vec<_> = result
            .rows
            .iter()
            .filter(|r| !r.pass)
            .map(|r| format!("Demo {}", r.demo_id))
            .collect();
        println!("Failed: {}", failed.join(", "));
    }
}

// =============================================================================
// Demo 9 Metrics Runner
// =============================================================================

fn run_demo9_metrics(config: &Config) -> Demo9Metrics {
    use crate::action::{ActionConfig, ActionPolicy};
    use crate::anchor::{
        AnchorBank, ConfidenceInfo, GateParams, KeyedMemoryConfig, KeyedMemoryMetrics,
        KeyedMemoryStore, KeyedRecallDecision, MemoryKey, ANCHOR_MARGIN_MIN,
    };
    use crate::causes::{get_top_k, Causes};
    use crate::echo::EchoChamber;
    use crate::memory::RollingWindow;
    use crate::mode::{ModePolicy, ModePolicyConfig};
    use crate::rng::Rng;

    if !config.enable_mode_policy || !config.enable_action_policy {
        return Demo9Metrics {
            coverage_pos: 0.0,
            selective_accuracy: 0.0,
            false_positive_rate: 0.0,
            perturb_rate: 0.0,
            stable_drop_ratio: 0.0,
            scan_rate: 0.0,
            focus_rate: 0.0,
            pass: false,
        };
    }

    let mode_policy_config = ModePolicyConfig::from_config(config);
    let mut mode_policy = ModePolicy::new(mode_policy_config);

    let action_config = ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    };
    let mut action_policy = ActionPolicy::new_with_floor(
        action_config,
        config.perturb_floor_window,
        config.perturb_floor_min_rate,
    );

    let mut rng = Rng::new(config.seed.wrapping_add(0x7A7A_7A7A));
    let mut chamber = EchoChamber::random_graph(config.clone(), &mut rng);
    let causes = Causes::new(config, &mut rng);

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

    let mut window = RollingWindow::new(config.num_nodes, config.num_ctx);
    let bind_ticks = config.competitive_bind_ticks();
    let mut global_tick: u64 = 0;

    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    print!("  Running {} episodes... ", config.competitive_episodes);

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

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);

            let proto_align = if anchor_id != 0xFFFF {
                if let Some(anchor) = anchor_bank.get_anchor(anchor_id) {
                    anchor.proto_score(&base_topk, config.proto_m)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);
            let mode = mode_policy.choose_mode(global_tick);

            let (action, trigger_reason) = action_policy.choose_action_with_triggers(
                mode,
                abs_td as f32,
                base_gate_passed,
                topk_margin as f32,
                proto_align,
                anchor_value,
                config,
            );

            action_policy.record_trigger(trigger_reason);
            action_policy.record_action_for_floor(action);

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
                    let neg_sig_mask =
                        flip_bits_simple(sig_mask, config.competitive_neg_flip_bits, rng.next_u64());
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

    let action_stats = &action_policy.stats;
    let stable_drop_ratio = anchor_bank.stable_drop_ratio();

    let coverage_pos = metrics.coverage_pos();
    let selective_accuracy = metrics.selective_accuracy();
    let false_positive_rate = metrics.false_positive_rate();
    let perturb_rate = action_stats.perturb_rate();
    let scan_rate = action_stats.scan_rate();
    let focus_rate = action_stats.focus_rate();

    // Acceptance checks
    let coverage_ok = coverage_pos >= 0.70;
    let selective_ok = selective_accuracy >= 0.80;
    let fp_ok = false_positive_rate == 0.0;
    let stable_drop_ok = stable_drop_ratio <= 0.005;
    let scan_rate_ok = scan_rate >= 0.05;
    let focus_rate_ok = focus_rate >= 0.50;
    let perturb_rate_ok = perturb_rate >= 0.005 && perturb_rate <= 0.05;

    let pass = coverage_ok
        && selective_ok
        && fp_ok
        && stable_drop_ok
        && scan_rate_ok
        && focus_rate_ok
        && perturb_rate_ok;

    println!(
        "  Results: cov={:.1}% sel={:.1}% FP={:.1}% perturb={:.2}%",
        coverage_pos * 100.0,
        selective_accuracy * 100.0,
        false_positive_rate * 100.0,
        perturb_rate * 100.0
    );

    Demo9Metrics {
        coverage_pos,
        selective_accuracy,
        false_positive_rate,
        perturb_rate,
        stable_drop_ratio,
        scan_rate,
        focus_rate,
        pass,
    }
}

// =============================================================================
// Demo 11 Metrics Runner
// =============================================================================

fn run_demo11_metrics(config: &Config) -> Demo11Metrics {
    use crate::action::{Action, ActionConfig, ActionPolicy};
    use crate::action_ablate::TriggerTrace;
    use crate::anchor::{
        AnchorBank, ConfidenceInfo, GateParams, KeyedMemoryConfig, KeyedMemoryMetrics,
        KeyedMemoryStore, KeyedRecallDecision, MemoryKey, ANCHOR_MARGIN_MIN,
    };
    use crate::causes::{get_top_k, Causes};
    use crate::echo::EchoChamber;
    use crate::memory::RollingWindow;
    use crate::mode::{ModePolicy, ModePolicyConfig};
    use crate::regret::{RegretConfig, RegretStats};
    use crate::rng::Rng;

    if !config.enable_mode_policy || !config.enable_action_policy {
        return Demo11Metrics {
            full_coverage_pos: 0.0,
            full_selective_accuracy: 0.0,
            full_false_positive_rate: 0.0,
            regret_wins: 0,
            trigger_match_ok: false,
            pass: false,
        };
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
    // Run FULL variant and capture trigger trace
    // =========================================================================
    print!("  Running FULL variant... ");

    let mode_policy_config = ModePolicyConfig::from_config(config);
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
    let causes = Causes::new(config, &mut rng);

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
    let mut keyed_memory = KeyedMemoryStore::new(keyed_config.clone());
    let mut metrics = KeyedMemoryMetrics::new();
    let mut regret_stats = RegretStats::new(&regret_config);

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

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);

            let mode = mode_policy.choose_mode(global_tick);
            let action = action_policy.choose_action(mode);

            // Record to trigger trace
            trigger_trace.record(action);

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Record regret stats
            regret_stats.observe_tick(
                &regret_config,
                global_tick,
                abs_td as f32,
                topk_margin,
                proto_align,
                anchor_value,
                gate_passed,
            );
            if action != Action::Focus {
                regret_stats.observe_action(global_tick, action);
            }

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
                    let neg_sig_mask =
                        flip_bits_simple(sig_mask, config.competitive_neg_flip_bits, rng.next_u64());
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

    let full_action_stats = &action_policy.stats;

    // Extract regret metrics directly from RegretStats
    let full_bad_state_share = regret_stats.bad_state_share();
    let full_td_spike_rate = regret_stats.td_spike_rate();
    let full_recovery_improve_mean = regret_stats.recovery_improve_mean();
    let full_regret_rate = regret_stats.regret_rate();

    let full_coverage = metrics.coverage_pos();
    let full_selective = metrics.selective_accuracy();
    let full_fp = metrics.false_positive_rate();
    let full_scan_rate = full_action_stats.scan_rate();
    let full_perturb_rate = full_action_stats.perturb_rate();

    println!("done. ({} triggers)", trigger_trace.trigger_count());

    // =========================================================================
    // Run RANDOM_TRIGGER_MATCHED variant
    // =========================================================================
    print!("  Running RANDOM_TRIGGER_MATCHED... ");

    // Reset everything for the matched run
    let mut mode_policy2 = ModePolicy::new(ModePolicyConfig::from_config(config));
    let mut action_policy2 = ActionPolicy::new(ActionConfig {
        scan_topk_scale: config.scan_topk_scale,
        focus_topk_scale: config.focus_topk_scale,
        scan_margin_scale: config.scan_margin_scale,
        focus_margin_scale: config.focus_margin_scale,
        perturb_noise_amp: config.perturb_noise_amp,
    });

    let mut rng2 = Rng::new(config.seed.wrapping_add(0x7A7A_7A7A));
    let mut chamber2 = EchoChamber::random_graph(config.clone(), &mut rng2);
    let causes2 = Causes::new(config, &mut rng2);

    // Pre-train (same)
    for _ in 0..10000 {
        let (active_mask, _) = causes2.sample_active(&mut rng2);
        let z_inj = causes2.compute_z_inj(active_mask);
        causes2.inject_for_tick(&mut rng2, &mut chamber2, active_mask);
        let topk = get_top_k(&chamber2, config.top_k);
        let topk_ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
        chamber2.tick_with_context_plasticity(z_inj, &topk_ids, true);
    }

    let mut anchor_bank2 = AnchorBank::new();
    let mut keyed_memory2 = KeyedMemoryStore::new(keyed_config.clone());
    let mut metrics2 = KeyedMemoryMetrics::new();
    let mut regret_stats2 = RegretStats::new(&regret_config);

    let mut window2 = RollingWindow::new(config.num_nodes, config.num_ctx);
    let mut global_tick2: u64 = 0;
    let mut tick_index: usize = 0;

    let mut prev_anchor_id2: u16 = 0xFFFF;
    let mut prev_power2: f64 = 0.0;
    let mut prev_topk_margin2: f64 = 0.0;
    let mut prev_proto_align2: f32 = 0.0;
    let mut reward_ema2: f32 = 0.0;

    let mut action_rng = Rng::new(config.seed.wrapping_add(0xABCD_1234));

    for _ep in 0..config.competitive_episodes {
        window2.reset();

        for t in 0..config.competitive_episode_ticks {
            let (active_mask, _) = causes2.sample_active(&mut rng2);
            let z_inj = causes2.compute_z_inj(active_mask);
            causes2.inject_for_tick(&mut rng2, &mut chamber2, active_mask);

            let base_topk = get_top_k(&chamber2, config.top_k);
            let tick_metrics = chamber2.tick_with_context_plasticity(z_inj, &[], false);

            let ctx_hat = tick_metrics.ctx.map(|c| c as u8);
            let topk_ids: Vec<usize> = base_topk.iter().map(|(id, _)| *id).collect();
            window2.push(&topk_ids, ctx_hat);

            if !window2.is_ready() {
                global_tick2 += 1;
                tick_index += 1;
                continue;
            }

            let current_sig = window2.competitive_sig();
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

            if anchor_bank2.should_merge(global_tick2) {
                let remaps = anchor_bank2.merge_similar(Some(config));
                if !remaps.is_empty() {
                    keyed_memory2.apply_remaps(&remaps);
                }
                anchor_bank2.mark_merge_done(global_tick2);
            }

            if anchor_bank2.should_scan_merges(global_tick2, config) {
                let remaps = anchor_bank2.scan_and_merge(config);
                if !remaps.is_empty() {
                    keyed_memory2.apply_remaps(&remaps);
                }
                anchor_bank2.mark_scan_done(global_tick2);
            }

            anchor_bank2.update_stability(global_tick2, config);

            let base_gate_params = if anchor_bank2.stable_mode {
                GateParams::stable(config)
            } else {
                GateParams::explore(config)
            };

            let (anchor_id, _is_new, _match_dist) =
                anchor_bank2.resolve_gated(sig_mask, global_tick2, Some(&confidence), Some(config));

            let anchor_value = if anchor_id != 0xFFFF {
                anchor_bank2.get_value(anchor_id)
            } else {
                0.0
            };

            let is_stable = if anchor_id != 0xFFFF {
                anchor_bank2
                    .get_anchor(anchor_id)
                    .map(|a| a.stable)
                    .unwrap_or(false)
            } else {
                false
            };

            let proto_align = if anchor_id != 0xFFFF {
                anchor_bank2
                    .get_anchor(anchor_id)
                    .map(|a| a.proto_score(&base_topk, config.proto_m))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let abs_td = if prev_anchor_id2 != 0xFFFF {
                let gate_passed = confidence.passes_gate_with_params(&base_gate_params);
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank2.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * base_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank2.get_value(prev_anchor_id2);
                let delta_power = total_power - prev_power2;
                let reward =
                    compute_reward(delta_power, prev_topk_margin2, prev_proto_align2, config);
                let td = reward + config.gamma_v * v_next - v_prev;
                td.abs()
            } else {
                0.0
            };

            let base_gate_passed = confidence.passes_gate_with_params(&base_gate_params);
            mode_policy2.observe(global_tick2, anchor_value, abs_td as f32, base_gate_passed);

            // TRIGGER-MATCHED: At trigger times from FULL, pick random action
            let action = if trigger_trace.should_act_at(tick_index) {
                if action_rng.next_f64() < 0.5 {
                    Action::Scan
                } else {
                    Action::Perturb
                }
            } else {
                Action::Focus
            };

            let action_overrides = action_policy2.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy2.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Record regret stats
            regret_stats2.observe_tick(
                &regret_config,
                global_tick2,
                abs_td as f32,
                topk_margin,
                proto_align,
                anchor_value,
                gate_passed,
            );
            if action != Action::Focus {
                regret_stats2.observe_action(global_tick2, action);
            }

            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber2.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng2);
            }

            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank2.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank2.update_anchor_proto(anchor_id, &base_topk, config);
            }

            if prev_anchor_id2 != 0xFFFF {
                let v_next = if gate_passed && anchor_id != 0xFFFF {
                    anchor_bank2.get_value(anchor_id)
                } else if topk_margin < ANCHOR_MARGIN_MIN * adjusted_gate_params.margin_mult {
                    config.v_abstain_margin
                } else {
                    0.0
                };
                let v_prev = anchor_bank2.get_value(prev_anchor_id2);
                let delta_power = total_power - prev_power2;
                let mut reward =
                    compute_reward(delta_power, prev_topk_margin2, prev_proto_align2, config);
                reward_ema2 =
                    (1.0 - config.reward_ema_beta) * reward_ema2 + config.reward_ema_beta * reward;
                if config.use_advantage_reward {
                    reward = reward - reward_ema2;
                }
                let td = reward + config.gamma_v * v_next - v_prev;
                anchor_bank2.update_anchor_value(prev_anchor_id2, td, config);
            }

            if gate_passed && anchor_id != 0xFFFF {
                prev_anchor_id2 = anchor_id;
                prev_power2 = total_power;
                prev_topk_margin2 = topk_margin;
                if let Some(anchor) = anchor_bank2.get_anchor(anchor_id) {
                    prev_proto_align2 = anchor.proto_score(&base_topk, config.proto_m);
                } else {
                    prev_proto_align2 = 0.0;
                }
            } else {
                prev_anchor_id2 = 0xFFFF;
            }

            let learned_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            let key = MemoryKey::new(anchor_id, learned_mask);

            if bind_ticks.contains(&t) {
                let label = current_sig.ctx_hat.unwrap_or(0) as u16;
                keyed_memory2.store(key, label);
            }

            if t >= config.competitive_recall_start && t % config.competitive_recall_stride == 0 {
                let is_negative = rng2.next_f64() < config.competitive_p_neg;

                if is_negative {
                    let neg_sig_mask =
                        flip_bits_simple(sig_mask, config.competitive_neg_flip_bits, rng2.next_u64());
                    let (neg_anchor_id, _, _) = anchor_bank2.resolve(neg_sig_mask, global_tick2);
                    let neg_key = MemoryKey::new(neg_anchor_id, learned_mask);
                    let decision = keyed_memory2.recall(neg_key);
                    metrics2.record_negative(&decision);
                } else {
                    let true_label = current_sig.ctx_hat.unwrap_or(255) as u16;
                    let decision = keyed_memory2.recall(key);
                    if let KeyedRecallDecision::Label(recalled_label, _) = &decision {
                        if *recalled_label == true_label {
                            anchor_bank2.record_win(anchor_id);
                        }
                    }
                    metrics2.record_positive(&decision, true_label);
                }
            }

            global_tick2 += 1;
            tick_index += 1;
        }
    }

    let trigger_action_stats = &action_policy2.stats;

    // Extract regret metrics directly from RegretStats
    let trigger_bad_state_share = regret_stats2.bad_state_share();
    let trigger_td_spike_rate = regret_stats2.td_spike_rate();
    let trigger_recovery_improve_mean = regret_stats2.recovery_improve_mean();
    let trigger_regret_rate = regret_stats2.regret_rate();

    println!("done.");

    // =========================================================================
    // Calculate acceptance
    // =========================================================================

    // Regret wins calculation
    let bad_state_drop = trigger_bad_state_share - full_bad_state_share;
    let bad_state_ok = bad_state_drop >= 0.02;

    let td_spike_improve = if trigger_td_spike_rate > 0.001 {
        (trigger_td_spike_rate - full_td_spike_rate) / trigger_td_spike_rate
    } else {
        0.0
    };
    let td_spike_ok = td_spike_improve >= 0.10;

    let recovery_diff = full_recovery_improve_mean - trigger_recovery_improve_mean;
    let recovery_ok =
        recovery_diff >= 0.10 || full_recovery_improve_mean >= trigger_recovery_improve_mean + 0.05;

    let regret_improve = if trigger_regret_rate > 0.001 {
        (trigger_regret_rate - full_regret_rate) / trigger_regret_rate
    } else {
        0.0
    };
    let regret_ok = regret_improve >= 0.10;

    let regret_wins = [bad_state_ok, td_spike_ok, recovery_ok, regret_ok]
        .iter()
        .filter(|&&x| x)
        .count();

    // Trigger match check
    let full_trigger_rate = (full_scan_rate + full_perturb_rate) * 100.0;
    let trigger_action_rate =
        (trigger_action_stats.scan_rate() + trigger_action_stats.perturb_rate()) * 100.0;
    let trigger_delta = (trigger_action_rate - full_trigger_rate).abs();
    let trigger_match_ok = trigger_delta < 1.0;

    // Regression guard
    let coverage_ok = full_coverage >= 0.70;
    let selective_ok = full_selective >= 0.80;
    let fp_ok = full_fp == 0.0;
    let directional_ok = regret_wins >= 2;

    let pass = trigger_match_ok && directional_ok && coverage_ok && selective_ok && fp_ok;

    println!(
        "  Results: cov={:.1}% sel={:.1}% FP={:.1}% regret_wins={}/4",
        full_coverage * 100.0,
        full_selective * 100.0,
        full_fp * 100.0,
        regret_wins
    );

    Demo11Metrics {
        full_coverage_pos: full_coverage,
        full_selective_accuracy: full_selective,
        full_false_positive_rate: full_fp,
        regret_wins,
        trigger_match_ok,
        pass,
    }
}

// =============================================================================
// Demo 13 Metrics Runner
// =============================================================================

fn run_demo13_metrics(config: &Config, quick: bool) -> Demo13Metrics {
    use crate::demo13;

    if !config.enable_mode_policy || !config.enable_action_policy {
        return Demo13Metrics {
            coverage_pos_mean: 0.0,
            selective_accuracy_mean: 0.0,
            false_positive_mean: 0.0,
            worst_coverage: 0.0,
            worst_sel_acc: 0.0,
            lift_wins: 0,
            max_rescues: 0,
            perturb_rate_mean: 0.0,
            chronic_mean: 0.0,
            pass: false,
        };
    }

    // Apply quick mode overrides
    let mut config = config.clone();
    if quick {
        config.competitive_episodes = 200;
        config.competitive_episode_ticks = 300;
    }

    let seeds: Vec<u64> = RELEASE_SEEDS.to_vec();
    let num_seeds = seeds.len();

    let lift_config = LiftConfig {
        bad_margin: config.lift_bad_margin,
        bad_proto: config.lift_bad_proto,
        bad_value: config.lift_bad_value,
        recovery_window: 10,
    };

    // Run FULL variant across seeds
    let mut full_runs: Vec<SeedRun> = Vec::new();
    let mut full_lifts: Vec<crate::lift::LiftStats> = Vec::new();
    let mut full_diagnostics: Vec<SeedDiagnostics> = Vec::new();

    for (i, &seed) in seeds.iter().enumerate() {
        print!("  Seed {}/{} (0x{:08X})... ", i + 1, num_seeds, seed);
        let (run, lift_stats, diag) = demo13::run_single_seed_full(&config, &lift_config, seed);
        println!(
            "cov={:.1}% sel={:.1}% rescues={}",
            run.coverage_pos * 100.0,
            run.selective_accuracy * 100.0,
            diag.rescue_count,
        );
        full_runs.push(run);
        full_lifts.push(lift_stats);
        full_diagnostics.push(diag);
    }

    let full_agg = multiseed::aggregate(&full_runs);
    let full_lift_agg = lift::aggregate_lift(&full_lifts);

    // Run RANDOM_BUDGETED for lift comparison
    let target_scan_rate = full_agg.scan_rate_mean;
    let target_perturb_rate = full_agg.perturb_rate_mean;

    let mut budgeted_lifts: Vec<crate::lift::LiftStats> = Vec::new();
    for &seed in &seeds {
        let (_, lift_stats) = demo13::run_single_seed_budgeted(
            &config,
            &lift_config,
            seed,
            target_scan_rate,
            target_perturb_rate,
        );
        budgeted_lifts.push(lift_stats);
    }
    let budgeted_lift_agg = lift::aggregate_lift(&budgeted_lifts);

    // Calculate metrics
    let (lift_wins, _) = lift::compare_lift(&full_lift_agg, &budgeted_lift_agg);

    let worst_coverage = full_runs
        .iter()
        .map(|r| r.coverage_pos)
        .fold(f64::INFINITY, f64::min);
    let worst_sel_acc = full_runs
        .iter()
        .map(|r| r.selective_accuracy)
        .fold(f64::INFINITY, f64::min);
    let max_rescues = full_diagnostics
        .iter()
        .map(|d| d.rescue_count)
        .max()
        .unwrap_or(0);
    let chronic_mean = full_diagnostics
        .iter()
        .map(|d| d.chronic_lock_share)
        .sum::<f64>()
        / full_diagnostics.len().max(1) as f64;

    // Acceptance checks
    let coverage_ok = full_agg.coverage_pos_mean >= 0.70;
    let selective_ok = full_agg.selective_accuracy_mean >= 0.80;
    let fp_ok = full_agg.false_positive_mean < 0.001;
    let policy_advantage_ok = lift_wins >= 2;
    let max_rescues_ok = max_rescues <= 15;
    let worst_coverage_ok = worst_coverage >= 0.65;
    let worst_sel_acc_ok = worst_sel_acc >= 0.75;
    let perturb_ok = full_agg.perturb_rate_mean <= 0.05;
    let chronic_ok = chronic_mean <= 0.50;

    let pass = coverage_ok
        && selective_ok
        && fp_ok
        && policy_advantage_ok
        && max_rescues_ok
        && worst_coverage_ok
        && worst_sel_acc_ok
        && perturb_ok
        && chronic_ok;

    println!(
        "  Aggregate: cov_mean={:.1}% sel_mean={:.1}% lift={}/3 worst_cov={:.1}%",
        full_agg.coverage_pos_mean * 100.0,
        full_agg.selective_accuracy_mean * 100.0,
        lift_wins,
        worst_coverage * 100.0
    );

    Demo13Metrics {
        coverage_pos_mean: full_agg.coverage_pos_mean,
        selective_accuracy_mean: full_agg.selective_accuracy_mean,
        false_positive_mean: full_agg.false_positive_mean,
        worst_coverage,
        worst_sel_acc,
        lift_wins,
        max_rescues,
        perturb_rate_mean: full_agg.perturb_rate_mean,
        chronic_mean,
        pass,
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute reward for value learning.
fn compute_reward(delta_power: f64, margin: f64, proto_align: f32, config: &Config) -> f32 {
    let power_term = (delta_power as f32).clamp(-config.r_p_clip, config.r_p_clip);
    let margin_term = (margin as f32 / config.margin_norm).clamp(0.0, 1.0);
    let proto_term = proto_align.clamp(0.0, 1.0);

    config.r_w_power * power_term
        + config.r_w_margin * margin_term
        + config.r_w_proto * proto_term
}

/// Flip bits for negative query generation.
fn flip_bits_simple(mask: u64, n_flip: u32, rand: u64) -> u64 {
    let mut result = mask;
    let mut bits_to_flip = n_flip;
    let mut rand_state = rand;

    while bits_to_flip > 0 {
        let bit_pos = (rand_state % 64) as u32;
        result ^= 1u64 << bit_pos;
        rand_state = rand_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        bits_to_flip -= 1;
    }

    result
}
