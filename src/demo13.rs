//! Demo 13: Phase 2.1 - Multi-seed evaluation + lift metrics
//!
//! Runs experiments across multiple seeds with aggregated reporting.
//! Compares FULL policy against RANDOM_BUDGETED baseline.

use crate::action::{Action, ActionConfig, ActionPolicy};
use crate::anchor::{
    AnchorBank, ConfidenceInfo, GateParams, KeyedMemoryConfig, KeyedMemoryMetrics,
    KeyedMemoryStore, KeyedRecallDecision, MemoryKey, ANCHOR_MARGIN_MIN,
};
use crate::causes::{get_top_k, Causes};
use crate::config::Config;
use crate::echo::EchoChamber;
use crate::lift::{self, LiftConfig, LiftStats};
use crate::memory::RollingWindow;
use crate::mode::{Mode, ModePolicy, ModePolicyConfig};
use crate::multiseed::{self, SeedRun};
use crate::rng::Rng;

/// Run Demo 13: Multi-seed evaluation with lift metrics.
pub fn run(config: &Config) {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEMO 13: Phase 2.1 - MULTI-SEED EVALUATION + LIFT METRICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if !config.enable_mode_policy || !config.enable_action_policy {
        println!("Mode or action policy disabled. Skipping Demo 13.");
        return;
    }

    let num_seeds = config.demo13_num_seeds;
    println!("Configuration:");
    println!("  num_seeds: {}", num_seeds);
    println!("  episodes_per_seed: {}", config.competitive_episodes);
    println!("  ticks_per_episode: {}", config.competitive_episode_ticks);
    println!();

    let lift_config = LiftConfig {
        bad_margin: config.lift_bad_margin,
        bad_proto: config.lift_bad_proto,
        bad_value: config.lift_bad_value,
        recovery_window: 10,
    };

    // Generate seeds
    let base_seed = config.seed;
    let seeds: Vec<u64> = (0..num_seeds)
        .map(|i| base_seed.wrapping_add(0x1000_0000 * i as u64))
        .collect();

    // ==========================================================================
    // Run FULL variant across all seeds
    // ==========================================================================
    println!("Running FULL variant across {} seeds...", num_seeds);
    let mut full_runs: Vec<SeedRun> = Vec::new();
    let mut full_lifts: Vec<LiftStats> = Vec::new();

    for (i, &seed) in seeds.iter().enumerate() {
        print!("  Seed {}/{} (0x{:08X})... ", i + 1, num_seeds, seed);
        let (run, lift_stats) = run_single_seed_full(config, &lift_config, seed);
        println!(
            "cov={:.1}% sel={:.1}% FP={:.1}%",
            run.coverage_pos * 100.0,
            run.selective_accuracy * 100.0,
            run.false_positive * 100.0
        );
        full_runs.push(run);
        full_lifts.push(lift_stats);
    }

    let full_agg = multiseed::aggregate(&full_runs);
    let full_lift_agg = lift::aggregate_lift(&full_lifts);

    // ==========================================================================
    // Run RANDOM_BUDGETED variant across all seeds
    // ==========================================================================
    println!();
    println!("Running RANDOM_BUDGETED variant across {} seeds...", num_seeds);

    // Use average action rates from FULL as budget targets
    let target_scan_rate = full_agg.scan_rate_mean;
    let target_perturb_rate = full_agg.perturb_rate_mean;

    let mut budgeted_runs: Vec<SeedRun> = Vec::new();
    let mut budgeted_lifts: Vec<LiftStats> = Vec::new();

    for (i, &seed) in seeds.iter().enumerate() {
        print!("  Seed {}/{} (0x{:08X})... ", i + 1, num_seeds, seed);
        let (run, lift_stats) = run_single_seed_budgeted(
            config,
            &lift_config,
            seed,
            target_scan_rate,
            target_perturb_rate,
        );
        println!(
            "cov={:.1}% sel={:.1}% FP={:.1}%",
            run.coverage_pos * 100.0,
            run.selective_accuracy * 100.0,
            run.false_positive * 100.0
        );
        budgeted_runs.push(run);
        budgeted_lifts.push(lift_stats);
    }

    let budgeted_agg = multiseed::aggregate(&budgeted_runs);
    let budgeted_lift_agg = lift::aggregate_lift(&budgeted_lifts);

    // ==========================================================================
    // Print Results
    // ==========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.1 RESULTS: MULTI-SEED EVALUATION");
    println!("═══════════════════════════════════════════════════════════════════");

    // Per-seed tables
    println!();
    println!("FULL per-seed:");
    multiseed::print_per_seed_table(&full_runs);

    println!();
    println!("RANDOM_BUDGETED per-seed:");
    multiseed::print_per_seed_table(&budgeted_runs);

    // Aggregate tables
    println!();
    println!("───────────────────────────────────────────────────────────────────");
    println!("Aggregate Metrics (Mean ± Std):");
    println!("───────────────────────────────────────────────────────────────────");
    println!();
    multiseed::print_aggregate_table("FULL", &full_agg);
    println!();
    multiseed::print_aggregate_table("RANDOM_BUDGETED", &budgeted_agg);

    // Lift tables
    println!();
    println!("───────────────────────────────────────────────────────────────────");
    println!("Lift Metrics (Policy Advantage):");
    println!("───────────────────────────────────────────────────────────────────");
    println!();
    lift::print_lift_table("FULL", &full_lift_agg);
    println!();
    lift::print_lift_table("RANDOM_BUDGETED", &budgeted_lift_agg);

    // Lift comparison
    println!();
    println!("───────────────────────────────────────────────────────────────────");
    println!("Lift Comparison (FULL vs RANDOM_BUDGETED):");
    println!("───────────────────────────────────────────────────────────────────");

    let (lift_wins, comparisons) = lift::compare_lift(&full_lift_agg, &budgeted_lift_agg);
    println!();
    for (metric, full_wins) in &comparisons {
        let symbol = if *full_wins { "✓" } else { "✗" };
        let direction = if *metric == "bad_state_share" {
            "lower"
        } else {
            "higher"
        };
        println!(
            "  [{}] FULL {} on {}: {} is better",
            symbol,
            if *full_wins { "wins" } else { "loses" },
            metric,
            direction
        );
    }
    println!();
    println!("  FULL wins on {}/3 lift metrics", lift_wins);

    // ==========================================================================
    // Acceptance Criteria
    // ==========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PHASE 2.1 ACCEPTANCE:");
    println!("═══════════════════════════════════════════════════════════════════");

    // C1) Regression guard on FULL
    println!();
    println!("C1) Regression guard (FULL mean across seeds):");

    let coverage_ok = full_agg.coverage_pos_mean >= 0.70;
    let selective_ok = full_agg.selective_accuracy_mean >= 0.80;
    let fp_ok = full_agg.false_positive_mean < 0.001; // Use epsilon for floating-point comparison

    println!(
        "  [{}] coverage_pos_mean >= 70%: {:.1}%",
        if coverage_ok { "✓" } else { "✗" },
        full_agg.coverage_pos_mean * 100.0
    );
    println!(
        "  [{}] selective_accuracy_mean >= 80%: {:.1}%",
        if selective_ok { "✓" } else { "✗" },
        full_agg.selective_accuracy_mean * 100.0
    );
    println!(
        "  [{}] false_positive_mean == 0%: {:.1}%",
        if fp_ok { "✓" } else { "✗" },
        full_agg.false_positive_mean * 100.0
    );

    let regression_ok = coverage_ok && selective_ok && fp_ok;

    // C2) Policy advantage (FULL beats RANDOM_BUDGETED on >= 2 lift metrics)
    println!();
    println!("C2) Policy advantage (FULL vs RANDOM_BUDGETED):");

    let policy_advantage_ok = lift_wins >= 2;
    println!(
        "  [{}] FULL beats RANDOM_BUDGETED on >= 2 lift metrics: {}/3",
        if policy_advantage_ok { "✓" } else { "✗" },
        lift_wins
    );

    // C3) Variability reporting
    println!();
    println!("C3) Variability (std devs reported):");

    let low_variability = full_agg.coverage_pos_std < 0.15 && full_agg.selective_accuracy_std < 0.15;
    println!(
        "  [{}] coverage_pos_std < 15%: {:.1}%",
        if full_agg.coverage_pos_std < 0.15 {
            "✓"
        } else {
            "~"
        },
        full_agg.coverage_pos_std * 100.0
    );
    println!(
        "  [{}] selective_accuracy_std < 15%: {:.1}%",
        if full_agg.selective_accuracy_std < 0.15 {
            "✓"
        } else {
            "~"
        },
        full_agg.selective_accuracy_std * 100.0
    );

    if !full_agg.failed_seeds.is_empty() {
        println!();
        println!(
            "  ⚠ Seeds violating regression guard: {}",
            full_agg
                .failed_seeds
                .iter()
                .map(|s| format!("0x{:08X}", s))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Summary
    let all_ok = regression_ok && policy_advantage_ok;
    println!();
    if all_ok {
        println!("  → Phase 2.1: ALL ACCEPTANCE CRITERIA MET!");
        if low_variability {
            println!("  → Low variability across seeds - results are robust.");
        }
    } else {
        if regression_ok {
            println!("  → Phase 2.1: Regression guard OK. Policy advantage needs work.");
        } else if policy_advantage_ok {
            println!("  → Phase 2.1: Policy advantage OK. Regression guard failed.");
        } else {
            println!("  → Phase 2.1: Multiple criteria not met. Tuning needed.");
        }
    }
}

/// Run a single seed with FULL policy.
fn run_single_seed_full(
    config: &Config,
    lift_config: &LiftConfig,
    seed: u64,
) -> (SeedRun, LiftStats) {
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
    let mut action_policy = ActionPolicy::new_with_floor(
        action_config,
        config.perturb_floor_window,
        config.perturb_floor_min_rate,
    );

    let mut rng = Rng::new(seed.wrapping_add(0x7A7A_7A7A));
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

    // Value learning state
    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    // Mode/action counters
    let mut explore_count: usize = 0;
    let mut exploit_count: usize = 0;
    let mut reset_count: usize = 0;
    let mut stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;

    // Lift stats
    let mut lift_stats = LiftStats::new();

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

            total_ticks += 1;
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

            if is_stable {
                stable_ticks += 1;
            }

            // Compute TD
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
                let reward = compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
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

            // Mode policy
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);
            let mode = mode_policy.choose_mode(global_tick);

            match mode {
                Mode::Explore => explore_count += 1,
                Mode::Exploit => exploit_count += 1,
                Mode::Reset => reset_count += 1,
            }

            // Action policy with triggers
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

            // Apply perturb noise
            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            // Update lift stats
            lift_stats.observe_tick(
                lift_config,
                global_tick,
                mode,
                action,
                anchor_id,
                abs_td as f32,
                topk_margin,
                proto_align,
                anchor_value,
                gate_passed,
            );

            // Update partition info
            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            // Update prototype
            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            // Value update
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
                let mut reward = compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
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

    lift_stats.finalize(0.0);

    let action_stats = &action_policy.stats;
    let mode_total = explore_count + exploit_count + reset_count;

    let mut run = SeedRun::new(seed);
    run.coverage_pos = metrics.coverage_pos();
    run.selective_accuracy = metrics.selective_accuracy();
    run.false_positive = metrics.false_positive_rate();
    run.stable_share = if total_ticks > 0 {
        stable_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    run.explore_rate = if mode_total > 0 {
        explore_count as f64 / mode_total as f64
    } else {
        0.0
    };
    run.exploit_rate = if mode_total > 0 {
        exploit_count as f64 / mode_total as f64
    } else {
        0.0
    };
    run.reset_rate = if mode_total > 0 {
        reset_count as f64 / mode_total as f64
    } else {
        0.0
    };

    run.scan_rate = action_stats.scan_rate();
    run.focus_rate = action_stats.focus_rate();
    run.perturb_rate = action_stats.perturb_rate();

    run.bad_state_share = Some(lift_stats.bad_state_share());
    run.recovery_improve = Some(lift_stats.recovery_after_perturb());

    (run, lift_stats)
}

/// Run a single seed with RANDOM_BUDGETED policy.
fn run_single_seed_budgeted(
    config: &Config,
    lift_config: &LiftConfig,
    seed: u64,
    target_scan_rate: f64,
    target_perturb_rate: f64,
) -> (SeedRun, LiftStats) {
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

    let mut rng = Rng::new(seed.wrapping_add(0x7A7A_7A7A));
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

    // Value learning state
    let mut prev_anchor_id: u16 = 0xFFFF;
    let mut prev_power: f64 = 0.0;
    let mut prev_topk_margin: f64 = 0.0;
    let mut prev_proto_align: f32 = 0.0;
    let mut reward_ema: f32 = 0.0;

    // Mode/action counters
    let mut explore_count: usize = 0;
    let mut exploit_count: usize = 0;
    let mut reset_count: usize = 0;
    let mut stable_ticks: usize = 0;
    let mut total_ticks: usize = 0;

    // Lift stats
    let mut lift_stats = LiftStats::new();

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

            total_ticks += 1;
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

            if is_stable {
                stable_ticks += 1;
            }

            // Compute TD
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
                let reward = compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
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

            // Mode policy (still use real mode for tracking)
            mode_policy.observe(global_tick, anchor_value, abs_td as f32, base_gate_passed);
            let mode = mode_policy.choose_mode(global_tick);

            match mode {
                Mode::Explore => explore_count += 1,
                Mode::Exploit => exploit_count += 1,
                Mode::Reset => reset_count += 1,
            }

            // RANDOM_BUDGETED: choose action randomly with budget-matched rates
            let r = rng.next_f64();
            let action = if r < target_perturb_rate {
                Action::Perturb
            } else if r < target_perturb_rate + target_scan_rate {
                Action::Scan
            } else {
                Action::Focus
            };

            let action_overrides = action_policy.get_overrides(action);

            let mut adjusted_gate_params = base_gate_params.clone();
            adjusted_gate_params.margin_mult *= action_overrides.margin_scale as f64;

            let gate_passed = confidence.passes_gate_with_params(&adjusted_gate_params);

            action_policy.record_tick(action, gate_passed, abs_td as f32, anchor_value, is_stable);

            // Apply perturb noise
            if action_overrides.apply_noise && action_overrides.noise_amp > 0.0 {
                let noise_nodes: Vec<usize> = base_topk
                    .iter()
                    .take(config.mode_reset_dampen_top_k)
                    .map(|(id, _)| *id)
                    .collect();
                chamber.apply_noise(&noise_nodes, action_overrides.noise_amp, &mut rng);
            }

            // Update lift stats
            lift_stats.observe_tick(
                lift_config,
                global_tick,
                mode,
                action,
                anchor_id,
                abs_td as f32,
                topk_margin,
                proto_align,
                anchor_value,
                gate_passed,
            );

            // Update partition info
            let partition_mask = current_sig.ctx_hat.unwrap_or(0) as u64;
            anchor_bank.update_anchor_partition(anchor_id, partition_mask, ctx_hat);

            // Update prototype
            if gate_passed && anchor_id != 0xFFFF {
                anchor_bank.update_anchor_proto(anchor_id, &base_topk, config);
            }

            // Value update
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
                let mut reward = compute_reward(delta_power, prev_topk_margin, prev_proto_align, config);
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

    lift_stats.finalize(0.0);

    let action_stats = &action_policy.stats;
    let mode_total = explore_count + exploit_count + reset_count;

    let mut run = SeedRun::new(seed);
    run.coverage_pos = metrics.coverage_pos();
    run.selective_accuracy = metrics.selective_accuracy();
    run.false_positive = metrics.false_positive_rate();
    run.stable_share = if total_ticks > 0 {
        stable_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    run.explore_rate = if mode_total > 0 {
        explore_count as f64 / mode_total as f64
    } else {
        0.0
    };
    run.exploit_rate = if mode_total > 0 {
        exploit_count as f64 / mode_total as f64
    } else {
        0.0
    };
    run.reset_rate = if mode_total > 0 {
        reset_count as f64 / mode_total as f64
    } else {
        0.0
    };

    run.scan_rate = action_stats.scan_rate();
    run.focus_rate = action_stats.focus_rate();
    run.perturb_rate = action_stats.perturb_rate();

    run.bad_state_share = Some(lift_stats.bad_state_share());
    run.recovery_improve = Some(lift_stats.recovery_after_perturb());

    (run, lift_stats)
}

/// Compute reward for value learning.
fn compute_reward(delta_power: f64, prev_topk_margin: f64, prev_proto_align: f32, config: &Config) -> f32 {
    let r_power = (delta_power / config.r_p_clip as f64).clamp(-1.0, 1.0) as f32;
    let r_margin = (prev_topk_margin as f32 / config.margin_norm).clamp(0.0, 1.0);
    let r_proto = prev_proto_align.clamp(0.0, 1.0);

    config.r_w_power * r_power
        + config.r_w_margin * r_margin
        + config.r_w_proto * r_proto
}

/// Simple bit flip for negative queries.
fn flip_bits_simple(mask: u64, num_flips: u32, rand_val: u64) -> u64 {
    let mut result = mask;
    let mut r = rand_val;
    for _ in 0..num_flips {
        let bit_pos = (r % 64) as u8;
        result ^= 1u64 << bit_pos;
        r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    result
}
