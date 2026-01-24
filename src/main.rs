//! Echo Chamber MVP: Emergent cancellation via complex signal interference.
//! Phase 1.9: CONSOLIDATION - Make merges happen + reduce stability flicker.

mod ablate;
mod action;
mod action_ablate;
mod anchor;
mod causes;
mod complex;
mod concepts;
mod config;
mod demos;
mod distill;
mod echo;
mod memory;
mod mode;
mod regret;
mod rng;

use config::Config;
use demos::{
    demo_10_action_ablations, demo_11_trigger_matched, demo_12_action_distillation,
    demo_7_mode_policy, demo_8_ablations, demo_9_action_loop, demo_competitive_binding,
    demo_label_binding, demo_latent_causes, demo_lie_triangle, demo_phase_1_5b_comparison,
    run_capacity_sweep,
};

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

    if config.run_demo_7 {
        println!();
        demo_7_mode_policy(&config);
    }

    if config.run_demo_8 {
        println!();
        demo_8_ablations(&config);
    }

    if config.run_demo_9 {
        println!();
        demo_9_action_loop(&config);
    }

    if config.run_demo_10 {
        println!();
        demo_10_action_ablations(&config);
    }

    if config.run_demo_11 {
        println!();
        demo_11_trigger_matched(&config);
    }

    if config.run_demo_12 {
        println!();
        demo_12_action_distillation(&config);
    }

    if config.run_capacity_sweep {
        println!();
        run_capacity_sweep(&config);
    }
}
