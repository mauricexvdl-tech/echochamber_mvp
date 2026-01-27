//! Echo Chamber MVP: Emergent cancellation via complex signal interference.
//! Phase 2.2: CLI + JSON results export + config hashing.

mod ablate;
mod action;
mod action_ablate;
mod anchor;
mod causes;
mod cli;
mod complex;
mod concepts;
mod config;
mod demo13;
mod demos;
mod distill;
mod echo;
mod lift;
mod memory;
mod mode;
mod multiseed;
mod regret;
mod release;
mod results;
mod rng;
mod topology;

use cli::CliArgs;
use config::Config;
use demo13::Demo13Options;
use demos::{
    demo_10_action_ablations, demo_11_trigger_matched, demo_12_action_distillation,
    demo_7_mode_policy, demo_8_ablations, demo_9_action_loop, demo_competitive_binding,
    demo_label_binding, demo_latent_causes, demo_lie_triangle, demo_phase_1_5b_comparison,
    run_capacity_sweep,
};

fn main() {
    let args = CliArgs::parse();

    // Handle --help
    if args.help {
        CliArgs::print_help();
        return;
    }

    let config = Config::default();

    // Handle --release (release harness mode)
    if args.release {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║  ECHO CHAMBER MVP - Phase 2.2a: Release Harness               ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Quick-Mode: disabled (full runs only)");
        println!("Config hash: {}", config.config_hash());
        println!("Base seed: 0x{:08X}", config.seed);
        println!();

        let result = release::run_release_suite(&config);
        release::print_release_table(&result);

        let exit_code = if result.pass { 0 } else { 1 };
        println!();
        println!("Exit code: {}", exit_code);
        std::process::exit(exit_code);
    }

    // Print header
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  ECHO CHAMBER MVP - Phase 2.2: CLI + JSON Export              ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Config hash: {}", config.config_hash());
    println!("Base seed: 0x{:08X}", config.seed);
    println!();

    // Single demo mode
    if let Some(demo_num) = args.demo {
        run_single_demo(&config, &args, demo_num);
        return;
    }

    // Full suite mode (default)
    run_full_suite(&config);
}

/// Run a single demo by number.
fn run_single_demo(config: &Config, args: &CliArgs, demo_num: usize) {
    match demo_num {
        1 => demo_lie_triangle(config),
        2 => demo_latent_causes(config),
        3 => demo_label_binding(config),
        4 => demo_competitive_binding(config),
        5 | 6 => demo_phase_1_5b_comparison(config),
        7 => {
            if config.run_demo_7 {
                demo_7_mode_policy(config);
            } else {
                println!("Demo 7 disabled in config.");
            }
        }
        8 => {
            if config.run_demo_8 {
                demo_8_ablations(config);
            } else {
                println!("Demo 8 disabled in config.");
            }
        }
        9 => {
            if config.run_demo_9 {
                demo_9_action_loop(config);
            } else {
                println!("Demo 9 disabled in config.");
            }
        }
        10 => {
            if config.run_demo_10 {
                demo_10_action_ablations(config);
            } else {
                println!("Demo 10 disabled in config.");
            }
        }
        11 => {
            if config.run_demo_11 {
                demo_11_trigger_matched(config);
            } else {
                println!("Demo 11 disabled in config.");
            }
        }
        12 => {
            if config.run_demo_12 {
                demo_12_action_distillation(config);
            } else {
                println!("Demo 12 disabled in config.");
            }
        }
        13 => {
            if config.run_demo_13 {
                println!("Quick-Mode: disabled (full runs only)");
                let options = Demo13Options {
                    seeds: args.seeds.clone(),
                    quick: false, // Quick mode permanently disabled
                    out_path: args.out.clone(),
                };
                demo13::run_with_options(config, options);
            } else {
                println!("Demo 13 disabled in config.");
            }
        }
        _ => {
            println!("Unknown demo number: {}. Valid range: 1-13.", demo_num);
        }
    }
}

/// Run the full demo suite.
fn run_full_suite(config: &Config) {
    println!("Quick-Mode: disabled (full runs only)");
    println!();
    demo_lie_triangle(config);
    println!();
    demo_latent_causes(config);
    println!();
    demo_label_binding(config);
    println!();
    demo_competitive_binding(config);
    println!();
    demo_phase_1_5b_comparison(config);

    if config.run_demo_7 {
        println!();
        demo_7_mode_policy(config);
    }

    if config.run_demo_8 {
        println!();
        demo_8_ablations(config);
    }

    if config.run_demo_9 {
        println!();
        demo_9_action_loop(config);
    }

    if config.run_demo_10 {
        println!();
        demo_10_action_ablations(config);
    }

    if config.run_demo_11 {
        println!();
        demo_11_trigger_matched(config);
    }

    if config.run_demo_12 {
        println!();
        demo_12_action_distillation(config);
    }

    if config.run_demo_13 {
        println!();
        demo13::run(config);
    }

    if config.run_capacity_sweep {
        println!();
        run_capacity_sweep(config);
    }
}
