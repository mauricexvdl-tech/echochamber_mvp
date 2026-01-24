//! Phase 2.2: Lightweight CLI argument parsing.
//!
//! Uses only std::env for minimal dependencies.

use std::env;

/// Parsed command-line arguments.
#[derive(Clone, Debug)]
pub struct CliArgs {
    /// Run only this demo (1-13), or None for full suite.
    pub demo: Option<usize>,
    /// Seeds for demo 13 multi-seed evaluation.
    pub seeds: Option<Vec<u64>>,
    /// Output path for JSON results.
    pub out: Option<String>,
    /// Config override file path (reserved for future use).
    pub config_path: Option<String>,
    /// Quick mode: reduced tick budgets for CI smoke tests.
    pub quick: bool,
    /// Show help and exit.
    pub help: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            demo: None,
            seeds: None,
            out: None,
            config_path: None,
            quick: false,
            help: false,
        }
    }
}

impl CliArgs {
    /// Parse command-line arguments.
    pub fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        Self::parse_from(&args[1..])
    }

    /// Parse from a slice of argument strings (for testing).
    pub fn parse_from(args: &[String]) -> Self {
        let mut result = Self::default();
        let mut i = 0;

        while i < args.len() {
            let arg = &args[i];

            match arg.as_str() {
                "--demo" => {
                    if i + 1 < args.len() {
                        if let Ok(n) = args[i + 1].parse::<usize>() {
                            result.demo = Some(n);
                        }
                        i += 1;
                    }
                }
                "--seeds" => {
                    if i + 1 < args.len() {
                        let seeds_str = &args[i + 1];
                        let seeds: Vec<u64> = seeds_str
                            .split(',')
                            .filter_map(|s| s.trim().parse::<u64>().ok())
                            .collect();
                        if !seeds.is_empty() {
                            result.seeds = Some(seeds);
                        }
                        i += 1;
                    }
                }
                "--out" => {
                    if i + 1 < args.len() {
                        result.out = Some(args[i + 1].clone());
                        i += 1;
                    }
                }
                "--config" => {
                    if i + 1 < args.len() {
                        result.config_path = Some(args[i + 1].clone());
                        i += 1;
                    }
                }
                "--quick" => {
                    result.quick = true;
                }
                "--help" | "-h" => {
                    result.help = true;
                }
                _ => {
                    // Unknown argument, ignore
                }
            }
            i += 1;
        }

        result
    }

    /// Print usage help.
    pub fn print_help() {
        println!("Echo Chamber MVP - Phase 2.2");
        println!();
        println!("USAGE:");
        println!("    cargo run --release [-- OPTIONS]");
        println!();
        println!("OPTIONS:");
        println!("    --demo <N>        Run only demo N (1-13)");
        println!("    --seeds <list>    Seeds for demo 13 (e.g. \"1,2,3,4,5\")");
        println!("    --out <path>      Write JSON results to path");
        println!("    --config <path>   Load config overrides (reserved)");
        println!("    --quick           Quick mode: reduced ticks for CI");
        println!("    --help, -h        Show this help");
        println!();
        println!("EXAMPLES:");
        println!("    cargo run --release");
        println!("        Run full demo suite (1-13)");
        println!();
        println!("    cargo run --release -- --demo 13 --seeds 1,2,3 --quick --out results/demo13_quick.json");
        println!("        Quick CI smoke test for demo 13");
        println!();
        println!("    cargo run --release -- --demo 9 --out results/demo9.json");
        println!("        Run demo 9 with JSON output");
    }
}
