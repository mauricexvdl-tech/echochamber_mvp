//! Phase 2.2: JSON Result Export
//!
//! Provides structs for machine-readable JSON output of demo results.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;

/// Metadata for a result file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResultMeta {
    /// Git commit hash (best-effort).
    pub git_commit: Option<String>,
    /// ISO timestamp when results were generated.
    pub timestamp: String,
    /// SHA256 hash of canonical config string.
    pub config_hash: String,
    /// Demo number.
    pub demo_id: usize,
    /// Seeds used (for multi-seed demos).
    pub seeds: Option<Vec<u64>>,
    /// Whether quick mode was enabled.
    pub quick_mode: bool,
}

impl ResultMeta {
    /// Create new metadata.
    pub fn new(demo_id: usize, config_hash: &str, seeds: Option<Vec<u64>>, quick_mode: bool) -> Self {
        Self {
            git_commit: get_git_commit(),
            timestamp: get_timestamp(),
            config_hash: config_hash.to_string(),
            demo_id,
            seeds,
            quick_mode,
        }
    }
}

/// Get current git commit hash (best-effort).
fn get_git_commit() -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

/// Get current ISO timestamp.
fn get_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Simple ISO-ish format without external deps
    format!("{}", secs)
}

/// Per-seed run result for Demo 13.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeedRunResult {
    pub seed: u64,
    pub coverage_pos: f64,
    pub selective_accuracy: f64,
    pub false_positive: f64,
    pub stable_share: f64,
    pub explore_rate: f64,
    pub exploit_rate: f64,
    pub reset_rate: f64,
    pub scan_rate: f64,
    pub focus_rate: f64,
    pub perturb_rate: f64,
    pub bad_state_share: Option<f64>,
    pub recovery_improve: Option<f64>,
}

/// Aggregate statistics (mean ± std).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregateResult {
    pub num_seeds: usize,
    pub coverage_pos_mean: f64,
    pub coverage_pos_std: f64,
    pub selective_accuracy_mean: f64,
    pub selective_accuracy_std: f64,
    pub false_positive_mean: f64,
    pub false_positive_std: f64,
    pub stable_share_mean: f64,
    pub stable_share_std: f64,
    pub scan_rate_mean: f64,
    pub scan_rate_std: f64,
    pub focus_rate_mean: f64,
    pub focus_rate_std: f64,
    pub perturb_rate_mean: f64,
    pub perturb_rate_std: f64,
    pub failed_seeds: Vec<u64>,
}

/// Lift metrics result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiftResult {
    pub exploit_focus_lift_mean: f64,
    pub exploit_focus_lift_std: f64,
    pub recovery_after_perturb_mean: f64,
    pub recovery_after_perturb_std: f64,
    pub bad_state_share_mean: f64,
    pub bad_state_share_std: f64,
    pub scan_diversity_mean: f64,
    pub scan_diversity_std: f64,
}

/// Variant result for Demo 13.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariantResult {
    pub name: String,
    pub runs: Vec<SeedRunResult>,
    pub aggregate: AggregateResult,
    pub lift: LiftResult,
}

/// Acceptance criteria results.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptanceResult {
    pub regression_guard_ok: bool,
    pub coverage_ok: bool,
    pub selective_accuracy_ok: bool,
    pub false_positive_ok: bool,
    pub policy_advantage_ok: bool,
    pub lift_wins: usize,
    pub all_pass: bool,
}

/// Demo 13 result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Demo13Result {
    pub meta: ResultMeta,
    pub full: VariantResult,
    pub random_budgeted: VariantResult,
    pub acceptance: AcceptanceResult,
}

/// Compact demo result for Demos 9/10/11/12.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactDemoResult {
    pub meta: ResultMeta,
    pub metrics: CompactMetrics,
    pub acceptance: CompactAcceptance,
}

/// Compact metrics for simpler demos.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactMetrics {
    pub coverage_pos: f64,
    pub selective_accuracy: f64,
    pub false_positive: f64,
    pub scan_rate: Option<f64>,
    pub focus_rate: Option<f64>,
    pub perturb_rate: Option<f64>,
    pub stable_share: Option<f64>,
}

/// Compact acceptance for simpler demos.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactAcceptance {
    pub all_pass: bool,
    pub details: Vec<AcceptanceCheck>,
}

/// Single acceptance check.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptanceCheck {
    pub name: String,
    pub passed: bool,
    pub value: String,
}

/// Write a result to JSON file.
pub fn write_json<T: Serialize>(result: &T, path: &str) -> Result<(), String> {
    // Create parent directories if needed
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }
    }

    let json = serde_json::to_string_pretty(result)
        .map_err(|e| format!("Failed to serialize JSON: {}", e))?;

    let mut file =
        fs::File::create(path).map_err(|e| format!("Failed to create file {}: {}", path, e))?;

    file.write_all(json.as_bytes())
        .map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(())
}

/// Convert multiseed::SeedRun to SeedRunResult.
impl From<&crate::multiseed::SeedRun> for SeedRunResult {
    fn from(run: &crate::multiseed::SeedRun) -> Self {
        Self {
            seed: run.seed,
            coverage_pos: run.coverage_pos,
            selective_accuracy: run.selective_accuracy,
            false_positive: run.false_positive,
            stable_share: run.stable_share,
            explore_rate: run.explore_rate,
            exploit_rate: run.exploit_rate,
            reset_rate: run.reset_rate,
            scan_rate: run.scan_rate,
            focus_rate: run.focus_rate,
            perturb_rate: run.perturb_rate,
            bad_state_share: run.bad_state_share,
            recovery_improve: run.recovery_improve,
        }
    }
}

/// Convert multiseed::Aggregate to AggregateResult.
impl From<&crate::multiseed::Aggregate> for AggregateResult {
    fn from(agg: &crate::multiseed::Aggregate) -> Self {
        Self {
            num_seeds: agg.num_seeds,
            coverage_pos_mean: agg.coverage_pos_mean,
            coverage_pos_std: agg.coverage_pos_std,
            selective_accuracy_mean: agg.selective_accuracy_mean,
            selective_accuracy_std: agg.selective_accuracy_std,
            false_positive_mean: agg.false_positive_mean,
            false_positive_std: agg.false_positive_std,
            stable_share_mean: agg.stable_share_mean,
            stable_share_std: agg.stable_share_std,
            scan_rate_mean: agg.scan_rate_mean,
            scan_rate_std: agg.scan_rate_std,
            focus_rate_mean: agg.focus_rate_mean,
            focus_rate_std: agg.focus_rate_std,
            perturb_rate_mean: agg.perturb_rate_mean,
            perturb_rate_std: agg.perturb_rate_std,
            failed_seeds: agg.failed_seeds.clone(),
        }
    }
}

/// Convert lift::LiftAggregate to LiftResult.
impl From<&crate::lift::LiftAggregate> for LiftResult {
    fn from(lift: &crate::lift::LiftAggregate) -> Self {
        Self {
            exploit_focus_lift_mean: lift.exploit_focus_lift_mean,
            exploit_focus_lift_std: lift.exploit_focus_lift_std,
            recovery_after_perturb_mean: lift.recovery_after_perturb_mean,
            recovery_after_perturb_std: lift.recovery_after_perturb_std,
            bad_state_share_mean: lift.bad_state_share_mean,
            bad_state_share_std: lift.bad_state_share_std,
            scan_diversity_mean: lift.scan_diversity_mean,
            scan_diversity_std: lift.scan_diversity_std,
        }
    }
}
