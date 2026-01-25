# CLAUDE.md — EchoChamber MVP (Project Contract)

This file is the **single source of truth** for how we build, test, and evolve the EchoChamber MVP.
If something conflicts with ad-hoc instructions or older demo notes, **CLAUDE.md wins**.

---

## 0) What this project is

**EchoChamber MVP** is a deterministic simulation + memory system with:
- **Echo physics** (complex signals + latent causes)
- **Anchors / prototypes / values** (memory substrate)
- **ModePolicy** (Explore / Exploit / Reset)
- **ActionPolicy** (Scan / Focus / Perturb)
- A demo-driven test suite (Demos 1–13) with acceptance gates

The goal is a **real-world usable policy loop** that:
1) avoids false positives (hard constraint),
2) shows causal advantage over fair baselines,
3) remains stable across seeds (robustness),
4) has escape/repair mechanisms for bad regimes (worst-seed floor).

---

## 1) Repo layout (current intent)

Recommended structure (you already started refactoring this way):

- `src/echo.rs` / `complex.rs` / `causes.rs`
  - Chamber physics: **DO NOT change** for policy experiments unless explicitly in a physics phase.
- `src/anchor.rs` / `memory.rs` / `concepts.rs`
  - Anchor lifecycle, merges, prototype updates, recall/binding.
- `src/mode.rs`
  - Mode selection, guardrails, chronic clamp, rescue logic.
- `src/action.rs`
  - Action selection (Scan/Focus/Perturb), triggers, perturb effectiveness tracking.
- `src/regret.rs`
  - Regret/recovery metrics used in fairness arguments.
- `src/distill.rs`
  - Teacher→student distillation infrastructure (Demo 12).
- `src/multiseed.rs` / `src/lift.rs` / `src/demo13.rs`
  - Multi-seed evaluation + policy advantage metrics (Demo 13).
- `src/demos/` + `src/legacy_demos/`
  - **Each demo gets its own file** when it grows. `demos.rs` should remain a thin dispatcher.

---

## 2) Hard constraints (never violate)

### 2.1 Safety / correctness invariants
- **False positive must remain 0%** (or within the explicitly defined tolerance, default: 0%).
- Determinism: seeded RNG where possible (especially Demo 13).
- No "unfair baselines": random comparisons must match **timing** and/or **budget** appropriately.

### 2.2 Scope rules
- **Do not change Echo physics / plasticity / merge logic** unless a phase explicitly says so.
- Prefer **small, isolated patches** (config + policy logic + metrics) over refactors.
- Demos must remain runnable; never delete demos. Move old stuff to `legacy_demos`.

---

## 3) What "done" means (Acceptance Gates)

### 3.1 Regression Guard (global)
Applies to any phase that touches policy/memory:
- `coverage_pos_mean >= 70%`
- `selective_accuracy_mean >= 80%`
- `false_positive_mean == 0%`
- Variability: `std < 15%` on core metrics (coverage/sel_acc)

### 3.2 Policy Advantage (causal proof)
FULL must beat fair baselines on **≥ 2/3 lift metrics**:
- `exploit_focus_lift` (Focus% in Exploit – Focus% in Explore) higher is better
- `recovery_after_perturb` higher is better
- `bad_state_share` lower is better

### 3.3 Worst-seed floor (robustness target)
This is the "real world doesn't care about your mean" gate:
- `worst_seed_coverage_pos >= 65%`
- `worst_seed_selective_accuracy >= 75%`
- No rescue spam (cap rescues per seed; typical gate: `<= 15` unless phase says otherwise)

> Note: Worst-seed is allowed to lag temporarily **only** if we are in an explicit tuning phase (2.1x) and we keep regression guard + advantage.

---

## 4) Phase roadmap (high-level)

### Phase 1.x — Foundation
- Echo cancellation + concept readout + episodic memory + competitive binding.
- Output: stable memory substrate + 0% FP.

### Phase 2.0a–2.0e — Policies become causal
- 2.0a: ModePolicy works (Explore/Exploit/Reset)
- 2.0b: Mode ablations (directional effects)
- 2.0c: Mode→Action loop online
- 2.0d: Action ablations + sensitivity sweep (fair budgets)
- 2.0e: Trigger-matched random + regret metrics (prove timing/type matters)

### Phase 2.0f — Distillation (Demo 12)
- Natural teacher viability (no synthetic teacher schedules)
- Student must learn mode-conditioned behavior (not just the action prior)
- Budget fairness must be two-sided or target-matched

### Phase 2.1 — Real-world robustness harness (Demo 13)
- Multi-seed eval
- Lift metrics
- Robustness targets (variability + worst-seed floor)
- "Quality repair" mechanisms allowed if they keep FP=0 and don't regress mean

---

## 5) Demo responsibilities (what each demo is FOR)

- Demo 9: **Mode→Action loop sanity** (regression smoke test)
  - Not where we optimize anymore; keep fast + stable.
- Demo 10: **Ablations + sensitivity curves** (prove knobs matter)
- Demo 11: **Causality proof** (trigger-matched random + regret/recovery)
- Demo 12: **Distillation** (teacher→student) + fairness
- Demo 13: **Product gate** (multi-seed + lift + worst-seed floor)

If a demo grows above ~400–600 lines: move it into `src/demos/demoXX.rs`.

---

## 6) Implementation strategy rules (to avoid "context disaster")

### 6.1 Small changes, measurable effect
Every patch should answer:
- Which metric is targeted?
- Which demo proves it?
- What is the expected directional change?

### 6.2 Baseline fairness
Any "random" baseline must specify which fairness it matches:
- Budget-matched (same rates, random timing)
- Trigger-matched (same timing, random action type)
- Target-matched limiter (two-sided matching)

No naive random baselines (they lie).

### 6.3 Instrumentation first
If a metric is failing (e.g., worst-seed floor), add a **minimal table**:
- per-seed: explore/exploit/reset, stable%, bad%, rescues, chronic%, burst triggers
Then tune.

---

## 7) How to run (developer workflow)

### Build
- `cargo fmt`
- `cargo build --release`

### Run full suite
- `cargo run --release`

### Extract specific demo output
- `cargo run --release | grep -A 250 "DEMO 13"`

### Multi-seed quick mode
Use the existing config knobs / quick mode if present, but ensure determinism.

---

## 8) "Real world" design principles

1) **Robustness > peak metrics**
   - Mean performance is not enough; worst-seed floor matters.

2) **Repair beats force**
   - When signal quality collapses (low stable%, high bad%), forcing Exploit/Focus tends to backfire.
   - Prefer targeted "quality repair" interventions (e.g., controlled perturb bursts) that improve stability.

3) **Learning must continue**
   - Never accidentally block prototype updates for long regimes (this caused regressions before).
   - If gating proto updates, do it with **quality gates** and clearly bounded rate limits.

---

## 9) Future direction: floats → bit logic (guideline, not immediate refactor)

We currently use floats for:
- complex signal arithmetic
- continuous thresholds (proto_align, margins, TD)

Bit-logic (or fixed-point / integer) may help:
- speed / cache behavior
- determinism across platforms
- eventual hardware targets

But do **not** prematurely convert the whole system.
Migration plan:
1) Identify the minimal "policy feature vector" subset that can become fixed-point.
2) Quantize features + thresholds with acceptance tests (Demo 13 must not regress).
3) Only then consider deeper signal quantization.

Rule: **No numeric representation changes without a demo gate proving equivalence or improvement.**

---

## 10) What to move into `legacy_demos/`

Safe candidates:
- old ablation variant lists that are not referenced by the current dispatcher
- replaced limiters/buffers that are not used in active demos
- older distillation scaffolding superseded by the current Demo 12 pipeline

Do **not** delete; move and mark with:
- `// LEGACY: kept for reference (Phase X.Y), not used by current demos`

---

## 11) Current priorities (what we do next)

1) Keep Demo 9–11 stable as regression gates.
2) Demo 12: ensure "natural teacher" + mode-conditioned student is real, not synthetic.
3) Demo 13: pass regression guard + advantage consistently.
4) Raise worst-seed floor using **quality repair** (bursts / targeted intervention) without FP regressions.
5) Only then: stable_count target (≥30) as a product hardening step.

---

## 12) Definition of "MVP complete"

MVP is complete when:
- Demo 13 passes:
  - regression guard
  - policy advantage
  - variability gate
  - worst-seed floor gate
  - FP=0
- And Demo 9–11 remain green as smoke/regression checks.

---

## 13) Communication contract (for Claude / assistants)

When proposing changes:
- Provide **exact config knob changes** and their intent.
- Provide **expected metric deltas** (directional).
- Provide **which demo proves it**.
- Avoid large refactors; prefer surgical patches.
- If tuning: propose a small sweep grid and stop when regression guard breaks.
