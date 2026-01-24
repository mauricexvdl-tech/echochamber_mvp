# ECHO CHAMBER MVP — CLAUDE.md

This file is the working contract for contributors (human + LLM).
It documents architecture, invariants, how to run, and the roadmap.

---

## 0) What this project is

**Echo Chamber MVP** is a small Rust simulation that demonstrates:
- interference dynamics (constructive vs destructive)
- concept emergence via context-specialized winners
- memory via anchors + masks + consolidation/merging
- actionable readout: ModePolicy (Explore/Exploit/Reset) and ActionPolicy (Scan/Focus/Perturb)
- fair baselines + regret metrics to prove policy causality

Core idea: **We do not "teach" a classifier directly.**
We shape network dynamics and then measure readout/memory/policy behavior.

---

## 1) Core invariants (DO NOT BREAK)

### Physics invariants
- **EchoChamber dynamics must remain stable**: no large refactors or behavior changes that silently change dynamics.
- Any "policy" logic must operate via **small, bounded knobs only**:
  - gating thresholds / scaling
  - bounded noise injection (very small)
  - bounded local dampening (e.g. buffer *= factor in (0,1])

### Metrics invariants
Existing acceptance metrics must not regress unless explicitly changing a phase target:
- coverage_pos
- selective_accuracy
- false_positive == 0%
- stable_drop_ratio <= 0.5%
- merges_done_proto + avg_merge_score still printed
- stability / stable_mass metrics still printed (if present)

### Code hygiene invariants
- **No mega-refactors mid-phase.** Minimal diffs.
- Every new phase adds:
  - a config section
  - a demo (or integrates into an existing demo intentionally)
  - explicit acceptance checks with ✓/✗
- Keep deterministic seeds where possible.

---

## 2) Repository structure (mental model)

### Simulation layer
- `src/echo.rs`
  EchoChamber physics (buffers, delivery, coherence/destruction, etc.)

- `src/causes.rs`
  Cause / context injection schedule and evaluation contexts.

### Memory & consolidation layer
- `src/anchor.rs`
  Anchors + prototypes + values + stability + merging infrastructure.

- `src/memory.rs` (or keyed memory if present)
  Signature/memory storage + recall + competitive matching or O(1) keyed lookup.

### Policies (readout/control, not physics)
- `src/mode.rs`
  ModePolicy: Explore / Exploit / Reset based on diagnostics.

- `src/action.rs`
  ActionPolicy: Scan / Focus / Perturb (may be Mode-conditioned).
  **Phase 2.0c-FIX adds**: ActionTriggers (streak-based), PerturbFloor (rolling window rate enforcement).

- `src/action_ablate.rs`, `src/ablate.rs`, `src/regret.rs`
  Baselines + ablations + regret/recovery metrics.

- `src/distill.rs`
  Distillation infrastructure (teacher/student, stratified replay, budget matching).

### Orchestration
- `src/config.rs`
  One place for tunables; phases add config knobs here.

- `src/main.rs`, `src/demos.rs`
  Demo runners, printing, acceptance checks.

---

## 3) Demos overview (what each proves)

### Phase 1.x (emergence + memory)
- Demo 1: cancellation sanity check
- Demo 2: concept readout + episodic memory
- Demo 3: intra-episode one-shot binding
- Demo 4: competitive binding + abstain (Phase 1.4c)
- Demo 5/6: consolidation + stability + merges (Phase 1.9*)

### Phase 2.0x (actionable readout + fairness baselines)
- Demo 7: Mode policy loop (Explore/Exploit/Reset)
- Demo 8: mode ablations (NO_RESET / NO_EXPLORE etc.)
- Demo 9: Mode → Action loop (Scan/Focus/Perturb) + action usage constraints + **perturb triggers**
- Demo 10: action ablations + sensitivity sweeps
- Demo 11: trigger-matched random + regret/recovery metrics (proves causal action type selection)
- Demo 12: distillation (teacher → student) with fairness/budgeting + natural exploit viability

---

## 4) How to run

### Build
```bash
cd ~/workspace/echo_chamber_mvp
cargo fmt
cargo build --release
```

### Run all demos
```bash
cargo run --release
# or
./target/release/echo_chamber_mvp
```

### Run specific demos
Control via config flags in `src/config.rs`:
- `run_demo_7` through `run_demo_12` (Phase 2.0 experiments)

---

## 5) Key terminology

| Term | Meaning |
|------|---------|
| **Top-K** | The K highest-amplitude nodes after propagation |
| **Margin** | Gap between 1st and 2nd highest node |
| **Gate** | Confidence filter based on margin + power thresholds |
| **Anchor** | Memory address with prototype + value + stability |
| **Prototype** | Sparse vector (12-node Top-K averages) |
| **Mode** | Explore / Exploit / Reset (state machine) |
| **Action** | Scan / Focus / Perturb (executive control) |
| **TD** | Temporal difference error for value learning |
| **Trigger** | Streak-based condition for detecting "bad states" |
| **Floor** | Rolling window mechanism enforcing minimum perturb rate |

---

## 6) Config key groups

- **Dynamics**: `decay_per_tick`, `clamp_max_amp`, `pow_target`
- **Memory**: `proto_m`, `proto_eta`, `memory_max_entries`
- **Value**: `alpha_v`, `gamma_v`, reward weights
- **Mode**: `mode_explore_v_max`, `mode_exploit_v_min`, `mode_reset_td_min`
- **Action**: `scan_topk_scale`, `focus_margin_scale`
- **Triggers** (Phase 2.0c-FIX): `perturb_extra_triggers`, `perturb_floor_enabled`, `perturb_trig_*` params

---

## 7) Phase history

| Phase | What it added |
|-------|---------------|
| 1.9 | Aggressive anchor consolidation |
| 2.0a | Mode policy (Explore/Exploit/Reset) |
| 2.0c | Action policy (Scan/Focus/Perturb) |
| 2.0c-FIX | **Perturb trigger reliability + budget floor** — ActionTriggers, PerturbFloor, streak-based triggers |
| 2.0f-B | Policy distillation with budget fairness |
| 2.0f-C | Mode-conditioned distillation + two-sided budget matching |
| 2.0f-D | Forced mode schedule ensuring Exploit mode occurs with Focus action |
| 2.0f-E | Natural Exploit emergence via signal quality (stable + proto_align + margin) |

**Current**: Phase 2.0c-FIX (Perturb Trigger Reliability + Budget Floor)

---

## 8) Current acceptance status

All demos passing (as of Phase 2.0c-FIX):

| Demo | Status | Key Metrics |
|------|--------|-------------|
| Demo 9 | ✓ | perturb_rate: 1.10% (target: 0.5%-5%), coverage: 81.4%, sel_acc: 87.8%, FP: 0% |
| Demo 10 | ✓ | Action ablations passing |
| Demo 11 | ✓ | Regret/recovery metrics prove causal action selection |
| Demo 12 | ✓ | Distillation with budget fairness |

### Demo 9 Perturb Trigger Breakdown
- by_high_td: ~78%
- by_mode_reset: ~19%
- by_value_drop: ~3%
- Perturb effectiveness: +38.9% TD reduction

---

## 9) Important patterns

**Gating**: Confidence-based filtering using `topk_margin` and `total_power`. Gate params vary by mode (Explore loosens, Exploit tightens).

**Prototype Learning**: Anchors maintain sparse vectors via exponential averaging of Top-K node appearances.

**Value Learning**: TD(0) with reward = f(power_delta, coherence, proto_alignment). Values guide mode transitions.

**Merging**: Every 50 ticks, anchors with `proto_score > 0.78` are consolidated to prevent fragmentation.

**Perturb Triggers** (Phase 2.0c-FIX): Multiple streak-based conditions can fire Perturb independently of mode:
- `high_td`: |TD| exceeds threshold
- `gate_fail`: Consecutive gate failures
- `low_margin`: Weak winner separation
- `off_proto`: Poor prototype alignment
- `value_drop`: Declining anchor value
- `floor`: Rolling window rate below minimum

---

## 10) Rules for changes

1. **Minimal diffs** — don't refactor unrelated code
2. **Never remove demos** — only add or modify
3. **Add config knobs** — don't hardcode magic numbers
4. **Print acceptance checks** — every demo shows ✓/✗
5. **Keep deterministic** — use fixed seeds where possible
6. **No silent behavior changes** — document what changed and why
