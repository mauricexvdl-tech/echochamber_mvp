# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo build --release          # Build optimized binary
cargo run --release            # Run all demos (Demo 1-12)
./target/release/echo_chamber_mvp  # Run compiled binary
```

No external dependencies - uses only Rust std library.

## Architecture Overview

**Echo Chamber MVP** demonstrates emergent signal cancellation through complex-valued interference. Signals with π phase difference naturally cancel when superposed (no explicit logic needed).

### Core Signal Flow

```
Causes (inject) → EchoChamber (propagate) → Top-K Sampling → Anchor/Memory → Mode/Action
```

1. **EchoChamber** (`echo.rs`): 32-node network with complex-valued buffers. Two-phase tick:
   - Distribute: split buffers across edges with phase rotations
   - Integrate: sum incoming signals (interference occurs here)

2. **AnchorBank** (`anchor.rs`): Memory addresses with:
   - Prototype vectors (sparse 12-node Top-K averages)
   - Values (TD(0) learned credit)
   - Stability hysteresis (enter/exit gates)
   - Aggressive merging when prototypes similar

3. **Mode Policy** (`mode.rs`): State machine (Explore/Exploit/Reset) based on value and TD signals

4. **Action Policy** (`action.rs`): Selects Scan/Focus/Perturb based on mode, adjusts Top-K and margin scales

### Key Modules

| Module | Purpose |
|--------|---------|
| `config.rs` | 150+ parameters controlling all behavior |
| `echo.rs` | Complex-valued signal propagation engine |
| `anchor.rs` | Memory addressing, prototype learning, value learning, merging |
| `memory.rs` | Episodic binding, competitive recall with abstain |
| `mode.rs` | Explore/Exploit/Reset state machine |
| `action.rs` | Scan/Focus/Perturb action selection |
| `distill.rs` | Policy distillation (student imitates teacher) |
| `causes.rs` | Latent cause injection (3 disjoint sets) |

### Demo Structure

`main.rs` runs 12 sequential demos. Control via config flags:
- `run_demo_7` through `run_demo_12` (Phase 2.0 experiments)
- Each demo follows pattern: setup → training loop → evaluation → metrics

### Important Patterns

**Gating**: Confidence-based filtering using `topk_margin` and `total_power`. Gate params vary by mode (Explore loosens, Exploit tightens).

**Prototype Learning**: Anchors maintain sparse vectors via exponential averaging of Top-K node appearances.

**Value Learning**: TD(0) with reward = f(power_delta, coherence, proto_alignment). Values guide mode transitions.

**Merging**: Every 50 ticks, anchors with `proto_score > 0.78` are consolidated to prevent fragmentation.

### Config Key Groups

- **Dynamics**: `decay_per_tick`, `clamp_max_amp`, `pow_target`
- **Memory**: `proto_m`, `proto_eta`, `memory_max_entries`
- **Value**: `alpha_v`, `gamma_v`, reward weights
- **Mode**: `mode_explore_v_max`, `mode_exploit_v_min`, `mode_reset_td_min`
- **Action**: `scan_topk_scale`, `focus_margin_scale`

### Phase History

Current: Phase 2.0f-E (natural teacher viability + exploit emergence)
- 1.9: Aggressive anchor consolidation
- 2.0a: Mode policy
- 2.0c: Action policy
- 2.0f-B: Policy distillation with budget fairness
- 2.0f-C: Mode-conditioned distillation + two-sided budget matching
- 2.0f-D: Forced mode schedule ensuring Exploit mode occurs with Focus action
- 2.0f-E: Natural Exploit emergence via signal quality (stable + proto_align + margin)
