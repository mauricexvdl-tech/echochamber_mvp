# Echo Chamber MVP

A minimal Rust demonstration of **emergent signal cancellation** through local complex-valued signal interference—no global logic checks required.

## What This Shows

- **Destructive interference emerges naturally** from local complex addition when signals arrive out of phase (e.g., π radians apart)
- **No explicit correctness checks**: cancellation happens via mathematics of complex numbers, not `if/else` logic
- **Phase constraints on edges** encode "consistency rules"—contradictory paths self-cancel
- **Two-phase tick processing** ensures order-independent, deterministic propagation
- **Power metric** (sum of |z|²) is the physically meaningful measure; amplitude sum can grow due to distribution
- **Meaning-Selector dynamics**: decay + clamp keeps power bounded while preserving interference patterns

## What This Does NOT Claim

- **Not quantum computing**: this is classical complex arithmetic, not quantum superposition
- **No quantum hardware**: runs on any CPU with standard f64 arithmetic
- **Not a theorem prover**: interference is a dynamical toy model, not formal verification
- **No parallelism guarantees**: single-threaded, synchronous execution
- **Phase shifts are arbitrary constraints**: they don't represent physical phenomena

## How It Works

1. Each node holds a complex-valued buffer (amplitude + phase)
2. On each tick:
   - (Demo 2 only) Apply global decay (3%/tick) and amplitude clamp (max 2.0)
   - Every node splits its buffer across outgoing edges
   - Each edge applies a phase rotation (constraint encoding)
   - Target nodes locally sum incoming signals
3. When signals from different paths have opposite phases, they cancel to ~0

## Phase 0.6: Disjoint Causes + Top-K Readout

This phase improves evaluation to avoid single-node winner collapse:

### Disjoint Injector Sets
- 3 causes × 6 injector nodes = 18 **unique** nodes (no overlap)
- Each cause injects only to its dedicated nodes
- Prevents hub dominance from receiving multiple cause signals

### Top-K Readout (K=5)
- Instead of single "winner" node, track Top-5 nodes by amplitude each tick
- **Metrics per node**:
  - `hits`: number of ticks node appears in Top-K
  - `dominance`: hits / total_ticks
  - `purity`: max_c(hits_when_c / hits) — cause selectivity
- **Coverage**: how many distinct nodes ever appear in Top-K

## Phase 1.0a: Destruction Ratio Instrumentation

This phase adds metrics to measure destructive interference correlation with cause selectivity.

### Metrics per Node per Tick

- **inflow_power[i]**: Sum of |signal|² for all packets delivered to node i during this tick
- **self_power[i]**: |buffer|² after all deliveries complete (before next process step)
- **destruction_ratio[i]**: `1 - self_power / (inflow_power + EPS)`, clamped to [0, 1]

### Interpretation

- **destruction_ratio ≈ 0**: Incoming signals are coherent (same phase) → constructive interference
- **destruction_ratio ≈ 1**: Incoming signals cancel out → destructive interference (contradiction)

### What We Expect

If coherent "meaning modes" exist:
- **Top-K nodes should have lower destruction** (they accumulate coherent signals)
- **Non-Top-K nodes should have higher destruction** (incoherent mix of signals)
- **Positive delta (non-topk - topk)** indicates Top-K winners are more coherent

### Aggregate Metrics

During evaluation:
- `avg_destr_topk`: Average destruction for nodes in Top-K
- `avg_destr_non_topk`: Average destruction for nodes NOT in Top-K (with inflow > 0)
- `delta`: non_topk - topk (positive = winners more coherent)
- Per-cause `avg_destr_topk`: Breakdown by which cause was active

## Instrumentation

- **Amplitude** (`norm`): |z| = sqrt(re² + im²)
- **Power** (`power`): |z|² = re² + im² — the physically meaningful metric
- **Phase** (`arg`): atan2(im, re) — printed as `φ=undef` when amplitude < 1e-9

## Demo 1: Lie Triangle

Four nodes demonstrating perfect cancellation:
```
0 (start) ──► 1 ──► 2 (target)    [path A: total phase = 0]
    └──────► 3 ──►                [path B: total phase = π]
```
At node 2, the two paths arrive with π phase difference → amplitude ≈ 0.

## Demo 2: Latent Cause Injection

32 nodes with random edges and phase shifts, plus:
- **Decay**: 3% amplitude loss per tick
- **Clamp**: Maximum amplitude of 2.0
- **Disjoint cause injection**: 3 causes with non-overlapping injector sets
- **Training**: 50,000 ticks with destruction metrics
- **Evaluation**: 10,000 ticks measuring Top-5 purity and destruction ratios

## Phase 1.7a: Anchors as Concept Tokens

Anchors now maintain a lightweight **prototype vector** that captures the typical Top-K node distribution when that anchor is active. This turns anchors from simple signature-based addresses into richer "concept tokens" that learn online.

**Key features:**
- Each anchor stores a sparse prototype of `PROTO_M=12` nodes with learned weights
- Prototypes update online via exponential averaging (`proto_eta=0.10`) with decay (`proto_decay=0.01`)
- Prototype similarity scoring enables better disambiguation when signatures are similar
- Entropy of prototypes decreases over time as concepts sharpen (measured via `proto_entropy_top10`)

**Configuration:** `proto_m`, `proto_eta`, `proto_decay`, `proto_beta`, `proto_insert_margin` in `Config`

## Build & Run

```bash
cargo run --release
```

## Files

- `src/complex.rs` — Complex number type (add, mul, from_polar, norm, power, arg)
- `src/rng.rs` — SplitMix64 PRNG for reproducible random graphs
- `src/echo.rs` — Node, Edge, EchoChamber, TickMetrics with destruction instrumentation
- `src/causes.rs` — Disjoint cause configuration, injection, and Top-K evaluation
- `src/main.rs` — Demo scenarios with training, evaluation, and destruction summary

## Requirements

- Rust (stable, edition 2021)
- No external crates (std only)
