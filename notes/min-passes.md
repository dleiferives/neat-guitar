# Minimum Passes from Topology

## What I changed
`src/neat/network.hpp` — added `int n_nodes` field, added `compute_min_passes(int max_passes)` declaration.
`src/neat/network.cpp` — added `compute_min_passes()` implementation (Bellman-Ford relaxation on
connection graph); call it at end of `from_genome()`; use `n_nodes` instead of `sums.size()` /
`values.size()` in `activate()`.
`bench/bench.cpp` — set `net.n_nodes` in `make_bench_network()`; call `compute_min_passes()` after
building synthetic connections so the bench reflects realistic n_passes.

## Why this should be faster (Casey Muratori reasoning)
`n_passes` was hardcoded to 4 (`cfg.activation_passes`) for every network regardless of topology.
For a depth-1 feedforward network (all connections go input→output or input→hidden directly), only
1 pass is required to fully propagate activations. Running 4 passes does 3× redundant work: the
sums-init loop, the scatter-add loop, and the sigmoid loop all execute again and produce identical
results after pass 1. Eliminating that work is pure elimination — no tradeoffs.

`compute_min_passes()` does Bellman-Ford relaxation: it propagates depth labels from input nodes
outward through the connection graph. For a DAG, it converges in `max_depth` iterations. The
computed depth becomes `n_passes`. For recurrent networks (cycles) the labels keep changing so the
algorithm runs up to `max_passes` iterations, preserving correctness.

## Benchmark results (15-round median)
Baseline = session start (bias-fuse already applied).

| Scenario | Before | After | Speedup |
|---|---|---|---|
| conns=0, hidden=0 | 308.2 ns | 97.0 ns | 3.18x |
| conns=50, hidden=10 | 541.9 ns | 156.3 ns | 3.47x |
| conns=100, hidden=20 | 767.3 ns | 223.2 ns | 3.44x |
| conns=300, hidden=50 | 1588.5 ns | 447.1 ns | 3.55x |
| conns=500, hidden=100 | 2609.1 ns | 695.5 ns | 3.75x |

## What I learned
D was the only change that mattered. Three other ideas were tested in the same batch:
- **B** (remove `node_id_to_idx` from Network struct) — neutral for `activate()`, only helps construction
- **C** (sort connections by `conn_out`) — actually hurts large networks; write locality improves but
  read side (`values[ci[c]]`) becomes more random, net negative
- **F** (`n_nodes` field instead of `.size()`) — neutral; compiler already hoisted the `.size()` call

Key lesson: individual micro-optimizations are swamped by benchmark noise (~±10%). Batching is
needed to see signal, but then you must decompose to find which change drove the result.

## Casey Muratori principle applied
**Eliminate work** — don't compute things you don't need. For feedforward topologies the extra
activation passes are completely redundant. Detecting the actual depth and doing only the required
passes eliminates 75% of the work in `activate()` for the common case.
