# Frame-Level Fitness Rewrite (2026-03-29)

## Problem

Training stuck at fitness ~0.33-0.37 for 2683 generations. Genomes bloated from 214 to 3300+ nodes.

### Root causes

1. **Wrong fitness objective**: 70% of fitness was onset-detection F1 (rising-edge detection
   with cooldown). NEAT can't optimize for precise temporal transitions — it has to randomly
   stumble onto the right pattern. Meanwhile frame-level loss (30%) provided dense per-frame
   signal but was barely weighted.

2. **Noisy evaluation**: Single shared RNG meant each genome saw different random segments.
   Small fitness improvements were invisible under evaluation noise.

3. **No complexity penalty**: Genomes grew unboundedly with zero fitness benefit.

4. **Weak weight mutations**: `weight_perturb_power=0.10` too small for 550+ connections.

### What we actually want

Per-frame "is this note active right now?" — not onset detection. The frame targets already
encode this as binary 1.0/0.0. Onset detection is a post-processing step, not a training
objective.

## Changes

### Fitness function (`src/fitness.cpp`)

- **Removed**: onset detection logic (was_active, cooldown, rising-edge, all_detected)
- **Removed**: note_f1() call, sparsity penalty, silence penalty
- **Added**: Frame-level TP/FP/FN counting per frame per output
- **New fitness**: `(0.5 * frame_f1 + 0.5 * frame_score) * parsimony`
  - `frame_f1`: per-frame precision/recall on "is note active?"
  - `frame_score`: exp(-0.5 * focal_loss) for confidence calibration
  - `parsimony`: `1/(1 + 0.0002 * (nodes + conns))` to penalize bloat

### Evaluation determinism (`src/main.cpp`)

- Per-generation seeded RNG: `std::mt19937(42 + gen_counter)`
- All genomes in same generation see same segments = fair comparison
- Different seed each generation = diverse evaluation over time

### Config (`src/neat/config.hpp`)

- `add_node_rate`: 0.03 -> 0.01 (less structural mutation)
- `add_conn_rate`: 0.05 -> 0.03
- `weight_perturb_power`: 0.10 -> 0.25 (stronger weight exploration)

### Inference (`src/main.cpp`, `cmd_infer`)

- **Removed**: max() aggregation across all frames
- **Added**: Per-frame threshold detection with active time ranges
- Output shows each note with start/end time and note name

## Expected impact

- Fitness should show consistent upward trend (not flat oscillation)
- Genome sizes should stabilize (< 500 nodes)
- Inference output now shows timed note events
- Should train from fresh population (old one is bloated)
