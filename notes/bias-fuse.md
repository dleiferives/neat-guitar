# Fuse Bias into Sums Initialization

## What I changed
`src/neat/network.cpp` — `Network::activate()` pass loop.

Old code:
```cpp
for (int i = n_inputs; i < (int)sums.size(); ++i)
    sums[i] = 0.0f;
// ...
values[i] = sigmoid(sums[i] + biases[i]);
```

New code:
```cpp
const float* __restrict__ bias = biases.data();
for (int i = n_inputs; i < (int)sums.size(); ++i)
    sums[i] = bias[i];
// ...
values[i] = sigmoid(sums[i]);
```

Also added `__restrict__` on the `bias` pointer to help the compiler prove no aliasing.

## Why this should be faster (Casey Muratori reasoning)
The sigmoid loop previously did two memory loads per node: `sums[i]` and `biases[i]`.
By pre-loading bias into `sums` at initialization time, the sigmoid loop only reads `sums[i]` — one load instead of two, and one add eliminated. For `n_passes=4` and 59 non-input nodes (outputs + hidden), that's 4 × 59 = 236 load+add pairs eliminated per `activate()` call. The init loop still touches `biases[]` once, but that's a sequential read vs the random-ish access pattern of the sigmoid loop.

## Benchmark results (15-round median)
| Scenario | Before | After | Speedup |
|---|---|---|---|
| activate() conns=0 hidden=0 | 352.9 ns | 324.7 ns | 1.09x |
| activate() conns=50 hidden=10 | 589.9 ns | 574.9 ns | 1.03x |
| activate() conns=100 hidden=20 | 866.3 ns | 824.0 ns | 1.05x |
| activate() conns=300 hidden=50 | 1730.5 ns | 1728.8 ns | ~neutral |
| activate() conns=500 hidden=100 | 2719.4 ns | 2523.9 ns | 1.08x |
| focal_loss() | 3.7 ns | 4.0 ns | neutral (noise) |

## What I learned
The gain is real but modest (~3–9%). The connection-scatter loop (`sums[co[c]] += values[ci[c]] * cw[c]`) dominates at larger network sizes, so reducing work in the sigmoid loop has diminishing returns as conn count grows. The `conns=300` case is basically neutral — scatter cost swamps everything else.

The `__restrict__` annotation on `bias` may have helped the compiler auto-vectorize the init loop but was hard to isolate.

## Casey Muratori principle applied
**Eliminate work** — removed one load and one add per node per pass from the hot sigmoid loop by folding the bias addition into the initialization step.
