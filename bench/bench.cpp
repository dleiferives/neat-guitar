// Micro-benchmark for the two hottest paths in neat-guitar:
//   1. Network::activate()   — ~645k calls/generation
//   2. focal_loss equivalent — ~63M calls/generation
//
// Build:
//   cmake -B build -DBUILD_BENCH=ON && cmake --build build --target neat_bench
// Run:
//   ./build/neat_bench

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "neat/network.hpp"
#include "neat/config.hpp"
#include "neat/genome.hpp"
#include "neat/innovation.hpp"

// ── Timing helpers ────────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ns(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

// ── Build a synthetic Network directly (no audio data needed) ─────────────────

static Network make_bench_network(int n_conns, int n_hidden) {
    NeatConfig cfg;
    Network net;
    net.n_inputs     = cfg.n_inputs();    // 165
    net.n_outputs    = cfg.n_outputs();   // 49
    net.output_start = cfg.n_inputs();

    int n_nodes = net.n_inputs + net.n_outputs + n_hidden;
    net.n_nodes = n_nodes;
    net.values.assign(n_nodes, 0.0f);
    net.sums.assign(n_nodes, 0.0f);
    net.biases.assign(n_nodes, 0.1f);

    // Distribute connections input→output and input→hidden
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> in_dist(0, net.n_inputs - 1);
    std::uniform_int_distribution<int> out_dist(net.n_inputs, n_nodes - 1);
    std::uniform_real_distribution<float> w_dist(-1.0f, 1.0f);

    net.conn_in.reserve(n_conns);
    net.conn_out.reserve(n_conns);
    net.conn_w.reserve(n_conns);
    for (int i = 0; i < n_conns; ++i) {
        net.conn_in.push_back(in_dist(rng));
        net.conn_out.push_back(out_dist(rng));
        net.conn_w.push_back(w_dist(rng));
    }

    // Compute minimum passes for this topology (feedforward bench nets → 1 pass).
    net.compute_min_passes(cfg.activation_passes);

    return net;
}

// ── focal_loss equivalent (same math as fitness.cpp) ─────────────────────────

static inline float fast_log_bench(float x) {
    union { float f; uint32_t i; } u = {x};
    int e = (int)((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x007FFFFFu) | 0x3F800000u;
    float m = u.f - 1.0f;
    return ((float)e + m * (1.0f + m * (-0.5f + m * 0.333333f))) * 0.693147180f;
}

static inline float focal_loss_bench(float target, float pred, float alpha) {
    if (pred < 1e-7f) pred = 1e-7f;
    if (pred > 1.0f - 1e-7f) pred = 1.0f - 1e-7f;
    float p_t     = target * pred + (1.0f - target) * (1.0f - pred);
    float alpha_t = target * alpha + (1.0f - target) * (1.0f - alpha);
    float omp     = 1.0f - p_t;
    return -alpha_t * omp * omp * fast_log_bench(p_t);
}

// ── Multi-round timing: run n_rounds rounds, report median ns/call ────────────

static double median_of(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) * 0.5;
}

// ── Benchmark: Network::activate() ───────────────────────────────────────────

static void bench_activate(int n_conns, int n_hidden, int n_iter, int n_rounds = 15) {
    Network net = make_bench_network(n_conns, n_hidden);
    NeatConfig cfg;

    std::vector<float> inp(cfg.n_inputs(), 0.5f);
    std::vector<float> out(cfg.n_outputs());

    // Warmup
    for (int i = 0; i < 1000; ++i) net.activate(inp.data(), out.data());

    volatile float sink = 0.0f;
    std::vector<double> samples(n_rounds);

    for (int r = 0; r < n_rounds; ++r) {
        auto t0 = Clock::now();
        for (int i = 0; i < n_iter; ++i) {
            inp[0] = (float)i * 0.000001f;
            net.activate(inp.data(), out.data());
            sink += out[0];
        }
        auto t1 = Clock::now();
        samples[r] = elapsed_ns(t0, t1) / n_iter;
    }

    double med = median_of(samples);
    printf("activate() | conns=%4d hidden=%3d | %7.1f ns/call  (sink=%.4f)\n",
           n_conns, n_hidden, med, (float)sink);
}

// ── Benchmark: focal_loss() ───────────────────────────────────────────────────

static void bench_focal_loss(int n_iter, int n_rounds = 15) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.01f, 0.99f);

    std::vector<float> targets(1024), preds(1024);
    for (int i = 0; i < 1024; ++i) {
        targets[i] = (rng() & 1) ? 1.0f : 0.0f;
        preds[i]   = dist(rng);
    }

    volatile float sink = 0.0f;
    std::vector<double> samples(n_rounds);

    for (int r = 0; r < n_rounds; ++r) {
        auto t0 = Clock::now();
        for (int i = 0; i < n_iter; ++i) {
            int idx = i & 1023;
            sink += focal_loss_bench(targets[idx], preds[idx], 0.75f);
        }
        auto t1 = Clock::now();
        samples[r] = elapsed_ns(t0, t1) / n_iter;
    }

    double med = median_of(samples);
    printf("focal_loss()                       | %7.1f ns/call  (sink=%.4f)\n",
           med, (float)sink);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    printf("=== neat-guitar micro-benchmark ===\n\n");

    printf("--- Network::activate() ---\n");
    bench_activate(  0,   0, 1000000);  // minimal (input→output direct)
    bench_activate( 50,  10, 1000000);  // typical early-generation genome
    bench_activate(100,  20, 1000000);  // medium genome
    bench_activate(300,  50,  500000);  // evolved genome with hidden nodes
    bench_activate(500, 100,  200000);  // large genome

    printf("\n--- focal_loss() ---\n");
    bench_focal_loss(10000000);

    printf("\n");
    return 0;
}
