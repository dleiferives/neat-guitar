#pragma once
#include <cstdint>

struct NeatConfig {
    // ── Population ──────────────────────────────────────────────────────────
    int   pop_size            = 300;
    int   generations         = 500;

    // ── Speciation ───────────────────────────────────────────────────────────
    float compat_threshold    = 3.0f;
    float c1                  = 1.0f;   // excess gene coefficient
    float c2                  = 1.0f;   // disjoint gene coefficient
    float c3                  = 0.4f;   // average weight diff coefficient
    int   target_species      = 15;     // dynamic threshold aims at this
    float compat_mod          = 0.1f;   // threshold adjustment per generation

    // ── Mutation ─────────────────────────────────────────────────────────────
    float weight_mutate_rate  = 0.80f;
    float weight_perturb_rate = 0.90f;  // perturb vs full replace
    float weight_perturb_power= 0.15f;
    float weight_init_range   = 1.0f;
    float add_conn_rate       = 0.08f;
    float add_node_rate       = 0.01f;
    float toggle_conn_rate    = 0.01f;
    int   add_conn_tries      = 50;     // max attempts to find a novel connection

    // ── Reproduction ─────────────────────────────────────────────────────────
    float crossover_rate      = 0.75f;
    float interspecies_rate   = 0.01f;
    float elitism_fraction    = 0.10f;  // top 10% per species survive unchanged
    float survival_threshold  = 0.30f;  // fraction allowed to reproduce
    int   stagnation_limit    = 30;     // generations before culling a species

    // ── Network evaluation ───────────────────────────────────────────────────
    int   activation_passes   = 4;      // forward-pass iterations (handles cycles)

    // ── Audio ────────────────────────────────────────────────────────────────
    int   n_cqt_bins          = 108;    // CQT_N_OCTAVES(9) x CQT_BINS_PER_OCTAVE(12)
    int   n_salience_bins     = 49;     // N_SALIENCE_BINS: one per semitone E2-E6
    int   hop_size            = 512;    // samples between frames
    int   sample_rate         = 22050;
    int   pitch_history       = 8;      // recent peak-salience values appended to input

    // ── MIDI / guitar range ───────────────────────────────────────────────────
    int   midi_min            = 40;     // E2
    int   midi_max            = 88;     // E6

    // ── Derived (call after setting the above) ───────────────────────────────
    int n_inputs()  const { return n_cqt_bins + n_salience_bins + pitch_history; }
    int n_outputs() const { return midi_max - midi_min + 1; }   // 49
    int first_output_node() const { return n_inputs(); }
    int first_hidden_node() const { return n_inputs() + n_outputs(); }
};
