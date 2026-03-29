#pragma once
#include <vector>
#include "genome.hpp"

// Compiled phenotype network built from a Genome.
// Uses multi-pass activation to handle arbitrary topologies including cycles.
struct Network {
    int n_inputs;
    int n_outputs;
    int n_nodes;             // total node count (inputs + outputs + hidden)
    int output_start;        // index in values[] where output nodes begin
    int n_passes;            // passes required (1 for feedforward, more for recurrent)

    // SoA connection layout: three parallel arrays, sorted by conn_out
    std::vector<int>   conn_in;   // source node index for each connection
    std::vector<int>   conn_out;  // destination node index for each connection
    std::vector<float> conn_w;    // weight for each connection

    std::vector<float> values;    // current node activations
    std::vector<float> sums;      // accumulator per node
    std::vector<float> biases;

    static Network from_genome(const Genome& g, const NeatConfig& cfg);

    // Compute minimum activation passes needed for this network's topology.
    // Call after conn_in/conn_out are populated. max_passes is the upper bound.
    void compute_min_passes(int max_passes);

    // Zero-allocation hot path: caller provides pre-allocated buffers.
    // inp must be n_inputs floats; out must be n_outputs floats.
    void activate(const float* __restrict__ inp, float* __restrict__ out) noexcept;

    // Convenience wrapper — allocates output vector (use for non-hot paths).
    std::vector<float> activate(const std::vector<float>& inputs);

    // Reset internal state (for recurrent networks between recordings).
    void reset();
};
