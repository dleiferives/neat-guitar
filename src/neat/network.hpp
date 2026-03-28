#pragma once
#include <vector>
#include "genome.hpp"

// Compiled phenotype network built from a Genome.
// Uses multi-pass activation to handle arbitrary topologies including cycles.
struct Network {
    struct ActiveConn {
        int   in_idx;
        int   out_idx;
        float weight;
    };

    int n_inputs;
    int n_outputs;
    int output_start;       // index in values[] where output nodes begin
    int n_passes;

    std::vector<ActiveConn> active_conns;
    std::vector<float>      values;     // current node activations
    std::vector<float>      sums;       // accumulator per node
    std::vector<float>      biases;
    std::vector<int>        node_id_to_idx; // sparse map: node_id → values[] index

    static Network from_genome(const Genome& g, const NeatConfig& cfg);

    // Run one inference step. Returns output node activations (size = n_outputs).
    std::vector<float> activate(const std::vector<float>& inputs);

    // Reset internal state (for recurrent networks between recordings).
    void reset();
};
