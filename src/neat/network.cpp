#include "network.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-4.9f * x));
}

Network Network::from_genome(const Genome& g, const NeatConfig& cfg) {
    Network net;
    net.n_inputs  = cfg.n_inputs();
    net.n_outputs = cfg.n_outputs();
    net.n_passes  = cfg.activation_passes;

    // Determine node ordering:
    // 0..n_inputs-1          → input nodes  (IDs 0..n_inputs-1)
    // n_inputs..n_inputs+n_outputs-1 → output nodes
    // n_inputs+n_outputs..   → hidden nodes (sorted by id for determinism)

    std::vector<int> hidden_ids;
    for (const auto& n : g.nodes)
        if (n.type == NodeType::HIDDEN)
            hidden_ids.push_back(n.id);
    std::sort(hidden_ids.begin(), hidden_ids.end());

    int n_nodes = net.n_inputs + net.n_outputs + (int)hidden_ids.size();
    net.values.assign(n_nodes, 0.0f);
    net.sums.assign(n_nodes, 0.0f);
    net.biases.assign(n_nodes, 0.0f);

    // Build node_id → index map
    // We need to handle potentially non-contiguous hidden node IDs
    int max_id = cfg.first_hidden_node() + (int)hidden_ids.size() + 10;
    for (const auto& n : g.nodes)
        max_id = std::max(max_id, n.id + 1);
    net.node_id_to_idx.assign(max_id, -1);

    // Inputs: id == index
    for (int i = 0; i < net.n_inputs; ++i)
        net.node_id_to_idx[i] = i;

    // Outputs: id == n_inputs + k → index == n_inputs + k
    net.output_start = net.n_inputs;
    for (int k = 0; k < net.n_outputs; ++k)
        net.node_id_to_idx[cfg.first_output_node() + k] = net.n_inputs + k;

    // Hidden nodes: appended after outputs
    int hidden_start = net.n_inputs + net.n_outputs;
    for (int i = 0; i < (int)hidden_ids.size(); ++i)
        net.node_id_to_idx[hidden_ids[i]] = hidden_start + i;

    // Copy biases
    for (const auto& n : g.nodes) {
        if (n.id >= (int)net.node_id_to_idx.size()) continue;
        int idx = net.node_id_to_idx[n.id];
        if (idx >= 0) net.biases[idx] = n.bias;
    }

    // Build active connection list (enabled only)
    for (const auto& c : g.conns) {
        if (!c.enabled) continue;
        if (c.in_node  >= (int)net.node_id_to_idx.size()) continue;
        if (c.out_node >= (int)net.node_id_to_idx.size()) continue;
        int in_idx  = net.node_id_to_idx[c.in_node];
        int out_idx = net.node_id_to_idx[c.out_node];
        if (in_idx < 0 || out_idx < 0) continue;
        net.active_conns.push_back({in_idx, out_idx, c.weight});
    }

    return net;
}

std::vector<float> Network::activate(const std::vector<float>& inputs) {
    if ((int)inputs.size() != n_inputs)
        throw std::runtime_error("Input size mismatch");

    // Set input node values
    for (int i = 0; i < n_inputs; ++i)
        values[i] = inputs[i];

    // Multi-pass activation
    for (int pass = 0; pass < n_passes; ++pass) {
        // Reset sums for non-input nodes
        for (int i = n_inputs; i < (int)sums.size(); ++i)
            sums[i] = 0.0f;

        // Accumulate weighted inputs
        for (const auto& c : active_conns)
            sums[c.out_idx] += values[c.in_idx] * c.weight;

        // Apply activation to non-input nodes
        for (int i = n_inputs; i < (int)values.size(); ++i)
            values[i] = sigmoid(sums[i] + biases[i]);
    }

    // Extract output nodes
    std::vector<float> out(n_outputs);
    for (int i = 0; i < n_outputs; ++i)
        out[i] = values[output_start + i];
    return out;
}

void Network::reset() {
    std::fill(values.begin(), values.end(), 0.0f);
    std::fill(sums.begin(),   sums.end(),   0.0f);
}
