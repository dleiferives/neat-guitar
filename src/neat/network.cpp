#include "network.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Schraudolph (1999) fast sigmoid via IEEE 754 bit-cast.
// Max error ~0.74% vs exact sigmoid — negligible for NEAT fitness ranking.
static inline float sigmoid(float x) {
    float y = -4.9f * x;
    if (y >  88.0f) return 0.0f;
    if (y < -88.0f) return 1.0f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * y + 1064866805.0f);
    return 1.0f / (1.0f + u.f);
}

Network Network::from_genome(const Genome& g, const NeatConfig& cfg) {
    Network net;
    net.n_inputs  = cfg.n_inputs();
    net.n_outputs = cfg.n_outputs();
    net.n_passes  = cfg.activation_passes;

    std::vector<int> hidden_ids;
    for (const auto& n : g.nodes)
        if (n.type == NodeType::HIDDEN)
            hidden_ids.push_back(n.id);
    std::sort(hidden_ids.begin(), hidden_ids.end());

    int n_nodes = net.n_inputs + net.n_outputs + (int)hidden_ids.size();
    net.values.assign(n_nodes, 0.0f);
    net.sums.assign(n_nodes, 0.0f);
    net.biases.assign(n_nodes, 0.0f);

    int max_id = cfg.first_hidden_node() + (int)hidden_ids.size() + 10;
    for (const auto& n : g.nodes)
        max_id = std::max(max_id, n.id + 1);
    net.node_id_to_idx.assign(max_id, -1);

    for (int i = 0; i < net.n_inputs; ++i)
        net.node_id_to_idx[i] = i;

    net.output_start = net.n_inputs;
    for (int k = 0; k < net.n_outputs; ++k)
        net.node_id_to_idx[cfg.first_output_node() + k] = net.n_inputs + k;

    int hidden_start = net.n_inputs + net.n_outputs;
    for (int i = 0; i < (int)hidden_ids.size(); ++i)
        net.node_id_to_idx[hidden_ids[i]] = hidden_start + i;

    for (const auto& n : g.nodes) {
        if (n.id >= (int)net.node_id_to_idx.size()) continue;
        int idx = net.node_id_to_idx[n.id];
        if (idx >= 0) net.biases[idx] = n.bias;
    }

    for (const auto& c : g.conns) {
        if (!c.enabled) continue;
        if (c.in_node  >= (int)net.node_id_to_idx.size()) continue;
        if (c.out_node >= (int)net.node_id_to_idx.size()) continue;
        int in_idx  = net.node_id_to_idx[c.in_node];
        int out_idx = net.node_id_to_idx[c.out_node];
        if (in_idx < 0 || out_idx < 0) continue;
        net.conn_in.push_back(in_idx);
        net.conn_out.push_back(out_idx);
        net.conn_w.push_back(c.weight);
    }

    return net;
}

void Network::activate(const float* __restrict__ inp,
                       float* __restrict__ out) noexcept {
    for (int i = 0; i < n_inputs; ++i)
        values[i] = inp[i];

    const int*   __restrict__ ci = conn_in.data();
    const int*   __restrict__ co = conn_out.data();
    const float* __restrict__ cw = conn_w.data();
    const int n_c = (int)conn_w.size();

    for (int pass = 0; pass < n_passes; ++pass) {
        for (int i = n_inputs; i < (int)sums.size(); ++i)
            sums[i] = 0.0f;

        for (int c = 0; c < n_c; ++c)
            sums[co[c]] += values[ci[c]] * cw[c];

        for (int i = n_inputs; i < (int)values.size(); ++i)
            values[i] = sigmoid(sums[i] + biases[i]);
    }

    for (int i = 0; i < n_outputs; ++i)
        out[i] = values[output_start + i];
}

std::vector<float> Network::activate(const std::vector<float>& inputs) {
    if ((int)inputs.size() != n_inputs)
        throw std::runtime_error("Input size mismatch");
    std::vector<float> result(n_outputs);
    activate(inputs.data(), result.data());
    return result;
}

void Network::reset() {
    std::fill(values.begin(), values.end(), 0.0f);
    std::fill(sums.begin(),   sums.end(),   0.0f);
}
