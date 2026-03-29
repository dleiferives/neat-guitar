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

    net.n_nodes = net.n_inputs + net.n_outputs + (int)hidden_ids.size();
    net.values.assign(net.n_nodes, 0.0f);
    net.sums.assign(net.n_nodes, 0.0f);
    net.biases.assign(net.n_nodes, 0.0f);

    int max_id = cfg.first_hidden_node() + (int)hidden_ids.size() + 10;
    for (const auto& n : g.nodes)
        max_id = std::max(max_id, n.id + 1);
    std::vector<int> node_id_to_idx(max_id, -1);

    for (int i = 0; i < net.n_inputs; ++i)
        node_id_to_idx[i] = i;

    net.output_start = net.n_inputs;
    for (int k = 0; k < net.n_outputs; ++k)
        node_id_to_idx[cfg.first_output_node() + k] = net.n_inputs + k;

    int hidden_start = net.n_inputs + net.n_outputs;
    for (int i = 0; i < (int)hidden_ids.size(); ++i)
        node_id_to_idx[hidden_ids[i]] = hidden_start + i;

    for (const auto& n : g.nodes) {
        if (n.id >= (int)node_id_to_idx.size()) continue;
        int idx = node_id_to_idx[n.id];
        if (idx >= 0) net.biases[idx] = n.bias;
    }

    for (const auto& c : g.conns) {
        if (!c.enabled) continue;
        if (c.in_node  >= (int)node_id_to_idx.size()) continue;
        if (c.out_node >= (int)node_id_to_idx.size()) continue;
        int in_idx  = node_id_to_idx[c.in_node];
        int out_idx = node_id_to_idx[c.out_node];
        if (in_idx < 0 || out_idx < 0) continue;
        net.conn_in.push_back(in_idx);
        net.conn_out.push_back(out_idx);
        net.conn_w.push_back(c.weight);
    }

    net.compute_min_passes(cfg.activation_passes);

    return net;
}

// Bellman-Ford relaxation: compute the minimum activation passes needed.
// Feedforward depth-1 networks → n_passes=1. Recurrent → up to max_passes.
void Network::compute_min_passes(int max_passes) {
    std::vector<int> depth(n_nodes, 0);
    bool changed = true;
    int iters = 0;
    while (changed && iters < max_passes) {
        changed = false;
        for (int c = 0; c < (int)conn_in.size(); ++c) {
            int new_depth = depth[conn_in[c]] + 1;
            if (new_depth > depth[conn_out[c]]) {
                depth[conn_out[c]] = new_depth;
                changed = true;
            }
        }
        ++iters;
    }
    int max_depth = 0;
    for (int i = n_inputs; i < n_nodes; ++i)
        max_depth = std::max(max_depth, depth[i]);
    n_passes = std::max(1, std::min(max_depth, max_passes));
}

void Network::activate(const float* __restrict__ inp,
                       float* __restrict__ out) noexcept {
    for (int i = 0; i < n_inputs; ++i)
        values[i] = inp[i];

    const int*   __restrict__ ci = conn_in.data();
    const int*   __restrict__ co = conn_out.data();
    const float* __restrict__ cw = conn_w.data();
    const int n_c = (int)conn_w.size();

    const float* __restrict__ bias = biases.data();

    for (int pass = 0; pass < n_passes; ++pass) {
        for (int i = n_inputs; i < n_nodes; ++i)
            sums[i] = bias[i];

        for (int c = 0; c < n_c; ++c)
            sums[co[c]] += values[ci[c]] * cw[c];

        for (int i = n_inputs; i < n_nodes; ++i)
            values[i] = sigmoid(sums[i]);
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
