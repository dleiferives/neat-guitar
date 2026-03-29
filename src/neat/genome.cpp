#include "genome.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

// ── Helpers ──────────────────────────────────────────────────────────────────

NodeGene* Genome::find_node(int id) {
    for (auto& n : nodes)
        if (n.id == id) return &n;
    return nullptr;
}

const NodeGene* Genome::find_node(int id) const {
    for (const auto& n : nodes)
        if (n.id == id) return &n;
    return nullptr;
}

bool Genome::has_connection(int in_node, int out_node) const {
    for (const auto& c : conns)
        if (c.in_node == in_node && c.out_node == out_node) return true;
    return false;
}

int Genome::max_node_id() const {
    int m = -1;
    for (const auto& n : nodes) m = std::max(m, n.id);
    return m;
}

// ── Construction ─────────────────────────────────────────────────────────────

Genome Genome::make_minimal(int genome_id,
                             const NeatConfig&  cfg,
                             InnovationTracker& innov,
                             std::mt19937&      rng) {
    Genome g;
    g.id = genome_id;

    int n_in  = cfg.n_inputs();
    int n_out = cfg.n_outputs();

    for (int i = 0; i < n_in; ++i)
        g.nodes.push_back({i, NodeType::INPUT, 0.0f});

    for (int i = 0; i < n_out; ++i)
        g.nodes.push_back({n_in + i, NodeType::OUTPUT, 0.0f});

    // Seed each output with a direct connection from its corresponding salience
    // input (salience_bin[k] → output[k]).  This gives the network the trivial
    // solution "note k is active when salience for note k is high" from gen 0.
    // Also add a few random connections for exploration.
    std::uniform_real_distribution<float> wdist(-cfg.weight_init_range,
                                                  cfg.weight_init_range);
    std::uniform_int_distribution<int>    in_dist(0, n_in - 1);

    int salience_offset = cfg.n_cqt_bins;  // salience bins start after CQT bins

    for (int k = 0; k < n_out; ++k) {
        int out_id = n_in + k;

        // Direct salience → output connection
        int sal_id = salience_offset + k;
        uint32_t inv = innov.get_conn(sal_id, out_id);
        g.conns.push_back({sal_id, out_id, wdist(rng), true, inv});

        // Plus 2 random connections for exploration
        for (int j = 0; j < 2; ++j) {
            int      in_id = in_dist(rng);
            uint32_t inv2  = innov.get_conn(in_id, out_id);
            if (!g.has_connection(in_id, out_id))
                g.conns.push_back({in_id, out_id, wdist(rng), true, inv2});
        }
    }

    return g;
}

// ── Mutations ────────────────────────────────────────────────────────────────

void Genome::mutate_weights(const NeatConfig& cfg, std::mt19937& rng) {
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::normal_distribution<float>       perturb(0.0f, cfg.weight_perturb_power);
    std::uniform_real_distribution<float> replace(-cfg.weight_init_range,
                                                    cfg.weight_init_range);
    // Mutate connection weights
    for (auto& c : conns) {
        if (unit(rng) < cfg.weight_mutate_rate) {
            if (unit(rng) < cfg.weight_perturb_rate)
                c.weight += perturb(rng);
            else
                c.weight = replace(rng);
        }
    }
    // Mutate biases (same rates as weights)
    for (auto& n : nodes) {
        if (n.type == NodeType::INPUT) continue;
        if (unit(rng) < cfg.weight_mutate_rate) {
            if (unit(rng) < cfg.weight_perturb_rate)
                n.bias += perturb(rng);
            else
                n.bias = replace(rng);
        }
    }
}

void Genome::mutate_add_connection(const NeatConfig&  cfg,
                                    InnovationTracker& innov,
                                    std::mt19937&      rng) {
    // Collect candidates: any node can be an in_node except output nodes
    // feeding back into input nodes is fine (recurrence)
    std::vector<int> all_ids;
    std::vector<int> non_input_ids;
    for (const auto& n : nodes) {
        all_ids.push_back(n.id);
        if (n.type != NodeType::INPUT)
            non_input_ids.push_back(n.id);
    }

    if (all_ids.empty() || non_input_ids.empty()) return;

    std::uniform_int_distribution<size_t> all_dist(0, all_ids.size() - 1);
    std::uniform_int_distribution<size_t> out_dist(0, non_input_ids.size() - 1);

    for (int attempt = 0; attempt < cfg.add_conn_tries; ++attempt) {
        int in_id  = all_ids[all_dist(rng)];
        int out_id = non_input_ids[out_dist(rng)];
        if (in_id == out_id) continue;
        if (has_connection(in_id, out_id)) continue;

        std::uniform_real_distribution<float> wdist(-cfg.weight_init_range,
                                                      cfg.weight_init_range);
        uint32_t inv = innov.get_conn(in_id, out_id);
        conns.push_back({in_id, out_id, wdist(rng), true, inv});
        return;
    }
}

void Genome::mutate_add_node(const NeatConfig&  cfg,
                              InnovationTracker& innov,
                              std::mt19937&      rng) {
    // Pick a random enabled connection to split
    std::vector<size_t> enabled_idx;
    for (size_t i = 0; i < conns.size(); ++i)
        if (conns[i].enabled) enabled_idx.push_back(i);

    if (enabled_idx.empty()) return;

    std::uniform_int_distribution<size_t> dist(0, enabled_idx.size() - 1);
    ConnGene& split = conns[enabled_idx[dist(rng)]];
    split.enabled   = false;

    auto [new_id, inv1, inv2] = innov.get_split(split.innov,
                                                  split.in_node,
                                                  split.out_node);

    // Bias initialised to 0
    nodes.push_back({new_id, NodeType::HIDDEN, 0.0f});
    conns.push_back({split.in_node, new_id,       1.0f,        true, inv1});
    conns.push_back({new_id,        split.out_node, split.weight, true, inv2});

    (void)cfg;
}

void Genome::mutate_toggle(std::mt19937& rng) {
    if (conns.empty()) return;
    std::uniform_int_distribution<size_t> dist(0, conns.size() - 1);
    auto& c   = conns[dist(rng)];
    c.enabled = !c.enabled;
}

void Genome::mutate(const NeatConfig&  cfg,
                    InnovationTracker& innov,
                    std::mt19937&      rng) {
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    mutate_weights(cfg, rng);

    if (unit(rng) < cfg.add_conn_rate)
        mutate_add_connection(cfg, innov, rng);

    if (unit(rng) < cfg.add_node_rate)
        mutate_add_node(cfg, innov, rng);

    if (unit(rng) < cfg.toggle_conn_rate)
        mutate_toggle(rng);
}

// ── Crossover ────────────────────────────────────────────────────────────────

Genome Genome::crossover(const Genome& p1,
                          const Genome& p2,
                          int           offspring_id,
                          std::mt19937& rng) {
    // p1 is the more fit (or equal) parent
    Genome child;
    child.id = offspring_id;

    // Build innovation → conn map for p2
    std::unordered_map<uint32_t, const ConnGene*> p2_map;
    for (const auto& c : p2.conns)
        p2_map[c.innov] = &c;

    std::bernoulli_distribution coin(0.5);

    for (const auto& c1 : p1.conns) {
        auto it = p2_map.find(c1.innov);
        if (it != p2_map.end()) {
            // Matching gene: inherit from either parent
            const ConnGene& chosen = coin(rng) ? c1 : *it->second;
            ConnGene gc            = chosen;
            // If either parent has it disabled, 25% chance offspring also disables it
            if (!c1.enabled || !it->second->enabled) {
                std::bernoulli_distribution dis(0.25);
                gc.enabled = !dis(rng);
            }
            child.conns.push_back(gc);
        } else {
            // Disjoint / excess: inherit from more fit parent (p1)
            child.conns.push_back(c1);
        }
    }

    // Collect all node ids referenced by connections + original p1 nodes
    std::unordered_map<int, NodeGene> node_map;
    for (const auto& n : p1.nodes) node_map[n.id] = n;
    for (const auto& n : p2.nodes) {
        if (!node_map.count(n.id)) node_map[n.id] = n;
    }
    for (const auto& c : child.conns) {
        if (!node_map.count(c.in_node))
            node_map[c.in_node] = {c.in_node, NodeType::HIDDEN, 0.0f};
        if (!node_map.count(c.out_node))
            node_map[c.out_node] = {c.out_node, NodeType::HIDDEN, 0.0f};
    }
    // Always include input/output nodes from p1
    for (const auto& n : p1.nodes)
        node_map[n.id] = n;

    for (auto& [id, ng] : node_map)
        child.nodes.push_back(ng);

    return child;
}

// ── Compatibility distance ───────────────────────────────────────────────────

float Genome::compat_distance(const Genome& other, const NeatConfig& cfg) const {
    if (conns.empty() && other.conns.empty()) return 0.0f;

    // Map innovation → weight for other
    std::unordered_map<uint32_t, float> other_map;
    uint32_t other_max_innov = 0;
    for (const auto& c : other.conns) {
        other_map[c.innov] = c.weight;
        other_max_innov    = std::max(other_max_innov, c.innov);
    }

    uint32_t self_max_innov = 0;
    for (const auto& c : conns)
        self_max_innov = std::max(self_max_innov, c.innov);

    uint32_t max_innov = std::max(self_max_innov, other_max_innov);

    int   matching  = 0;
    int   disjoint  = 0;
    int   excess    = 0;
    float weight_diff_sum = 0.0f;

    for (const auto& c : conns) {
        auto it = other_map.find(c.innov);
        if (it != other_map.end()) {
            ++matching;
            weight_diff_sum += std::abs(c.weight - it->second);
        } else {
            if (c.innov <= other_max_innov) ++disjoint;
            else                            ++excess;
        }
    }

    // Build self innov set for O(1) lookup instead of O(n) any_of
    std::unordered_map<uint32_t, bool> self_map;
    for (const auto& c : conns) self_map[c.innov] = true;

    for (const auto& c : other.conns) {
        if (!self_map.count(c.innov)) {
            if (c.innov <= self_max_innov) ++disjoint;
            else                           ++excess;
        }
    }

    float N = std::max(1, std::max((int)conns.size(), (int)other.conns.size()));
    float W = matching > 0 ? weight_diff_sum / matching : 0.0f;

    (void)max_innov;
    return cfg.c1 * excess / N + cfg.c2 * disjoint / N + cfg.c3 * W;
}

// ── Serialization ────────────────────────────────────────────────────────────

void Genome::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path + " for writing");

    f << "NEAT_GENOME\n";
    f << "id "      << id      << "\n";
    f << "fitness " << fitness << "\n";

    f << "nodes " << nodes.size() << "\n";
    for (const auto& n : nodes)
        f << n.id << " " << (int)n.type << " " << n.bias << "\n";

    f << "conns " << conns.size() << "\n";
    for (const auto& c : conns)
        f << c.in_node  << " "
          << c.out_node << " "
          << c.weight   << " "
          << (int)c.enabled << " "
          << c.innov    << "\n";
}

Genome Genome::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);

    std::string tag;
    f >> tag;
    if (tag != "NEAT_GENOME") throw std::runtime_error("Bad genome file");

    Genome g;
    f >> tag >> g.id;
    f >> tag >> g.fitness;

    int n_nodes;
    f >> tag >> n_nodes;
    g.nodes.resize(n_nodes);
    for (auto& n : g.nodes) {
        int t;
        f >> n.id >> t >> n.bias;
        n.type = static_cast<NodeType>(t);
    }

    int n_conns;
    f >> tag >> n_conns;
    g.conns.resize(n_conns);
    for (auto& c : g.conns) {
        int en;
        f >> c.in_node >> c.out_node >> c.weight >> en >> c.innov;
        c.enabled = en != 0;
    }

    return g;
}
