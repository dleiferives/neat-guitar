#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "config.hpp"
#include "innovation.hpp"

enum class NodeType : uint8_t { INPUT = 0, HIDDEN = 1, OUTPUT = 2 };

struct NodeGene {
    int      id;
    NodeType type;
    float    bias = 0.0f;
};

struct ConnGene {
    int      in_node;
    int      out_node;
    float    weight;
    bool     enabled;
    uint32_t innov;
};

struct Genome {
    int                   id          = 0;
    std::vector<NodeGene> nodes;
    std::vector<ConnGene> conns;
    float                 fitness     = 0.0f;
    float                 adj_fitness = 0.0f;
    int                   species_id  = -1;
    bool                  is_elite    = false;  // skip re-evaluation

    // ── Construction ─────────────────────────────────────────────────────────
    static Genome make_minimal(int genome_id,
                               const NeatConfig&    cfg,
                               InnovationTracker&   innov,
                               std::mt19937&        rng);

    // ── Mutation ─────────────────────────────────────────────────────────────
    void mutate(const NeatConfig& cfg,
                InnovationTracker& innov,
                std::mt19937& rng);

    // ── Crossover ────────────────────────────────────────────────────────────
    // parent1 must be the more fit (or equally fit) parent.
    static Genome crossover(const Genome& parent1,
                            const Genome& parent2,
                            int           offspring_id,
                            std::mt19937& rng);

    // ── Speciation ───────────────────────────────────────────────────────────
    float compat_distance(const Genome& other, const NeatConfig& cfg) const;

    // ── Serialization ────────────────────────────────────────────────────────
    void save(const std::string& path) const;
    static Genome load(const std::string& path);

    // ── Helpers ──────────────────────────────────────────────────────────────
    NodeGene*       find_node(int id);
    const NodeGene* find_node(int id) const;
    bool            has_connection(int in_node, int out_node) const;
    int             max_node_id() const;

private:
    void mutate_weights(const NeatConfig& cfg, std::mt19937& rng);
    void mutate_add_connection(const NeatConfig& cfg,
                               InnovationTracker& innov,
                               std::mt19937& rng);
    void mutate_add_node(const NeatConfig& cfg,
                         InnovationTracker& innov,
                         std::mt19937& rng);
    void mutate_toggle(std::mt19937& rng);
};
