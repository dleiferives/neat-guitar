#include "population.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

// ── Construction ─────────────────────────────────────────────────────────────

Population::Population(NeatConfig c, uint64_t seed)
    : cfg(std::move(c)), rng(seed) {

    innov.next_node_id = cfg.first_hidden_node();

    genomes.reserve(cfg.pop_size);
    for (int i = 0; i < cfg.pop_size; ++i) {
        genomes.push_back(Genome::make_minimal(next_genome_id++, cfg, innov, rng));
    }
    innov.reset_generation();
}

Population Population::from_file(const std::string& path, NeatConfig cfg, uint64_t seed) {
    Population pop;
    pop.cfg = std::move(cfg);
    pop.rng = std::mt19937(seed);

    // Load genomes and generation number directly from the file.
    {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open " + path);
        std::string tag;
        f >> tag;
        if (tag != "NEAT_POPULATION") throw std::runtime_error("Bad population file");
        f >> tag;
        if (tag == "generation") {
            f >> pop.generation;
            f >> tag; // "count"
        }
        int count; f >> count;
        pop.genomes.reserve(count);
        for (int gi = 0; gi < count; ++gi) {
            f >> tag;
            if (tag != "NEAT_GENOME") throw std::runtime_error("Expected NEAT_GENOME");
            Genome g;
            f >> tag >> g.id;
            f >> tag >> g.fitness;
            int n_nodes; f >> tag >> n_nodes;
            g.nodes.resize(n_nodes);
            for (auto& n : g.nodes) {
                int t; f >> n.id >> t >> n.bias;
                n.type = static_cast<NodeType>(t);
            }
            int n_conns; f >> tag >> n_conns;
            g.conns.resize(n_conns);
            for (auto& c : g.conns) {
                int en; f >> c.in_node >> c.out_node >> c.weight >> en >> c.innov;
                c.enabled = en != 0;
            }
            pop.genomes.push_back(std::move(g));
        }
    }

    // Reconstruct innovation state from the loaded genomes.
    for (const auto& g : pop.genomes) {
        pop.next_genome_id = std::max(pop.next_genome_id, g.id + 1);
        for (const auto& n : g.nodes)
            pop.innov.next_node_id = std::max(pop.innov.next_node_id, n.id + 1);
        for (const auto& c : g.conns)
            pop.innov.next_innov = std::max(pop.innov.next_innov, c.innov + 1);
    }

    pop.speciate();
    return pop;
}

// ── Public interface ──────────────────────────────────────────────────────────

void Population::evolve(FitnessFn fit_fn,
                         std::function<void(int, float, const Genome&)> on_gen) {
    for (int g = 0; g < cfg.generations; ++g) {
        step(fit_fn);
        const Genome& best = best_genome();
        if (on_gen) on_gen(generation, best.fitness, best);
        if (best.fitness >= cfg.fitness_threshold) break;
    }
}

void Population::step(FitnessFn fit_fn) {
    innov.reset_generation();
    evaluate(fit_fn);
    speciate();
    reproduce();
    ++generation;
}

const Genome& Population::best_genome() const {
    return *std::max_element(genomes.begin(), genomes.end(),
        [](const Genome& a, const Genome& b){ return a.fitness < b.fitness; });
}

// ── Evaluation ───────────────────────────────────────────────────────────────

void Population::evaluate(FitnessFn fit_fn) {
    for (auto& g : genomes)
        g.fitness = fit_fn(g);
}

// ── Speciation ───────────────────────────────────────────────────────────────

void Population::speciate() {
    // Clear member lists; keep representatives from last generation
    for (auto& sp : species) sp.member_indices.clear();

    for (int gi = 0; gi < (int)genomes.size(); ++gi) {
        const Genome& g = genomes[gi];
        bool placed = false;
        for (auto& sp : species) {
            float dist = g.compat_distance(sp.representative, cfg);
            if (dist < cfg.compat_threshold) {
                sp.member_indices.push_back(gi);
                placed = true;
                break;
            }
        }
        if (!placed) {
            Species sp;
            sp.id             = next_species_id++;
            sp.representative = g;
            sp.member_indices.push_back(gi);
            species.push_back(std::move(sp));
        }
    }

    // Remove empty species
    species.erase(
        std::remove_if(species.begin(), species.end(),
            [](const Species& s){ return s.member_indices.empty(); }),
        species.end());

    // Update representatives, staleness, best fitness
    for (auto& sp : species) {
        // New representative: random member
        std::uniform_int_distribution<size_t> rd(0, sp.member_indices.size() - 1);
        sp.representative = genomes[sp.member_indices[rd(rng)]];

        float sp_best = 0.0f;
        for (int idx : sp.member_indices)
            sp_best = std::max(sp_best, genomes[idx].fitness);

        // Use mean of current and previous best to smooth noise.
        // Replaces the old all-time-max which could never be beaten under
        // noisy evaluation, making staleness always increment.
        float smoothed = 0.5f * sp_best + 0.5f * sp.best_fitness;
        if (sp_best > sp.best_fitness) {
            sp.best_fitness = sp_best;
            sp.staleness    = 0;
        } else if (smoothed >= sp.best_fitness * 0.99f) {
            // Within 1% — don't penalize, evaluation noise could explain it
            sp.best_fitness = smoothed;
        } else {
            ++sp.staleness;
        }
    }

    // Adjusted (shared) fitness
    for (auto& sp : species) {
        float sz = (float)sp.member_indices.size();
        for (int idx : sp.member_indices)
            genomes[idx].adj_fitness = genomes[idx].fitness / sz;
    }

    adjust_compat_threshold();
}

void Population::adjust_compat_threshold() {
    int n_sp = (int)species.size();
    if (n_sp < cfg.target_species)       cfg.compat_threshold -= cfg.compat_mod;
    else if (n_sp > cfg.target_species)  cfg.compat_threshold += cfg.compat_mod;
    cfg.compat_threshold = std::max(0.3f, cfg.compat_threshold);
}

// ── Reproduction ─────────────────────────────────────────────────────────────

int Population::pick_parent_index(const Species& sp) const {
    // Roulette selection within species by raw fitness (not adj_fitness —
    // adj_fitness divides by species size which is constant within a species,
    // making selection nearly uniform).
    float total = 0.0f;
    for (int idx : sp.member_indices) total += genomes[idx].fitness;
    if (total <= 0.0f) {
        return sp.member_indices[const_cast<Population*>(this)->rng() % sp.member_indices.size()];
    }
    std::uniform_real_distribution<float> roulette(0.0f, total);
    float r   = roulette(const_cast<Population*>(this)->rng);
    float acc = 0.0f;
    for (int idx : sp.member_indices) {
        acc += genomes[idx].fitness;
        if (acc >= r) return idx;
    }
    return sp.member_indices.back();
}

Genome Population::make_offspring(const Species& sp, std::mt19937& local_rng) {
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    if (unit(local_rng) < cfg.crossover_rate && sp.member_indices.size() > 1) {
        int idx1 = pick_parent_index(sp);
        int idx2 = pick_parent_index(sp);
        // Ensure different parents
        for (int t = 0; t < 5 && idx2 == idx1; ++t) idx2 = pick_parent_index(sp);

        const Genome* p1 = &genomes[idx1];
        const Genome* p2 = &genomes[idx2];
        if (p1->fitness < p2->fitness) std::swap(p1, p2);

        Genome child = Genome::crossover(*p1, *p2, next_genome_id++, local_rng);
        child.mutate(cfg, innov, local_rng);
        return child;
    } else {
        int    idx   = pick_parent_index(sp);
        Genome child = genomes[idx];
        child.id     = next_genome_id++;
        child.mutate(cfg, innov, local_rng);
        return child;
    }
}

void Population::reproduce() {
    // Cull stagnant species (keep at least 2, never cull the species
    // containing the global best genome)
    if ((int)species.size() > 2) {
        float global_best = -1e30f;
        int best_species_id = -1;
        for (const auto& sp : species) {
            for (int idx : sp.member_indices) {
                if (genomes[idx].fitness > global_best) {
                    global_best = genomes[idx].fitness;
                    best_species_id = sp.id;
                }
            }
        }
        species.erase(
            std::remove_if(species.begin(), species.end(),
                [&](const Species& s){
                    return s.staleness >= cfg.stagnation_limit
                        && s.id != best_species_id;
                }),
            species.end());
    }
    if (species.empty()) return;

    // Compute offspring allocation per species proportional to adj_fitness sum
    std::vector<float> sp_fitness(species.size(), 0.0f);
    float total_adj = 0.0f;
    for (size_t si = 0; si < species.size(); ++si) {
        for (int idx : species[si].member_indices)
            sp_fitness[si] += genomes[idx].adj_fitness;
        total_adj += sp_fitness[si];
    }

    std::vector<int> offspring_count(species.size(), 0);
    if (total_adj > 0.0f) {
        int allocated = 0;
        for (size_t si = 0; si < species.size(); ++si) {
            offspring_count[si] = (int)std::round(
                sp_fitness[si] / total_adj * cfg.pop_size);
            allocated += offspring_count[si];
        }
        // Clamp to pop_size
        while (allocated > cfg.pop_size) {
            int max_si = (int)(std::max_element(offspring_count.begin(),
                                                 offspring_count.end())
                               - offspring_count.begin());
            --offspring_count[max_si]; --allocated;
        }
        while (allocated < cfg.pop_size) {
            int max_si = (int)(std::max_element(offspring_count.begin(),
                                                 offspring_count.end())
                               - offspring_count.begin());
            ++offspring_count[max_si]; ++allocated;
        }
    } else {
        offspring_count[0] = cfg.pop_size;
    }

    // Build next generation
    std::vector<Genome> next_gen;
    next_gen.reserve(cfg.pop_size);

    for (size_t si = 0; si < species.size(); ++si) {
        Species& sp = species[si];
        if (sp.member_indices.empty()) continue;

        int alloc = std::max(0, offspring_count[si]);

        // Sort members by fitness descending
        std::sort(sp.member_indices.begin(), sp.member_indices.end(),
            [&](int a, int b){ return genomes[a].fitness > genomes[b].fitness; });

        // Elitism: carry over top fraction unchanged
        int elite = std::max(1, (int)(sp.member_indices.size() * cfg.elitism_fraction));
        elite     = std::min(elite, alloc);
        for (int e = 0; e < elite; ++e)
            next_gen.push_back(genomes[sp.member_indices[e]]);

        // Cull low performers
        int keep = std::max(1, (int)(sp.member_indices.size() * cfg.survival_threshold));
        sp.member_indices.resize(keep);

        // Fill remaining slots with offspring
        for (int o = elite; o < alloc; ++o)
            next_gen.push_back(make_offspring(sp, rng));
    }

    // Pad if rounding left us short
    while ((int)next_gen.size() < cfg.pop_size) {
        const Species& sp = species[rng() % species.size()];
        next_gen.push_back(make_offspring(sp, rng));
    }
    next_gen.resize(cfg.pop_size);

    genomes = std::move(next_gen);
}

// ── Population serialization ──────────────────────────────────────────────────

void Population::save_all(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path + " for writing");
    f << "NEAT_POPULATION\n";
    f << "generation " << generation << "\n";
    f << "count " << genomes.size() << "\n";
    for (const auto& g : genomes) {
        f << "NEAT_GENOME\n";
        f << "id "      << g.id      << "\n";
        f << "fitness " << g.fitness << "\n";
        f << "nodes "   << g.nodes.size() << "\n";
        for (const auto& n : g.nodes)
            f << n.id << " " << (int)n.type << " " << n.bias << "\n";
        f << "conns " << g.conns.size() << "\n";
        for (const auto& c : g.conns)
            f << c.in_node  << " "
              << c.out_node << " "
              << c.weight   << " "
              << (int)c.enabled << " "
              << c.innov    << "\n";
    }
}

std::vector<Genome> Population::load_all(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::string tag;
    f >> tag;
    if (tag != "NEAT_POPULATION") throw std::runtime_error("Bad population file");
    int count;
    // generation line is optional for backwards compatibility
    f >> tag;
    if (tag == "generation") {
        // (generation number is reconstructed in from_file, not used here)
        f >> tag; // consume the value
        f >> tag; // now tag should be "count"
    }
    f >> count;
    std::vector<Genome> result;
    result.reserve(count);
    for (int gi = 0; gi < count; ++gi) {
        f >> tag;
        if (tag != "NEAT_GENOME") throw std::runtime_error("Expected NEAT_GENOME");
        Genome g;
        f >> tag >> g.id;
        f >> tag >> g.fitness;
        int n_nodes;
        f >> tag >> n_nodes;
        g.nodes.resize(n_nodes);
        for (auto& n : g.nodes) {
            int t; f >> n.id >> t >> n.bias;
            n.type = static_cast<NodeType>(t);
        }
        int n_conns;
        f >> tag >> n_conns;
        g.conns.resize(n_conns);
        for (auto& c : g.conns) {
            int en; f >> c.in_node >> c.out_node >> c.weight >> en >> c.innov;
            c.enabled = en != 0;
        }
        result.push_back(std::move(g));
    }
    return result;
}
