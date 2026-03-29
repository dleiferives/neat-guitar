#pragma once
#include <functional>
#include <random>
#include <vector>

#include "config.hpp"
#include "genome.hpp"
#include "innovation.hpp"

struct Species {
    int              id;
    Genome           representative;
    std::vector<int> member_indices;  // indices into Population::genomes
    float            best_fitness   = 0.0f;
    int              staleness      = 0;
};

// Fitness function: receives a genome, returns a scalar fitness (unbounded).
using FitnessFn = std::function<float(const Genome&)>;

struct Population {
    NeatConfig           cfg;
    std::vector<Genome>  genomes;
    std::vector<Species> species;
    InnovationTracker    innov;
    std::mt19937         rng;
    int                  generation      = 0;
    int                  next_genome_id  = 0;
    int                  next_species_id = 0;

    // Auto-tuning: stagnation-responsive mutation boost
    float                global_best_fitness = 0.0f;
    int                  global_stagnation   = 0;
    int                  gens_since_rotation = 100;

    explicit Population(NeatConfig cfg, uint64_t seed = 42);

    // Reconstruct a Population from a saved population file.
    // Rebuilds innovation state from the genomes and re-speciates.
    static Population from_file(const std::string& path, NeatConfig cfg, uint64_t seed = 42);

    // Call after rotating the track to enter addition-only mode for 100 gens.
    void signal_rotation() { gens_since_rotation = 0; }

    // Run the full evolutionary loop.
    // on_generation called at end of each generation with (gen, best_fitness, best_genome).
    void evolve(FitnessFn fit_fn,
                std::function<void(int, float, const Genome&)> on_generation = nullptr);

    // Run one generation: evaluate → speciate → reproduce.
    void step(FitnessFn fit_fn);

    const Genome& best_genome() const;

    // Save/load the entire genome vector to a single file.
    void save_all(const std::string& path) const;
    static std::vector<Genome> load_all(const std::string& path);

private:
    Population() = default;

    void evaluate(FitnessFn fit_fn);
    void speciate();
    void reproduce();
    void adjust_compat_threshold();

    Genome make_offspring(const Species& sp, std::mt19937& rng);
    int    pick_parent_index(const Species& sp) const;
};
