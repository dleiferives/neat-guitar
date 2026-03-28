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

// Fitness function: receives a genome, returns a scalar fitness in [0, 1].
using FitnessFn = std::function<float(const Genome&)>;

struct Population {
    NeatConfig           cfg;
    std::vector<Genome>  genomes;
    std::vector<Species> species;
    InnovationTracker    innov;
    std::mt19937         rng;
    int                  generation     = 0;
    int                  next_genome_id = 0;
    int                  next_species_id= 0;

    explicit Population(NeatConfig cfg, uint64_t seed = 42);

    // Run the full evolutionary loop.
    // on_generation called at end of each generation with (gen, best_fitness, best_genome).
    void evolve(FitnessFn fit_fn,
                std::function<void(int, float, const Genome&)> on_generation = nullptr);

    // Run one generation: evaluate → speciate → reproduce.
    void step(FitnessFn fit_fn);

    const Genome& best_genome() const;

private:
    void evaluate(FitnessFn fit_fn);
    void speciate();
    void reproduce();
    void adjust_compat_threshold();

    Genome make_offspring(const Species& sp, std::mt19937& rng);
    int    pick_parent_index(const Species& sp) const;
};
