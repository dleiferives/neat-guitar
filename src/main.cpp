// main.cpp
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

#include "audio/dataset.hpp"
#include "fitness.hpp"
#include "neat/config.hpp"
#include "neat/population.hpp"

namespace fs = std::filesystem;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --data <path>       Path to data directory (required)\n"
              << "  --cache <path>      Override cache file location\n"
              << "  --no-cache          Disable caching\n"
              << "  --clear-cache       Delete existing cache and recompute\n"
              << "  --generations <n>   Number of generations (default: config)\n"
              << "  --population <n>    Population size (default: config)\n"
              << "  -h, --help          Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string data_dir;
    std::string cache_override;
    bool use_cache = true;
    bool clear_cache = false;
    int gen_override = -1;
    int pop_override = -1;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--cache") == 0 && i + 1 < argc) {
            cache_override = argv[++i];
        } else if (std::strcmp(argv[i], "--no-cache") == 0) {
            use_cache = false;
        } else if (std::strcmp(argv[i], "--clear-cache") == 0) {
            clear_cache = true;
        } else if (std::strcmp(argv[i], "--generations") == 0 && i + 1 < argc) {
            gen_override = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--population") == 0 && i + 1 < argc) {
            pop_override = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (data_dir.empty()) {
        std::cerr << "Error: --data <path> is required\n";
        print_usage(argv[0]);
        return 1;
    }

    NeatConfig cfg;
    if (gen_override > 0) cfg.generations = gen_override;
    if (pop_override > 0) cfg.pop_size = pop_override;

    // Determine cache path
    fs::path cache_path;
    if (!cache_override.empty()) {
        cache_path = cache_override;
    } else {
        cache_path = fs::path(data_dir) / "frames.cache";
    }

    // Handle --clear-cache
    if (clear_cache && fs::exists(cache_path)) {
        std::cout << "[cache] Removing " << cache_path << "\n";
        fs::remove(cache_path);
    }

    // Load data
    std::vector<RecordingFrames> data;

    if (use_cache && fs::exists(cache_path) && !clear_cache) {
        std::cout << "[cache] Loading from " << cache_path << "...\n";
        if (!load_frames_cache(data, cache_path.string(), cfg)) {
            std::cout << "[cache] Cache invalid, recomputing...\n";
        }
    }

    if (data.empty()) {
        std::cout << "Loading recordings from " << data_dir << "...\n";
        auto recs = load_recordings(data_dir, cfg.sample_rate);
        if (recs.empty()) {
            std::cerr << "No recordings found.\n";
            return 1;
        }

        std::cout << "Precomputing CQT frames...\n";
        data = precompute_frames(recs, cfg);

        if (use_cache) {
            save_frames_cache(data, cache_path.string());
        }
    }

    std::cout << "\nNetwork: " << cfg.n_inputs() << " inputs, "
              << cfg.n_outputs() << " outputs ("
              << cfg.midi_min << "-" << cfg.midi_max << " MIDI)\n";
    std::cout << "Population: " << cfg.pop_size
              << "  Generations: " << cfg.generations << "\n\n";

    Population pop(cfg, 42);
    std::mt19937 rng(std::random_device{}());

    auto fit_fn = [&](const Genome& g) {
        return evaluate_genome(g, data, cfg, rng);
    };

    for (int gen = 0; gen < cfg.generations; ++gen) {
        pop.evaluate(fit_fn);
        auto [best_fitness, best_genome] = pop.get_best();

        std::cout << "Gen " << gen << ": best=" << best_fitness
                  << " species=" << pop.num_species() << "\n";

        pop.evolve();
    }

    return 0;
}
