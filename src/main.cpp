#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <sndfile.h>

#include "audio/dataset.hpp"
#include "audio/processing.hpp"
#include "fitness.hpp"
#include "neat/config.hpp"
#include "neat/genome.hpp"
#include "neat/network.hpp"
#include "neat/population.hpp"

namespace fs = std::filesystem;

// ── Helpers ──────────────────────────────────────────────────────────────────

static void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " train  <recordings_dir> [options]\n"
        << "  " << argv0 << " eval   <genome.txt> <recordings_dir> [options]\n"
        << "  " << argv0 << " infer  <genome.txt> <audio.wav>\n"
        << "\n"
        << "Train/Eval options:\n"
        << "  --save <path>       Save best genome to path (default: best_genome.txt)\n"
        << "  --cache <path>      Override cache file location\n"
        << "  --no-cache          Disable caching\n"
        << "  --clear-cache       Delete existing cache and recompute\n"
        << "  --generations <n>   Number of generations\n"
        << "  --population <n>    Population size\n";
}

// ── Cache-aware data loading ─────────────────────────────────────────────────

struct DataLoadOptions {
    std::string cache_path;
    bool use_cache = true;
    bool clear_cache = false;
};

static std::vector<RecordingFrames> load_data(const std::string& data_dir,
                                               const NeatConfig& cfg,
                                               const DataLoadOptions& opts) {
    fs::path cache_path;
    if (!opts.cache_path.empty()) {
        cache_path = opts.cache_path;
    } else {
        cache_path = fs::path(data_dir) / "frames.cache";
    }

    // Handle --clear-cache
    if (opts.clear_cache && fs::exists(cache_path)) {
        std::cout << "[cache] Removing " << cache_path << "\n";
        fs::remove(cache_path);
    }

    std::vector<RecordingFrames> data;

    // Try loading from cache
    if (opts.use_cache && fs::exists(cache_path) && !opts.clear_cache) {
        std::cout << "[cache] Loading from " << cache_path << "...\n";
        if (load_frames_cache(data, cache_path.string(), cfg)) {
            return data;
        }
        std::cout << "[cache] Cache invalid, recomputing...\n";
    }

    // Compute fresh
    std::cout << "Loading recordings from " << data_dir << "...\n";
    auto recs = load_recordings(data_dir, cfg.sample_rate);
    if (recs.empty()) {
        return {};
    }

    std::cout << "Precomputing CQT frames...\n";
    data = precompute_frames(recs, cfg);

    // Save cache
    if (opts.use_cache && !data.empty()) {
        save_frames_cache(data, cache_path.string());
    }

    return data;
}

// ── train ────────────────────────────────────────────────────────────────────

static int cmd_train(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "train: need recordings_dir\n";
        return 1;
    }

    std::string rec_dir = args[0];
    std::string save_path = "best_genome.txt";
    DataLoadOptions load_opts;
    int gen_override = -1;
    int pop_override = -1;

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--save" && i + 1 < args.size()) {
            save_path = args[++i];
        } else if (args[i] == "--cache" && i + 1 < args.size()) {
            load_opts.cache_path = args[++i];
        } else if (args[i] == "--no-cache") {
            load_opts.use_cache = false;
        } else if (args[i] == "--clear-cache") {
            load_opts.clear_cache = true;
        } else if (args[i] == "--generations" && i + 1 < args.size()) {
            gen_override = std::stoi(args[++i]);
        } else if (args[i] == "--population" && i + 1 < args.size()) {
            pop_override = std::stoi(args[++i]);
        }
    }

    NeatConfig cfg;
    if (gen_override > 0) cfg.generations = gen_override;
    if (pop_override > 0) cfg.pop_size = pop_override;

    auto data = load_data(rec_dir, cfg, load_opts);
    if (data.empty()) {
        std::cerr << "No recordings found.\n";
        return 1;
    }

    std::cout << "Network: " << cfg.n_inputs() << " inputs, "
              << cfg.n_outputs() << " outputs ("
              << cfg.midi_min << "-" << cfg.midi_max << " MIDI)\n";
    std::cout << "Population: " << cfg.pop_size
              << "  Generations: " << cfg.generations << "\n\n";

    Population pop(cfg, 42);
    std::mt19937 rng(std::random_device{}());

    auto fit_fn = [&](const Genome& g) {
        return evaluate_genome(g, data, cfg, rng);
    };

    auto on_gen = [&](int gen, float best_fit, const Genome& best) {
        std::cout << "Gen " << gen
                  << "  best=" << best_fit
                  << "  species=" << pop.species.size()
                  << "  nodes=" << best.nodes.size()
                  << "  conns=" << best.conns.size()
                  << "\n";
        if (gen % 10 == 0) {
            best.save(save_path);
            std::cout << "  [saved -> " << save_path << "]\n";
        }
    };

    auto t0 = std::chrono::steady_clock::now();
    pop.evolve(fit_fn, on_gen);
    auto t1 = std::chrono::steady_clock::now();

    const Genome& best = pop.best_genome();
    best.save(save_path);

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nDone in " << secs << "s\n";
    std::cout << "Best fitness: " << best.fitness << "\n";
    std::cout << "Saved to: " << save_path << "\n";
    return 0;
}

// ── eval ─────────────────────────────────────────────────────────────────────

static int cmd_eval(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "eval: need <genome.txt> <recordings_dir>\n";
        return 1;
    }

    std::string genome_path = args[0];
    std::string rec_dir = args[1];
    DataLoadOptions load_opts;

    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--cache" && i + 1 < args.size()) {
            load_opts.cache_path = args[++i];
        } else if (args[i] == "--no-cache") {
            load_opts.use_cache = false;
        } else if (args[i] == "--clear-cache") {
            load_opts.clear_cache = true;
        }
    }

    NeatConfig cfg;
    Genome g = Genome::load(genome_path);

    auto data = load_data(rec_dir, cfg, load_opts);
    if (data.empty()) {
        std::cerr << "No recordings.\n";
        return 1;
    }

    constexpr float threshold = 0.5f;
    constexpr int cooldown_frames = 5;

    for (const auto& rf : data) {
        Network net = Network::from_genome(g, cfg);
        net.reset();

        int n_out = cfg.n_outputs();
        std::vector<bool> was_active(n_out, false);
        std::vector<int> cooldown(n_out, 0);
        std::vector<DetectedNote> detected;

        for (int fi = 0; fi < (int)rf.frames.size(); ++fi) {
            auto inp = build_input(rf.frames, fi, cfg.pitch_history);
            auto out = net.activate(inp);
            for (int k = 0; k < n_out; ++k) {
                bool active = (out[k] >= threshold);
                if (cooldown[k] > 0) {
                    --cooldown[k];
                } else if (active && !was_active[k]) {
                    detected.push_back({cfg.midi_min + k, fi * rf.hop_secs});
                    cooldown[k] = cooldown_frames;
                }
                was_active[k] = active;
            }
        }

        float f1 = note_f1(rf.notes, detected);
        std::cout << rf.name << "  f1=" << f1
                  << "  detected=" << detected.size()
                  << "  truth=" << rf.notes.size() << "\n"
                  << "  predicted: [";
        for (size_t i = 0; i < detected.size(); ++i)
            std::cout << detected[i].midi << "@" << detected[i].time
                      << (i + 1 < detected.size() ? "," : "");
        std::cout << "]\n  truth:     [";
        for (size_t i = 0; i < rf.notes.size(); ++i)
            std::cout << rf.notes[i].midi << "@" << rf.notes[i].time
                      << (i + 1 < rf.notes.size() ? "," : "");
        std::cout << "]\n";
    }

    std::mt19937 rng(std::random_device{}());
    float fit = evaluate_genome(g, data, cfg, rng, 100.0f, threshold, cooldown_frames);
    std::cout << "\nOverall F1: " << fit << "\n";
    return 0;
}

// ── infer ────────────────────────────────────────────────────────────────────

static int cmd_infer(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "infer: need <genome.txt> <audio.wav>\n";
        return 1;
    }

    NeatConfig cfg;
    Genome g = Genome::load(args[0]);
    Recording rec;
    rec.name = args[1];

    {
        SF_INFO info{};
        SNDFILE* sf = sf_open(args[1].c_str(), SFM_READ, &info);
        if (!sf) {
            std::cerr << "Cannot open " << args[1] << "\n";
            return 1;
        }
        rec.sample_rate = info.samplerate;
        rec.audio.resize(info.frames * info.channels);
        sf_read_float(sf, rec.audio.data(), rec.audio.size());
        sf_close(sf);
        if (info.channels > 1) {
            int ch = info.channels;
            std::vector<float> mono(info.frames);
            for (int f = 0; f < (int)info.frames; ++f) {
                float s = 0;
                for (int c = 0; c < ch; ++c) s += rec.audio[f * ch + c];
                mono[f] = s / ch;
            }
            rec.audio = std::move(mono);
        }
    }

    auto frames = extract_frames(rec.audio, rec.sample_rate, cfg.hop_size);

    Network net = Network::from_genome(g, cfg);
    net.reset();

    int n_out = cfg.n_outputs();
    std::vector<float> max_act(n_out, 0.0f);
    for (int fi = 0; fi < (int)frames.size(); ++fi) {
        auto inp = build_input(frames, fi, cfg.pitch_history);
        auto out = net.activate(inp);
        for (int k = 0; k < n_out; ++k)
            max_act[k] = std::max(max_act[k], out[k]);
    }

    std::cout << "Detected notes (MIDI): [";
    bool first = true;
    for (int k = 0; k < n_out; ++k) {
        if (max_act[k] >= 0.5f) {
            if (!first) std::cout << ", ";
            std::cout << (cfg.midi_min + k) << " (" << max_act[k] << ")";
            first = false;
        }
    }
    std::cout << "]\n";
    return 0;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    std::string cmd = argv[1];
    std::vector<std::string> args(argv + 2, argv + argc);

    try {
        if (cmd == "train") return cmd_train(args);
        if (cmd == "eval") return cmd_eval(args);
        if (cmd == "infer") return cmd_infer(args);
        std::cerr << "Unknown command: " << cmd << "\n";
        usage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
