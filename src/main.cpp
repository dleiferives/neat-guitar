#include <algorithm>
#include <chrono>
#include <cstdio>
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
        << "  --save-pop <path>    Save entire final population to path\n"
        << "  --save-pop-every <n> Also save population every N generations\n"
        << "  --load-pop <path>    Resume from a saved population file\n"
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
    std::string save_pop_path;
    std::string load_pop_path;
    int save_pop_every = 0;
    DataLoadOptions load_opts;
    int gen_override = -1;
    int pop_override = -1;

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--save" && i + 1 < args.size()) {
            save_path = args[++i];
        } else if (args[i] == "--save-pop" && i + 1 < args.size()) {
            save_pop_path = args[++i];
        } else if (args[i] == "--load-pop" && i + 1 < args.size()) {
            load_pop_path = args[++i];
        } else if (args[i] == "--save-pop-every" && i + 1 < args.size()) {
            save_pop_every = std::stoi(args[++i]);
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
    Population pop = load_pop_path.empty()
        ? Population(cfg, 42)
        : Population::from_file(load_pop_path, cfg, 42);

    std::cout << "Population: " << cfg.pop_size
              << "  Generations: " << cfg.generations;
    if (!load_pop_path.empty())
        std::cout << "  (resuming from gen " << pop.generation << ")";
    std::cout << "\n\n";
    int gen_counter = pop.generation;

    auto fit_fn = [&](const Genome& g) {
        // Same seed for all genomes in a generation = fair comparison.
        // Different seed each generation = diverse evaluation over time.
        std::mt19937 eval_rng(42 + gen_counter);
        return evaluate_genome(g, data, cfg, eval_rng);
    };

    float best_val_ever = 0.0f;
    Genome best_val_genome;

    auto on_gen = [&](int gen, float best_fit, const Genome& best) {
        ++gen_counter;
        // Evaluate best on fixed validation set (seed 0, always the same segments)
        std::mt19937 val_rng(0);
        float val_fit = evaluate_genome(best, data, cfg, val_rng);
        if (val_fit > best_val_ever) {
            best_val_ever = val_fit;
            best_val_genome = best;
            best_val_genome.save(save_path);
            std::cout << "  [saved -> " << save_path << " (new best val)]\n";
        }
        std::cout << "Gen " << gen
                  << "  best=" << best_fit
                  << "  val=" << val_fit
                  << "  best_val=" << best_val_ever
                  << "  species=" << pop.species.size()
                  << "  nodes=" << best.nodes.size()
                  << "  conns=" << best.conns.size()
                  << "\n";
        if (!save_pop_path.empty() && save_pop_every > 0 && gen % save_pop_every == 0) {
            pop.save_all(save_pop_path);
            std::cout << "  [pop saved -> " << save_pop_path << "]\n";
        }
    };

    auto t0 = std::chrono::steady_clock::now();
    pop.evolve(fit_fn, on_gen);
    auto t1 = std::chrono::steady_clock::now();

    // Save best-on-validation genome (already saved incrementally, but ensure final save)
    if (best_val_ever > 0.0f)
        best_val_genome.save(save_path);

    if (!save_pop_path.empty()) {
        pop.save_all(save_pop_path);
        std::cout << "Population saved to: " << save_pop_path << "\n";
    }

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nDone in " << secs << "s\n";
    std::cout << "Best val fitness: " << best_val_ever << "\n";
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
    int n_out = cfg.n_outputs();

    int total_tp = 0, total_fp = 0, total_fn = 0;

    for (const auto& rf : data) {
        Network net = Network::from_genome(g, cfg);
        net.reset();

        int tp = 0, fp = 0, fn = 0;

        for (int fi = 0; fi < (int)rf.frames.size(); ++fi) {
            auto inp = build_input(rf.frames, fi, cfg.pitch_history);
            auto out = net.activate(inp);
            for (int k = 0; k < n_out; ++k) {
                bool predicted = (out[k] >= threshold);
                bool target    = (rf.frame_targets[fi][k] >= 0.5f);
                tp += ( predicted &&  target);
                fp += ( predicted && !target);
                fn += (!predicted &&  target);
            }
        }

        float prec = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
        float rec  = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
        float f1   = (prec + rec > 0) ? 2.0f * prec * rec / (prec + rec) : 0.0f;

        std::printf("%-40s  f1=%.4f  prec=%.4f  rec=%.4f  tp=%d fp=%d fn=%d\n",
                    rf.name.c_str(), f1, prec, rec, tp, fp, fn);

        total_tp += tp; total_fp += fp; total_fn += fn;
    }

    float prec = (total_tp + total_fp > 0) ? (float)total_tp / (total_tp + total_fp) : 0.0f;
    float rec  = (total_tp + total_fn > 0) ? (float)total_tp / (total_tp + total_fn) : 0.0f;
    float f1   = (prec + rec > 0) ? 2.0f * prec * rec / (prec + rec) : 0.0f;

    std::printf("\nOverall:  f1=%.4f  prec=%.4f  rec=%.4f  tp=%d fp=%d fn=%d\n",
                f1, prec, rec, total_tp, total_fp, total_fn);

    std::mt19937 rng(std::random_device{}());
    float fit = evaluate_genome(g, data, cfg, rng, 100.0f, threshold);
    std::printf("Fitness (100s window): %.4f\n", fit);
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
    float hop_secs = (float)cfg.hop_size / (float)rec.sample_rate;
    constexpr float threshold = 0.5f;

    static const char* NOTE_NAMES[] = {
        "C","C#","D","D#","E","F","F#","G","G#","A","A#","B"
    };

    // Track active note ranges
    struct NoteRange { int midi; float start; float end; };
    std::vector<NoteRange> ranges;
    std::vector<bool> active(n_out, false);
    std::vector<float> start_time(n_out, 0.0f);

    for (int fi = 0; fi < (int)frames.size(); ++fi) {
        auto inp = build_input(frames, fi, cfg.pitch_history);
        auto out = net.activate(inp);
        float t = fi * hop_secs;

        for (int k = 0; k < n_out; ++k) {
            bool on = (out[k] >= threshold);
            if (on && !active[k]) {
                start_time[k] = t;
                active[k] = true;
            } else if (!on && active[k]) {
                ranges.push_back({cfg.midi_min + k, start_time[k], t});
                active[k] = false;
            }
        }
    }
    // Close any still-active notes
    float end_t = (float)frames.size() * hop_secs;
    for (int k = 0; k < n_out; ++k) {
        if (active[k])
            ranges.push_back({cfg.midi_min + k, start_time[k], end_t});
    }

    // Sort by start time
    std::sort(ranges.begin(), ranges.end(),
              [](const NoteRange& a, const NoteRange& b) { return a.start < b.start; });

    // Print
    std::cout << "Detected notes (" << ranges.size() << "):\n";
    for (const auto& r : ranges) {
        int note = r.midi % 12;
        int octave = r.midi / 12 - 1;
        std::printf("  %6.2fs - %6.2fs  %s%d (MIDI %d)\n",
                    r.start, r.end, NOTE_NAMES[note], octave, r.midi);
    }

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
