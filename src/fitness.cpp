#include "fitness.hpp"
#include "neat/network.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

namespace fs = std::filesystem;

// ── Binary I/O helpers ────────────────────────────────────────────────────────

template <typename T>
static void write_pod(std::ostream& os, const T& val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template <typename T>
static void read_pod(std::istream& is, T& val) {
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
}

static void write_string(std::ostream& os, const std::string& s) {
    uint32_t len = (uint32_t)s.size();
    write_pod(os, len);
    os.write(s.data(), len);
}

static void read_string(std::istream& is, std::string& s) {
    uint32_t len;
    read_pod(is, len);
    s.resize(len);
    is.read(s.data(), len);
}

template <typename T>
static void write_vector(std::ostream& os, const std::vector<T>& v) {
    uint64_t size = v.size();
    write_pod(os, size);
    if (!v.empty()) {
        os.write(reinterpret_cast<const char*>(v.data()), size * sizeof(T));
    }
}

template <typename T>
static void read_vector(std::istream& is, std::vector<T>& v) {
    uint64_t size;
    read_pod(is, size);
    v.resize(size);
    if (size > 0) {
        is.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
    }
}

static void write_vector_2d(std::ostream& os,
                            const std::vector<std::vector<float>>& v) {
    uint64_t rows = v.size();
    write_pod(os, rows);
    if (rows == 0) return;
    uint64_t cols = v[0].size();
    write_pod(os, cols);
    for (const auto& row : v) {
        os.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(float));
    }
}

static void read_vector_2d(std::istream& is,
                           std::vector<std::vector<float>>& v) {
    uint64_t rows, cols;
    read_pod(is, rows);
    if (rows == 0) {
        v.clear();
        return;
    }
    read_pod(is, cols);
    v.resize(rows);
    for (auto& row : v) {
        row.resize(cols);
        is.read(reinterpret_cast<char*>(row.data()), cols * sizeof(float));
    }
}

// In fitness.cpp - replace the AudioFrame serialization

static void write_audio_frame(std::ostream& os, const AudioFrame& frame) {
    write_vector(os, frame.cqt_bins);
    write_vector(os, frame.salience);
    write_pod(os, frame.peak_salience);
}

static void read_audio_frame(std::istream& is, AudioFrame& frame) {
    read_vector(is, frame.cqt_bins);
    read_vector(is, frame.salience);
    read_pod(is, frame.peak_salience);
}

bool save_frames_cache(const std::vector<RecordingFrames>& data,
                       const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "[cache] Failed to open " << path << " for writing\n";
        return false;
    }

    write_pod(out, CACHE_MAGIC);
    write_pod(out, CACHE_VERSION);

    uint32_t n_recs = (uint32_t)data.size();
    write_pod(out, n_recs);

    if (!data.empty()) {
        write_pod(out, data[0].midi_min);
        write_pod(out, data[0].midi_max);
        write_pod(out, data[0].hop_secs);
    }

    for (const auto& rf : data) {
        write_string(out, rf.name);
        write_pod(out, rf.midi_min);
        write_pod(out, rf.midi_max);
        write_pod(out, rf.hop_secs);

        // Frames - serialize each AudioFrame properly
        uint64_t n_frames = rf.frames.size();
        write_pod(out, n_frames);
        for (const auto& frame : rf.frames) {
            write_audio_frame(out, frame);
        }

        // Notes
        uint64_t n_notes = rf.notes.size();
        write_pod(out, n_notes);
        for (const auto& n : rf.notes) {
            write_pod(out, n.midi);
            write_pod(out, n.time);
            write_pod(out, n.duration);
        }

        // Targets
        write_vector_2d(out, rf.onset_targets);
        write_vector_2d(out, rf.frame_targets);
    }

    std::cout << "[cache] Saved " << n_recs << " recordings to " << path << "\n";
    return out.good();
}

bool load_frames_cache(std::vector<RecordingFrames>& data,
                       const std::string& path,
                       const NeatConfig& cfg) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    uint32_t magic, version;
    read_pod(in, magic);
    read_pod(in, version);

    if (magic != CACHE_MAGIC) {
        std::cerr << "[cache] Invalid magic number\n";
        return false;
    }
    if (version != CACHE_VERSION) {
        std::cerr << "[cache] Version mismatch (got " << version
                  << ", expected " << CACHE_VERSION << ")\n";
        return false;
    }

    uint32_t n_recs;
    read_pod(in, n_recs);

    if (n_recs > 0) {
        int cached_midi_min, cached_midi_max;
        float cached_hop_secs;
        read_pod(in, cached_midi_min);
        read_pod(in, cached_midi_max);
        read_pod(in, cached_hop_secs);

        float expected_hop = (float)cfg.hop_size / (float)cfg.sample_rate;
        if (cached_midi_min != cfg.midi_min || cached_midi_max != cfg.midi_max ||
            std::abs(cached_hop_secs - expected_hop) > 1e-6f) {
            std::cerr << "[cache] Config mismatch, recomputing\n";
            return false;
        }
    }

    data.clear();
    data.reserve(n_recs);

    for (uint32_t i = 0; i < n_recs; ++i) {
        RecordingFrames rf;
        read_string(in, rf.name);
        read_pod(in, rf.midi_min);
        read_pod(in, rf.midi_max);
        read_pod(in, rf.hop_secs);

        // Frames - deserialize each AudioFrame properly
        uint64_t n_frames;
        read_pod(in, n_frames);
        rf.frames.resize(n_frames);
        for (auto& frame : rf.frames) {
            read_audio_frame(in, frame);
        }

        // Notes
        uint64_t n_notes;
        read_pod(in, n_notes);
        rf.notes.resize(n_notes);
        for (auto& n : rf.notes) {
            read_pod(in, n.midi);
            read_pod(in, n.time);
            read_pod(in, n.duration);
        }

        // Targets
        read_vector_2d(in, rf.onset_targets);
        read_vector_2d(in, rf.frame_targets);

        data.push_back(std::move(rf));
    }

    if (!in.good()) {
        std::cerr << "[cache] Read error\n";
        data.clear();
        return false;
    }

    std::cout << "[cache] Loaded " << n_recs << " recordings from " << path << "\n";
    return true;
}

// ── Frame target building ─────────────────────────────────────────────────────

static void build_frame_targets(RecordingFrames& rf, int n_pitches) {
    int n_frames = (int)rf.frames.size();
    rf.onset_targets.assign(n_frames, std::vector<float>(n_pitches, 0.0f));
    rf.frame_targets.assign(n_frames, std::vector<float>(n_pitches, 0.0f));

    for (const auto& note : rf.notes) {
        int pitch_idx = note.midi - rf.midi_min;
        if (pitch_idx < 0 || pitch_idx >= n_pitches) continue;

        int onset_frame = (int)(note.time / rf.hop_secs);
        int offset_frame = (int)((note.time + note.duration) / rf.hop_secs);

        float sigma = 1.5f;
        for (int f = std::max(0, onset_frame - 3);
             f < std::min(n_frames, onset_frame + 4); ++f) {
            float dist = (float)(f - onset_frame);
            float val = std::exp(-(dist * dist) / (2.0f * sigma * sigma));
            rf.onset_targets[f][pitch_idx] =
                std::max(rf.onset_targets[f][pitch_idx], val);
        }

        for (int f = onset_frame; f <= std::min(offset_frame, n_frames - 1); ++f) {
            rf.frame_targets[f][pitch_idx] = 1.0f;
        }
    }
}

// ── Precompute frames ─────────────────────────────────────────────────────────

std::vector<RecordingFrames> precompute_frames(const std::vector<Recording>& recs,
                                                const NeatConfig& cfg) {
    std::vector<RecordingFrames> out;
    out.reserve(recs.size());
    int n_pitches = cfg.midi_max - cfg.midi_min + 1;

    for (const auto& r : recs) {
        RecordingFrames rf;
        rf.name     = r.name;
        rf.notes    = r.notes;
        rf.midi_min = cfg.midi_min;
        rf.midi_max = cfg.midi_max;
        rf.hop_secs = (float)cfg.hop_size / (float)cfg.sample_rate;
        rf.frames   = extract_frames(r.audio, r.sample_rate, cfg.hop_size);

        build_frame_targets(rf, n_pitches);
        out.push_back(std::move(rf));
    }
    return out;
}

std::vector<RecordingFrames> load_or_compute_frames(const std::string& data_dir,
                                                     const NeatConfig& cfg) {
    fs::path cache_path = fs::path(data_dir) / "frames.cache";
    std::vector<RecordingFrames> data;

    // Try loading cache first
    if (fs::exists(cache_path)) {
        std::cout << "[cache] Found " << cache_path << ", attempting load...\n";
        if (load_frames_cache(data, cache_path.string(), cfg)) {
            return data;
        }
        std::cout << "[cache] Cache invalid, recomputing...\n";
    }

    // Compute fresh
    std::cout << "Loading recordings from " << data_dir << " ...\n";
    auto recs = load_recordings(data_dir, cfg.sample_rate);
    if (recs.empty()) {
        std::cerr << "No recordings found.\n";
        return {};
    }

    std::cout << "Precomputing CQT frames...\n";
    data = precompute_frames(recs, cfg);

    // Save cache
    save_frames_cache(data, cache_path.string());

    return data;
}

// ── Focal loss for fitness (handles class imbalance) ─────────────────────────
// FL(p_t) = -alpha * (1 - p_t)^2 * log(p_t)
// gamma is always 2.0 at all call sites, so pow(x,2) is replaced with x*x.
// log is replaced with a fast IEEE 754 bit-manipulation approximation (~2% error).

static inline float fast_log(float x) {
    union { float f; uint32_t i; } u = {x};
    int e = (int)((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x007FFFFFu) | 0x3F800000u;
    float m = u.f - 1.0f;
    return ((float)e + m * (1.0f + m * (-0.5f + m * 0.333333f))) * 0.693147180f;
}

static inline float focal_loss(float target, float pred, float alpha) {
    pred = std::clamp(pred, 1e-7f, 1.0f - 1e-7f);
    float p_t     = target * pred + (1.0f - target) * (1.0f - pred);
    float alpha_t = target * alpha + (1.0f - target) * (1.0f - alpha);
    float omp     = 1.0f - p_t;
    return -alpha_t * omp * omp * fast_log(p_t);
}

// ── Main fitness function ─────────────────────────────────────────────────────

float evaluate_genome(const Genome& g,
                      const std::vector<RecordingFrames>& data,
                      const NeatConfig& cfg,
                      std::mt19937& rng,
                      float window_secs,
                      float threshold) {
    if (data.empty()) return 0.0f;

    float hop_secs = data[0].hop_secs;
    int target_frames = (int)(window_secs / hop_secs);
    int n_out = cfg.n_outputs();

    // Build segments
    struct Segment { int rec_idx, start, end; };
    std::vector<Segment> segments;
    std::uniform_int_distribution<int> rec_dist(0, (int)data.size() - 1);
    int remaining = target_frames;

    while (remaining > 0) {
        int ri = rec_dist(rng);
        int total = (int)data[ri].frames.size();
        if (total == 0) continue;

        std::uniform_int_distribution<int> frame_dist(0, total - 1);
        int start = frame_dist(rng);
        int use = std::min(total - start, remaining);
        segments.push_back({ri, start, start + use});
        remaining -= use;
    }

    // Run network and accumulate frame-level metrics
    Network net = Network::from_genome(g, cfg);

    // Pre-allocated buffers: reused every frame, zero heap allocs in the hot loop.
    std::vector<float> inp_buf(cfg.n_inputs());
    std::vector<float> out_buf(cfg.n_outputs());

    float total_onset_loss = 0.0f;
    float total_frame_loss = 0.0f;
    int total_frames = 0;

    // Frame-level TP/FP/FN for per-frame "is this note active?" F1
    int tp = 0, fp = 0, fn = 0;

    for (const auto& seg : segments) {
        const auto& rf = data[seg.rec_idx];

        net.reset();

        for (int fi = seg.start; fi < seg.end; ++fi) {
            build_input(rf.frames, fi, cfg.pitch_history, inp_buf.data());
            net.activate(inp_buf.data(), out_buf.data());

            const float* o = out_buf.data();
            const float* on_tgt = rf.onset_targets[fi].data();
            const float* fr_tgt = rf.frame_targets[fi].data();
            for (int p = 0; p < n_out; ++p) {
                total_onset_loss += focal_loss(on_tgt[p], o[p], 0.9f);
                total_frame_loss += focal_loss(fr_tgt[p], o[p], 0.75f);

                bool predicted = (o[p] >= threshold);
                bool target    = (fr_tgt[p] >= 0.5f);
                tp += ( predicted &&  target);
                fp += ( predicted && !target);
                fn += (!predicted &&  target);
            }
            ++total_frames;
        }
    }

    // === COMBINE FITNESS COMPONENTS ===

    // 1. Frame-level loss: low loss = good (confidence calibration via focal loss)
    float avg_onset_loss = total_onset_loss / (total_frames * n_out);
    float avg_frame_loss = total_frame_loss / (total_frames * n_out);
    float frame_score = std::exp(-0.5f * (avg_onset_loss + avg_frame_loss));

    // 2. Frame-level F1: per-frame "is this note active?" precision/recall
    float frame_f1 = 0.0f;
    if (tp + fp + fn > 0)
        frame_f1 = (2.0f * tp) / (2.0f * tp + fp + fn);

    // 3. Complexity penalty: discourage bloat
//  float complexity = (float)(g.nodes.size() + g.conns.size());
//  float parsimony = 1.0f / (1.0f + 0.0002f * complexity);

    float fitness = (0.5f * frame_f1 + 0.5f * frame_score); // * parsimony;

    return std::max(0.01f, fitness);
}

// ── Racing fitness ────────────────────────────────────────────────────────────
// Each frame's contribution to fitness = rolling F1 at that point.
// A model at F1=0.9 earns 0.9/frame; at F1=0.0 earns nothing.
// Distance and quality are naturally unified — can't game one at the other's expense.
// Hard kill at low threshold (0.2) is just a compute optimization.

float racing_theoretical_max(const std::vector<RecordingFrames>& data_sorted) {
    int total_frames = 0;
    for (const auto& rf : data_sorted) total_frames += (int)rf.frames.size();
    int n_files = (int)data_sorted.size();
    // Perfect F1 (1.0) every frame, all files completed, no parsimony penalty
    return (float)total_frames * 1.0f * (1.0f + 0.10f * (float)n_files);
}

// fitness.cpp - update the implementations

RacingResult evaluate_genome_racing_detailed(const Genome& g,
                                              const std::vector<RecordingFrames>& data_sorted,
                                              const NeatConfig& cfg,
                                              int start_file_idx,
                                              float threshold,
                                              float kill_threshold,
                                              float window_secs) {
    RacingResult result{0.01f, 0, 0, 0, (int)data_sorted.size(), 0.0f};
    if (data_sorted.empty()) return result;

    float hop_secs = data_sorted[0].hop_secs;
    int window_frames = std::max(1, (int)(window_secs / hop_secs));
    int n_out = cfg.n_outputs();
    int n_files = (int)data_sorted.size();

    for (const auto& rf : data_sorted) result.total_frames += (int)rf.frames.size();
    if (result.total_frames == 0) return result;

    Network net = Network::from_genome(g, cfg);
    std::vector<float> inp_buf(cfg.n_inputs());
    std::vector<float> out_buf(cfg.n_outputs());

    struct FrameTPFPFN { int tp, fp, fn; };
    std::deque<FrameTPFPFN> win;
    int win_tp = 0, win_fp = 0, win_fn = 0;

    double fitness_accum = 0.0;
    int frames_processed = 0;
    int files_completed = 0;

    for (int file_offset = 0; file_offset < n_files; ++file_offset) {
        int file_idx = (start_file_idx + file_offset) % n_files;
        const auto& rf = data_sorted[file_idx];

        net.reset();
        bool file_alive = true;

        for (int fi = 0; fi < (int)rf.frames.size(); ++fi) {
            build_input(rf.frames, fi, cfg.pitch_history, inp_buf.data());
            net.activate(inp_buf.data(), out_buf.data());

            const float* fr_tgt = rf.frame_targets[fi].data();
            FrameTPFPFN m{0, 0, 0};
            for (int p = 0; p < n_out; ++p) {
                bool predicted = (out_buf[p] >= threshold);
                bool target    = (fr_tgt[p] >= 0.5f);
                m.tp += ( predicted &&  target);
                m.fp += ( predicted && !target);
                m.fn += (!predicted &&  target);
            }

            win.push_back(m);
            win_tp += m.tp; win_fp += m.fp; win_fn += m.fn;
            if ((int)win.size() > window_frames) {
                win_tp -= win.front().tp;
                win_fp -= win.front().fp;
                win_fn -= win.front().fn;
                win.pop_front();
            }

            int denom = 2 * win_tp + win_fp + win_fn;
            float rolling_f1 = (denom > 0)
                ? (float)(2 * win_tp) / (float)denom
                : 1.0f;

            fitness_accum += (double)rolling_f1;
            ++frames_processed;

            if ((int)win.size() == window_frames && rolling_f1 < kill_threshold) {
                file_alive = false;
                break;
            }
        }

        if (!file_alive) break;
        ++files_completed;
    }

    if (frames_processed == 0) return result;

    float fitness = (float)fitness_accum;
    fitness *= (1.0f + 0.10f * (float)files_completed);

    float complexity = (float)(g.nodes.size() + g.conns.size());
    fitness *= 1.0f / (1.0f + 0.0002f * complexity);

    result.fitness          = std::max(0.01f, fitness);
    result.frames_processed = frames_processed;
    result.files_completed  = files_completed;
    result.avg_accuracy     = (float)(fitness_accum / frames_processed);
    return result;
}

float evaluate_genome_racing(const Genome& g,
                              const std::vector<RecordingFrames>& data_sorted,
                              const NeatConfig& cfg,
                              int start_file_idx,
                              float threshold,
                              float kill_threshold,
                              float window_secs) {
    return evaluate_genome_racing_detailed(g, data_sorted, cfg, start_file_idx,
                                           threshold, kill_threshold, window_secs).fitness;
}
