#include "fitness.hpp"
#include "neat/network.hpp"
#include <algorithm>
#include <cmath>
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
// FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
// This down-weights easy negatives and focuses on hard examples

static float focal_loss(float target, float pred, float alpha = 0.75f,
                        float gamma = 2.0f) {
    pred = std::clamp(pred, 1e-7f, 1.0f - 1e-7f);
    float p_t = target * pred + (1.0f - target) * (1.0f - pred);
    float alpha_t = target * alpha + (1.0f - target) * (1.0f - alpha);
    return -alpha_t * std::pow(1.0f - p_t, gamma) * std::log(p_t);
}

// ── Frame-level fitness component ─────────────────────────────────────────────
// Evaluates how well the network's raw outputs match the target piano roll

struct FrameMetrics {
    float onset_loss = 0.0f;
    float frame_loss = 0.0f;
    int   n_frames = 0;
};

static FrameMetrics compute_frame_metrics(
    const std::vector<std::vector<float>>& outputs,   // [frame][pitch]
    const std::vector<std::vector<float>>& onset_tgt,
    const std::vector<std::vector<float>>& frame_tgt,
    int start_frame, int end_frame) {

    FrameMetrics m;
    int n_pitches = (int)onset_tgt[0].size();

    for (int f = start_frame; f < end_frame; ++f) {
        for (int p = 0; p < n_pitches; ++p) {
            float pred = outputs[f - start_frame][p];
            float onset_target = onset_tgt[f][p];
            float frame_target = frame_tgt[f][p];

            // Onset loss: heavily weight positives (alpha=0.9)
            m.onset_loss += focal_loss(onset_target, pred, 0.9f, 2.0f);

            // Frame loss: standard focal
            m.frame_loss += focal_loss(frame_target, pred, 0.75f, 2.0f);
        }
        m.n_frames++;
    }

    return m;
}

// ── Note-level F1 with improved matching ──────────────────────────────────────

struct MatchCandidate {
    size_t truth_idx;
    size_t det_idx;
    float  distance;
};

float note_f1(const std::vector<NoteEvent>& truth,
              const std::vector<DetectedNote>& detected,
              float onset_tol_secs) {
    if (truth.empty() && detected.empty()) return 1.0f;
    if (truth.empty() || detected.empty()) return 0.0f;

    // Build all valid match candidates sorted by distance
    std::vector<MatchCandidate> candidates;
    for (size_t i = 0; i < truth.size(); ++i) {
        const auto& gt = truth[i];
        float win_lo = gt.time - onset_tol_secs;
        float win_hi = gt.time + gt.duration + onset_tol_secs;

        for (size_t j = 0; j < detected.size(); ++j) {
            if (detected[j].midi != gt.midi) continue;
            float t = detected[j].time;
            if (t >= win_lo && t <= win_hi) {
                float dist = std::abs(t - gt.time);
                candidates.push_back({i, j, dist});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.distance < b.distance; });

    std::vector<bool> truth_matched(truth.size(), false);
    std::vector<bool> det_matched(detected.size(), false);
    int tp = 0;

    for (const auto& c : candidates) {
        if (truth_matched[c.truth_idx] || det_matched[c.det_idx]) continue;
        truth_matched[c.truth_idx] = true;
        det_matched[c.det_idx] = true;
        ++tp;
    }

    float precision = (float)tp / (float)detected.size();
    float recall = (float)tp / (float)truth.size();
    if (precision + recall < 1e-9f) return 0.0f;
    return 2.0f * precision * recall / (precision + recall);
}

// ── Main fitness function ─────────────────────────────────────────────────────

float evaluate_genome(const Genome& g,
                      const std::vector<RecordingFrames>& data,
                      const NeatConfig& cfg,
                      std::mt19937& rng,
                      float window_secs,
                      float threshold,
                      int cooldown_frames) {
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

    // Run network and collect both frame outputs and detected notes
    Network net = Network::from_genome(g, cfg);
    std::vector<bool> was_active(n_out, false);
    std::vector<int> cooldown(n_out, 0);
    std::vector<DetectedNote> all_detected;
    std::vector<NoteEvent> all_truth;

    float total_onset_loss = 0.0f;
    float total_frame_loss = 0.0f;
    int total_frames = 0;

    float time_offset = 0.0f;
    for (const auto& seg : segments) {
        const auto& rf = data[seg.rec_idx];
        float seg_start_time = seg.start * rf.hop_secs;
        float seg_end_time = seg.end * rf.hop_secs;

        // Collect ground truth notes for this segment
        for (const auto& n : rf.notes) {
            if (n.time >= seg_start_time && n.time < seg_end_time) {
                all_truth.push_back({n.midi, n.time - seg_start_time + time_offset,
                                     n.duration});
            }
        }

        net.reset();
        std::fill(was_active.begin(), was_active.end(), false);
        std::fill(cooldown.begin(), cooldown.end(), 0);

        // Collect outputs for frame-level evaluation
        std::vector<std::vector<float>> seg_outputs;

        for (int fi = seg.start; fi < seg.end; ++fi) {
            auto inp = build_input(rf.frames, fi, cfg.pitch_history);
            auto out = net.activate(inp);
            seg_outputs.push_back(out);

            float frame_time = (fi - seg.start) * rf.hop_secs + time_offset;

            // Onset detection with cooldown
            for (int k = 0; k < n_out; ++k) {
                bool active = (out[k] >= threshold);
                if (cooldown[k] > 0) {
                    --cooldown[k];
                } else if (active && !was_active[k]) {
                    all_detected.push_back({cfg.midi_min + k, frame_time});
                    cooldown[k] = cooldown_frames;
                }
                was_active[k] = active;
            }
        }

        // Compute frame-level losses
        auto metrics = compute_frame_metrics(seg_outputs, rf.onset_targets,
                                             rf.frame_targets, seg.start, seg.end);
        total_onset_loss += metrics.onset_loss;
        total_frame_loss += metrics.frame_loss;
        total_frames += metrics.n_frames;

        time_offset += (seg.end - seg.start) * rf.hop_secs;
    }

    // === COMBINE FITNESS COMPONENTS ===

    // 1. Frame-level: low loss = good
    float avg_onset_loss = total_onset_loss / (total_frames * n_out);
    float avg_frame_loss = total_frame_loss / (total_frames * n_out);
    float frame_score = std::exp(-0.5f * (avg_onset_loss + avg_frame_loss));

    // 2. Note-level F1
    float f1 = note_f1(all_truth, all_detected, 0.05f);

    // 3. Sparsity penalty: severely punish over-detection
    float sparsity_penalty = 1.0f;
    if (!all_truth.empty() && !all_detected.empty()) {
        float ratio = (float)all_detected.size() / (float)all_truth.size();
        if (ratio > 1.5f) {
            // Quadratic penalty for over-detection
            sparsity_penalty = 1.0f / (1.0f + 0.5f * (ratio - 1.5f) * (ratio - 1.5f));
        }
    }

    // 4. Silence penalty: if network does nothing when notes exist
    if (all_detected.empty() && !all_truth.empty()) {
        return 0.01f;  // Very low but non-zero to maintain gradient
    }

    // Weighted combination:
    // - frame_score: ensures network learns note patterns
    // - f1: ensures correct onset detection
    // - sparsity: prevents spam
    float fitness = (0.3f * frame_score + 0.7f * f1) * sparsity_penalty;

    return std::max(0.01f, fitness);
}
