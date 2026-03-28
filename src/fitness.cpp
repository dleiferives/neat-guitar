#include "fitness.hpp"
#include "neat/network.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

// ── Frame precomputation ──────────────────────────────────────────────────────

std::vector<RecordingFrames> precompute_frames(const std::vector<Recording>& recs,
                                                const NeatConfig& cfg) {
    std::vector<RecordingFrames> out;
    out.reserve(recs.size());
    for (const auto& r : recs) {
        RecordingFrames rf;
        rf.name      = r.name;
        rf.notes     = r.notes;
        rf.midi_min  = cfg.midi_min;
        rf.midi_max  = cfg.midi_max;
        rf.hop_secs  = (float)cfg.hop_size / (float)cfg.sample_rate;
        rf.frames    = extract_frames(r.audio, r.sample_rate,
                                      cfg.fft_size, cfg.hop_size,
                                      cfg.n_fft_bins);
        out.push_back(std::move(rf));
    }
    return out;
}

// ── Tolerance-window F1 ───────────────────────────────────────────────────────
// A detected note (midi, time) matches a ground truth NoteEvent if:
//   detected.midi == gt.midi
//   detected.time in [gt.time - tol, gt.time + gt.duration + tol]
// Greedy one-to-one: each event can participate in at most one match.

float note_f1(const std::vector<NoteEvent>&   truth,
              const std::vector<DetectedNote>& detected,
              float onset_tol_secs) {
    if (truth.empty() && detected.empty()) return 1.0f;
    if (truth.empty() || detected.empty()) return 0.0f;

    std::vector<bool> truth_matched(truth.size(), false);
    std::vector<bool> det_matched(detected.size(), false);

    for (size_t i = 0; i < truth.size(); ++i) {
        const auto& gt   = truth[i];
        float win_lo = gt.time - onset_tol_secs;
        float win_hi = gt.time + gt.duration + onset_tol_secs;

        for (size_t j = 0; j < detected.size(); ++j) {
            if (det_matched[j]) continue;
            if (detected[j].midi != gt.midi) continue;
            if (detected[j].time >= win_lo && detected[j].time <= win_hi) {
                truth_matched[i] = true;
                det_matched[j]   = true;
                break;
            }
        }
    }

    int tp = (int)std::count(truth_matched.begin(), truth_matched.end(), true);
    float precision = (float)tp / (float)detected.size();
    float recall    = (float)tp / (float)truth.size();
    if (precision + recall < 1e-9f) return 0.0f;
    return 2.0f * precision * recall / (precision + recall);
}

// ── Genome fitness ────────────────────────────────────────────────────────────
// Accumulates random segments from random recordings until `window_secs` of
// audio has been processed, then returns a single F1 over all segments.

float evaluate_genome(const Genome& g,
                       const std::vector<RecordingFrames>& data,
                       const NeatConfig& cfg,
                       std::mt19937& rng,
                       float window_secs,
                       float threshold,
                       int   cooldown_frames) {
    if (data.empty()) return 0.0f;

    float hop_secs    = data[0].hop_secs;
    int target_frames = (int)(window_secs / hop_secs);

    // Build segments: pick random recording + random start, repeat until 100s filled
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
        int use   = std::min(total - start, remaining);
        segments.push_back({ri, start, start + use});
        remaining -= use;
    }

    // Run network across all segments, mapping each onto a virtual timeline
    Network net = Network::from_genome(g, cfg);
    int n_out = cfg.n_outputs();
    std::vector<bool> was_active(n_out, false);
    std::vector<int>  cooldown(n_out, 0);
    std::vector<DetectedNote> all_detected;
    std::vector<NoteEvent>    all_truth;

    float time_offset = 0.0f;
    for (const auto& seg : segments) {
        const auto& rf       = data[seg.rec_idx];
        float seg_start_time = seg.start * rf.hop_secs;
        float seg_end_time   = seg.end   * rf.hop_secs;

        for (const auto& n : rf.notes)
            if (n.time >= seg_start_time && n.time < seg_end_time)
                all_truth.push_back({n.midi,
                                     n.time - seg_start_time + time_offset,
                                     n.duration});

        // Reset network and detector state at each segment boundary
        net.reset();
        std::fill(was_active.begin(), was_active.end(), false);
        std::fill(cooldown.begin(), cooldown.end(), 0);

        for (int fi = seg.start; fi < seg.end; ++fi) {
            auto inp = build_input(rf.frames, fi, cfg.pitch_history);
            auto out = net.activate(inp);
            float frame_time = (fi - seg.start) * rf.hop_secs + time_offset;

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

        time_offset += (seg.end - seg.start) * rf.hop_secs;
    }

    return note_f1(all_truth, all_detected);
}
