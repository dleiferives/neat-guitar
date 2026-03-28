#pragma once
#include <random>
#include <vector>
#include "neat/config.hpp"
#include "neat/genome.hpp"
#include "audio/dataset.hpp"
#include "audio/processing.hpp"

// Precomputed frames per recording (avoids recomputing FFT every genome eval)
struct RecordingFrames {
    std::string              name;
    std::vector<AudioFrame>  frames;
    std::vector<NoteEvent>   notes;       // ground truth, sorted by time
    int                      midi_min;
    int                      midi_max;
    float                    hop_secs;    // hop_size / sample_rate
};

// A note onset detected by the network (no duration — network produces onsets only)
struct DetectedNote {
    int   midi;
    float time;  // onset in seconds
};

std::vector<RecordingFrames> precompute_frames(const std::vector<Recording>& recs,
                                                const NeatConfig& cfg);

// Tolerance-window F1.
// A detected note matches a ground truth note with the same MIDI pitch if
// the detection time falls within [gt.time - tol, gt.time + gt.duration + tol].
// Matching is greedy one-to-one (each event can match at most once).
float note_f1(const std::vector<NoteEvent>&   truth,
              const std::vector<DetectedNote>& detected,
              float onset_tol_secs = 0.05f);

// Evaluate a genome against all recordings.
// Each recording is trimmed to a random window of `window_secs` seconds.
// Returns fitness in [0, 1].
float evaluate_genome(const Genome& g,
                      const std::vector<RecordingFrames>& data,
                      const NeatConfig& cfg,
                      std::mt19937& rng,
                      float window_secs         = 100.0f,
                      float activation_threshold = 0.5f,
                      int   cooldown_frames      = 5);
