// fitness.hpp
#pragma once
#include <random>
#include <vector>
#include "neat/config.hpp"
#include "neat/genome.hpp"
#include "audio/dataset.hpp"
#include "audio/processing.hpp"

struct RecordingFrames {
    std::string              name;
    std::vector<AudioFrame>  frames;
    std::vector<NoteEvent>   notes;
    int                      midi_min;
    int                      midi_max;
    float                    hop_secs;

    // Precomputed frame-level targets
    std::vector<std::vector<float>> onset_targets;  // [frame][pitch] soft onset
    std::vector<std::vector<float>> frame_targets;  // [frame][pitch] note active
};

struct DetectedNote {
    int   midi;
    float time;
};

std::vector<RecordingFrames> precompute_frames(const std::vector<Recording>& recs,
                                                const NeatConfig& cfg);

float note_f1(const std::vector<NoteEvent>& truth,
              const std::vector<DetectedNote>& detected,
              float onset_tol_secs = 0.05f);

float evaluate_genome(const Genome& g,
                      const std::vector<RecordingFrames>& data,
                      const NeatConfig& cfg,
                      std::mt19937& rng,
                      float window_secs = 100.0f,
                      float threshold = 0.5f,
                      int cooldown_frames = 5);
