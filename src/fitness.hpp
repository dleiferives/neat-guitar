// fitness.hpp - add cache declarations
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

    std::vector<std::vector<float>> onset_targets;
    std::vector<std::vector<float>> frame_targets;
};

// Cache management
constexpr uint32_t CACHE_MAGIC   = 0x4E454154;  // "NEAT"
constexpr uint32_t CACHE_VERSION = 2;

bool save_frames_cache(const std::vector<RecordingFrames>& data,
                       const std::string& path);

bool load_frames_cache(std::vector<RecordingFrames>& data,
                       const std::string& path,
                       const NeatConfig& cfg);

std::vector<RecordingFrames> precompute_frames(const std::vector<Recording>& recs,
                                                const NeatConfig& cfg);

// Load with automatic caching
std::vector<RecordingFrames> load_or_compute_frames(const std::string& data_dir,
                                                     const NeatConfig& cfg);

float evaluate_genome(const Genome& g,
                      const std::vector<RecordingFrames>& data,
                      const NeatConfig& cfg,
                      std::mt19937& rng,
                      float window_secs = 100.0f,
                      float threshold = 0.5f);
