#pragma once
#include <vector>

// CQT parameters (must match processing.cpp internals)
inline constexpr int CQT_BINS_PER_OCTAVE = 12;
inline constexpr int CQT_N_OCTAVES       = 9;
inline constexpr int CQT_TOTAL_BINS      = CQT_BINS_PER_OCTAVE * CQT_N_OCTAVES;  // 108

// Salience covers the full guitar MIDI range
inline constexpr int SALIENCE_MIDI_MIN = 40;  // E2
inline constexpr int SALIENCE_MIDI_MAX = 88;  // E6
inline constexpr int N_SALIENCE_BINS   = SALIENCE_MIDI_MAX - SALIENCE_MIDI_MIN + 1;  // 49

// One analysis frame
struct AudioFrame {
    std::vector<float> cqt_bins;      // CQT_TOTAL_BINS values, peak-normalised [0, 1]
    std::vector<float> salience;      // N_SALIENCE_BINS harmonic-salience values, [0, 1]
    float              peak_salience; // max(salience), used as temporal history signal
};

// Slice an entire recording into overlapping frames using the Constant-Q Transform.
// hop_size: samples between successive frames
std::vector<AudioFrame> extract_frames(const std::vector<float>& audio,
                                        int sample_rate,
                                        int hop_size);

// Build a single NEAT input vector: cqt_bins + salience + recent peak_salience history.
std::vector<float> build_input(const std::vector<AudioFrame>& frames,
                                int frame_idx,
                                int pitch_history);

// Zero-allocation variant: writes directly into caller-provided buffer.
// out must point to at least (CQT_TOTAL_BINS + N_SALIENCE_BINS + pitch_history) floats.
void build_input(const std::vector<AudioFrame>& frames,
                 int frame_idx,
                 int pitch_history,
                 float* __restrict__ out) noexcept;
