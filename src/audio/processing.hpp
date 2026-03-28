#pragma once
#include <vector>

// One analysis frame: FFT magnitude bins + dominant MIDI pitch estimate
struct AudioFrame {
    std::vector<float> fft_bins;   // n_fft_bins values, normalized [0, 1]
    float              pitch_midi; // dominant frequency as MIDI note, 0 = silence
};

// Compute FFT magnitude spectrum for one hop of 'fft_size' samples.
// Returns fft_size/2 magnitude bins, peak-normalised.
// Uses single-precision fftw3f internally.
std::vector<float> compute_fft_magnitudes(const float* samples, int fft_size);

// Estimate dominant MIDI pitch from magnitude spectrum.
// bin_hz = sample_rate / fft_size
float dominant_pitch_midi(const std::vector<float>& mags, float bin_hz,
                           float midi_min = 40.0f, float midi_max = 88.0f);

// Slice an entire recording into overlapping frames.
// hop_size: samples between successive frames
// fft_size: analysis window length (must be power of 2, >= hop_size)
std::vector<AudioFrame> extract_frames(const std::vector<float>& audio,
                                        int sample_rate,
                                        int fft_size,
                                        int hop_size,
                                        int n_fft_bins);

// Build a single NEAT input vector from a frame window.
// Appends the last 'pitch_history' dominant pitch values (normalised) after fft_bins.
std::vector<float> build_input(const std::vector<AudioFrame>& frames,
                                int frame_idx,
                                int pitch_history);
