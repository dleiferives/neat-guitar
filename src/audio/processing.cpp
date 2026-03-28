#include "processing.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fftw3.h>

// ── FFT ──────────────────────────────────────────────────────────────────────
// Reusable FFT context to avoid per-frame plan creation overhead.
struct FftContext {
    int            fft_size = 0;
    float*         in       = nullptr;
    fftwf_complex* out      = nullptr;
    fftwf_plan     plan     = nullptr;
    std::vector<float> window;

    void init(int size) {
        if (fft_size == size) return;
        destroy();
        fft_size = size;
        in  = fftwf_alloc_real(size);
        out = fftwf_alloc_complex(size / 2 + 1);
        plan = fftwf_plan_dft_r2c_1d(size, in, out, FFTW_MEASURE);
        window.resize(size);
        for (int i = 0; i < size; ++i)
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    }
    void destroy() {
        if (plan) { fftwf_destroy_plan(plan); plan = nullptr; }
        if (in)   { fftwf_free(in);  in  = nullptr; }
        if (out)  { fftwf_free(out); out = nullptr; }
        fft_size = 0;
    }
    ~FftContext() { destroy(); }
};

static FftContext g_fft_ctx;

std::vector<float> compute_fft_magnitudes(const float* samples, int fft_size) {
    g_fft_ctx.init(fft_size);
    int n_bins = fft_size / 2;

    for (int i = 0; i < fft_size; ++i)
        g_fft_ctx.in[i] = samples[i] * g_fft_ctx.window[i];

    fftwf_execute(g_fft_ctx.plan);

    std::vector<float> mags(n_bins);
    float peak = 1e-9f;
    for (int i = 0; i < n_bins; ++i) {
        float re = g_fft_ctx.out[i][0];
        float im = g_fft_ctx.out[i][1];
        mags[i]  = std::sqrt(re * re + im * im);
        peak     = std::max(peak, mags[i]);
    }
    for (auto& v : mags) v /= peak;
    return mags;
}

// ── Pitch estimate ────────────────────────────────────────────────────────────

float dominant_pitch_midi(const std::vector<float>& mags, float bin_hz,
                           float midi_min, float midi_max) {
    // Find the highest-magnitude bin within the guitar frequency range
    // E2 (40) ≈ 82.4 Hz,  E6 (88) ≈ 1318.5 Hz
    float freq_min = 440.0f * std::pow(2.0f, (midi_min - 69.0f) / 12.0f);
    float freq_max = 440.0f * std::pow(2.0f, (midi_max - 69.0f) / 12.0f);

    int bin_lo = std::max(1, (int)std::floor(freq_min / bin_hz));
    int bin_hi = std::min((int)mags.size() - 1, (int)std::ceil(freq_max / bin_hz));

    int   best_bin = bin_lo;
    float best_mag = 0.0f;
    for (int b = bin_lo; b <= bin_hi; ++b) {
        if (mags[b] > best_mag) { best_mag = mags[b]; best_bin = b; }
    }

    if (best_mag < 0.01f) return 0.0f;  // silence

    float freq_hz = best_bin * bin_hz;
    return 69.0f + 12.0f * std::log2(freq_hz / 440.0f);
}

// ── Frame extraction ─────────────────────────────────────────────────────────

std::vector<AudioFrame> extract_frames(const std::vector<float>& audio,
                                        int sample_rate,
                                        int fft_size,
                                        int hop_size,
                                        int n_fft_bins) {
    float bin_hz = (float)sample_rate / fft_size;
    std::vector<AudioFrame> frames;

    int n_hops = ((int)audio.size() - fft_size) / hop_size;
    if (n_hops <= 0) return frames;

    frames.reserve(n_hops);

    // Pad input so we can always read fft_size samples starting at each hop
    std::vector<float> padded(audio.begin(), audio.end());
    padded.resize(padded.size() + fft_size, 0.0f);

    for (int h = 0; h < n_hops; ++h) {
        const float* ptr = padded.data() + h * hop_size;

        AudioFrame fr;
        fr.fft_bins  = compute_fft_magnitudes(ptr, fft_size);
        // Trim or pad to exactly n_fft_bins
        fr.fft_bins.resize(n_fft_bins, 0.0f);
        fr.pitch_midi = dominant_pitch_midi(fr.fft_bins, bin_hz);
        frames.push_back(std::move(fr));
    }

    return frames;
}

// ── Input builder ─────────────────────────────────────────────────────────────

std::vector<float> build_input(const std::vector<AudioFrame>& frames,
                                int frame_idx,
                                int pitch_history) {
    const AudioFrame& cur = frames[frame_idx];
    std::vector<float> inp;
    inp.reserve(cur.fft_bins.size() + pitch_history);

    // FFT magnitudes
    inp.insert(inp.end(), cur.fft_bins.begin(), cur.fft_bins.end());

    // Recent pitch stream: normalised MIDI/127 (0 = silence)
    for (int k = pitch_history - 1; k >= 0; --k) {
        int fi = frame_idx - k;
        float p = (fi >= 0) ? frames[fi].pitch_midi / 127.0f : 0.0f;
        inp.push_back(std::clamp(p, 0.0f, 1.0f));
    }

    return inp;
}
