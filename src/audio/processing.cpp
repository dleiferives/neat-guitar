#include "processing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include "SlidingCqt.h"

static constexpr int    BINS_PER_OCTAVE = CQT_BINS_PER_OCTAVE;
static constexpr int    N_OCTAVES       = CQT_N_OCTAVES;
static constexpr int    N_TOTAL_BINS    = CQT_TOTAL_BINS;
static constexpr double CONCERT_PITCH   = 440.0;

// ── Salience map ──────────────────────────────────────────────────────────────
// For each of the 49 guitar semitones (MIDI 40–88), stores the CQT bin indices
// for harmonics 1–5 (−1 if the harmonic falls outside the CQT frequency range).
// Weights follow the natural harmonic decay: 1, 1/2, 1/4, 1/8, 1/16.

static constexpr float HARM_WEIGHTS[5] = {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f};

struct SalienceMap {
    std::array<std::array<int, 5>, N_SALIENCE_BINS> harm_bins;

    static int closest_bin(const std::vector<double>& freq_table, double target) {
        int    best      = 0;
        double best_dist = 1e18;
        for (int i = 0; i < static_cast<int>(freq_table.size()); ++i) {
            if (freq_table[i] <= 0.0) continue;
            double d = std::abs(std::log2(freq_table[i] / target));
            if (d < best_dist) { best_dist = d; best = i; }
        }
        return best;
    }

    void build(const std::vector<double>& freq_table) {
        for (int k = 0; k < N_SALIENCE_BINS; ++k) {
            double fund = 440.0 * std::pow(2.0, (SALIENCE_MIDI_MIN + k - 69) / 12.0);
            for (int h = 0; h < 5; ++h) {
                double target = fund * (h + 1);
                int    bin    = closest_bin(freq_table, target);
                // Accept if within 50 cents of the target frequency
                double cents = std::abs(1200.0 * std::log2(freq_table[bin] / target));
                harm_bins[k][h] = (cents < 50.0) ? bin : -1;
            }
        }
    }
};

// ── CQT context ───────────────────────────────────────────────────────────────

struct CqtContext {
    std::unique_ptr<Cqt::SlidingCqt<BINS_PER_OCTAVE, N_OCTAVES>> cqt;
    std::vector<double> freq_table;
    SalienceMap         sal_map;

    void init(int sample_rate, int block_size) {
        cqt = std::make_unique<Cqt::SlidingCqt<BINS_PER_OCTAVE, N_OCTAVES>>();
        cqt->init(static_cast<double>(sample_rate), block_size);
        cqt->setConcertPitch(CONCERT_PITCH);
        cqt->recalculateKernels();

        freq_table.resize(N_TOTAL_BINS);
        for (int oct = 0; oct < N_OCTAVES; ++oct) {
            double* freqs = cqt->getOctaveBinFreqs(oct);
            for (int b = 0; b < BINS_PER_OCTAVE; ++b)
                freq_table[oct * BINS_PER_OCTAVE + b] = freqs[b];
        }

        sal_map.build(freq_table);
    }
};

// ── Per-frame salience ────────────────────────────────────────────────────────

static std::vector<float> compute_salience(const std::vector<float>& cqt_bins,
                                            const SalienceMap&        sal_map) {
    std::vector<float> sal(N_SALIENCE_BINS, 0.0f);
    for (int k = 0; k < N_SALIENCE_BINS; ++k) {
        float s = 0.0f;
        for (int h = 0; h < 5; ++h) {
            int bin = sal_map.harm_bins[k][h];
            if (bin >= 0)
                s += HARM_WEIGHTS[h] * cqt_bins[bin];
        }
        sal[k] = s;
    }
    float peak = *std::max_element(sal.begin(), sal.end());
    if (peak > 1e-9f)
        for (auto& v : sal) v /= peak;
    return sal;
}

// ── Frame extraction ──────────────────────────────────────────────────────────

std::vector<AudioFrame> extract_frames(const std::vector<float>& audio,
                                        int sample_rate,
                                        int hop_size) {
    CqtContext ctx;
    ctx.init(sample_rate, hop_size);

    int n_hops = static_cast<int>(audio.size()) / hop_size;
    std::vector<AudioFrame> frames;
    frames.reserve(n_hops);

    std::vector<double>               audio_double(hop_size);
    std::vector<std::complex<double>> cqt_buf;

    for (int h = 0; h < n_hops; ++h) {
        const float* ptr = audio.data() + h * hop_size;
        for (int i = 0; i < hop_size; ++i)
            audio_double[i] = ptr[i];
        ctx.cqt->inputBlock(audio_double.data(), hop_size);

        // Extract per-bin RMS magnitude for all octaves
        std::vector<float> cqt_bins(N_TOTAL_BINS, 0.0f);
        for (int oct = 0; oct < N_OCTAVES; ++oct) {
            int n_samp = ctx.cqt->getSamplesToProcess(oct);
            if (n_samp <= 0) continue;
            cqt_buf.resize(n_samp);
            for (int b = 0; b < BINS_PER_OCTAVE; ++b) {
                ctx.cqt->pullBinCqtData(oct, b, cqt_buf.data());
                double sum_sq = 0.0;
                for (int s = 0; s < n_samp; ++s)
                    sum_sq += std::norm(cqt_buf[s]);
                cqt_bins[oct * BINS_PER_OCTAVE + b] =
                    static_cast<float>(std::sqrt(sum_sq / n_samp));
            }
        }
        float peak = *std::max_element(cqt_bins.begin(), cqt_bins.end());
        if (peak > 1e-9f)
            for (auto& v : cqt_bins) v /= peak;

        auto sal = compute_salience(cqt_bins, ctx.sal_map);
        float peak_sal = *std::max_element(sal.begin(), sal.end());

        AudioFrame fr;
        fr.cqt_bins      = std::move(cqt_bins);
        fr.salience      = std::move(sal);
        fr.peak_salience = peak_sal;
        frames.push_back(std::move(fr));
    }

    return frames;
}

// ── Input builder ─────────────────────────────────────────────────────────────

void build_input(const std::vector<AudioFrame>& frames,
                 int frame_idx,
                 int pitch_history,
                 float* __restrict__ out) noexcept {
    const AudioFrame& cur = frames[frame_idx];
    float* p = out;

    const float* cqt = cur.cqt_bins.data();
    int n_cqt = (int)cur.cqt_bins.size();
    for (int i = 0; i < n_cqt; ++i) p[i] = cqt[i];
    p += n_cqt;

    const float* sal = cur.salience.data();
    int n_sal = (int)cur.salience.size();
    for (int i = 0; i < n_sal; ++i) p[i] = sal[i];
    p += n_sal;

    for (int k = pitch_history - 1; k >= 0; --k) {
        int fi = frame_idx - k;
        *p++ = (fi >= 0) ? frames[fi].peak_salience : 0.0f;
    }
}

std::vector<float> build_input(const std::vector<AudioFrame>& frames,
                                int frame_idx,
                                int pitch_history) {
    const AudioFrame& cur = frames[frame_idx];
    std::vector<float> inp(cur.cqt_bins.size() + cur.salience.size() + pitch_history);
    build_input(frames, frame_idx, pitch_history, inp.data());
    return inp;
}
