#include "dataset.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sndfile.h>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

// ── WAV loading ───────────────────────────────────────────────────────────────

static std::vector<float> load_wav_mono(const std::string& path, int& out_sr) {
    SF_INFO info{};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
    if (!sf) throw std::runtime_error("Cannot open " + path + ": " + sf_strerror(nullptr));

    out_sr = info.samplerate;
    sf_count_t total = info.frames * info.channels;
    std::vector<float> buf(total);
    sf_read_float(sf, buf.data(), total);
    sf_close(sf);

    if (info.channels == 1) return buf;

    std::vector<float> mono(info.frames);
    int ch = info.channels;
    for (sf_count_t f = 0; f < info.frames; ++f) {
        float sum = 0.0f;
        for (int c = 0; c < ch; ++c) sum += buf[f * ch + c];
        mono[f] = sum / ch;
    }
    return mono;
}

static std::vector<float> resample_linear(const std::vector<float>& src,
                                           int src_sr, int dst_sr) {
    if (src_sr == dst_sr) return src;
    double ratio = (double)src_sr / dst_sr;
    size_t dst_len = (size_t)(src.size() / ratio);
    std::vector<float> dst(dst_len);
    for (size_t i = 0; i < dst_len; ++i) {
        double pos  = i * ratio;
        size_t lo   = (size_t)pos;
        float  frac = (float)(pos - lo);
        size_t hi   = std::min(lo + 1, src.size() - 1);
        dst[i] = src[lo] * (1.0f - frac) + src[hi] * frac;
    }
    return dst;
}

// ── Directory scan ────────────────────────────────────────────────────────────

std::vector<Recording> load_recordings(const std::string& data_root, int target_sr) {
    fs::path wav_dir  = fs::path(data_root) / "audio_mono-mic";
    fs::path json_dir = fs::path(data_root) / "processed";

    std::vector<Recording> recs;

    for (const auto& entry : fs::directory_iterator(wav_dir)) {
        const std::string filename = entry.path().filename().string();

        // Only process *_mic.wav files
        const std::string suffix = "_mic.wav";
        if (filename.size() <= suffix.size() ||
            filename.substr(filename.size() - suffix.size()) != suffix)
            continue;

        // Strip _mic.wav to get stem (e.g. "00_BN1-129-Eb_comp")
        std::string stem = filename.substr(0, filename.size() - suffix.size());
        fs::path json_path = json_dir / (stem + ".json");

        if (!fs::exists(json_path)) {
            std::cerr << "[skip] " << stem << " — no matching processed JSON\n";
            continue;
        }

        Recording rec;
        rec.name = stem;

        try {
            int file_sr = 0;
            rec.audio   = load_wav_mono(entry.path().string(), file_sr);
            if (file_sr != target_sr)
                rec.audio = resample_linear(rec.audio, file_sr, target_sr);
            rec.sample_rate = target_sr;
        } catch (const std::exception& e) {
            std::cerr << "[skip] " << stem << " (wav): " << e.what() << "\n";
            continue;
        }

        try {
            std::ifstream jf(json_path);
            nlohmann::json j;
            jf >> j;
            for (const auto& n : j.at("notes")) {
                rec.notes.push_back({
                    n.at("midi").get<int>(),
                    n.at("time").get<float>(),
                    n.at("duration").get<float>(),
                });
            }
        } catch (const std::exception& e) {
            std::cerr << "[skip] " << stem << " (json): " << e.what() << "\n";
            continue;
        }

        std::cout << "  " << stem << ": "
                  << rec.audio.size() / target_sr << "s  "
                  << rec.notes.size() << " notes\n";

        recs.push_back(std::move(rec));
    }

    return recs;
}
