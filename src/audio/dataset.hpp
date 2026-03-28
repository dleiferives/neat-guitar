#pragma once
#include <string>
#include <vector>

struct NoteEvent {
    int   midi;      // rounded MIDI pitch
    float time;      // onset in seconds (relative to audio file start)
    float duration;  // duration in seconds
};

struct Recording {
    std::string            name;
    std::vector<float>     audio;       // mono float32 at cfg sample_rate
    std::vector<NoteEvent> notes;       // ground truth note events, sorted by time
    int                    sample_rate;
};

// Load all recordings from data_root.
// Expects:  data_root/audio_mono-mic/<stem>_mic.wav
//           data_root/processed/<stem>.json
// Resamples audio to target_sr if needed.
std::vector<Recording> load_recordings(const std::string& data_root,
                                        int target_sr = 22050);
