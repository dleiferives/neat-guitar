# preprocess.py
"""Pre-compute and cache CQT features for fast training."""

import argparse
import json
import pickle
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


def compute_cqt(audio: np.ndarray, sr: int, hop_length: int, n_bins: int = 72) -> np.ndarray:
    cqt = librosa.cqt(audio, sr=sr, hop_length=hop_length, n_bins=n_bins,
                      bins_per_octave=12, fmin=librosa.note_to_hz('C2'))
    mag = np.abs(cqt).T.astype(np.float32)
    peak = mag.max(axis=1, keepdims=True) + 1e-9
    return mag / peak


def compute_salience(cqt: np.ndarray, midi_min: int, midi_max: int) -> np.ndarray:
    freqs = librosa.cqt_frequencies(n_bins=cqt.shape[1], fmin=librosa.note_to_hz('C2'))
    n_frames = cqt.shape[0]
    n_pitches = midi_max - midi_min + 1
    salience = np.zeros((n_frames, n_pitches), dtype=np.float32)
    harm_weights = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])

    for k in range(n_pitches):
        fund_hz = 440.0 * (2.0 ** ((midi_min + k - 69) / 12.0))
        for h, w in enumerate(harm_weights):
            target_hz = fund_hz * (h + 1)
            cents = 1200 * np.abs(np.log2(freqs / target_hz + 1e-10))
            best_bin = np.argmin(cents)
            if cents[best_bin] < 50:
                salience[:, k] += w * cqt[:, best_bin]

    peak = salience.max(axis=1, keepdims=True) + 1e-9
    return salience / peak


def build_targets(notes: list, n_frames: int, hop_length: int, sr: int,
                  midi_min: int, midi_max: int) -> tuple[np.ndarray, np.ndarray]:
    hop_secs = hop_length / sr
    n_pitches = midi_max - midi_min + 1
    onset_t = np.zeros((n_frames, n_pitches), dtype=np.float32)
    frame_t = np.zeros((n_frames, n_pitches), dtype=np.float32)

    for note in notes:
        midi = note["midi"]
        if midi < midi_min or midi > midi_max:
            continue
        k = midi - midi_min
        onset_frame = int(note["time"] / hop_secs)
        offset_frame = int((note["time"] + note["duration"]) / hop_secs)

        if 0 <= onset_frame < n_frames:
            onset_t[onset_frame, k] = 1.0
        for fi in range(max(0, onset_frame), min(n_frames, offset_frame + 1)):
            frame_t[fi, k] = 1.0

    return onset_t, frame_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-cqt", type=int, default=72)
    parser.add_argument("--midi-min", type=int, default=40)
    parser.add_argument("--midi-max", type=int, default=88)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    audio_dir = data_dir / "audio_mono-mic"
    json_dir = data_dir / "processed"
    cache_path = data_dir / "features_cache.pkl"

    recordings = []
    wav_files = sorted(audio_dir.glob("*_mic.wav"))

    print(f"Pre-processing {len(wav_files)} recordings...")
    for wav_path in tqdm(wav_files):
        stem = wav_path.stem.replace("_mic", "")
        json_path = json_dir / f"{stem}.json"
        if not json_path.exists():
            continue

        audio, _ = librosa.load(wav_path, sr=args.sample_rate)
        with open(json_path) as f:
            notes = json.load(f)["notes"]

        cqt = compute_cqt(audio, args.sample_rate, args.hop_length, args.n_cqt)
        salience = compute_salience(cqt, args.midi_min, args.midi_max)
        features = np.concatenate([cqt, salience], axis=1)

        onset_t, frame_t = build_targets(
            notes, len(features), args.hop_length, args.sample_rate,
            args.midi_min, args.midi_max
        )

        recordings.append({
            "name": stem,
            "features": features,
            "onset_targets": onset_t,
            "frame_targets": frame_t,
        })

    with open(cache_path, 'wb') as f:
        pickle.dump(recordings, f)

    print(f"Cached {len(recordings)} recordings to {cache_path}")
    total_frames = sum(r["features"].shape[0] for r in recordings)
    print(f"Total frames: {total_frames:,}")


if __name__ == "__main__":
    main()
