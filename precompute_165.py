#!/usr/bin/env python3
"""Precompute dense 165-d guitar features.

Output:
  <data_dir>/features_165.pkl

Each recording is stored as:
{
    "name": str,
    "features": float32 array of shape (T, 165),
    "onset_targets": float32 array of shape (T, 49),
    "frame_targets": float32 array of shape (T, 49),
}
"""

import argparse
import json
import pickle
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


AUDIO_SR = 22050
AUDIO_HOP = 512

FEAT_SR = 44100
FEAT_HOP = 1024

N_CQT = 108
BINS_PER_OCTAVE = 12
FMIN = librosa.note_to_hz("C1")

MIDI_MIN = 40
MIDI_MAX = 88
N_PITCHES = MIDI_MAX - MIDI_MIN + 1
HISTORY = 8

HARM_WEIGHTS = np.array(
    [1.0, 0.5, 0.25, 0.125, 0.0625],
    dtype=np.float32,
)


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def build_salience_lookup() -> list[list[tuple[int, float]]]:
    freqs = librosa.cqt_frequencies(
        n_bins=N_CQT,
        fmin=FMIN,
        bins_per_octave=BINS_PER_OCTAVE,
    )

    lookup: list[list[tuple[int, float]]] = []

    for midi in range(MIDI_MIN, MIDI_MAX + 1):
        fund_hz = midi_to_hz(midi)
        bins: list[tuple[int, float]] = []

        for harm_idx, weight in enumerate(HARM_WEIGHTS, start=1):
            target_hz = fund_hz * harm_idx
            ratio = np.maximum(freqs / target_hz, 1e-10)
            cents = 1200.0 * np.abs(np.log2(ratio))
            best_bin = int(np.argmin(cents))

            if cents[best_bin] < 50.0:
                bins.append((best_bin, float(weight)))

        lookup.append(bins)

    return lookup


SALIENCE_LOOKUP = build_salience_lookup()


def load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def compute_cqt(audio: np.ndarray) -> np.ndarray:
    cqt = librosa.cqt(
        audio,
        sr=FEAT_SR,
        hop_length=FEAT_HOP,
        n_bins=N_CQT,
        bins_per_octave=BINS_PER_OCTAVE,
        fmin=FMIN,
    )

    mag = np.abs(cqt).T.astype(np.float32)
    peak = np.maximum(mag.max(axis=1, keepdims=True), 1e-9)
    return mag / peak


def compute_salience(cqt_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_frames = cqt_norm.shape[0]
    salience_raw = np.zeros((n_frames, N_PITCHES), dtype=np.float32)

    for pitch_idx, mapping in enumerate(SALIENCE_LOOKUP):
        for bin_idx, weight in mapping:
            salience_raw[:, pitch_idx] += weight * cqt_norm[:, bin_idx]

    peak_salience = salience_raw.max(axis=1).astype(np.float32)
    denom = np.maximum(peak_salience[:, None], 1e-9)
    salience = salience_raw / denom

    return salience.astype(np.float32), peak_salience.astype(np.float32)


def build_history(peak_salience: np.ndarray, history_len: int) -> np.ndarray:
    n_frames = len(peak_salience)
    hist = np.zeros((n_frames, history_len), dtype=np.float32)

    for t in range(n_frames):
        for k in range(history_len):
            src = t - (history_len - 1 - k)
            if src >= 0:
                hist[t, k] = peak_salience[src]

    return hist


def load_notes(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    notes = []
    for note in data["notes"]:
        notes.append(
            {
                "midi": int(note["midi"]),
                "time": float(note["time"]),
                "duration": float(note["duration"]),
            }
        )

    return notes


def build_targets(notes: list[dict], n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    hop_secs = AUDIO_HOP / AUDIO_SR

    onset_targets = np.zeros((n_frames, N_PITCHES), dtype=np.float32)
    frame_targets = np.zeros((n_frames, N_PITCHES), dtype=np.float32)

    for note in notes:
        midi = note["midi"]
        if midi < MIDI_MIN or midi > MIDI_MAX:
            continue

        k = midi - MIDI_MIN
        start = int(note["time"] / hop_secs)
        end = int((note["time"] + note["duration"]) / hop_secs)

        if 0 <= start < n_frames:
            onset_targets[start, k] = 1.0

        for fi in range(max(0, start), min(n_frames, end + 1)):
            frame_targets[fi, k] = 1.0

    return onset_targets, frame_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--output", default="features_165.pkl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    audio_dir = data_dir / "audio_mono-mic"
    json_dir = data_dir / "processed"
    out_path = data_dir / args.output

    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio dir: {audio_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"Missing processed dir: {json_dir}")

    recordings = []

    wav_files = sorted(audio_dir.glob("*_mic.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No *_mic.wav files found in {audio_dir}")

    print(f"Precomputing 165-d features for {len(wav_files)} recordings...")

    for wav_path in tqdm(wav_files):
        stem = wav_path.stem.replace("_mic", "")
        json_path = json_dir / f"{stem}.json"

        if not json_path.exists():
            print(f"[skip] {stem}: missing {json_path.name}")
            continue

        try:
            audio = load_audio_mono(wav_path, FEAT_SR)
            notes = load_notes(json_path)

            cqt = compute_cqt(audio)
            salience, peak_salience = compute_salience(cqt)
            history = build_history(peak_salience, HISTORY)

            features = np.concatenate([cqt, salience, history], axis=1)
            onset_targets, frame_targets = build_targets(notes, len(features))

            if features.shape[1] != 165:
                raise ValueError(
                    f"{stem}: expected 165 features, got {features.shape[1]}"
                )

            recordings.append(
                {
                    "name": stem,
                    "features": features.astype(np.float32),
                    "onset_targets": onset_targets.astype(np.float32),
                    "frame_targets": frame_targets.astype(np.float32),
                }
            )
        except Exception as e:
            print(f"[skip] {stem}: {e}")

    if not recordings:
        raise RuntimeError("No recordings were successfully precomputed.")

    with open(out_path, "wb") as f:
        pickle.dump(recordings, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_frames = sum(rec["features"].shape[0] for rec in recordings)
    print(f"Saved {len(recordings)} recordings to {out_path}")
    print(f"Total frames: {total_frames:,}")


if __name__ == "__main__":
    main()
