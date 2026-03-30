"""
Build dataset for guitar note verification model.

Each sample = (CQT of audio segment, multi-hot note vector) -> label (1=correct, 0=wrong)

Positive samples:  real audio segment + ground-truth notes
Negative samples:  same audio segment + corrupted notes (3x positive count)
  Corruption types: add random note(s), remove random note(s), swap note(s)

Writes output incrementally to an HDF5 file to avoid OOM.
"""

import os
import json
import glob
import random
import numpy as np
import librosa
import h5py
from pathlib import Path

# Guitar MIDI range E2 (40) through ~C6 (88), clamp to dataset
MIDI_MIN = 40
MIDI_MAX = 88
N_NOTES = MIDI_MAX - MIDI_MIN + 1  # 49

SEGMENT_DURATION = 0.5   # seconds
HOP_DURATION = 0.25      # seconds (50% overlap)
SR = 22050
HOP_LENGTH = 512
N_BINS = 84              # CQT bins (7 octaves)
BINS_PER_OCTAVE = 12

DATA_DIR = Path(__file__).parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio_mono-mic"
ANNOT_DIR = DATA_DIR / "annotation"
OUT_PATH = Path(__file__).parent / "dataset.h5"

SYNTH_RATIO = 3          # negative samples per positive sample
RANDOM_SEED = 42

# CQT frames per segment (fixed)
SEG_FRAMES = int(np.ceil(SEGMENT_DURATION * SR / HOP_LENGTH))   # ~22


def midi_to_idx(midi_note):
    return int(round(midi_note)) - MIDI_MIN


def notes_to_multihot(notes):
    vec = np.zeros(N_NOTES, dtype=np.float32)
    for n in notes:
        idx = midi_to_idx(n)
        if 0 <= idx < N_NOTES:
            vec[idx] = 1.0
    return vec


def corrupt_notes(notes):
    """Return a corrupted version of the note set. Always changes something."""
    all_note_range = list(range(MIDI_MIN, MIDI_MAX + 1))
    notes = list(notes)
    rng = random.random()

    if len(notes) == 0:
        k = random.randint(1, 3)
        return random.sample(all_note_range, k)

    if rng < 0.33:
        # Add 1-2 random notes not already present
        candidates = [n for n in all_note_range if n not in notes]
        k = min(random.randint(1, 2), len(candidates))
        if k > 0:
            notes = notes + random.sample(candidates, k)
    elif rng < 0.66:
        # Remove 1-2 notes
        k = min(random.randint(1, 2), len(notes))
        to_remove = random.sample(notes, k)
        for n in to_remove:
            notes.remove(n)
        if len(notes) == 0:
            notes = [random.choice(all_note_range)]
    else:
        # Swap 1-2 notes for different ones
        k = min(random.randint(1, 2), len(notes))
        for _ in range(k):
            old = random.choice(notes)
            notes.remove(old)
            candidates = [n for n in all_note_range if n not in notes and n != old]
            if candidates:
                notes.append(random.choice(candidates))

    return notes


def load_annotations(jams_path):
    """Return list of (time, duration, midi_note) from note_midi namespace."""
    with open(jams_path) as f:
        d = json.load(f)
    notes = []
    for ann in d["annotations"]:
        if ann["namespace"] == "note_midi":
            for item in ann["data"]:
                notes.append((item["time"], item["duration"], item["value"]))
    return notes


def get_active_notes(annotations, t_start, t_end):
    """Notes that overlap with [t_start, t_end]."""
    return [
        int(round(pitch))
        for (onset, dur, pitch) in annotations
        if onset < t_end and (onset + dur) > t_start
    ]


def compute_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=N_BINS,
                      bins_per_octave=BINS_PER_OCTAVE)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).astype(np.float32)
    return cqt_db  # (N_BINS, T)


def iter_segments(wav_path, jams_path):
    """Yield (cqt_seg, active_notes) for each window in one audio file."""
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    duration = len(y) / sr
    annotations = load_annotations(jams_path)
    cqt_full = compute_cqt(y, sr)

    hop_frames = int(np.ceil(HOP_DURATION * sr / HOP_LENGTH))

    t = 0.0
    while t + SEGMENT_DURATION <= duration:
        frame_start = int(t * sr / HOP_LENGTH)
        frame_end = frame_start + SEG_FRAMES

        cqt_seg = cqt_full[:, frame_start:frame_end]
        if cqt_seg.shape[1] < SEG_FRAMES:
            cqt_seg = np.pad(cqt_seg, ((0, 0), (0, SEG_FRAMES - cqt_seg.shape[1])))

        active = get_active_notes(annotations, t, t + SEGMENT_DURATION)
        yield cqt_seg, active
        t += HOP_DURATION


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(wav_files)} audio files")

    # First pass: count total samples
    # Approx: each file ~ duration/hop_dur segments
    # We'll just write incrementally with resizable HDF5 datasets

    if OUT_PATH.exists():
        OUT_PATH.unlink()

    with h5py.File(OUT_PATH, "w") as hf:
        cqt_ds = hf.create_dataset("X_cqt", shape=(0, N_BINS, SEG_FRAMES),
                                   maxshape=(None, N_BINS, SEG_FRAMES),
                                   dtype=np.float32, chunks=(256, N_BINS, SEG_FRAMES))
        notes_ds = hf.create_dataset("X_notes", shape=(0, N_NOTES),
                                     maxshape=(None, N_NOTES),
                                     dtype=np.float32, chunks=(256, N_NOTES))
        label_ds = hf.create_dataset("y", shape=(0,), maxshape=(None,),
                                     dtype=np.float32, chunks=(256,))

        n_pos = 0
        n_neg = 0

        for i, wav_path in enumerate(wav_files):
            stem = wav_path.stem.replace("_mic", "")
            jams_path = ANNOT_DIR / f"{stem}.jams"
            if not jams_path.exists():
                print(f"  SKIP (no annotation): {wav_path.name}")
                continue

            batch_cqt = []
            batch_notes = []
            batch_labels = []

            for cqt_seg, active in iter_segments(wav_path, jams_path):
                # Positive
                batch_cqt.append(cqt_seg)
                batch_notes.append(notes_to_multihot(active))
                batch_labels.append(1.0)
                n_pos += 1

                # Negatives
                for _ in range(SYNTH_RATIO):
                    bad = corrupt_notes(active[:])
                    batch_cqt.append(cqt_seg)
                    batch_notes.append(notes_to_multihot(bad))
                    batch_labels.append(0.0)
                    n_neg += 1

            if not batch_cqt:
                continue

            bc = np.stack(batch_cqt).astype(np.float32)
            bn = np.stack(batch_notes).astype(np.float32)
            bl = np.array(batch_labels, dtype=np.float32)

            cur = cqt_ds.shape[0]
            new_size = cur + len(bl)
            cqt_ds.resize(new_size, axis=0)
            notes_ds.resize(new_size, axis=0)
            label_ds.resize(new_size, axis=0)

            cqt_ds[cur:new_size] = bc
            notes_ds[cur:new_size] = bn
            label_ds[cur:new_size] = bl

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(wav_files)}] pos={n_pos}  neg={n_neg}  total={n_pos+n_neg}")

        total = n_pos + n_neg
        print(f"\nTotal positives: {n_pos}")
        print(f"Total negatives: {n_neg}")
        print(f"CQT shape: {cqt_ds.shape}  Notes: {notes_ds.shape}")

        # Save a shuffled index for train/val splitting (don't shuffle the HDF5 itself)
        idx = np.random.permutation(total)
        hf.create_dataset("shuffle_idx", data=idx)

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
