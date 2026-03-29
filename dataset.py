# dataset.py
"""Dataset loading."""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf

from config import Config
from processing import extract_frames, AudioFrame


@dataclass
class NoteEvent:
    midi: int
    time: float
    duration: float


@dataclass
class RecordingFrames:
    name: str
    frames: List[AudioFrame]
    notes: List[NoteEvent]
    midi_min: int
    midi_max: int
    hop_secs: float
    onset_targets: np.ndarray = field(default_factory=lambda: np.array([]))
    frame_targets: np.ndarray = field(default_factory=lambda: np.array([]))


def load_recording(wav_path: Path, json_path: Path, cfg: Config) -> Optional[RecordingFrames]:
    """Load a single recording."""
    try:
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != cfg.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.sample_rate)

        with open(json_path) as f:
            data = json.load(f)

        notes = [
            NoteEvent(n['midi'], n['time'], n['duration'])
            for n in data['notes']
        ]

        frames = extract_frames(audio.astype(np.float32), cfg.sample_rate, cfg.hop_size)
        hop_secs = cfg.hop_size / cfg.sample_rate
        n_frames = len(frames)
        n_outputs = cfg.n_outputs

        # Build frame-level targets
        frame_targets = np.zeros((n_frames, n_outputs), dtype=np.float32)
        for note in notes:
            if note.midi < cfg.midi_min or note.midi > cfg.midi_max:
                continue
            k = note.midi - cfg.midi_min
            start_frame = int(note.time / hop_secs)
            end_frame = int((note.time + note.duration) / hop_secs)
            for fi in range(max(0, start_frame), min(n_frames, end_frame + 1)):
                frame_targets[fi, k] = 1.0

        return RecordingFrames(
            name=wav_path.stem,
            frames=frames,
            notes=notes,
            midi_min=cfg.midi_min,
            midi_max=cfg.midi_max,
            hop_secs=hop_secs,
            frame_targets=frame_targets,
        )
    except Exception as e:
        print(f"[skip] {wav_path.stem}: {e}")
        return None


def load_recordings(data_dir: str, cfg: Config) -> List[RecordingFrames]:
    """Load all recordings from directory."""
    data_path = Path(data_dir)
    wav_dir = data_path / "audio_mono-mic"
    json_dir = data_path / "processed"
    cache_path = data_path / "frames.pkl"

    # Try cache first
    if cache_path.exists():
        print(f"[cache] Loading from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[cache] Failed: {e}, recomputing...")

    recordings = []
    for wav_path in sorted(wav_dir.glob("*_mic.wav")):
        stem = wav_path.stem.replace("_mic", "")
        json_path = json_dir / f"{stem}.json"
        if not json_path.exists():
            print(f"[skip] {stem} — no JSON")
            continue

        rec = load_recording(wav_path, json_path, cfg)
        if rec:
            print(f"  {stem}: {len(rec.frames)} frames, {len(rec.notes)} notes")
            recordings.append(rec)

    # Save cache
    if recordings:
        with open(cache_path, 'wb') as f:
            pickle.dump(recordings, f)
        print(f"[cache] Saved to {cache_path}")

    return recordings
