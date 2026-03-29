# cache_reader.py
"""Read C++ binary cache format."""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, BinaryIO

import numpy as np


CACHE_MAGIC = 0x4E454154  # "NEAT"
CACHE_VERSION = 2


@dataclass
class AudioFrame:
    cqt_bins: np.ndarray
    salience: np.ndarray
    peak_salience: float


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


def read_pod(f: BinaryIO, fmt: str):
    """Read a POD type."""
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return struct.unpack(fmt, data)[0]


def read_string(f: BinaryIO) -> str:
    """Read a length-prefixed string."""
    length = read_pod(f, '<I')  # uint32_t
    data = f.read(length)
    if len(data) != length:
        raise EOFError(f"Expected {length} bytes for string, got {len(data)}")
    return data.decode('utf-8')


def read_vector_float(f: BinaryIO) -> np.ndarray:
    """Read a vector<float>."""
    size = read_pod(f, '<Q')  # uint64_t
    if size == 0:
        return np.array([], dtype=np.float32)
    data = f.read(size * 4)
    return np.frombuffer(data, dtype=np.float32).copy()


def read_audio_frame(f: BinaryIO) -> AudioFrame:
    """Read an AudioFrame."""
    cqt_bins = read_vector_float(f)
    salience = read_vector_float(f)
    peak_salience = read_pod(f, '<f')
    return AudioFrame(cqt_bins, salience, peak_salience)


def read_vector_2d(f: BinaryIO) -> np.ndarray:
    """Read a vector<vector<float>>."""
    rows = read_pod(f, '<Q')  # uint64_t
    if rows == 0:
        return np.array([], dtype=np.float32).reshape(0, 0)
    cols = read_pod(f, '<Q')  # uint64_t
    data = f.read(int(rows * cols * 4))
    return np.frombuffer(data, dtype=np.float32).copy().reshape(rows, cols)


def load_cache(path: str, midi_min: int = 40, midi_max: int = 88,
               hop_size: int = 512, sample_rate: int = 22050) -> List[RecordingFrames]:
    """Load the C++ binary cache file."""
    expected_hop_secs = hop_size / sample_rate

    with open(path, 'rb') as f:
        magic = read_pod(f, '<I')
        if magic != CACHE_MAGIC:
            raise ValueError(f"Invalid magic: {magic:#x}, expected {CACHE_MAGIC:#x}")

        version = read_pod(f, '<I')
        if version != CACHE_VERSION:
            raise ValueError(f"Version mismatch: {version}, expected {CACHE_VERSION}")

        n_recs = read_pod(f, '<I')
        print(f"[cache] Loading {n_recs} recordings...")

        if n_recs > 0:
            cached_midi_min = read_pod(f, '<i')
            cached_midi_max = read_pod(f, '<i')
            cached_hop_secs = read_pod(f, '<f')

            if cached_midi_min != midi_min or cached_midi_max != midi_max:
                raise ValueError(f"MIDI range mismatch: cache has {cached_midi_min}-{cached_midi_max}")
            if abs(cached_hop_secs - expected_hop_secs) > 1e-6:
                raise ValueError(f"Hop secs mismatch: {cached_hop_secs} vs {expected_hop_secs}")

        recordings = []
        for i in range(n_recs):
            name = read_string(f)
            rec_midi_min = read_pod(f, '<i')
            rec_midi_max = read_pod(f, '<i')
            rec_hop_secs = read_pod(f, '<f')

            # Frames
            n_frames = read_pod(f, '<Q')
            frames = [read_audio_frame(f) for _ in range(n_frames)]

            # Notes
            n_notes = read_pod(f, '<Q')
            notes = []
            for _ in range(n_notes):
                midi = read_pod(f, '<i')
                time = read_pod(f, '<f')
                duration = read_pod(f, '<f')
                notes.append(NoteEvent(midi, time, duration))

            # Targets
            onset_targets = read_vector_2d(f)
            frame_targets = read_vector_2d(f)

            recordings.append(RecordingFrames(
                name=name,
                frames=frames,
                notes=notes,
                midi_min=rec_midi_min,
                midi_max=rec_midi_max,
                hop_secs=rec_hop_secs,
                onset_targets=onset_targets,
                frame_targets=frame_targets,
            ))
            print(f"  {name}: {len(frames)} frames, {len(notes)} notes")

        return recordings
