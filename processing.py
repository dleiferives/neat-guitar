# processing.py
"""Audio feature extraction using librosa."""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List

SALIENCE_MIDI_MIN = 40
SALIENCE_MIDI_MAX = 88
N_SALIENCE_BINS = SALIENCE_MIDI_MAX - SALIENCE_MIDI_MIN + 1
HARM_WEIGHTS = np.array([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=np.float32)


@dataclass
class AudioFrame:
    cqt_bins: np.ndarray      # (108,) peak-normalized
    salience: np.ndarray      # (49,) harmonic salience
    peak_salience: float


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def compute_salience(cqt_mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Compute harmonic salience for each MIDI pitch."""
    salience = np.zeros(N_SALIENCE_BINS, dtype=np.float32)

    for k in range(N_SALIENCE_BINS):
        fund_hz = midi_to_hz(SALIENCE_MIDI_MIN + k)
        s = 0.0
        for h, weight in enumerate(HARM_WEIGHTS):
            target_hz = fund_hz * (h + 1)
            # Find closest CQT bin
            cents = 1200 * np.abs(np.log2(freqs / target_hz + 1e-10))
            best_bin = np.argmin(cents)
            if cents[best_bin] < 50:  # Within 50 cents
                s += weight * cqt_mag[best_bin]
        salience[k] = s

    peak = salience.max()
    if peak > 1e-9:
        salience /= peak
    return salience


def extract_frames(audio: np.ndarray, sr: int, hop_size: int) -> List[AudioFrame]:
    """Extract CQT frames from audio."""
    # Compute CQT: 9 octaves, 12 bins per octave, starting from C1 (~32 Hz)
    cqt = librosa.cqt(
        audio,
        sr=sr,
        hop_length=hop_size,
        n_bins=108,
        bins_per_octave=12,
        fmin=librosa.note_to_hz('C1'),
    )
    cqt_mag = np.abs(cqt).T.astype(np.float32)  # (n_frames, 108)

    # Get CQT frequencies for salience computation
    freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))

    frames = []
    for i in range(cqt_mag.shape[0]):
        bins = cqt_mag[i].copy()
        peak = bins.max()
        if peak > 1e-9:
            bins /= peak

        sal = compute_salience(bins, freqs)
        peak_sal = sal.max()

        frames.append(AudioFrame(
            cqt_bins=bins,
            salience=sal,
            peak_salience=float(peak_sal),
        ))

    return frames


def build_input(frames: List[AudioFrame], idx: int, pitch_history: int) -> np.ndarray:
    """Build network input vector."""
    cur = frames[idx]
    parts = [cur.cqt_bins, cur.salience]

    # Append recent peak salience history
    history = []
    for k in range(pitch_history - 1, -1, -1):
        fi = idx - k
        history.append(frames[fi].peak_salience if fi >= 0 else 0.0)
    parts.append(np.array(history, dtype=np.float32))

    return np.concatenate(parts)
