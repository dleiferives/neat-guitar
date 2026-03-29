# fitness.py
"""Racing-style fitness evaluation matching C++ implementation."""

from dataclasses import dataclass
from typing import List
from collections import deque

import numpy as np

from config import Config
from cache_reader import RecordingFrames, AudioFrame


@dataclass
class RacingResult:
    fitness: float
    frames_processed: int
    total_frames: int
    files_completed: int
    total_files: int
    avg_accuracy: float


def build_input(frames: List[AudioFrame], idx: int, pitch_history: int) -> np.ndarray:
    """Build network input vector."""
    cur = frames[idx]
    parts = [cur.cqt_bins, cur.salience]

    history = []
    for k in range(pitch_history - 1, -1, -1):
        fi = idx - k
        history.append(frames[fi].peak_salience if fi >= 0 else 0.0)
    parts.append(np.array(history, dtype=np.float32))

    return np.concatenate(parts)


def evaluate_genome_racing(
    net,
    data: List[RecordingFrames],
    cfg: Config,
    start_file_idx: int = 0,
    threshold: float = 0.5,
    kill_threshold: float = 0.2,
    window_secs: float = 5.0,
) -> RacingResult:
    """Racing fitness matching C++ implementation."""
    result = RacingResult(0.01, 0, 0, 0, len(data), 0.0)
    if not data:
        return result

    hop_secs = data[0].hop_secs
    window_frames = max(1, int(window_secs / hop_secs))
    n_out = cfg.n_outputs

    result.total_frames = sum(len(rf.frames) for rf in data)
    if result.total_frames == 0:
        return result

    # Rolling window
    win = deque()
    win_tp = win_fp = win_fn = 0

    fitness_accum = 0.0
    frames_processed = 0
    files_completed = 0
    n_files = len(data)

    for file_offset in range(n_files):
        file_idx = (start_file_idx + file_offset) % n_files
        rf = data[file_idx]

        net.reset()
        file_alive = True

        for fi in range(len(rf.frames)):
            inp = build_input(rf.frames, fi, cfg.pitch_history)
            out = net.activate(inp.tolist())

            # Per-frame metrics
            tp = fp = fn = 0
            for k in range(n_out):
                predicted = out[k] >= threshold
                target = rf.frame_targets[fi, k] >= 0.5
                tp += predicted and target
                fp += predicted and not target
                fn += not predicted and target

            # Update rolling window
            win.append((tp, fp, fn))
            win_tp += tp
            win_fp += fp
            win_fn += fn
            if len(win) > window_frames:
                old = win.popleft()
                win_tp -= old[0]
                win_fp -= old[1]
                win_fn -= old[2]

            # Rolling F1
            denom = 2 * win_tp + win_fp + win_fn
            rolling_f1 = (2 * win_tp / denom) if denom > 0 else 1.0

            fitness_accum += rolling_f1
            frames_processed += 1

            # Hard kill
            if len(win) == window_frames and rolling_f1 < kill_threshold:
                file_alive = False
                break

        if not file_alive:
            break
        files_completed += 1

    if frames_processed == 0:
        return result

    fitness = fitness_accum
    fitness *= (1.0 + 0.10 * files_completed)

    # Note: parsimony penalty applied separately in eval_genomes
    result.fitness = max(0.01, fitness)
    result.frames_processed = frames_processed
    result.files_completed = files_completed
    result.avg_accuracy = fitness_accum / frames_processed
    return result


def racing_theoretical_max(data: List[RecordingFrames]) -> float:
    """Compute theoretical maximum fitness."""
    total_frames = sum(len(rf.frames) for rf in data)
    n_files = len(data)
    return float(total_frames) * 1.0 * (1.0 + 0.10 * n_files)
