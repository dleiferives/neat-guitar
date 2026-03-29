# fitness.py
"""Racing-style fitness evaluation."""

from dataclasses import dataclass
from typing import List

import numpy as np

from config import Config
from dataset import RecordingFrames
from processing import build_input


@dataclass
class RacingResult:
    fitness: float
    frames_processed: int
    total_frames: int
    files_completed: int
    total_files: int
    avg_accuracy: float


def evaluate_genome_racing(
    net,  # NEAT network with activate() method
    data: List[RecordingFrames],
    cfg: Config,
    start_file_idx: int = 0,
    threshold: float = 0.5,
    kill_threshold: float = 0.2,
    window_secs: float = 5.0,
) -> RacingResult:
    """Racing-style fitness: each frame earns its rolling F1."""
    n_files = len(data)
    n_outputs = cfg.n_outputs
    hop_secs = cfg.hop_size / cfg.sample_rate
    window_frames = int(window_secs / hop_secs)

    total_fitness = 0.0
    total_frames = 0
    frames_processed = 0
    files_completed = 0
    accuracy_sum = 0.0
    accuracy_count = 0

    # Rolling window stats
    tp_window = np.zeros(window_frames, dtype=np.int32)
    fp_window = np.zeros(window_frames, dtype=np.int32)
    fn_window = np.zeros(window_frames, dtype=np.int32)
    window_idx = 0

    for file_offset in range(n_files):
        file_idx = (start_file_idx + file_offset) % n_files
        rec = data[file_idx]
        n_frames = len(rec.frames)
        total_frames += n_frames

        net.reset()  # Reset network state between files

        file_killed = False
        for fi in range(n_frames):
            inp = build_input(rec.frames, fi, cfg.pitch_history)
            out = net.activate(inp.tolist())

            # Compute frame metrics
            tp = fp = fn = 0
            for k in range(n_outputs):
                predicted = out[k] >= threshold
                target = rec.frame_targets[fi, k] >= 0.5
                tp += predicted and target
                fp += predicted and not target
                fn += not predicted and target

            # Update rolling window
            tp_window[window_idx] = tp
            fp_window[window_idx] = fp
            fn_window[window_idx] = fn
            window_idx = (window_idx + 1) % window_frames

            # Compute rolling F1
            sum_tp = tp_window.sum()
            sum_fp = fp_window.sum()
            sum_fn = fn_window.sum()

            prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
            rec_val = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
            f1 = 2 * prec * rec_val / (prec + rec_val) if (prec + rec_val) > 0 else 0.0

            total_fitness += f1
            frames_processed += 1
            accuracy_sum += f1
            accuracy_count += 1

            # Early termination check
            if frames_processed >= window_frames and f1 < kill_threshold:
                file_killed = True
                break

        if not file_killed:
            files_completed += 1

    # File completion bonus
    file_bonus = 1.0 + 0.10 * files_completed
    total_fitness *= file_bonus

    avg_acc = accuracy_sum / accuracy_count if accuracy_count > 0 else 0.0

    return RacingResult(
        fitness=total_fitness,
        frames_processed=frames_processed,
        total_frames=total_frames,
        files_completed=files_completed,
        total_files=n_files,
        avg_accuracy=avg_acc,
    )
