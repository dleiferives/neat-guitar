"""Dataset loading from dense 165-d cached features."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GuitarSetDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        segment_frames: int = 256,
        step_frames: int | None = None,
        cache_name: str = "features_165.pkl",
    ):
        self.data_dir = Path(data_dir)
        self.segment_frames = segment_frames
        self.step_frames = (
            step_frames if step_frames is not None else max(1, segment_frames // 2)
        )

        cache_path = self.data_dir / cache_name
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing {cache_path}. Run:\n"
                f"  python precompute_165.py {self.data_dir}"
            )

        with open(cache_path, "rb") as f:
            self.recordings = pickle.load(f)

        if not self.recordings:
            raise ValueError(f"No recordings found in {cache_path}")

        self.input_dim = int(self.recordings[0]["features"].shape[1])
        self.n_pitches = int(self.recordings[0]["frame_targets"].shape[1])

        self.index: list[tuple[int, int]] = []

        for rec_idx, rec in enumerate(self.recordings):
            n_frames = rec["features"].shape[0]
            last_start = max(0, n_frames - self.segment_frames)

            starts = list(range(0, last_start + 1, self.step_frames))
            if not starts or starts[-1] != last_start:
                starts.append(last_start)

            for start in starts:
                self.index.append((rec_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        rec_idx, start = self.index[idx]
        rec = self.recordings[rec_idx]
        end = start + self.segment_frames

        features = rec["features"][start:end]
        onset_targets = rec["onset_targets"][start:end]
        frame_targets = rec["frame_targets"][start:end]

        if features.shape[0] < self.segment_frames:
            pad = self.segment_frames - features.shape[0]
            features = np.pad(features, ((0, pad), (0, 0)))
            onset_targets = np.pad(onset_targets, ((0, pad), (0, 0)))
            frame_targets = np.pad(frame_targets, ((0, pad), (0, 0)))

        return {
            "features": torch.from_numpy(features.copy()),
            "onset_targets": torch.from_numpy(onset_targets.copy()),
            "frame_targets": torch.from_numpy(frame_targets.copy()),
        }
