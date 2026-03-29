# dataset.py
"""Fast dataset loading from pre-computed cache."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GuitarSetDataset(Dataset):
    """Load pre-computed features from cache."""

    def __init__(self, data_dir: str, segment_frames: int = 256):
        self.segment_frames = segment_frames
        cache_path = Path(data_dir) / "features_cache.pkl"

        with open(cache_path, 'rb') as f:
            self.recordings = pickle.load(f)

        # Build index: (recording_idx, frame_offset)
        self.index = []
        for ri, rec in enumerate(self.recordings):
            n_frames = rec["features"].shape[0]
            for start in range(0, max(1, n_frames - segment_frames + 1), segment_frames // 2):
                self.index.append((ri, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ri, start = self.index[idx]
        rec = self.recordings[ri]
        n_frames = rec["features"].shape[0]
        end = min(start + self.segment_frames, n_frames)

        features = rec["features"][start:end]
        onset_t = rec["onset_targets"][start:end]
        frame_t = rec["frame_targets"][start:end]

        # Pad if needed
        if features.shape[0] < self.segment_frames:
            pad = self.segment_frames - features.shape[0]
            features = np.pad(features, ((0, pad), (0, 0)))
            onset_t = np.pad(onset_t, ((0, pad), (0, 0)))
            frame_t = np.pad(frame_t, ((0, pad), (0, 0)))

        return {
            "features": torch.from_numpy(features.copy()),
            "onset_targets": torch.from_numpy(onset_t.copy()),
            "frame_targets": torch.from_numpy(frame_t.copy()),
        }
