"""
Train the NoteVerifier model on the prebuilt HDF5 dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py

from model import NoteVerifier

DATASET_PATH = Path(__file__).parent / "dataset.h5"
CHECKPOINT_PATH = Path(__file__).parent / "note_verifier.pt"

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HDF5Dataset(Dataset):
    """Loads samples from HDF5 file using a pre-computed shuffled index."""

    def __init__(self, h5_path, indices):
        self.h5_path = str(h5_path)
        # Sort indices so HDF5 reads are sequential (much faster)
        self.indices = np.sort(indices)
        self._hf = None

    def _open(self):
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._open()
        idx = int(self.indices[i])
        cqt = torch.from_numpy(self._hf["X_cqt"][idx])
        notes = torch.from_numpy(self._hf["X_notes"][idx])
        label = torch.tensor(self._hf["y"][idx], dtype=torch.float32)
        return cqt, notes, label


def worker_init_fn(worker_id):
    """Each DataLoader worker needs its own HDF5 handle."""
    pass  # __getitem__ lazily opens the file per-process


def main():
    print(f"Using device: {DEVICE}")

    with h5py.File(DATASET_PATH, "r") as hf:
        total = hf["y"].shape[0]
        shuffle_idx = hf["shuffle_idx"][:]
        n_bins = hf["X_cqt"].shape[1]
        n_notes = hf["X_notes"].shape[1]
        seg_frames = hf["X_cqt"].shape[2]
        pos_count = int(hf["y"][:].sum())

    print(f"Dataset: {total} samples  CQT={n_bins}x{seg_frames}  Notes={n_notes}")
    print(f"Positive: {pos_count}  Negative: {total - pos_count}")

    n_val = int(total * VAL_SPLIT)
    n_train = total - n_val
    train_idx = shuffle_idx[:n_train]
    val_idx = shuffle_idx[n_train:]

    train_set = HDF5Dataset(DATASET_PATH, train_idx)
    val_set = HDF5Dataset(DATASET_PATH, val_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=(DEVICE == "cuda"))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=(DEVICE == "cuda"))

    model = NoteVerifier(n_bins=n_bins, n_notes=n_notes).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Pos weight to handle class imbalance (1 pos : 3 neg)
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for cqt, notes, labels in train_loader:
            cqt, notes, labels = cqt.to(DEVICE), notes.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(cqt, notes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            preds = (logits.sigmoid() > 0.5).float()
            train_correct += (preds == labels).sum().item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tp = val_fp = val_tn = val_fn = 0
        with torch.no_grad():
            for cqt, notes, labels in val_loader:
                cqt, notes, labels = cqt.to(DEVICE), notes.to(DEVICE), labels.to(DEVICE)
                logits = model(cqt, notes)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = (logits.sigmoid() > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        train_acc = train_correct / n_train
        val_acc = val_correct / n_val
        prec = val_tp / (val_tp + val_fp + 1e-8)
        rec = val_tp / (val_tp + val_fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss/n_train:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss/n_val:.4f}  val_acc={val_acc:.4f}  "
              f"prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "n_bins": n_bins, "n_notes": n_notes},
                       CHECKPOINT_PATH)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
