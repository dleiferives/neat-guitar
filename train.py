# train.py
"""Fast training script."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm

from model import HybridGuitarTranscriber


from dataset import GuitarSetDataset


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)  # Increased
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", choices=["cnn", "rnn"], default="cnn")
    parser.add_argument("--workers", type=int, default=8)  # More workers
    parser.add_argument("--save", default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = GuitarSetDataset(args.data_dir)
    print(f"Samples: {len(dataset):,}")

    n_val = max(1, len(dataset) // 5)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,         # Fast GPU transfer
        persistent_workers=True,  # Don't restart workers
        prefetch_factor=4,        # Prefetch more batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = HybridGuitarTranscriber(
        n_cqt=108,
        n_salience=49,
        n_history=8,
        n_pitches=49,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, fused=True)  # Fused optimizer
    pos_weight = torch.tensor([5.0], device=device)

    # Use automatic mixed precision for speed
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            x = batch["features"].to(device, non_blocking=True)
            onset_t = batch["onset_targets"].to(device, non_blocking=True)
            frame_t = batch["frame_targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                onset_out, frame_out = model(x)
                loss = (
                    nn.functional.binary_cross_entropy_with_logits(onset_out, onset_t, pos_weight=pos_weight) +
                    nn.functional.binary_cross_entropy_with_logits(frame_out, frame_t, pos_weight=pos_weight)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validate
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device, non_blocking=True)
                frame_t = batch["frame_targets"].to(device, non_blocking=True)
                _, frame_out = model(x)
                pred = torch.sigmoid(frame_out) > 0.5
                true = frame_t > 0.5
                tp += (pred & true).sum().item()
                fp += (pred & ~true).sum().item()
                fn += (~pred & true).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | "
              f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.save)
            print(f"  → Saved (F1={best_f1:.3f})")

    print(f"\nBest F1: {best_f1:.3f}")


if __name__ == "__main__":
    train()
