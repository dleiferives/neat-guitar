# train.py
"""Training script for CausalGuitarTranscriber (teacher model).

Supports:
- AMP (mixed precision)
- OneCycleLR scheduler with warmup
- Focal-weighted BCE loss (better for sparse labels)
- Label smoothing on positives
- Best-F1 model saving
- Optional distillation mode placeholder comment
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from model import CausalGuitarTranscriber
from dataset import GuitarSetDataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Focal binary cross-entropy with label smoothing on positives.

    Focal loss down-weights easy negatives so the model focuses on hard
    onsets — critical for sparse guitar labels.
    """
    # Label smoothing: pull positives slightly toward 0.95
    smooth_targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(
        logits, smooth_targets, pos_weight=pos_weight, reduction="none"
    )
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_prf(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        frame_t = batch["frame_targets"].to(device, non_blocking=True)
        _, frame_out, _ = model(x)
        pred = torch.sigmoid(frame_out) > threshold
        true = frame_t > 0.5
        tp += (pred & true).sum().item()
        fp += (pred & ~true).sum().item()
        fn += (~pred & true).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train() -> None:
    parser = argparse.ArgumentParser(
        description="Train CausalGuitarTranscriber teacher model"
    )
    parser.add_argument("data_dir")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save", default="teacher_model.pt")
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--gru-hidden", type=int, default=192)
    parser.add_argument("--n-tcn-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal loss gamma (0 = plain BCE)")
    parser.add_argument(
        "--segment-frames", type=int, default=256,
        help="Frames per training segment"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Fraction of dataset used for validation"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset ---
    dataset = GuitarSetDataset(
        args.data_dir, segment_frames=args.segment_frames
    )
    print(f"Total samples: {len(dataset):,}")

    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    # --- Model ---
    model = CausalGuitarTranscriber(
        n_cqt=108,
        n_salience=49,
        n_history=8,
        n_pitches=49,
        model_dim=args.model_dim,
        gru_hidden=args.gru_hidden,
        n_tcn_blocks=args.n_tcn_blocks,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- Optimizer + scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        fused=device.type == "cuda",
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,          # 10% warmup
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    # Positive class weight to handle class imbalance (notes are sparse)
    pos_weight = torch.tensor([7.0], device=device)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}")
        for batch in pbar:
            x = batch["features"].to(device, non_blocking=True)
            onset_t = batch["onset_targets"].to(device, non_blocking=True)
            frame_t = batch["frame_targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                onset_out, frame_out, _ = model(x)
                loss_onset = focal_bce(
                    onset_out, onset_t, pos_weight, gamma=args.gamma
                )
                loss_frame = focal_bce(
                    frame_out, frame_t, pos_weight, gamma=args.gamma
                )
                # Weight onset loss higher: accurate onsets → better notes
                loss = 1.5 * loss_onset + loss_frame

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_loss = epoch_loss / len(train_loader)
        precision, recall, f1 = compute_prf(model, val_loader, device)

        print(
            f"  Loss: {avg_loss:.4f} | "
            f"P: {precision:.3f}  R: {recall:.3f}  F1: {f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "f1": best_f1,
                    "args": vars(args),
                },
                args.save,
            )
            print(f"  ✓ Saved (F1={best_f1:.3f})")

    print(f"\nBest F1: {best_f1:.3f}  →  {args.save}")


if __name__ == "__main__":
    train()
