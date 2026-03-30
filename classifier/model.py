"""
Guitar note verification model.

Input:
  - CQT segment: (N_BINS=84, T=44) -> flattened or processed by CNN
  - Note vector: (N_NOTES=49,) multi-hot

Output:
  - Scalar probability: did this audio actually contain these notes?
"""

import torch
import torch.nn as nn


class CQTEncoder(nn.Module):
    """Small CNN to encode CQT segment into a fixed-size embedding."""

    def __init__(self, n_bins=84, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # -> (32, n_bins/2, T/2)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # -> (64, n_bins/4, T/4)

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> (128, 4, 4) = 2048
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, n_bins, T)
        x = x.unsqueeze(1)          # (B, 1, n_bins, T)
        x = self.conv(x)            # (B, 128, 4, 4)
        x = x.flatten(1)            # (B, 2048)
        return self.fc(x)           # (B, embed_dim)


class NoteEncoder(nn.Module):
    """Small MLP to encode multi-hot note vector."""

    def __init__(self, n_notes=49, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_notes, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class NoteVerifier(nn.Module):
    """
    Takes (cqt_segment, note_vector) and outputs P(notes match audio).
    """

    def __init__(self, n_bins=84, n_notes=49, cqt_embed=128, note_embed=64):
        super().__init__()
        self.cqt_enc = CQTEncoder(n_bins=n_bins, embed_dim=cqt_embed)
        self.note_enc = NoteEncoder(n_notes=n_notes, embed_dim=note_embed)

        combined = cqt_embed + note_embed
        self.classifier = nn.Sequential(
            nn.Linear(combined, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cqt, notes):
        cqt_emb = self.cqt_enc(cqt)        # (B, cqt_embed)
        note_emb = self.note_enc(notes)     # (B, note_embed)
        combined = torch.cat([cqt_emb, note_emb], dim=1)
        return self.classifier(combined).squeeze(1)  # (B,) logits
