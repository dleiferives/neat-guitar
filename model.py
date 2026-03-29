# model.py
"""Lightweight guitar transcription model."""

import torch
import torch.nn as nn


class GuitarTranscriber(nn.Module):
    def __init__(
        self,
        n_cqt: int = 72,  # Updated
        n_salience: int = 49,
        n_pitches: int = 49,
        hidden: int = 128,
    ):
        super().__init__()
        n_input = n_cqt + n_salience  # 121

        self.conv = nn.Sequential(
            nn.Conv1d(n_input, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        self.onset_head = nn.Conv1d(hidden, n_pitches, kernel_size=1)
        self.frame_head = nn.Conv1d(hidden, n_pitches, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        onset = self.onset_head(x).permute(0, 2, 1)
        frame = self.frame_head(x).permute(0, 2, 1)
        return onset, frame


class GuitarTranscriberRNN(nn.Module):
    def __init__(
        self,
        n_cqt: int = 72,  # Updated
        n_salience: int = 49,
        n_pitches: int = 49,
        hidden: int = 128,
    ):
        super().__init__()
        n_input = n_cqt + n_salience

        self.fc1 = nn.Linear(n_input, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.onset_head = nn.Linear(hidden * 2, n_pitches)
        self.frame_head = nn.Linear(hidden * 2, n_pitches)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x, _ = self.gru(x)
        return self.onset_head(x), self.frame_head(x)
