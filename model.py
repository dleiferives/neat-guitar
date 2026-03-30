"""Improved transcription model for fixed guitar features.

Input per frame:
- 108 CQT bins
- 49 salience bins
- 8 history values

Output per frame:
- 49 onset logits
- 49 frame logits
"""

import torch
import torch.nn as nn


class FeatureStem(nn.Module):
    """Small per-feature-group encoder."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SqueezeExcite1d(nn.Module):
    """Channel attention for temporal conv features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        scale = x.mean(dim=-1)
        scale = self.net(scale).unsqueeze(-1)
        return x * scale


class ResidualTemporalBlock(nn.Module):
    """Depthwise-separable dilated temporal block."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(channels)
        self.pointwise_in = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            channels * 2,
            channels * 2,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels * 2,
        )
        self.glu = nn.GLU(dim=1)
        self.pointwise_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.se = SqueezeExcite1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x

        y = self.norm(x)
        y = y.transpose(1, 2)  # (B, C, T)
        y = self.pointwise_in(y)
        y = self.depthwise(y)
        y = self.glu(y)  # (B, C, T)
        y = self.pointwise_out(y)
        y = self.se(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)  # (B, T, C)

        return residual + y


class ResidualMLPBlock(nn.Module):
    """Residual feed-forward block."""

    def __init__(
        self,
        dim: int,
        expansion: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class PitchRefinement(nn.Module):
    """Small pitch-axis conv to reduce harmonic/octave confusion."""

    def __init__(self, n_pitches: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_pitches = n_pitches
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=5, padding=2),
        )

    def forward(
        self,
        onset_prob: torch.Tensor,
        frame_logits: torch.Tensor,
    ) -> torch.Tensor:
        # onset_prob: (B, T, P)
        # frame_logits: (B, T, P)
        b, t, p = frame_logits.shape

        x = torch.stack([frame_logits, onset_prob], dim=2)
        x = x.reshape(b * t, 2, p)

        correction = self.net(x).squeeze(1)
        correction = correction.reshape(b, t, p)

        return frame_logits + correction


class HybridGuitarTranscriber(nn.Module):
    """Hybrid TCN + BiGRU guitar transcription model.

    This keeps your existing feature layout and 49-note output space, but
    uses a much better internal organization than a flat CNN/MLP.
    """

    def __init__(
        self,
        n_cqt: int = 108,
        n_salience: int = 49,
        n_history: int = 8,
        n_pitches: int = 49,
        model_dim: int = 128,
        gru_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_cqt = n_cqt
        self.n_salience = n_salience
        self.n_history = n_history
        self.n_pitches = n_pitches

        # Separate stems for semantically different inputs.
        self.cqt_stem = FeatureStem(n_cqt, 96, dropout)
        self.salience_stem = FeatureStem(n_salience, 48, dropout)
        self.history_stem = FeatureStem(n_history, 16, dropout)

        fused_dim = 96 + 48 + 16

        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Multi-scale local temporal modeling.
        self.temporal_blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(model_dim, dilation=1, dropout=dropout),
                ResidualTemporalBlock(model_dim, dilation=2, dropout=dropout),
                ResidualTemporalBlock(model_dim, dilation=4, dropout=dropout),
                ResidualTemporalBlock(model_dim, dilation=8, dropout=dropout),
                ResidualTemporalBlock(model_dim, dilation=16, dropout=dropout),
            ]
        )

        # Long-range temporal context.
        self.bigru = nn.GRU(
            input_size=model_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.shared = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Onset head.
        self.onset_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, n_pitches),
        )

        # Frame head conditioned on predicted onsets.
        self.frame_in = nn.Sequential(
            nn.LayerNorm(model_dim + n_pitches),
            nn.Linear(model_dim + n_pitches, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.frame_block_1 = ResidualMLPBlock(model_dim, dropout=dropout)
        self.frame_block_2 = ResidualMLPBlock(model_dim, dropout=dropout)
        self.frame_out = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, n_pitches),
        )

        self.pitch_refine = PitchRefinement(
            n_pitches=n_pitches,
            hidden=32,
            dropout=dropout,
        )

    def _split_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.n_cqt
        j = i + self.n_salience
        k = j + self.n_history

        cqt = x[..., :i]
        salience = x[..., i:j]
        history = x[..., j:k]

        return cqt, salience, history

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 165)
        cqt, salience, history = self._split_features(x)

        x = torch.cat(
            [
                self.cqt_stem(cqt),
                self.salience_stem(salience),
                self.history_stem(history),
            ],
            dim=-1,
        )

        x = self.fuse(x)

        for block in self.temporal_blocks:
            x = block(x)

        x, _ = self.bigru(x)
        x = self.shared(x)

        onset_logits = self.onset_head(x)
        onset_prob = torch.sigmoid(onset_logits).detach()

        frame = torch.cat([x, onset_prob], dim=-1)
        frame = self.frame_in(frame)
        frame = self.frame_block_1(frame)
        frame = self.frame_block_2(frame)

        frame_logits = self.frame_out(frame)
        frame_logits = self.pitch_refine(onset_prob, frame_logits)

        return onset_logits, frame_logits
