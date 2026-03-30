# model.py
"""Causal guitar transcription teacher model for knowledge distillation.

Design principles (informed by ISMIR 2025 real-time AMT research):
- Strictly causal: every operation looks only at t and past frames
- Autoregressive onset conditioning: previous-frame onset predictions
  are fed back as input to the frame head (biggest win for causal models)
- Causal TCN: left-only padding so no future leakage
- No Squeeze-Excite over the time axis (non-causal global mean)
- Unidirectional GRU for stateful long-range context
- Causal local self-attention (past-only window) for medium-range context
- Exposes hidden representations for distillation

Input: (B, T, 165) — 108 CQT + 49 salience + 8 peak-salience history
Output: onset_logits (B, T, 49), frame_logits (B, T, 49),
        hidden (B, T, model_dim)  ← used by distillation student
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class FeatureStem(nn.Module):
    """Per-feature-group linear encoder with LayerNorm."""

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


class CausalDepthwiseConv1d(nn.Module):
    """Strictly causal depthwise conv: pads only on the left."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            padding=0,  # we handle padding manually
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class PerFrameChannelGate(nn.Module):
    """Lightweight per-frame channel attention (causal-safe).

    Unlike SqueezeExcite which pools over T (non-causal), this applies
    a small MLP independently to each frame's channel vector.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — gate is applied per frame, fully causal
        return x * self.gate(x)


class CausalResidualTCNBlock(nn.Module):
    """Causal dilated depthwise-separable residual block.

    All convolutions use left-only padding → strictly causal.
    """

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.pointwise_in = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.depthwise = CausalDepthwiseConv1d(
            channels * 2, kernel_size=kernel_size, dilation=dilation
        )
        self.glu = nn.GLU(dim=1)
        self.pointwise_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.gate = PerFrameChannelGate(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        y = self.norm(x).transpose(1, 2)  # (B, C, T)
        y = self.pointwise_in(y)
        y = self.depthwise(y)
        y = self.glu(y)  # (B, C, T)
        y = self.pointwise_out(y).transpose(1, 2)  # (B, T, C)
        y = self.gate(y)
        y = self.dropout(y)
        return residual + y


class CausalLocalSelfAttention(nn.Module):
    """Multi-head self-attention restricted to a past-only window.

    Each position t attends to [t-window+1 … t]. This gives medium-range
    context without future leakage and without quadratic cost over long seqs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        x = self.norm(x)
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, T, D)

        # Build causal mask: allow attending to past `window` frames only
        # mask[i, j] = True means "block this position"
        idx = torch.arange(T, device=x.device)
        # j can be attended from i if i - window < j <= i
        mask = (idx.unsqueeze(0) - idx.unsqueeze(1)) > self.window  # (T, T)
        # also block future (j > i)
        mask = mask | (idx.unsqueeze(1) < idx.unsqueeze(0))  # causal

        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return residual + self.proj(out)


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class PitchAxisRefine(nn.Module):
    """1-D conv over the pitch axis to reduce octave/harmonic confusion."""

    def __init__(self, n_pitches: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=5, padding=2),
        )

    def forward(
        self, onset_prob: torch.Tensor, frame_logits: torch.Tensor
    ) -> torch.Tensor:
        # onset_prob, frame_logits: (B, T, P)
        b, t, p = frame_logits.shape
        x = torch.stack([frame_logits, onset_prob], dim=2)  # (B, T, 2, P)
        x = x.reshape(b * t, 2, p)
        correction = self.net(x).squeeze(1).reshape(b, t, p)
        return frame_logits + correction


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class CausalGuitarTranscriber(nn.Module):
    """Causal hybrid TCN + attention + GRU guitar transcription teacher.

    Strictly causal: suitable for real-time use and as a distillation teacher.

    Architecture flow:
      features(B,T,165)
        → per-group stems → fuse → (B,T,model_dim)
        → N causal TCN blocks (local short-range, exponential dilation)
        → causal local self-attention (medium-range, past-only window)
        → causal unidirectional GRU (long-range stateful context)
        → shared projection
        → onset head → onset_logits
        → frame head conditioned on onset_prob + prev-frame onset history
        → pitch axis refinement
        → (onset_logits, frame_logits, hidden)

    The `hidden` tensor is used by distillation: students KL-match logits
    and optionally MSE-match the hidden representations.
    """

    def __init__(
        self,
        n_cqt: int = 108,
        n_salience: int = 49,
        n_history: int = 8,
        n_pitches: int = 49,
        model_dim: int = 192,
        gru_hidden: int = 192,
        attn_heads: int = 4,
        attn_window: int = 48,
        n_tcn_blocks: int = 6,
        dropout: float = 0.1,
        onset_history_frames: int = 8,
    ) -> None:
        super().__init__()

        self.n_cqt = n_cqt
        self.n_salience = n_salience
        self.n_history = n_history
        self.n_pitches = n_pitches
        self.onset_history_frames = onset_history_frames

        # --- Input stems ---
        self.cqt_stem = FeatureStem(n_cqt, 128, dropout)
        self.salience_stem = FeatureStem(n_salience, 48, dropout)
        self.history_stem = FeatureStem(n_history, 16, dropout)

        fused_dim = 128 + 48 + 16  # 192

        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Causal TCN: exponential dilation schedule ---
        dilations = [2**i for i in range(n_tcn_blocks)]
        self.tcn_blocks = nn.ModuleList(
            [
                CausalResidualTCNBlock(model_dim, dilation=d, dropout=dropout)
                for d in dilations
            ]
        )

        # --- Causal local self-attention ---
        self.attn = CausalLocalSelfAttention(
            model_dim, num_heads=attn_heads, window=attn_window, dropout=dropout
        )
        self.attn_mlp = ResidualMLP(model_dim, dropout=dropout)

        # --- Causal unidirectional GRU ---
        self.gru = nn.GRU(
            input_size=model_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,  # strictly causal
            dropout=dropout,
        )

        self.shared = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Linear(gru_hidden, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Onset head ---
        self.onset_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, n_pitches),
        )

        # --- Autoregressive onset history embedding ---
        # The model learns to embed its own previous onset predictions
        # and use them as a conditioning signal for frame detection.
        onset_cond_dim = onset_history_frames * n_pitches
        self.onset_history_proj = nn.Sequential(
            nn.LayerNorm(onset_cond_dim),
            nn.Linear(onset_cond_dim, model_dim // 4),
            nn.GELU(),
        )

        # --- Frame head conditioned on onset_prob + onset history ---
        frame_in_dim = model_dim + n_pitches + model_dim // 4
        self.frame_in = nn.Sequential(
            nn.LayerNorm(frame_in_dim),
            nn.Linear(frame_in_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.frame_block_1 = ResidualMLP(model_dim, dropout=dropout)
        self.frame_block_2 = ResidualMLP(model_dim, dropout=dropout)
        self.frame_out = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, n_pitches),
        )

        self.pitch_refine = PitchAxisRefine(n_pitches, hidden=32, dropout=dropout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.n_cqt
        j = i + self.n_salience
        k = j + self.n_history
        return x[..., :i], x[..., i:j], x[..., j:k]

    def _build_onset_history(self, onset_probs: torch.Tensor) -> torch.Tensor:
        """Build causal onset history conditioning.

        For each frame t, concatenate onset_probs[t-K … t-1].
        onset_probs: (B, T, P)
        returns:     (B, T, K*P)
        """
        B, T, P = onset_probs.shape
        K = self.onset_history_frames
        # Pad K zeros on the left (no future leakage, no past leakage)
        padded = F.pad(onset_probs, (0, 0, K, 0))  # (B, T+K, P)
        # Stack K previous frames for each t
        # padded[:, t : t+K, :] are frames [t-K … t-1] for output frame t
        chunks = [padded[:, t : t + K, :] for t in range(T)]
        history = torch.stack(chunks, dim=1)  # (B, T, K, P)
        return history.reshape(B, T, K * P)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 165)

        Returns:
            onset_logits: (B, T, 49)
            frame_logits: (B, T, 49)
            hidden:       (B, T, model_dim)  — for distillation
        """
        cqt, salience, history = self._split_features(x)

        feat = torch.cat(
            [
                self.cqt_stem(cqt),
                self.salience_stem(salience),
                self.history_stem(history),
            ],
            dim=-1,
        )
        feat = self.fuse(feat)  # (B, T, model_dim)

        for block in self.tcn_blocks:
            feat = block(feat)

        feat = self.attn(feat)
        feat = self.attn_mlp(feat)

        feat, _ = self.gru(feat)  # (B, T, gru_hidden)
        feat = self.shared(feat)  # (B, T, model_dim)
        hidden = feat  # expose for distillation

        # Onset prediction
        onset_logits = self.onset_head(feat)  # (B, T, P)
        onset_prob = torch.sigmoid(onset_logits).detach()

        # Build causal onset history conditioning
        onset_hist = self._build_onset_history(onset_prob)  # (B, T, K*P)
        onset_hist_emb = self.onset_history_proj(onset_hist)  # (B, T, D/4)

        # Frame prediction conditioned on current onset + past onset history
        frame = torch.cat([feat, onset_prob, onset_hist_emb], dim=-1)
        frame = self.frame_in(frame)
        frame = self.frame_block_1(frame)
        frame = self.frame_block_2(frame)
        frame_logits = self.frame_out(frame)
        frame_logits = self.pitch_refine(onset_prob, frame_logits)

        return onset_logits, frame_logits, hidden
