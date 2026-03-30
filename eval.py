# eval.py
"""Evaluate a trained CausalGuitarTranscriber on a WAV file → MIDI.

Usage:
    python eval.py path/to/audio.wav --checkpoint teacher_model.pt
    python eval.py path/to/audio.wav --checkpoint teacher_model.pt \
        --onset-thresh 0.35 --frame-thresh 0.4 --out output.mid

Algorithm (frame-level → note-level):
  1. Extract 165-d features from WAV (same pipeline as precompute_165.py).
  2. Run model in causal chunks to produce per-frame onset/frame logits.
  3. Convert piano-roll to note events using onset gating:
       - A note ON fires when onset_prob > onset_thresh.
       - A note stays active while frame_prob > frame_thresh.
       - A note OFF fires when frame_prob drops below frame_thresh.
  4. Write MIDI via pretty_midi.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import pretty_midi

from model import CausalGuitarTranscriber

# ── Feature extraction constants (must match precompute_165.py) ──────────
FEAT_SR = 44100
FEAT_HOP = 1024
N_CQT = 108
BINS_PER_OCTAVE = 12
FMIN = librosa.note_to_hz("C1")

MIDI_MIN = 40
MIDI_MAX = 88
N_PITCHES = MIDI_MAX - MIDI_MIN + 1
HISTORY = 8

HARM_WEIGHTS = np.array([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=np.float32)


# ── Feature extraction (mirrors precompute_165.py exactly) ───────────────


def _build_salience_lookup() -> list[list[tuple[int, float]]]:
    freqs = librosa.cqt_frequencies(
        n_bins=N_CQT, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE
    )
    lookup: list[list[tuple[int, float]]] = []
    for midi in range(MIDI_MIN, MIDI_MAX + 1):
        fund_hz = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        bins: list[tuple[int, float]] = []
        for harm_idx, weight in enumerate(HARM_WEIGHTS, start=1):
            target_hz = fund_hz * harm_idx
            ratio = np.maximum(freqs / target_hz, 1e-10)
            cents = 1200.0 * np.abs(np.log2(ratio))
            best_bin = int(np.argmin(cents))
            if cents[best_bin] < 50.0:
                bins.append((best_bin, float(weight)))
        lookup.append(bins)
    return lookup


_SALIENCE_LOOKUP = _build_salience_lookup()


def extract_features(wav_path: Path) -> np.ndarray:
    """Return (T, 165) float32 feature array for the given WAV file."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != FEAT_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=FEAT_SR)

    # CQT
    cqt = librosa.cqt(
        audio,
        sr=FEAT_SR,
        hop_length=FEAT_HOP,
        n_bins=N_CQT,
        bins_per_octave=BINS_PER_OCTAVE,
        fmin=FMIN,
    )
    cqt_mag = np.abs(cqt).T.astype(np.float32)
    peak = np.maximum(cqt_mag.max(axis=1, keepdims=True), 1e-9)
    cqt_norm = cqt_mag / peak

    # Salience
    n_frames = cqt_norm.shape[0]
    salience_raw = np.zeros((n_frames, N_PITCHES), dtype=np.float32)
    for pitch_idx, mapping in enumerate(_SALIENCE_LOOKUP):
        for bin_idx, weight in mapping:
            salience_raw[:, pitch_idx] += weight * cqt_norm[:, bin_idx]
    denom = np.maximum(salience_raw.max(axis=1, keepdims=True), 1e-9)
    salience = salience_raw / denom

    # Peak-salience history
    peak_salience = salience_raw.max(axis=1).astype(np.float32)
    history = np.zeros((n_frames, HISTORY), dtype=np.float32)
    for t in range(n_frames):
        for k in range(HISTORY):
            src = t - (HISTORY - 1 - k)
            if src >= 0:
                history[t, k] = peak_salience[src]

    features = np.concatenate([cqt_norm, salience, history], axis=1)
    assert features.shape[1] == 165, f"Expected 165 cols, got {features.shape[1]}"
    return features.astype(np.float32)


# ── Inference ────────────────────────────────────────────────────────────


@torch.no_grad()
def run_model(
    features: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Run trained model on features, return (onset_probs, frame_probs)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Support both bare state-dict and wrapped checkpoint
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        saved_args = ckpt.get("args", {})
    else:
        state = ckpt
        saved_args = {}

    model = CausalGuitarTranscriber(
        n_cqt=108,
        n_salience=49,
        n_history=8,
        n_pitches=49,
        model_dim=saved_args.get("model_dim", 192),
        gru_hidden=saved_args.get("gru_hidden", 192),
        n_tcn_blocks=saved_args.get("n_tcn_blocks", 6),
        dropout=0.0,  # disable dropout at eval time
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    n_frames, feat_dim = features.shape
    onset_all = np.zeros((n_frames, N_PITCHES), dtype=np.float32)
    frame_all = np.zeros((n_frames, N_PITCHES), dtype=np.float32)

    # Process in chunks; GRU state carries across chunks for proper causality
    gru_state: torch.Tensor | None = None

    # Temporarily hook into the GRU to pass/receive state across chunks.
    # We do this by monkey-patching forward for chunked inference.
    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        chunk = torch.from_numpy(features[start:end]).unsqueeze(0).to(device)

        # ── manual forward with GRU state threading ──
        cqt, salience, history = model._split_features(chunk)
        feat = torch.cat(
            [
                model.cqt_stem(cqt),
                model.salience_stem(salience),
                model.history_stem(history),
            ],
            dim=-1,
        )
        feat = model.fuse(feat)
        for block in model.tcn_blocks:
            feat = block(feat)
        feat = model.attn(feat)
        feat = model.attn_mlp(feat)
        feat, gru_state = model.gru(feat, gru_state)
        feat = model.shared(feat)

        onset_logits = model.onset_head(feat)
        onset_prob = torch.sigmoid(onset_logits)

        onset_hist = model._build_onset_history(onset_prob.detach())
        onset_hist_emb = model.onset_history_proj(onset_hist)
        frame = torch.cat([feat, onset_prob.detach(), onset_hist_emb], dim=-1)
        frame = model.frame_in(frame)
        frame = model.frame_block_1(frame)
        frame = model.frame_block_2(frame)
        frame_logits = model.frame_out(frame)
        frame_logits = model.pitch_refine(onset_prob.detach(), frame_logits)

        onset_all[start:end] = (
            torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()
        )
        frame_all[start:end] = (
            torch.sigmoid(frame_logits).squeeze(0).cpu().numpy()
        )
        start = end

    return onset_all, frame_all


# ── Piano-roll → MIDI ────────────────────────────────────────────────────


def pianoroll_to_midi(
    onset_probs: np.ndarray,
    frame_probs: np.ndarray,
    onset_thresh: float,
    frame_thresh: float,
    hop_secs: float,
    midi_min: int = MIDI_MIN,
    program: int = 25,   # acoustic guitar (nylon) — GM program 25
    velocity: int = 80,
) -> pretty_midi.PrettyMIDI:
    """Convert per-frame probability arrays to a PrettyMIDI object.

    Note onset gating algorithm:
      - A pitch activates when onset_prob > onset_thresh AND frame_prob > frame_thresh.
      - It deactivates when frame_prob drops below frame_thresh.
      - Re-triggered only if a new onset fires during an active note.
    """
    n_frames, n_pitches = frame_probs.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, name="Guitar")

    active_start: dict[int, float] = {}  # pitch → onset time (seconds)

    def _close(pitch: int, t_frame: int) -> None:
        if pitch in active_start:
            t_on = active_start.pop(pitch)
            t_off = t_frame * hop_secs
            if t_off > t_on + 1e-3:  # ignore sub-ms notes
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch + midi_min,
                    start=t_on,
                    end=t_off,
                )
                instrument.notes.append(note)

    for t in range(n_frames):
        for p in range(n_pitches):
            is_active = p in active_start
            frame_on = frame_probs[t, p] > frame_thresh
            onset_on = onset_probs[t, p] > onset_thresh

            if not is_active:
                # Start note only on onset + frame agreement
                if onset_on and frame_on:
                    active_start[p] = t * hop_secs
            else:
                if onset_on:
                    # Re-trigger: close old, open new
                    _close(p, t)
                    active_start[p] = t * hop_secs
                elif not frame_on:
                    _close(p, t)

    # Close all still-active notes at end of audio
    for p in list(active_start.keys()):
        _close(p, n_frames)

    instrument.notes.sort(key=lambda n: n.start)
    pm.instruments.append(instrument)
    return pm


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV file to MIDI using a trained model."
    )
    parser.add_argument("wav", type=Path, help="Input WAV file")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("teacher_model.pt")
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output MIDI path (default: <wav stem>.mid)"
    )
    parser.add_argument(
        "--onset-thresh", type=float, default=0.35,
        help="Onset probability threshold (lower → more notes)"
    )
    parser.add_argument(
        "--frame-thresh", type=float, default=0.40,
        help="Frame probability threshold (lower → longer notes)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="Frames per inference chunk (trades memory vs. speed)"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    out_path = args.out or args.wav.with_suffix(".mid")

    print(f"Extracting features from {args.wav} …")
    features = extract_features(args.wav)
    n_frames = features.shape[0]
    duration = n_frames * FEAT_HOP / FEAT_SR
    print(
        f"  {n_frames:,} frames  ({duration:.1f}s)  "
        f"@ {FEAT_SR}Hz / hop {FEAT_HOP}"
    )

    print(f"Running model ({device}) …")
    onset_probs, frame_probs = run_model(
        features, args.checkpoint, device, chunk_size=args.chunk_size
    )

    hop_secs = FEAT_HOP / FEAT_SR
    print(
        f"Converting to MIDI  "
        f"(onset_thresh={args.onset_thresh}, frame_thresh={args.frame_thresh}) …"
    )
    pm = pianoroll_to_midi(
        onset_probs,
        frame_probs,
        onset_thresh=args.onset_thresh,
        frame_thresh=args.frame_thresh,
        hop_secs=hop_secs,
    )

    n_notes = sum(len(i.notes) for i in pm.instruments)
    print(f"  Detected {n_notes} notes")

    pm.write(str(out_path))
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
