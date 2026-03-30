#!/usr/bin/env python3
"""package_for_training.py

Bundles everything needed to train on a remote server into a single .tar.gz:
  - Precomputed feature cache (features_165.pkl) — NOT raw audio
  - Model/training source files
  - requirements.txt (auto-generated from your current environment)

Usage:
    python package_for_training.py <data_dir> [options]

    # Basic
    python package_for_training.py ./guitarset_data

    # Custom output path
    python package_for_training.py ./guitarset_data --out ~/uploads/run1.tar.gz

    # Different cache file name
    python package_for_training.py ./guitarset_data --cache features_165.pkl
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# ── Source files to include (relative to this script's directory) ─────────

SOURCE_FILES = [
    "model.py",
    "train.py",
    "dataset.py",
]

# These are pulled in automatically if present alongside this script
OPTIONAL_SOURCE_FILES = [
    "precompute_165.py",
    "processing.py",
    "eval.py",
]

# ── Pinned requirements (everything the training loop actually needs) ─────

REQUIREMENTS = """\
torch>=2.2.0
numpy>=1.24
librosa>=0.10
soundfile>=0.12
tqdm>=4.65
pretty_midi>=0.2.10
"""


# ── Helpers ───────────────────────────────────────────────────────────────


def find_file(base: Path, name: str) -> Path:
    p = base / name
    if not p.exists():
        raise FileNotFoundError(
            f"Expected {p}\n"
            f"Run:  python precompute_165.py {base}"
        )
    return p


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def human_size(path: Path) -> str:
    n = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def make_requirements(dst: Path) -> None:
    dst.write_text(REQUIREMENTS)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package training artefacts into a portable .tar.gz"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing the precomputed feature cache",
    )
    parser.add_argument(
        "--cache",
        default="features_165.pkl",
        help="Cache filename inside data_dir (default: features_165.pkl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .tar.gz path (default: ./training_bundle.tar.gz)",
    )
    parser.add_argument(
        "--no-source",
        action="store_true",
        help="Omit Python source files (cache + requirements only)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_path = (args.out or Path("training_bundle.tar.gz")).resolve()
    src_dir = script_dir()

    print("─" * 56)
    print("  Guitar transcription — training bundle packager")
    print("─" * 56)

    # ── 1. Locate the feature cache ───────────────────────────────────────
    cache_path = find_file(data_dir, args.cache)
    print(f"  cache   {cache_path}  ({human_size(cache_path)})")

    # ── 2. Locate source files ────────────────────────────────────────────
    sources: list[Path] = []
    if not args.no_source:
        missing_required = []
        for name in SOURCE_FILES:
            p = src_dir / name
            if p.exists():
                sources.append(p)
                print(f"  source  {p.name}")
            else:
                missing_required.append(name)

        if missing_required:
            print(
                f"\n  ERROR: required source files not found in {src_dir}:\n"
                + "".join(f"    • {n}\n" for n in missing_required),
                file=sys.stderr,
            )
            sys.exit(1)

        for name in OPTIONAL_SOURCE_FILES:
            p = src_dir / name
            if p.exists():
                sources.append(p)
                print(f"  source  {p.name}  (optional)")

    # ── 3. Build the archive in a temp dir ────────────────────────────────
    print(f"\n  Building → {out_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        staging = tmp_path / "training_bundle"
        staging.mkdir()

        # Feature cache goes inside a "data/" sub-folder so the training
        # script can be called as:  python train.py data/
        data_staging = staging / "data"
        data_staging.mkdir()
        shutil.copy2(cache_path, data_staging / args.cache)

        # Python source files sit at the root of the bundle
        for src in sources:
            shutil.copy2(src, staging / src.name)

        # requirements.txt
        req_path = staging / "requirements.txt"
        make_requirements(req_path)
        print(f"  wrote   requirements.txt")

        # Minimal README so the server operator knows what to run
        readme = staging / "README.txt"
        readme.write_text(
            "Guitar transcription — training bundle\n"
            "======================================\n\n"
            "1. Install dependencies:\n"
            "     pip install -r requirements.txt\n\n"
            "2. Precompute features (already done — cache is bundled):\n"
            f"     data/{args.cache}  ← use this\n\n"
            "3. Train:\n"
            "     python train.py data/\n\n"
            "4. Evaluate:\n"
            "     python eval.py recording.wav --checkpoint teacher_model.pt\n"
        )

        # Create the tar
        with tarfile.open(out_path, "w:gz", compresslevel=6) as tar:
            tar.add(staging, arcname="training_bundle")

    # ── 4. Summary ────────────────────────────────────────────────────────
    print(f"\n  ✓  Done  ({human_size(out_path)})")
    print(f"\n  To upload and unpack on your server:\n")
    print(f"    scp {out_path} user@host:~/")
    print(f"    ssh user@host")
    print(f"    tar -xzf {out_path.name}")
    print(f"    cd training_bundle")
    print(f"    pip install -r requirements.txt")
    print(f"    python train.py data/")
    print()


if __name__ == "__main__":
    main()
