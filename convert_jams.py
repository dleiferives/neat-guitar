#!/usr/bin/env python3
"""
Convert JAMS annotation files to a simple per-recording JSON dataset.

Each JAMS file may contain up to 6 note_midi annotation tracks (one per
guitar string). This script merges all tracks, rounds the continuous MIDI
pitch values to the nearest integer, and writes one output JSON per recording:

  data/processed/<stem>.json
  {
    "name": "<stem>",
    "notes": [
      {"midi": 44, "time": 0.049, "duration": 0.424},
      ...
    ]
  }

Note times in JAMS are already relative to the start of the audio clip,
so no offset correction is needed. The annotation-level "time" field is
the clip's position within the full session recording and is ignored here.

Usage:
  python3 convert_jams.py [data_root]   (default: ./data)
"""

import json
import os
import sys
from pathlib import Path


def convert_jams(jams_path: Path) -> list[dict]:
    """Extract and merge all note_midi events from a JAMS file."""
    with open(jams_path) as f:
        jams = json.load(f)

    notes = []
    for ann in jams.get("annotations", []):
        if ann.get("namespace") != "note_midi":
            continue

        data = ann.get("data", [])
        if not isinstance(data, list):
            # Some annotations use columnar dict format (e.g. pitch_contour);
            # note_midi always uses list format, so skip anything else.
            continue

        for entry in data:
            midi_val = entry.get("value")
            time_val = entry.get("time")
            dur_val  = entry.get("duration")

            if midi_val is None or time_val is None or dur_val is None:
                continue

            notes.append({
                "midi":     round(float(midi_val)),
                "time":     float(time_val),
                "duration": float(dur_val),
            })

    # Sort by onset time so the C++ loader gets a clean ordered list
    notes.sort(key=lambda n: n["time"])
    return notes


def main():
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    ann_dir   = data_root / "annotation"
    out_dir   = data_root / "processed"

    if not ann_dir.exists():
        print(f"ERROR: annotation directory not found: {ann_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(exist_ok=True)

    jams_files = sorted(ann_dir.glob("*.jams"))
    if not jams_files:
        print(f"No .jams files found in {ann_dir}", file=sys.stderr)
        sys.exit(1)

    converted = 0
    skipped   = 0

    for jams_path in jams_files:
        stem = jams_path.stem  # e.g. "00_BN1-129-Eb_comp"

        try:
            notes = convert_jams(jams_path)
        except Exception as e:
            print(f"[skip] {stem}: {e}")
            skipped += 1
            continue

        out_path = out_dir / f"{stem}.json"
        with open(out_path, "w") as f:
            json.dump({"name": stem, "notes": notes}, f, indent=2)

        midi_set = sorted({n["midi"] for n in notes})
        print(f"  {stem}: {len(notes)} notes  midi={midi_set[0]}-{midi_set[-1] if midi_set else '?'}")
        converted += 1

    print(f"\nDone: {converted} converted, {skipped} skipped  ->  {out_dir}/")


if __name__ == "__main__":
    main()
