"""Build segment-level features from a raw dataset (local, no Google Drive).

Expects raw audio laid out as:
    data/raw/<raw_label>/*.wav      e.g. data/raw/belly_pain/0a1.wav

Runs the full notebook pipeline end to end and writes:
    features/raw_features.npy   (N_segments, 346)
    features/labels.npy         (N_segments,)  collapsed class names
    features/groups.npy         (N_segments,)  file_id for GroupKFold

Then train with:  python -m voxinfant.train

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --max-distress 8000 --max-pain 1500 --max-burping 1500
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Make the package importable when run as a plain script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voxinfant import dsp, features, segmentation          # noqa: E402
from voxinfant.config import DATA_DIR, FEATURES_DIR, get_config  # noqa: E402

CFG = get_config()
RAW_DIR = os.path.join(DATA_DIR, "raw")


def build_metadata() -> pd.DataFrame:
    rows, counter = [], 0
    for raw_label in sorted(os.listdir(RAW_DIR)):
        class_path = os.path.join(RAW_DIR, raw_label)
        if not os.path.isdir(class_path) or raw_label not in CFG.class_map:
            continue
        mapped = CFG.class_map[raw_label]
        for fname in os.listdir(class_path):
            if not fname.endswith(".wav"):
                continue
            rows.append({
                "file_path": os.path.join(class_path, fname),
                "label": mapped,
                "file_id": f"file_{counter}",
            })
            counter += 1
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No .wav files found under {RAW_DIR} matching class_map. Check layout.")
    print("Files:", len(df))
    print(df["label"].value_counts())
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-distress", type=int, default=None)
    ap.add_argument("--max-pain", type=int, default=None)
    ap.add_argument("--max-burping", type=int, default=None)
    ap.add_argument("--rich", action="store_true", help="extract the richer 134-d feature set")
    ap.add_argument("--out", default="raw_features", help="output basename under features/")
    args = ap.parse_args()

    os.makedirs(FEATURES_DIR, exist_ok=True)
    df = build_metadata()

    # Process minority classes first (so caps never starve pain/burping), and
    # shuffle within so a distress cap samples across ALL distress subtypes
    # (cold_hot/discomfort/hungry/lonely/scared/tired), not just the first folders.
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["_order"] = df["label"].map({"pain": 0, "burping": 1, "distress": 2}).fillna(3)
    df = df.sort_values("_order", kind="stable").reset_index(drop=True)

    X, y, groups = [], [], []
    caps = {"distress": args.max_distress, "pain": args.max_pain, "burping": args.max_burping}
    seg_counts = {k: 0 for k in caps}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="files"):
        y_audio, sr = dsp.process_audio(row["file_path"])
        if y_audio is None:
            continue

        y_voiced = segmentation.apply_vad(y_audio, sr)
        for seg in segmentation.segment_audio(y_voiced, sr):
            label = row["label"]
            cap = caps.get(label)
            if cap is not None and seg_counts[label] >= cap:
                continue
            try:
                vec = features.build_segment_vector(seg, sr, rich=args.rich)
            except Exception as e:  # noqa: BLE001
                print("segment feature error:", e)
                continue
            X.append(vec); y.append(label); groups.append(row["file_id"])
            seg_counts[label] += 1

        # Early stop: distress is processed last (see _order). Once its cap is
        # met, every remaining file is distress and would be skipped anyway, so
        # stop decoding them (saves a large tail of wasted DSP).
        dcap = caps.get("distress")
        if dcap is not None and seg_counts["distress"] >= dcap:
            break

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    print("Feature matrix:", X.shape, "(expected dim:", CFG.expected_dim, ")")

    # Default basename keeps the legacy file names; a custom --out gets suffixed
    # label/group files so experiments never clobber the deployed 62-d arrays.
    if args.out == "raw_features":
        fn, ln, gn = "raw_features.npy", "labels.npy", "groups.npy"
    else:
        fn, ln, gn = f"{args.out}.npy", f"{args.out}_labels.npy", f"{args.out}_groups.npy"
    np.save(os.path.join(FEATURES_DIR, fn), X)
    np.save(os.path.join(FEATURES_DIR, ln), y)
    np.save(os.path.join(FEATURES_DIR, gn), groups)
    print("Saved features to", FEATURES_DIR, "as", fn, ln, gn)


if __name__ == "__main__":
    main()
