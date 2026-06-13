"""Augmentation-aware feature extraction for the minority classes.

For pain & burping: each source file yields ORIGINAL segments plus segments from
pitch/time/noise-augmented copies of the audio. For distress: originals only.
Every row is tagged with is_aug (0/1) and its source file_id (group), so the CV
harness can train on originals+augmented but TEST ON ORIGINALS ONLY (no leakage).

Writes (62-d, fast):
    features/feat_aug.npy / _labels.npy / _groups.npy / _isaug.npy
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voxinfant import dsp, features, segmentation  # noqa: E402
from voxinfant.config import DATA_DIR, FEATURES_DIR, get_config  # noqa: E402

CFG = get_config()
RAW_DIR = os.path.join(DATA_DIR, "raw")
SEED = 42
MAX_DISTRESS = 6000


def variants(y, sr, augment: bool):
    """Return [(tag, audio), ...]; original first. Deterministic."""
    import librosa
    out = [("orig", y)]
    if not augment:
        return out
    out.append(("ps_up", librosa.effects.pitch_shift(y, sr=sr, n_steps=2)))
    out.append(("ps_dn", librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)))
    out.append(("ts_slow", librosa.effects.time_stretch(y, rate=0.9)))
    peak = float(np.max(np.abs(y))) or 1.0
    noise = np.random.RandomState(SEED).randn(len(y)).astype(np.float32) * 0.005 * peak
    out.append(("noise", (y + noise).astype(np.float32)))
    return out


def main() -> None:
    rows = []
    for raw_label in sorted(os.listdir(RAW_DIR)):
        cp = os.path.join(RAW_DIR, raw_label)
        if not os.path.isdir(cp) or raw_label not in CFG.class_map:
            continue
        mapped = CFG.class_map[raw_label]
        for fn in os.listdir(cp):
            if fn.endswith(".wav"):
                rows.append({"path": os.path.join(cp, fn), "label": mapped})
    df = pd.DataFrame(rows)
    # minority first; shuffle so distress cap samples across subtypes
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df["_o"] = df["label"].map({"pain": 0, "burping": 1, "distress": 2}).fillna(3)
    df = df.sort_values("_o", kind="stable").reset_index(drop=True)
    print(df["label"].value_counts().to_dict())

    X, y, groups, isaug = [], [], [], []
    d_count = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="files"):
        label = row["label"]
        if label == "distress" and d_count >= MAX_DISTRESS:
            break  # distress is last; nothing useful remains
        y_audio, sr = dsp.process_audio(row["path"])
        if y_audio is None:
            continue
        fid = f"file_{i}"
        augment = label in ("pain", "burping")
        for tag, ya in variants(y_audio, sr, augment):
            yv = segmentation.apply_vad(ya, sr)
            for seg in segmentation.segment_audio(yv, sr):
                if label == "distress" and d_count >= MAX_DISTRESS:
                    break
                try:
                    vec = features.build_segment_vector(seg, sr, rich=False)
                except Exception:  # noqa: BLE001
                    continue
                X.append(vec); y.append(label); groups.append(fid)
                isaug.append(0 if tag == "orig" else 1)
                if label == "distress":
                    d_count += 1

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y); groups = np.asarray(groups); isaug = np.asarray(isaug, dtype=np.int8)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.save(os.path.join(FEATURES_DIR, "feat_aug.npy"), X)
    np.save(os.path.join(FEATURES_DIR, "feat_aug_labels.npy"), y)
    np.save(os.path.join(FEATURES_DIR, "feat_aug_groups.npy"), groups)
    np.save(os.path.join(FEATURES_DIR, "feat_aug_isaug.npy"), isaug)
    n_orig = int((isaug == 0).sum()); n_aug = int((isaug == 1).sum())
    print(f"X={X.shape}  originals={n_orig}  augmented={n_aug}")
    import collections
    print("orig per class:", collections.Counter(y[isaug == 0]))
    print("aug  per class:", collections.Counter(y[isaug == 1]))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
