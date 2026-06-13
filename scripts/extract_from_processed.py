"""Re-extract 62-d features LOCALLY from the DSP'd processed_audio files.

Why this exists: features computed in Colab do not reproduce at inference on a
different machine (library drift in spafe/transformers). Re-extracting with the
SAME code that serves inference makes train == inference exactly.

Input:
    data/processed_audio/<file_id>.wav   (already DSP'd, 16 kHz -- do NOT re-DSP)
    data/segment_metadata.csv            (file_id -> label)

Output:
    features/raw_features.npy   (N, 62)
    features/labels.npy         (N,)
    features/groups.npy         (N,)  file_id

Then:  python -m voxinfant.train
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voxinfant import features as F             # noqa: E402
from voxinfant import segmentation              # noqa: E402
from voxinfant.config import DATA_DIR, FEATURES_DIR, get_config  # noqa: E402

CFG = get_config()
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_audio")
META_CSV = os.path.join(DATA_DIR, "segment_metadata.csv")

DISTRESS_CAP = 8000   # cap distress segments (matches notebook subset); keep all pain/burping
SEED = 42


def main() -> None:
    import librosa

    meta = pd.read_csv(META_CSV)
    label_of = meta.groupby("file_id")["label"].first().to_dict()

    # Deterministic file order; process minority classes first so caps never
    # starve them.
    files = []
    for fid, label in label_of.items():
        path = os.path.join(PROCESSED_DIR, f"{fid}.wav")
        if os.path.exists(path):
            files.append((fid, label, path))
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)
    files.sort(key=lambda t: {"pain": 0, "burping": 1, "distress": 2}.get(t[1], 3))

    X, y, groups = [], [], []
    distress_segs = 0

    for fid, label, path in tqdm(files, desc="files"):
        try:
            audio, sr = librosa.load(path, sr=CFG.target_sr)  # already DSP'd
        except Exception as e:  # noqa: BLE001
            print("load error", fid, e)
            continue

        voiced = segmentation.apply_vad(audio, sr)
        for seg in segmentation.segment_audio(voiced, sr):
            if label == "distress" and distress_segs >= DISTRESS_CAP:
                continue
            try:
                vec = F.build_segment_vector(seg, sr)
            except Exception as e:  # noqa: BLE001
                print("feat error", fid, e)
                continue
            X.append(vec); y.append(label); groups.append(fid)
            if label == "distress":
                distress_segs += 1

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    print("Feature matrix:", X.shape, "(expected dim", CFG.expected_dim, ")")
    from collections import Counter
    print("class counts:", dict(Counter(y.tolist())))

    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.save(os.path.join(FEATURES_DIR, "raw_features.npy"), X)
    np.save(os.path.join(FEATURES_DIR, "labels.npy"), y)
    np.save(os.path.join(FEATURES_DIR, "groups.npy"), groups)
    print("Saved to", FEATURES_DIR)


if __name__ == "__main__":
    main()
