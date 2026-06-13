"""Retraining entrypoint — regenerates the deployable artifacts.

Reads precomputed 62-d segment features (features/raw_features.npy, labels.npy,
groups.npy — from scripts/prepare_data.py), trains the HIERARCHICAL classifier
(distress-vs-rest, then pain-vs-burping), reports GroupKFold metrics at both the
segment and file level, fits the final model on all data, and writes:

    models/voxinfant_ensemble.pkl   (a HierarchicalCryClassifier)
    models/scaler.pkl
    models/label_classes.json

The hierarchical model was chosen over the flat soft-vote ensemble: it lifts
pain/burping recall markedly at the file level (the metric users experience)
instead of defaulting everything to distress. See docs/experiments.md.

Usage:
    python -m voxinfant.train
"""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import FEATURES_DIR, MODELS_DIR, get_config
from .ensemble import HierarchicalCryClassifier

SEED = 42
EXPECTED_DIM = get_config().expected_dim  # 62 (acoustic + gfcc)


def _xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.6,
        reg_lambda=3.0, reg_alpha=2.0, gamma=2.0,
        tree_method="hist", eval_metric="mlogloss", random_state=SEED,
    )


def fit_hierarchical(X, y_enc, distress_idx: int, n_classes: int) -> HierarchicalCryClassifier:
    """Stage 1: distress-vs-rest (balanced so minorities aren't drowned).
    Stage 2: pain-vs-burping on the rest branch (remapped to 0..k-1)."""
    is_d = (y_enc == distress_idx).astype(int)
    # 3-class balanced weights up-weight pain+burping vs distress -> better recall.
    sw1 = compute_sample_weight("balanced", y_enc)
    s1 = _xgb(); s1.fit(X, is_d, sample_weight=sw1, verbose=False)

    mask = y_enc != distress_idx
    rest = sorted(set(y_enc[mask].tolist()))            # global indices, e.g. [0, 2]
    remap = {c: i for i, c in enumerate(rest)}
    y2 = np.array([remap[c] for c in y_enc[mask]])
    s2 = _xgb(); s2.fit(X[mask], y2,
                        sample_weight=compute_sample_weight("balanced", y2), verbose=False)
    return HierarchicalCryClassifier(s1, s2, distress_idx, rest, n_classes)


def load_features(features_dir: str = FEATURES_DIR):
    X = np.load(os.path.join(features_dir, "raw_features.npy"))
    y = np.load(os.path.join(features_dir, "labels.npy"), allow_pickle=True)
    groups = np.load(os.path.join(features_dir, "groups.npy"), allow_pickle=True)
    if X.shape[1] > EXPECTED_DIM:          # tolerate legacy wider feature files
        X = X[:, :EXPECTED_DIM]
    return X.astype(np.float32), y, groups


def _file_level(groups, proba, y_true):
    """Mean-prob over each file's segments -> argmax. Returns (true, pred)."""
    agg = defaultdict(lambda: np.zeros(proba.shape[1])); cnt = defaultdict(int); truth = {}
    for g, p, yt in zip(groups, proba, y_true):
        agg[g] += p; cnt[g] += 1; truth[g] = yt
    t, pr = [], []
    for g in agg:
        t.append(truth[g]); pr.append(int(np.argmax(agg[g] / cnt[g])))
    return np.array(t), np.array(pr)


def cross_validate(X, y_enc, groups, le) -> None:
    distress_idx = le.transform(["distress"])[0]
    nC = len(le.classes_)
    gkf = GroupKFold(n_splits=5)
    seg_t, seg_p, file_t, file_p, seg_f1s, file_f1s = [], [], [], [], [], []
    for fold, (tr, va) in enumerate(gkf.split(X, y_enc, groups), 1):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        model = fit_hierarchical(Xtr, y_enc[tr], distress_idx, nC)
        proba = model.predict_proba(Xva)
        sp = np.argmax(proba, axis=1)
        ft, fp = _file_level(groups[va], proba, y_enc[va])
        seg_f1s.append(f1_score(y_enc[va], sp, average="macro"))
        file_f1s.append(f1_score(ft, fp, average="macro"))
        seg_t.extend(y_enc[va]); seg_p.extend(sp); file_t.extend(ft); file_p.extend(fp)
        print(f"  fold {fold}: seg-F1={seg_f1s[-1]:.4f}  file-F1={file_f1s[-1]:.4f}")

    seg_t, seg_p = np.array(seg_t), np.array(seg_p)
    file_t, file_p = np.array(file_t), np.array(file_p)
    print(f"\nSEGMENT  macro-F1 = {f1_score(seg_t, seg_p, average='macro'):.4f}  "
          f"acc = {accuracy_score(seg_t, seg_p):.4f}")
    print(f"FILE     macro-F1 = {f1_score(file_t, file_p, average='macro'):.4f}  "
          f"acc = {accuracy_score(file_t, file_p):.4f}   (the metric users experience)\n")
    print("File-level report:")
    print(classification_report(file_t, file_p, target_names=le.classes_, digits=3))
    print("File-level confusion (rows=true):", le.classes_.tolist())
    for c, r in zip(le.classes_, confusion_matrix(file_t, file_p)):
        print(f"  {c[:7]:>7}", r.tolist())


def save_artifacts(model, scaler, le, models_dir: str = MODELS_DIR) -> None:
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "voxinfant_ensemble.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, "label_classes.json"), "w") as f:
        json.dump(list(le.classes_), f)
    print(f"\nSaved artifacts to {models_dir}/")


def main() -> None:
    np.random.seed(SEED)
    X, y, groups = load_features()
    print(f"Loaded features: X={X.shape}, classes={sorted(set(y))}")

    le = LabelEncoder(); y_enc = le.fit_transform(y)
    distress_idx = le.transform(["distress"])[0]

    cross_validate(X, y_enc, groups, le)

    # Final fit on ALL data: one global scaler (also used at inference) + model.
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    model = fit_hierarchical(X_scaled, y_enc, distress_idx, len(le.classes_))
    save_artifacts(model, scaler, le)


if __name__ == "__main__":
    main()
