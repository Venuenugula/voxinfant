"""Reusable GroupKFold evaluator for feature-set experiments.

Reproduces the deployable pipeline's methodology (downsample distress to the
minority count, global StandardScaler, XGB+MLP soft-vote, argmax decision) and
reports aggregate out-of-fold macro-F1, per-class report, and confusion matrix.

Point it at any feature matrix to compare experiments apples-to-apples:

    python scripts/cv_eval.py --features features/raw_features.npy            # baseline (62-d)
    python scripts/cv_eval.py --features features/feat_v2.npy                 # richer set
    python scripts/cv_eval.py --features features/raw_features.npy --dim 62   # slice legacy dims
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voxinfant.train import _balance, _xgb, _mlp, _weights_by_index  # noqa: E402
from voxinfant.ensemble import WeightedSoftVoteEnsemble  # noqa: E402

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/raw_features.npy")
    ap.add_argument("--labels", default="features/labels.npy")
    ap.add_argument("--groups", default="features/groups.npy")
    ap.add_argument("--dim", type=int, default=None, help="optional: slice to first N columns")
    args = ap.parse_args()

    np.random.seed(SEED)
    X = np.load(args.features).astype(np.float32)
    y = np.load(args.labels, allow_pickle=True)
    groups = np.load(args.groups, allow_pickle=True)
    if args.dim:
        X = X[:, : args.dim]
    print(f"Features: {args.features}  X={X.shape}")

    le = LabelEncoder(); y_enc = le.fit_transform(y)
    distress_c = le.transform(["distress"])[0]
    Xb, yb, gb = _balance(X, y_enc, groups, distress_c)
    print(f"Balanced: {Xb.shape}  classes={list(le.classes_)}")

    # Global scaling (matches train.main).
    scaler = StandardScaler(); Xb = scaler.fit_transform(Xb)
    weights = _weights_by_index(le)

    gkf = GroupKFold(n_splits=5)
    oof_true, oof_pred = [], []
    f1s = []
    for fold, (tr, va) in enumerate(gkf.split(Xb, yb, gb), 1):
        sw = np.array([weights[i] for i in yb[tr]])
        xgb = _xgb(); xgb.fit(Xb[tr], yb[tr], sample_weight=sw,
                              eval_set=[(Xb[va], yb[va])], verbose=False)
        mlp = _mlp(); mlp.fit(Xb[tr], yb[tr])
        ens = WeightedSoftVoteEnsemble(xgb, mlp)
        pred = ens.predict(Xb[va])
        f1s.append(f1_score(yb[va], pred, average="macro"))
        oof_true.extend(yb[va]); oof_pred.extend(pred)
        print(f"  fold {fold}: macro-F1={f1s[-1]:.4f}")

    oof_true = np.array(oof_true); oof_pred = np.array(oof_pred)
    print(f"\nMean fold macro-F1 = {np.mean(f1s):.4f} (+/-{np.std(f1s):.4f})")
    print(f"OOF macro-F1 = {f1_score(oof_true, oof_pred, average='macro'):.4f}  "
          f"acc = {accuracy_score(oof_true, oof_pred):.4f}\n")
    print(classification_report(oof_true, oof_pred, target_names=le.classes_, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{c[:7]:>7}" for c in le.classes_))
    cm = confusion_matrix(oof_true, oof_pred)
    for c, row in zip(le.classes_, cm):
        print(f"{c[:7]:>7} " + "  ".join(f"{v:>7}" for v in row))


if __name__ == "__main__":
    main()
