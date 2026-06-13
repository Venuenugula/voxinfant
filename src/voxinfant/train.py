"""Retraining entrypoint — regenerates the deployable artifacts.

Reads precomputed segment features (features/raw_features.npy, labels.npy,
groups.npy — produced by scripts/prepare_data.py), reproduces the notebook's
FINAL PUBLISHABLE pipeline, reports 5-fold GroupKFold metrics, then fits a single
final ensemble on all (balanced) data and writes:

    models/voxinfant_ensemble.pkl
    models/scaler.pkl
    models/label_classes.json

Usage:
    python -m voxinfant.train
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier

from .config import FEATURES_DIR, MODELS_DIR, get_config
from .ensemble import WeightedSoftVoteEnsemble

SEED = 42
EXPECTED_DIM = get_config().expected_dim  # 62 (acoustic + gfcc)

# Gentle class weights. The aggressive {b4,d1,p5} + pain-oversampling combo from
# the notebook over-triggers pain at inference (argmax). A mild lift to the
# minority classes scores best by argmax macro-F1 (~0.555 vs ~0.535).
CLASS_WEIGHTS = {"burping": 1.5, "distress": 1.0, "pain": 1.5}


def _xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.6,
        reg_lambda=3.0, reg_alpha=2.0, gamma=2.0,
        tree_method="hist", eval_metric="mlogloss", random_state=SEED,
    )


def _mlp() -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=150,
                         early_stopping=True, random_state=SEED)


def _balance(X, y_encoded, groups, distress_c):
    """Downsample distress to the combined minority count (no oversampling)."""
    data = list(zip(X, y_encoded, groups))
    distress = [d for d in data if d[1] == distress_c]
    others = [d for d in data if d[1] != distress_c]

    # Downsample distress to match the combined minority count. No pain
    # oversampling -- it hurt argmax macro-F1 by over-representing pain.
    distress_down = resample(distress, replace=False, n_samples=len(others), random_state=SEED)
    balanced = distress_down + others

    X_b = np.array([d[0] for d in balanced])
    y_b = np.array([d[1] for d in balanced])
    g_b = np.array([d[2] for d in balanced])
    return X_b, y_b, g_b


def load_features(features_dir: str = FEATURES_DIR):
    X = np.load(os.path.join(features_dir, "raw_features.npy"))
    y = np.load(os.path.join(features_dir, "labels.npy"), allow_pickle=True)
    groups = np.load(os.path.join(features_dir, "groups.npy"), allow_pickle=True)
    # The saved raw_features.npy may carry legacy wav2vec dims (318). The
    # deployable model uses only the first EXPECTED_DIM reproducible features
    # (acoustic + gfcc). Slicing here keeps old feature files usable.
    if X.shape[1] > EXPECTED_DIM:
        X = X[:, :EXPECTED_DIM]
    return X.astype(np.float32), y, groups


def _weights_by_index(le) -> dict:
    return {le.transform([name])[0]: w for name, w in CLASS_WEIGHTS.items()}


def cross_validate(X_bal, y_bal, groups_bal, le) -> None:
    class_weights = _weights_by_index(le)

    gkf = GroupKFold(n_splits=5)
    f1s, accs = [], []
    for fold, (tr, va) in enumerate(gkf.split(X_bal, y_bal, groups_bal), 1):
        Xtr, Xva, ytr, yva = X_bal[tr], X_bal[va], y_bal[tr], y_bal[va]
        sw = np.array([class_weights[i] for i in ytr])

        xgb = _xgb(); xgb.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
        mlp = _mlp(); mlp.fit(Xtr, ytr)
        ens = WeightedSoftVoteEnsemble(xgb, mlp)

        # Evaluate with argmax -- exactly what CryAnalyzer uses at inference.
        preds = ens.predict(Xva)
        f1 = f1_score(yva, preds, average="macro"); acc = accuracy_score(yva, preds)
        f1s.append(f1); accs.append(acc)
        print(f"\nFold {fold}: macro-F1={f1:.4f} acc={acc:.4f}")
        print(classification_report(yva, preds, target_names=le.classes_, zero_division=0))

    print(f"\nFINAL CV (argmax): mean macro-F1={np.mean(f1s):.4f} (+/-{np.std(f1s):.4f}), "
          f"mean acc={np.mean(accs):.4f}")


def fit_final(X_bal, y_bal, le):
    """Fit XGB+MLP on all balanced data and wrap as the deployable ensemble."""
    weights = _weights_by_index(le)
    sw = np.array([weights[i] for i in y_bal])
    xgb = _xgb(); xgb.fit(X_bal, y_bal, sample_weight=sw, verbose=False)
    mlp = _mlp(); mlp.fit(X_bal, y_bal)
    return WeightedSoftVoteEnsemble(xgb, mlp)


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

    le = LabelEncoder(); y_encoded = le.fit_transform(y)
    distress_c = le.transform(["distress"])[0]

    X_bal, y_bal, groups_bal = _balance(X, y_encoded, groups, distress_c)
    print(f"Balanced: {X_bal.shape}")

    # Global scaling fit on balanced data (matches notebook).
    scaler = StandardScaler()
    X_bal_scaled = scaler.fit_transform(X_bal)

    cross_validate(X_bal_scaled, y_bal, groups_bal, le)
    model = fit_final(X_bal_scaled, y_bal, le)
    save_artifacts(model, scaler, le)


if __name__ == "__main__":
    main()
