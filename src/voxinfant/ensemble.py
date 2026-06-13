"""Weighted soft-vote ensemble (XGBoost + MLP).

Defined in its own module so a pickled instance can be unpickled at inference
time (pickle stores the import path, not the class body). Mirrors the notebook's
final ensemble: ``probs = 0.65 * xgb + 0.35 * mlp``.
"""
from __future__ import annotations

import numpy as np


class WeightedSoftVoteEnsemble:
    def __init__(self, xgb, mlp, w_xgb: float = 0.65, w_mlp: float = 0.35):
        self.xgb = xgb
        self.mlp = mlp
        self.w_xgb = w_xgb
        self.w_mlp = w_mlp
        # Shared encoded-class order (both estimators are trained on the same y).
        self.classes_ = xgb.classes_

    def predict_proba(self, X) -> np.ndarray:
        return self.w_xgb * self.xgb.predict_proba(X) + self.w_mlp * self.mlp.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_with_thresholds(self, X, pain_class: int, burping_class: int,
                                distress_class: int, thr: float = 0.20) -> np.ndarray:
        """Notebook's soft-threshold decision rule (recall-biased toward pain/burping)."""
        probs = self.predict_proba(X)
        out = []
        for p in probs:
            if p[pain_class] > thr:
                out.append(pain_class)
            elif p[burping_class] > thr:
                out.append(burping_class)
            else:
                out.append(distress_class)
        return np.array(out)


class HierarchicalCryClassifier:
    """Two-stage classifier that distinguishes minority cries instead of
    defaulting to distress (the dominant class):

        stage 1: distress  vs  rest        (binary)
        stage 2: pain       vs  burping     (on the 'rest' branch)

    ``predict_proba`` returns columns in the GLOBAL encoded class order so it is
    a drop-in for the inference path (mean-prob over segments -> argmax). Picked
    over the flat ensemble because it lifts pain/burping recall markedly at the
    file level (see docs/experiments.md). Defined here so pickled instances
    unpickle at inference time.
    """

    def __init__(self, stage1, stage2, distress_idx: int, rest_classes, n_classes: int):
        self.stage1 = stage1            # binary estimator, label 1 == distress
        self.stage2 = stage2            # estimator over remapped rest classes 0..k-1
        self.distress_idx = int(distress_idx)
        self.rest_classes = list(rest_classes)   # global indices, e.g. [0, 2]
        self.n_classes = int(n_classes)
        self.classes_ = np.arange(n_classes)

    def predict_proba(self, X) -> np.ndarray:
        p_d = self.stage1.predict_proba(X)[:, list(self.stage1.classes_).index(1)]
        p_rest = 1.0 - p_d
        p2 = self.stage2.predict_proba(X)            # columns align with sorted(rest_classes)
        out = np.zeros((X.shape[0], self.n_classes), dtype=float)
        out[:, self.distress_idx] = p_d
        for j, c in enumerate(self.rest_classes):
            out[:, c] = p_rest * p2[:, j]
        return out

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
