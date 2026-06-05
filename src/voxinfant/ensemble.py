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
