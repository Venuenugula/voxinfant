"""Diagnostic probe: do model capacity or file-level aggregation change the story?

For each feature set (62-d baseline vs 134-d rich) and a few models, report BOTH
segment-level and file-level (mean-prob over a file's segments, argmax) macro-F1
under GroupKFold. File-level is what the product actually does.
"""
from __future__ import annotations

import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voxinfant.train import _balance, _xgb, _mlp, _weights_by_index  # noqa: E402
from voxinfant.ensemble import WeightedSoftVoteEnsemble  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

SEED = 42
FEAT = "features/feat_v2.npy"
LAB = "features/feat_v2_labels.npy"
GRP = "features/feat_v2_groups.npy"


def file_level_f1(groups_va, proba, y_true_seg, n_classes):
    """Aggregate segment probabilities per file -> argmax -> file-level truth/pred."""
    agg = defaultdict(lambda: np.zeros(n_classes))
    cnt = defaultdict(int)
    truth = {}
    for g, p, yt in zip(groups_va, proba, y_true_seg):
        agg[g] += p; cnt[g] += 1; truth[g] = yt
    yt, yp = [], []
    for g in agg:
        yt.append(truth[g]); yp.append(int(np.argmax(agg[g] / cnt[g])))
    return np.array(yt), np.array(yp)


def make_model(kind, le):
    if kind == "ensemble":
        return ("ens", None)
    if kind == "xgb_strong":
        return ("clf", XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", eval_metric="mlogloss", random_state=SEED))
    if kind == "rf":
        return ("clf", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=SEED))


def run(dim, kind):
    X = np.load(FEAT).astype(np.float32)
    if dim: X = X[:, :dim]
    y = np.load(LAB, allow_pickle=True); g = np.load(GRP, allow_pickle=True)
    le = LabelEncoder(); ye = le.fit_transform(y)
    dc = le.transform(["distress"])[0]
    Xb, yb, gb = _balance(X, ye, g, dc)
    Xb = StandardScaler().fit_transform(Xb)
    weights = _weights_by_index(le)
    nC = len(le.classes_)

    gkf = GroupKFold(5)
    seg_t, seg_p, file_t, file_p = [], [], [], []
    for tr, va in gkf.split(Xb, yb, gb):
        sw = np.array([weights[i] for i in yb[tr]])
        kindtype, clf = make_model(kind, le)
        if kindtype == "ens":
            xgb = _xgb(); xgb.fit(Xb[tr], yb[tr], sample_weight=sw, verbose=False)
            mlp = _mlp(); mlp.fit(Xb[tr], yb[tr])
            ens = WeightedSoftVoteEnsemble(xgb, mlp)
            proba = ens.predict_proba(Xb[va]); pred = np.argmax(proba, axis=1)
        else:
            try:
                clf.fit(Xb[tr], yb[tr], sample_weight=sw)
            except TypeError:
                clf.fit(Xb[tr], yb[tr])
            proba = clf.predict_proba(Xb[va]); pred = np.argmax(proba, axis=1)
        seg_t.extend(yb[va]); seg_p.extend(pred)
        ft, fp = file_level_f1(gb[va], proba, yb[va], nC)
        file_t.extend(ft); file_p.extend(fp)

    seg_f1 = f1_score(seg_t, seg_p, average="macro")
    file_f1 = f1_score(file_t, file_p, average="macro")
    file_acc = accuracy_score(file_t, file_p)
    print(f"{kind:>12} dim={dim or 134:>3} | seg-F1={seg_f1:.4f} | "
          f"FILE-F1={file_f1:.4f} file-acc={file_acc:.4f} (n_files={len(file_t)})")


if __name__ == "__main__":
    print("model         feat |  segment   |  file-level (what users get)")
    for kind in ["ensemble", "xgb_strong", "rf"]:
        for dim in [62, None]:
            run(dim, kind)
