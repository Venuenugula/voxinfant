"""Augmentation/tuning CV harness. Tests on ORIGINAL segments only (no leakage).

Flags let us A/B the levers on identical test folds:
    --no-aug          drop augmented rows from TRAIN too (isolates augmentation)
    --hierarchical    distress-vs-rest then pain-vs-burping
    --balanced-prior  divide file probs by train class freq before argmax

File-level macro-F1 (mean-prob over a file's segments) is the headline.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voxinfant.train import _xgb, _mlp  # noqa: E402
from voxinfant.ensemble import WeightedSoftVoteEnsemble  # noqa: E402

SEED = 42
F = "features/feat_aug.npy"


def fit_flat(Xtr, ytr, sw):
    xgb = _xgb(); xgb.fit(Xtr, ytr, sample_weight=sw, verbose=False)
    mlp = _mlp(); mlp.fit(Xtr, ytr)
    return WeightedSoftVoteEnsemble(xgb, mlp)


def proba_flat(model, X):
    return model.predict_proba(X)


def fit_hier(Xtr, ytr, sw, d_idx, le):
    """Stage1: distress vs rest. Stage2: pain vs burping (remapped to 0..k-1)."""
    is_d = (ytr == d_idx).astype(int)
    s1 = _xgb(); s1.fit(Xtr, is_d, sample_weight=sw, verbose=False)
    mask = ytr != d_idx
    rest = sorted(set(ytr[mask].tolist()))          # e.g. [0, 2]
    remap = {c: i for i, c in enumerate(rest)}
    y2 = np.array([remap[c] for c in ytr[mask]])
    s2 = _xgb(); s2.fit(Xtr[mask], y2,
                        sample_weight=compute_sample_weight("balanced", y2), verbose=False)
    return (s1, s2, d_idx, rest)


def proba_hier(model, X, n_classes):
    s1, s2, d_idx, rest = model
    p_d = s1.predict_proba(X)[:, list(s1.classes_).index(1)]
    p_rest = 1.0 - p_d
    p2 = s2.predict_proba(X)                         # cols align with sorted(rest)
    out = np.zeros((X.shape[0], n_classes))
    out[:, d_idx] = p_d
    for j, c in enumerate(rest):
        out[:, c] = p_rest * p2[:, j]
    return out


def file_agg(groups, proba, y_seg, prior=None):
    agg = defaultdict(lambda: np.zeros(proba.shape[1])); cnt = defaultdict(int); truth = {}
    for g, p, yt in zip(groups, proba, y_seg):
        agg[g] += p; cnt[g] += 1; truth[g] = yt
    yt, yp = [], []
    for g in agg:
        m = agg[g] / cnt[g]
        if prior is not None:
            m = m / prior
        yt.append(truth[g]); yp.append(int(np.argmax(m)))
    return np.array(yt), np.array(yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("--balanced-prior", action="store_true")
    args = ap.parse_args()

    X = np.load(F).astype(np.float32)
    y = np.load("features/feat_aug_labels.npy", allow_pickle=True)
    g = np.load("features/feat_aug_groups.npy", allow_pickle=True)
    isaug = np.load("features/feat_aug_isaug.npy")
    le = LabelEncoder(); ye = le.fit_transform(y); d_idx = le.transform(["distress"])[0]
    nC = len(le.classes_)
    tag = f"aug={'OFF' if args.no_aug else 'ON'} hier={args.hierarchical} prior={args.balanced_prior}"

    gkf = GroupKFold(5)
    seg_t, seg_p, file_t, file_p = [], [], [], []
    for tr, te in gkf.split(X, ye, g):
        tr_mask = np.ones(len(tr), bool) if not args.no_aug else (isaug[tr] == 0)
        tr = tr[tr_mask]
        te = te[isaug[te] == 0]  # ALWAYS test on originals only

        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        ytr = ye[tr]
        sw = compute_sample_weight("balanced", ytr)
        prior = None
        if args.balanced_prior:
            prior = np.array([(ytr == c).mean() for c in range(nC)])

        if args.hierarchical:
            model = fit_hier(Xtr, ytr, sw, d_idx, le)
            proba = proba_hier(model, Xte, nC)
        else:
            model = fit_flat(Xtr, ytr, sw)
            proba = proba_flat(model, Xte)

        seg_t.extend(ye[te]); seg_p.extend(np.argmax(proba, axis=1))
        ft, fp = file_agg(g[te], proba, ye[te], prior)
        file_t.extend(ft); file_p.extend(fp)

    seg_t, seg_p = np.array(seg_t), np.array(seg_p)
    file_t, file_p = np.array(file_t), np.array(file_p)
    print(f"[{tag}]")
    print(f"  seg-F1 ={f1_score(seg_t, seg_p, average='macro'):.4f}")
    print(f"  FILE-F1={f1_score(file_t, file_p, average='macro'):.4f}  "
          f"file-acc={accuracy_score(file_t, file_p):.4f}  n_files={len(file_t)}")
    print(classification_report(file_t, file_p, target_names=le.classes_, digits=3))
    cm = confusion_matrix(file_t, file_p)
    print("  file confusion (rows=true):", le.classes_.tolist())
    for c, r in zip(le.classes_, cm):
        print(f"    {c[:7]:>7}", r.tolist())


if __name__ == "__main__":
    main()
