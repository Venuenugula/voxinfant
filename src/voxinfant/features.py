"""Feature extraction (acoustic + GFCC).

A segment-level feature vector is the concatenation of two reproducible blocks:

    [ acoustic (36) | gfcc (26) ]  ->  62 dims

These are computed identically at training and inference time (librosa + spafe,
both numerically stable across environments). The earlier wav2vec block was
dropped from the deployable model -- see docs/feature_contract.md.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import get_config

CFG = get_config()

# Block sizes (kept explicit so tests can assert them): 36 + 26 = 62.
ACOUSTIC_DIM = 36   # mfcc mean/std (26) + spectral block (10)
GFCC_DIM = 26       # spafe gfcc default 13 ceps -> mean(13)+std(13)


# ---------------------------------------------------------------------------
# Acoustic + prosodic  (the "fast" feature set)
# ---------------------------------------------------------------------------
def extract_acoustic_features(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    features: List[float] = []

    # MFCC (13) mean + std -> 26
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG.n_mfcc)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Spectral block -> 10
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    features.extend([
        np.mean(centroid), np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff), np.std(rolloff),
        np.mean(zcr), np.std(zcr),
        np.mean(rms), np.std(rms),
    ])

    return np.asarray(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# GFCC (gammatone cepstral coefficients)
# ---------------------------------------------------------------------------
def extract_gfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
    from spafe.features.gfcc import gfcc  # requires `spafe`

    gfcc_feat = gfcc(y, fs=sr)
    return np.concatenate([
        np.mean(gfcc_feat, axis=0),
        np.std(gfcc_feat, axis=0),
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Rich extras (Phase 1 experiment) — pitch, MFCC dynamics, spectral texture.
# All librosa, deterministic, reproducible at inference. Appended AFTER the
# original 62 so columns [:62] stay byte-identical to the baseline feature set.
# Layout: f0(6) + mfcc_delta(26) + mfcc_delta2(26) + contrast(10) + flatness(2)
#         + onset(2)  = 72  ->  full vector = 134 dims.
# ---------------------------------------------------------------------------
RICH_DIM = 72


def extract_rich_extras(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    feats: List[float] = []

    # --- F0 / pitch via pYIN (cry range). Distinguishes pain (high, unstable)
    #     from general distress (lower, sustained). 6 dims.
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1200, sr=sr, hop_length=1024)
        fv = f0[~np.isnan(f0)]
        if fv.size:
            feats += [float(np.mean(fv)), float(np.std(fv)), float(np.min(fv)),
                      float(np.max(fv)), float(np.max(fv) - np.min(fv)),
                      float(np.mean(voiced))]
        else:
            feats += [0.0] * 6
    except Exception:  # noqa: BLE001
        feats += [0.0] * 6

    # --- MFCC delta + delta-delta (temporal dynamics / onsets). 26 + 26.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG.n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats += list(np.mean(d1, axis=1)); feats += list(np.std(d1, axis=1))
    feats += list(np.mean(d2, axis=1)); feats += list(np.std(d2, axis=1))

    # --- Spectral contrast (tonal vs noisy texture). n_bands=4 keeps the top
    #     band below Nyquist for our 200-4000 Hz band. 5 rows -> mean+std = 10.
    sc = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4)
    feats += list(np.mean(sc, axis=1)); feats += list(np.std(sc, axis=1))

    # --- Spectral flatness (noisiness). 2 dims.
    fl = librosa.feature.spectral_flatness(y=y)
    feats += [float(np.mean(fl)), float(np.std(fl))]

    # --- Onset strength (cry burst sharpness). 2 dims.
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    feats += [float(np.mean(onset)), float(np.std(onset))]

    return np.asarray(feats, dtype=np.float32)


def build_segment_vector(segment: np.ndarray, sr: int, rich: bool = False) -> np.ndarray:
    """Concatenate feature blocks for a single segment.

    rich=False -> (62,)  acoustic + gfcc (deployed contract).
    rich=True  -> (134,) baseline 62 + rich extras (Phase 1 experiment).
    """
    base = np.concatenate([
        extract_acoustic_features(segment, sr),
        extract_gfcc_features(segment, sr),
    ]).astype(np.float32)
    if not rich:
        return base
    return np.concatenate([base, extract_rich_extras(segment, sr)]).astype(np.float32)
