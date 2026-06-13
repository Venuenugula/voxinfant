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


def build_segment_vector(segment: np.ndarray, sr: int) -> np.ndarray:
    """Concatenate the two blocks for a single segment -> (62,) vector."""
    return np.concatenate([
        extract_acoustic_features(segment, sr),
        extract_gfcc_features(segment, sr),
    ]).astype(np.float32)
