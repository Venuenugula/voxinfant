"""Feature extraction (notebook STEP: acoustic + GFCC + wav2vec).

A segment-level feature vector is the concatenation of three blocks:

    [ acoustic (64) | gfcc (26) | wav2vec (256) ]  ->  346 dims

The wav2vec block is computed once per FILE (not per segment) and broadcast to
every segment of that file — this matches the training pipeline exactly.
See docs/feature_contract.md for the dimension breakdown.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import get_config

CFG = get_config()

# Block sizes (kept explicit so tests can assert them).
ACOUSTIC_DIM = 64   # mfcc mean/std (26) + delta mean (13) + delta2 mean (13) + spectral (10) + pitch (2)
GFCC_DIM = 26       # spafe gfcc default 13 ceps -> mean(13)+std(13)
WAV2VEC_DIM = CFG.wav2vec_dim  # 256


# ---------------------------------------------------------------------------
# Acoustic + prosodic
# ---------------------------------------------------------------------------
def extract_acoustic_features(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    features: List[float] = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG.n_mfcc)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta, axis=1))
    features.extend(np.mean(delta2, axis=1))

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

    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
        f0 = f0[~np.isnan(f0)]
        features.extend([np.mean(f0), np.std(f0)] if len(f0) else [0, 0])
    except Exception:  # noqa: BLE001
        features.extend([0, 0])

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
# wav2vec2 file-level embedding (lazy singleton model)
# ---------------------------------------------------------------------------
class Wav2VecEmbedder:
    """Loads facebook/wav2vec2-base once and produces a mean-pooled embedding.

    Returns the first ``CFG.wav2vec_dim`` (256) dimensions of the mean over the
    time axis of ``last_hidden_state`` — identical to the notebook.
    """

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Wav2Vec2Processor.from_pretrained(CFG.wav2vec_model)
        self._model = Wav2Vec2Model.from_pretrained(CFG.wav2vec_model).to(self._device)
        self._model.eval()

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        import torch

        self._ensure_loaded()
        inputs = self._processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(inputs.input_values.to(self._device))
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        return emb[:CFG.wav2vec_dim].astype(np.float32)


# Process-wide singleton (model load is expensive).
_EMBEDDER = Wav2VecEmbedder()


def file_wav2vec(y: np.ndarray, sr: int) -> np.ndarray:
    return _EMBEDDER.embed(y, sr)


def build_segment_vector(segment: np.ndarray, sr: int, file_embedding: np.ndarray) -> np.ndarray:
    """Concatenate the three blocks for a single segment -> (346,) vector."""
    feat = np.concatenate([
        extract_acoustic_features(segment, sr),
        extract_gfcc_features(segment, sr),
        file_embedding,
    ]).astype(np.float32)
    return feat
