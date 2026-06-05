"""End-to-end inference: a .wav file -> a prediction dict.

Pipeline (mirrors training so the feature contract is identical):

    wav -> DSP -> file-level wav2vec embedding -> VAD -> segment
        -> per-segment [acoustic|gfcc|wav2vec] vectors
        -> StandardScaler -> model.predict_proba per segment
        -> average probabilities across segments -> final class + alternatives

Requires the trained artifacts in ``models/``:
    - voxinfant_ensemble.pkl   (the fitted estimator, e.g. XGB or soft-vote ensemble)
    - scaler.pkl               (StandardScaler fitted on training features)
    - label_classes.json       (class order from the LabelEncoder)
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import dsp, features, segmentation
from .config import MODELS_DIR, get_config

CFG = get_config()

MODEL_FILE = "voxinfant_ensemble.pkl"
SCALER_FILE = "scaler.pkl"
CLASSES_FILE = "label_classes.json"


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    n_segments: int
    confident: bool

    def to_dict(self) -> dict:
        alternatives = sorted(
            ((k, v) for k, v in self.probabilities.items() if k != self.label),
            key=lambda kv: kv[1], reverse=True,
        )
        return {
            "prediction": self.label,
            "confidence": round(self.confidence, 4),
            "confident": self.confident,
            "alternatives": [{"label": k, "probability": round(v, 4)} for k, v in alternatives],
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "segments_analyzed": self.n_segments,
        }


class CryAnalyzer:
    """Loads artifacts once and serves predictions."""

    def __init__(self, model_dir: str = MODELS_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.classes: List[str] = []

    # -- loading -----------------------------------------------------------
    def load(self) -> "CryAnalyzer":
        model_path = os.path.join(self.model_dir, MODEL_FILE)
        scaler_path = os.path.join(self.model_dir, SCALER_FILE)
        classes_path = os.path.join(self.model_dir, CLASSES_FILE)

        missing = [p for p in (model_path, scaler_path, classes_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing trained artifacts: " + ", ".join(missing) +
                "\nRun `python -m voxinfant.train` (after preparing features) to generate them."
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(classes_path, "r") as f:
            self.classes = json.load(f)
        return self

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None and bool(self.classes)

    # -- prediction --------------------------------------------------------
    def predict(self, wav_path: str) -> Prediction:
        if not self.is_loaded:
            raise RuntimeError("CryAnalyzer.load() must be called before predict().")

        # 1. DSP on the whole file.
        y, sr = dsp.process_audio(wav_path)
        if y is None:
            raise ValueError(f"Could not process audio: {wav_path}")

        # 2. File-level wav2vec embedding (shared by all segments).
        file_emb = features.file_wav2vec(y, sr)

        # 3. VAD + segmentation.
        y_voiced = segmentation.apply_vad(y, sr)
        segments = segmentation.segment_audio(y_voiced, sr)
        if not segments:
            # Fall back to a single window if VAD removed everything.
            segments = [y[: int(sr * CFG.segment_duration_s)]]

        # 4. Per-segment feature vectors.
        X = np.vstack([
            features.build_segment_vector(seg, sr, file_emb) for seg in segments
        ])

        # 5. Scale + predict, then average probabilities across segments.
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)          # (n_segments, n_classes)
        mean_proba = proba.mean(axis=0)

        order = self._class_order()
        probs = {cls: float(mean_proba[i]) for i, cls in enumerate(order)}
        top_idx = int(np.argmax(mean_proba))
        top_label = order[top_idx]
        confidence = float(mean_proba[top_idx])

        return Prediction(
            label=top_label,
            confidence=confidence,
            probabilities=probs,
            n_segments=len(segments),
            confident=confidence >= CFG.confidence_threshold,
        )

    def _class_order(self) -> List[str]:
        """Map model column index -> class name.

        Prefers the estimator's own ``classes_`` (encoded ints) resolved through
        the saved label list; falls back to the saved list directly.
        """
        if hasattr(self.model, "classes_"):
            return [self.classes[int(c)] for c in self.model.classes_]
        return list(self.classes)


_DEFAULT: Optional[CryAnalyzer] = None


def get_analyzer() -> CryAnalyzer:
    """Lazily construct and load a process-wide analyzer."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CryAnalyzer().load()
    return _DEFAULT
