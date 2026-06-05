"""VAD + fixed-window segmentation (notebook STEP 3).

RMS-based voice activity detection, then 1.25 s windows with 50% overlap.
Segments shorter than one full window are dropped (matches training).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import get_config

CFG = get_config()


def apply_vad(y: np.ndarray, sr: int,
              frame_length: Optional[int] = None,
              hop_length: Optional[int] = None,
              threshold: Optional[float] = None) -> np.ndarray:
    """Keep only frames whose normalized RMS exceeds ``threshold``."""
    import librosa

    frame_length = CFG.vad_frame_length if frame_length is None else frame_length
    hop_length = CFG.vad_hop_length if hop_length is None else hop_length
    threshold = CFG.vad_threshold if threshold is None else threshold

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms = rms / (np.max(rms) + 1e-6)

    frames = np.where(rms > threshold)[0]
    if len(frames) == 0:
        return y  # fallback: keep everything

    indices = librosa.frames_to_samples(frames, hop_length=hop_length)
    voiced: List[float] = []
    for idx in indices:
        start, end = idx, idx + frame_length
        if end <= len(y):
            voiced.extend(y[start:end])
    return np.array(voiced)


def segment_audio(y: np.ndarray, sr: int,
                  segment_duration: Optional[float] = None,
                  overlap: Optional[float] = None) -> List[np.ndarray]:
    """Slice into overlapping fixed-length windows; drop the trailing short tail."""
    segment_duration = CFG.segment_duration_s if segment_duration is None else segment_duration
    overlap = CFG.overlap if overlap is None else overlap

    segment_length = int(sr * segment_duration)
    hop_length = int(segment_length * (1 - overlap))

    segments: List[np.ndarray] = []
    for start in range(0, len(y) - segment_length, hop_length):
        seg = y[start:start + segment_length]
        if len(seg) >= segment_length:
            segments.append(seg)
    return segments
