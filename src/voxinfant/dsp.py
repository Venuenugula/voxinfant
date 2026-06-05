"""DSP front-end.

Faithful extraction of the notebook's STEP 2 DSP functions:
normalize -> pre-emphasis -> bandpass (200-4000 Hz) -> spectral noise reduction.
All constants come from ``config.yaml`` via :mod:`voxinfant.config`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, lfilter

from .config import get_config

CFG = get_config()


def pre_emphasis(signal: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
    alpha = CFG.pre_emphasis_alpha if alpha is None else alpha
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def bandpass_filter(signal: np.ndarray, sr: int,
                    low: Optional[int] = None, high: Optional[int] = None) -> np.ndarray:
    low = CFG.bandpass_low_hz if low is None else low
    high = CFG.bandpass_high_hz if high is None else high
    nyquist = sr / 2
    b, a = butter(CFG.bandpass_order, [low / nyquist, high / nyquist], btype="band")
    return lfilter(b, a, signal)


def process_audio(file_path: str, target_sr: Optional[int] = None
                  ) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Load a file and run the full DSP chain. Returns (signal, sr) or (None, None)."""
    import librosa  # heavy import deferred

    target_sr = CFG.target_sr if target_sr is None else target_sr
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
        return process_signal(y, sr), sr
    except Exception as e:  # noqa: BLE001 - mirror notebook's tolerant behaviour
        print(f"Error processing {file_path}: {e}")
        return None, None


def process_signal(y: np.ndarray, sr: int) -> np.ndarray:
    """Run the DSP chain on an in-memory signal (same steps as process_audio)."""
    import noisereduce as nr  # heavy import deferred

    y = y / (np.max(np.abs(y)) + 1e-6)        # normalize
    y = pre_emphasis(y)                        # pre-emphasis
    y = bandpass_filter(y, sr)                 # bandpass 200-4000 Hz
    y = nr.reduce_noise(y=y, sr=sr)            # spectral noise reduction
    return y
