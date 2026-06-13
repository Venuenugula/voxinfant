"""Smoke tests: import cleanly and assert the feature contract dimensions.

These run without trained artifacts. Tests needing `spafe`/`librosa`/`torch`
are skipped when those deps are absent, so the suite passes on a bare checkout.
"""
import importlib

import numpy as np
import pytest

from voxinfant.config import get_config
from voxinfant import features as F


def test_config_expected_dim():
    cfg = get_config()
    assert cfg.expected_dim == F.ACOUSTIC_DIM + F.GFCC_DIM == 62


def test_class_order_matches_label_encoder_convention():
    # sklearn LabelEncoder sorts alphabetically; config must match.
    cfg = get_config()
    assert cfg.class_order == sorted(cfg.class_order)


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have("librosa"), reason="librosa not installed")
def test_acoustic_dim_on_synthetic_signal():
    sr = 16000
    y = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)  # 1 s tone
    feat = F.extract_acoustic_features(y, sr)
    assert feat.shape == (F.ACOUSTIC_DIM,)


@pytest.mark.skipif(not _have("spafe"), reason="spafe not installed")
def test_gfcc_dim_on_synthetic_signal():
    sr = 16000
    y = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    feat = F.extract_gfcc_features(y, sr)
    assert feat.shape == (F.GFCC_DIM,)


@pytest.mark.skipif(not (_have("librosa") and _have("spafe")), reason="deps missing")
def test_segment_vector_is_62():
    sr = 16000
    y = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    assert F.build_segment_vector(y, sr).shape == (62,)
