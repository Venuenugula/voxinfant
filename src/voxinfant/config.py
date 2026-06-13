"""Central configuration for the VoxInfant pipeline.

Loads ``config.yaml`` from the project root and exposes typed accessors.
This module imports cleanly with no heavy dependencies so it can be used by
any other module (DSP, features, inference, training) as the single source of
truth for constants.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

import yaml

# Project root = two levels up from this file (src/voxinfant/config.py -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


@dataclass(frozen=True)
class Config:
    raw: dict
    target_sr: int
    pre_emphasis_alpha: float
    bandpass_low_hz: int
    bandpass_high_hz: int
    bandpass_order: int
    segment_duration_s: float
    overlap: float
    vad_frame_length: int
    vad_hop_length: int
    vad_threshold: float
    n_mfcc: int
    expected_dim: int
    class_order: List[str]
    class_map: Dict[str, str]
    confidence_threshold: float


@lru_cache(maxsize=1)
def get_config(path: str = CONFIG_PATH) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    a, s, fe, c, inf = (
        raw["audio"],
        raw["segmentation"],
        raw["features"],
        raw["classes"],
        raw["inference"],
    )
    return Config(
        raw=raw,
        target_sr=a["target_sr"],
        pre_emphasis_alpha=a["pre_emphasis_alpha"],
        bandpass_low_hz=a["bandpass_low_hz"],
        bandpass_high_hz=a["bandpass_high_hz"],
        bandpass_order=a["bandpass_order"],
        segment_duration_s=s["segment_duration_s"],
        overlap=s["overlap"],
        vad_frame_length=s["vad_frame_length"],
        vad_hop_length=s["vad_hop_length"],
        vad_threshold=s["vad_threshold"],
        n_mfcc=fe["n_mfcc"],
        expected_dim=fe["expected_dim"],
        class_order=list(c["order"]),
        class_map=dict(raw["class_map"]),
        confidence_threshold=inf["confidence_threshold"],
    )
