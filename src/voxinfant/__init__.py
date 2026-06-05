"""VoxInfant: infant cry analysis pipeline (DSP + segment-level features + ensemble)."""

__version__ = "0.1.0"

from .config import get_config  # noqa: F401

__all__ = ["get_config", "__version__"]
