"""VoxInfant: infant cry analysis pipeline (DSP + segment-level features + ensemble).

The deployable model uses reproducible acoustic + GFCC features only (no wav2vec),
so there is no torch dependency or OpenMP/CUDA workaround in the inference path.
"""

__version__ = "0.2.0"

from .config import get_config  # noqa: F401

__all__ = ["get_config", "__version__"]
