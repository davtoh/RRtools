"""
Restoration Package for RRtools.

Modern restoration algorithms including multi-frame burst super-resolution,
deep single-image upscaling, and glare/reflection suppression.
"""
from .super_resolution import (
    SuperResolutionEngine,
    MultiFrameSuperResolution,
    SingleImageSuperResolution,
    SuperResolutionResult,
    compute_sharpness,
)

__all__ = [
    "SuperResolutionEngine",
    "MultiFrameSuperResolution",
    "SingleImageSuperResolution",
    "SuperResolutionResult",
    "compute_sharpness",
]
