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
    align_ecc_subpixel,
)
from .registration import (
    accumulate_warped_frame,
    boost_high_frequency_details,
    align_and_fuse_frames,
)

__all__ = [
    "SuperResolutionEngine",
    "MultiFrameSuperResolution",
    "SingleImageSuperResolution",
    "SuperResolutionResult",
    "compute_sharpness",
    "align_ecc_subpixel",
    "accumulate_warped_frame",
    "boost_high_frequency_details",
    "align_and_fuse_frames",
]
