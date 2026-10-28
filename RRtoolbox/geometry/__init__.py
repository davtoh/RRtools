"""
Geometry and Scene Reconstruction Subpackage for RRtools.

Exposes:
- Modern Feature Engine (SIFT, FLANN, and neural correspondence hooks)
- Equirectangular and Cylindrical 360° Spherical Stitchers
- Multi-Band Laplacian Pyramid Seam Blenders
"""
from RRtoolbox.geometry.features import (
    FeatureEngine,
    FeatureType,
    KeypointMatches,
    MatcherType,
    get_neural_matcher,
    list_neural_matchers,
    register_neural_matcher,
)
from RRtoolbox.geometry.panoramic import (
    BlendMode,
    PanoramaResult,
    PanoramicStitcher,
    ProjectionType,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    crop_panorama_canvas,
    cylindrical_warp,
    estimate_focal_length,
    feather_blend,
    multiband_blend,
    reconstruct_from_laplacian,
    spherical_warp,
)

__all__ = [
    # Features
    "FeatureEngine",
    "FeatureType",
    "KeypointMatches",
    "MatcherType",
    "register_neural_matcher",
    "get_neural_matcher",
    "list_neural_matchers",
    # Panoramic
    "PanoramicStitcher",
    "PanoramaResult",
    "ProjectionType",
    "BlendMode",
    "estimate_focal_length",
    "spherical_warp",
    "cylindrical_warp",
    "build_gaussian_pyramid",
    "build_laplacian_pyramid",
    "reconstruct_from_laplacian",
    "multiband_blend",
    "feather_blend",
    "crop_panorama_canvas",
]
