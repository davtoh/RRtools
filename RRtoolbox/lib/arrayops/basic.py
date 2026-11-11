# -*- coding: utf-8 -*-
"""
Basic array operations module.

This module acts as a backward-compatible facade re-exporting all symbols from
its focused domain submodules:
- `overlay`: Alpha transparency overlays, affine superimposition, and homography stitching.
- `geometry`: Coordinate frames, point transformations, and spatial geometry.
- `contours`: Contour analytics, polygon area calculations, and mask generation.
- `canvas`: Layout, padding, and shape conversions.
- `signal`: Signal processing, vector metrics, histograms, and noise operations.
"""

import cv2
import numpy as np

from .canvas import (
    boxPads,
    getdataVH,
    im2imFormat,
    im2shapeFormat,
    isnumpy,
    makeVis,
    padVH,
)
from .contours import (
    contours2mask,
    contoursArea,
    convexityRatio,
    findContours,
    points2mask,
    polygonArea,
    polygonArea_calcule,
    polygonArea_contour,
    polygonArea_fill,
)
from .geometry import (
    angleXY,
    axesIntercept,
    centerM,
    centerS,
    centerSM,
    convertXY,
    getTransformedCorners,
    invertM,
    invertSM,
    matrixIntercept,
    points_generator,
    quadrant,
    random_points,
    relativeQuadrants,
    separePointsByAxis,
    splitPoints,
    standarizePoints,
    transformPoint,
    transformPoints,
    vectorsQuadrants,
)
from .overlay import (
    getTransparency,
    multiple_superpose,
    overlay,
    overlay2,
    overlaypng,
    overlayXY,
    pad_to_fit_H,
    recursiveMap,
    superpose,
)
from .signal import (
    angle,
    angle2D,
    anorm,
    anorm2,
    entroyTest,
    find_near,
    findmaxima,
    findminima,
    get_x_space,
    getOtsuThresh,
    histogram,
    instability_bf,
    noisy,
    normalize,
    normalize2,
    normalizeCustom,
    process_as_blocks,
    relativeVectors,
    rescale,
    unit_vector,
    vectorsAngles,
    vertexesAngles,
    view_as_blocks,
    view_as_windows,
)

__all__ = [
    # Canvas
    "boxPads",
    "getdataVH",
    "im2imFormat",
    "im2shapeFormat",
    "isnumpy",
    "makeVis",
    "padVH",
    # Contours
    "contours2mask",
    "contoursArea",
    "convexityRatio",
    "findContours",
    "points2mask",
    "polygonArea",
    "polygonArea_calcule",
    "polygonArea_contour",
    "polygonArea_fill",
    # Geometry
    "angleXY",
    "axesIntercept",
    "centerM",
    "centerS",
    "centerSM",
    "convertXY",
    "getTransformedCorners",
    "invertM",
    "invertSM",
    "matrixIntercept",
    "points_generator",
    "quadrant",
    "random_points",
    "relativeQuadrants",
    "separePointsByAxis",
    "splitPoints",
    "standarizePoints",
    "transformPoint",
    "transformPoints",
    "vectorsQuadrants",
    # Overlay
    "getTransparency",
    "multiple_superpose",
    "overlay",
    "overlay2",
    "overlaypng",
    "overlayXY",
    "pad_to_fit_H",
    "recursiveMap",
    "superpose",
    # Signal
    "angle",
    "angle2D",
    "anorm",
    "anorm2",
    "entroyTest",
    "find_near",
    "findmaxima",
    "findminima",
    "get_x_space",
    "getOtsuThresh",
    "histogram",
    "instability_bf",
    "noisy",
    "normalize",
    "normalize2",
    "normalizeCustom",
    "process_as_blocks",
    "relativeVectors",
    "rescale",
    "unit_vector",
    "vectorsAngles",
    "vertexesAngles",
    "view_as_blocks",
    "view_as_windows",
    # Common namespaces
    "cv2",
    "np",
]
