# -*- coding: utf-8 -*-
r"""
Bundle of methods for handling images. Rather than manipulating specialized
operations in images methods in this module are used for loading, outputting
and format-converting methods, as well as color manipulation.

SUPPORTED FORMATS

see http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#imread

+---------------------------+--------------------------------------------------+
| Format                    | Extension                                        |
+===========================+==================================================+
| Windows bitmaps           | \*.bmp, \*.dib (always supported)                |
+---------------------------+--------------------------------------------------+
| JPEG files                | \*.jpeg, \*.jpg, \*.jpe (see the Notes section)  |
+---------------------------+--------------------------------------------------+
| JPEG 2000 files           | \*.jp2 (see the Notes section)                   |
+---------------------------+--------------------------------------------------+
| Portable Network Graphics | \*.png (see the Notes section)                   |
+---------------------------+--------------------------------------------------+
| Portable image format     | \*.pbm, \*.pgm, \*.ppm (always supported)        |
+---------------------------+--------------------------------------------------+
| Sun rasters               | \*.sr, \*.ras (always supported)                 |
+---------------------------+--------------------------------------------------+
| TIFF files                | \*.tiff, \*.tif (see the Notes section)          |
+---------------------------+--------------------------------------------------+
"""
from __future__ import division
import os
import cv2
import numpy as np
import matplotlib.axes
import matplotlib.figure
from collections.abc import MutableSequence

# Re-exported legacy dependencies
from ..directory import (getData, mkPath, getPath, increment_if_exits,
                         changedir, checkFile, getFileHandle, strdifference)
from ..config import FLOAT, INT, MANAGER
from ..arrayops.basic import (anorm, polygonArea, im2shapeFormat, angle,
                              vectorsAngles, overlay, standarizePoints,
                              splitPoints)
from ..root import glob
from ..cache import cache, ResourceManager
from ..plotter import Plotim, limitaxis
from ..serverServices import parseString, string_is_socket_address

# Domain submodules
from .conversions import (
    transposeIm, supported_formats, green, red, white, orange, black, blue,
    colors, bgra_dtype, plt2bgr, plt2bgra, fig2bgr, np2str, str2np, fig2bgra,
    qi2np, np2qi, gray2qi, rgb2qi, bgra2bgr, hist_match
)
from .io import (
    checkLoaded, loadcv, loadsfrom, interpretImage, convertAs
)
from .loaders import (
    ImFactory, loadFunc, ImLoader, PathLoader, LoaderDict, try_loads
)
from .coordinates import (
    ImCoors, drawcoorpoints, myline, drawcooraxes, drawcoorpolyline,
    drawcoorarea, drawcoorpolyArrow, drawcoorperspective, limitaxispoints
)
from .interactive import (
    GetCoors, getcoors, separe, getrectcoors, quadrants, getgeometrycoors,
    random_color
)
from .core import Image

__all__ = [
    # Conversions & Colors
    "transposeIm", "supported_formats", "green", "red", "white", "orange",
    "black", "blue", "colors", "bgra_dtype", "plt2bgr", "plt2bgra", "fig2bgr",
    "np2str", "str2np", "fig2bgra", "qi2np", "np2qi", "gray2qi", "rgb2qi",
    "bgra2bgr", "hist_match",
    # I/O
    "checkLoaded", "loadcv", "loadsfrom", "interpretImage", "convertAs",
    # Loaders
    "ImFactory", "loadFunc", "ImLoader", "PathLoader", "LoaderDict", "try_loads",
    # Coordinates & Geometry
    "ImCoors", "drawcoorpoints", "myline", "drawcooraxes", "drawcoorpolyline",
    "drawcoorarea", "drawcoorpolyArrow", "drawcoorperspective", "limitaxispoints",
    # Interactive
    "GetCoors", "getcoors", "separe", "getrectcoors", "quadrants",
    "getgeometrycoors", "random_color",
    # Core
    "Image",
]

