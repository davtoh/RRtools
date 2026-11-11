# -*- coding: utf-8 -*-
"""
Visualization package providing plotting routines, OpenCV interactive viewers,
parameter tuners, and feature correspondence visualizers.
"""

import matplotlib.pyplot as plt

from .helpers import (
    echo,
    background,
    limitaxis,
    convert2bgr,
    convert2bgra,
)
from .matplotlib_helpers import (
    wins,
    servertimeout,
    fastplt,
    graph_filter,
    plotPointsContour,
)
from .commands import CommandMixin
from .overlays import OverlayMixin
from .coordinates_mixin import CoordinateMixin
from .viewer import Plotim
from .match_explorer import MatchExplorer
from .edger import Edger
from .imtester import Imtester

__all__ = [
    # Classes
    "Plotim",
    "MatchExplorer",
    "Edger",
    "Imtester",
    # Mixins
    "CommandMixin",
    "OverlayMixin",
    "CoordinateMixin",
    # Functions
    "fastplt",
    "graph_filter",
    "plotPointsContour",
    "echo",
    "background",
    "limitaxis",
    "convert2bgr",
    "convert2bgra",
    # Globals
    "servertimeout",
    "wins",
    "plt",
]
