# -*- coding: utf-8 -*-
"""
This module holds the plotting and data-visualization tools. Motto: don't know how it is interpreted? i'll show you!

#Plotim example
filename = "t2.jpg"
win = "test"
img = cv2.resize(cv2.imread(filename), (400, 400))  # (height, width)
plot = Plotim(win,img)
plot.show()

.. note::
    This module serves as an authoritative backward-compatible re-export facade.
    Implementation logic is decomposed into :mod:`RRtoolbox.lib.visualization`.
"""

from .visualization import (
    # Classes
    Plotim,
    MatchExplorer,
    Edger,
    Imtester,
    # Mixins
    CommandMixin,
    OverlayMixin,
    CoordinateMixin,
    # Functions
    fastplt,
    graph_filter,
    plotPointsContour,
    echo,
    background,
    limitaxis,
    convert2bgr,
    convert2bgra,
    # Globals
    servertimeout,
    wins,
    plt,
)

__all__ = [
    # Classes
    "Plotim",
    "MatchExplorer",
    "Edger",
    "Imtester",
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
