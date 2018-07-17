#!/usr/bin/env python
"""
This is a mini program called fastplt made using RRtoolbox
 to plot most arrays and images using matplotlib
"""
# -*- coding: utf-8 -*-
# (C) 2017 David Toro <davsamirtor@gmail.com>
# compatibility with python 2 and 3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# import build-in modules
import sys

# import third party modules

# special variables
# __all__ = []
__author__ = "David Toro"
# __copyright__ = "Copyright 2017, The <name> Project"
# __credits__ = [""]
__license__ = "GPL"
# __version__ = "1.0.0"
__maintainer__ = "David Toro"
__email__ = "davsamirtor@gmail.com"
# __status__ = "Pre-release"


if __name__ == "__main__":
    import argparse
    from RRtoolbox.lib.serverServices import parseString as _parseString
    from RRtoolbox.lib.plotter import fastplt, wins, servertimeout, Plotim
    from RRtoolbox.lib.config import FLAG_DEBUG
    #import sys
    # if FLAG_DEBUG: print sys.argv
    parser = argparse.ArgumentParser(description='fast plot of images.')
    parser.add_argument('image', metavar='N',  # action='append',
                        help='path to image or numpy string', nargs="+")
    parser.add_argument('-m', '--cmap', dest='cmap', action='store',
                        help='map to use in matplotlib')
    parser.add_argument('-t', '--title', dest='title', action='store', default="visualazor",
                        help='title of subplot')
    parser.add_argument('-w', '--win', dest='win', action='store',
                        help='title of window')
    parser.add_argument('-n', '--num', dest='num', action='store', type=int, default=0,
                        help='number of Figure')
    parser.add_argument('-f', '--frames', dest='frames', action='store', type=int, default=None,
                        help='number of Figure')
    parser.add_argument('-b', '--block', dest='block', action='store_true', default=False,
                        help='number of Figure')
    parser.add_argument('-d', '--daemon', dest='daemon', action='store_true', default=False,
                        help='number of Figure')
    args = parser.parse_args()
    images = _parseString(args.image, servertimeout)
    wins[-1] = args.num - 1
    for image, kw in images:
        # pickled, so normal comparisons do not work
        if type(image).__name__ == Plotim.__name__:
            image.show(args.frames, args.block, args.daemon, **kw)
        else:
            fastplt(image, args.cmap, args.title,
                    args.win, args.block, args.daemon, **kw)
    if FLAG_DEBUG:
        print("leaving plotter module...")
