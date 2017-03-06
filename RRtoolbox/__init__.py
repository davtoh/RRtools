#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (C) 2016-2017 David Toro <davsamirtor@gmail.com>
"""
RRtoolbox is a python package which contains source code designed to process
images built mainly using OpenCV.
"""
from __future__ import absolute_import
#__all__ = []
__author__ = "David Toro"
#__copyright__ = "Copyright 2017, The <name> Project"
#__credits__ = [""]
__license__ = "BSD-3-Clause"
__version__ = '1.1.0a1.post1' # jump from '1.0.0a2.post3' when cv2_mock added
__email__ = "davsamirtor@gmail.com"
#__status__ = "Pre-release"


import sys
import os
try:
    import cv2
except ImportError:
    import warnings
    #from importlib import import_module
    #cv2_mock = import_module(".cv2_mock","RRtoolbox")
    from . import cv2_mock
    warnings.simplefilter('always', ImportWarning)
    warnings.warn(
        "OpenCV is not installed, a mock module will be used but wont work "
        "for functions that use OpenCV",
        ImportWarning
    )
    # solves ImportError: No module named cv2
    # this changes the behaviour of the module by mocking cv2
    # this is done so it can be documented in readthedocs.org and
    # in the future it will be replaced with binaries to
    # facilitate users that do not know how to install openCV
    if os.name == 'nt':
        sys.modules["cv2"] = cv2_mock
    elif os.name == 'posix':
        sys.modules["cv2"] = cv2_mock