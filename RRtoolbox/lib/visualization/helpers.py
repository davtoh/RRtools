# -*- coding: utf-8 -*-
"""
Helper and conversion utilities for visualization routines.
"""

import cv2
import numpy as np
from ..arrayops.basic import overlay

__all__ = [
    "echo",
    "background",
    "limitaxis",
    "convert2bgr",
    "convert2bgra",
]


def echo(obj):
    """
    Printer (used when user wants to print an object from Plotim)

    :param obj: object
    """
    print(obj)


def background(color, x=1, y=1, flag=0):
    """
    Creates background rectangle.

    :param color: main color.
    :param x: x pixels in axis x.
    :param y: y pixels in axis y.
    :param flag: Not implemented.
    :return: image of shape y,x and ndim == color.ndim.
    """
    try:
        back = np.zeros((y, x, len(color)), np.uint8)
    except Exception:
        back = np.zeros((y, x), np.uint8)
    back[:, :] = color
    return back


def limitaxis(c, maxc, minc=0):
    """
    Limit value in axis.

    :param c: value
    :param maxc: max c value.
    :param minc: min c value.
    :return: limited c value c E [minc,maxc]
    """
    if c > maxc:
        c = maxc
    if c < minc:
        c = minc
    return c


def convert2bgr(src, bgrcolor=None):
    """
    Tries to convert any image format to BGR.

    :param src: source image.
    :param bgrcolor: background or transparent color.
    :return: BGR array image.
    """
    if not np.max(src) > 1:
        im = src * 255
        im = im.astype("uint8")
    else:
        im = src.astype("uint8")
    # ERROR: Source image must have 1, 3 or 4 channels in function
    # cvConvertImage
    if len(im.shape) == 2:  # 1 channel
        if np.max(im) > 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        else:
            im = cv2.cvtColor(im * 255, cv2.COLOR_GRAY2BGR)
    elif len(im.shape) == 3 and im.shape[2] == 2:  # 2 channels
        im = cv2.cvtColor(im, cv2.COLOR_BGR5552BGR)
    elif len(im.shape) == 3 and im.shape[2] == 4:  # 4 channels
        if bgrcolor is None:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        else:
            temp = im.shape
            data = np.zeros((temp[0], temp[1], 3), np.uint8)
            data[:, :, :] = bgrcolor
            im = overlay(data, im)
    # else 3 channels or error for 2 channels)
    return im


def convert2bgra(src, bgracolor=None, transparency=None):
    """
    Tries to convert any image format to BGRA.

    :param src: source image.
    :param bgracolor: background or transparent color.
    :param transparency: mask or A channel.
            (typically source image has not A channel, so user can provide it)
    :return: BGRA array image.
    """
    if not np.max(src) > 1:
        im = src * 255
        im = im.astype("uint8")
    else:
        im = src.astype("uint8")
    # ERROR: Source image must have 1, 3 or 4 channels in function
    # cvConvertImage
    if len(im.shape) == 2:  # 1 channel
        if np.max(im) > 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        else:
            im = cv2.cvtColor(im * 255, cv2.COLOR_GRAY2BGRA)
    elif len(im.shape) == 3 and im.shape[2] == 2:  # 2 channels
        im = cv2.cvtColor(im, cv2.COLOR_BGR5552BGRA)
    elif len(im.shape) == 3 and im.shape[2] == 3:  # 3 channels
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    elif len(im.shape) == 3 and im.shape[2] == 4:  # 4 channels
        if bgracolor is not None:
            temp = im.shape
            data = np.zeros((temp[0], temp[1], 4), np.uint8)
            data[:, :] = bgracolor
            im = overlay(data, im)
    if transparency is not None:
        im[:, :, 3] = transparency
    return im

