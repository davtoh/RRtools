# -*- coding: utf-8 -*-
"""
Alpha transparency overlays, affine superimposition, and homography stitching operations.
"""

import cv2
import numpy as np

from .canvas import boxPads, im2imFormat
from .geometry import getTransformedCorners, matrixIntercept


def getTransparency(array):
    """
    Convert foreground to background.

    :param array: image array.
    :return: alfa (int or array)
    """
    try:
        temp = array.shape
        if len(temp) == 3 and temp[2] == 4:  # BGRA or RGBA
            trans = array[:, :, 3]
        else:
            trans = np.ones(temp[0:2], dtype=np.uint8)
    except Exception:
        trans = 1
    return trans


def overlaypng(back, fore, alpha=None, alfainverted=False, under=False, flag=0):
    """
    Overlay only BGRA.

    :param back: BGRA background image.
    :param fore: BGRA foreground image.
    :param alpha: transparency channel.
    :param alfainverted: if True inverts alpha transparency.
    :param under: if True, place back as fore and fore as back.
    :param flag: (experimental)

            0. Normally replace inverted transparency of alpha in back (N);
                superpose alpha in back (V).
            1. Bloat and replace inverted transparency of alpha in back;
                superpose bgr in back (V).
            2. Superpose inverted transparent COLOR of alpha in back.
            3. Superpose inverted transparent COLOR of alpha in back.
            4. Superpose transparent of alpha in back;
                superpose transparent COLOR of alpha in back.
            5. Superpose transparent of alpha in back;
                superpose transparent COLOR of alpha in back.
    :return: overlayed array

    .. seealso:: :func:`overlay`, :func:`overlay2`
    """
    if alpha is None:
        alpha = getTransparency(fore)
    if np.max(alpha) > 1:
        alpha = alpha / 255.0

    def png(back, fore, dst):
        for chanel in range(3):
            dst[:, :, chanel] = (fore[:, :, chanel] *
                (alpha) + back[:, :, chanel] *
                (1 - alpha))
        temp = dst.shape
        if not under and len(temp) == 3 and temp[2] > 3:
            if flag == 0:
                dst[:, :, 3] = fore[:, :, 3] + back[:, :, 3] * (1 - alpha)
            elif flag == 1:
                dst[:, :, 3] = (fore[:, :, 3] *
                    (alpha) + back[:, :, 3] * (1 - alpha))
            elif flag == 2:
                dst[:, :, 3] = back[:, :, 3] * (1 - alpha)
            elif flag == 3:
                dst[:, :, 3] = fore[:, :, 3] * (1 - alpha)
            elif flag == 4:
                dst[:, :, 3] = back[:, :, 3] * (alpha)
            elif flag == 5:
                dst[:, :, 3] = fore[:, :, 3] * (alpha)

    if alfainverted:
        png(fore, back, back)
    else:
        png(back, fore, back)
    return back


def overlay(back, fore, alpha=None, alfainverted=False, under=False, flag=0):
    """
    Try to Overlay any dimension array.

    :param back: BGRA background image.
    :param fore: BGRA foreground image.
    :param alpha: transparency channel.
    :param alfainverted: if True inverts alpha transparency.
    :param under: if True, place back as fore and fore as back.
    :param flag: (experimental)
    :return: overlayed array

    .. seealso:: :func:`overlay2`
    """
    if alpha is None:
        alpha = getTransparency(fore)
    if np.max(alpha) > 1:
        alpha = alpha / 255.0
    temp = back.shape
    temp2 = fore.shape
    if len(temp) == 2:  # 1 channel
        fore = im2imFormat(fore, back)
        if alfainverted:
            back[:, :] = fore * (1 - alpha) + back * (alpha)
        else:
            back[:, :] = fore * (alpha) + back * (1 - alpha)
    # 4 channels
    elif len(temp) == 3 and temp[2] >= 3 and len(temp2) == 3 and temp2[2] > 3:
        overlaypng(back, fore, alpha, alfainverted, under, flag)
    elif len(temp) == 3:  # more than one channel but 4
        fore = im2imFormat(fore, back)
        if alfainverted:
            for chanel in range(temp[2]):
                back[:, :, chanel] = (fore[:, :, chanel] *
                    (1 - alpha) + back[:, :, chanel] * (alpha))
        else:
            for chanel in range(temp[2]):
                back[:, :, chanel] = (fore[:, :, chanel] *
                    (alpha) + back[:, :, chanel] * (1 - alpha))
    return back


def overlay2(back, fore):
    """
    Overlay foreground to x,y coordinates in background image.

    :param back: background image (numpy array dim 3).
    :param fore: foreground image (numpy array dim 4). the fourth
                dimension is used for transparency.
    :return: back (with overlay).

    .. seealso:: :func:`overlay`
    """
    data = fore[:, :, 3] / 255.0
    for chanel in (0, 1, 2):
        back[:, :, chanel] = (fore[:, :, chanel] *
            data + back[:, :, chanel] * (1 - data))
    return back


def overlayXY(x, y, back, fore, alfa=None, alfainverted=False, under=False, flag=0):
    """
    Overlay foreground image to x,y coordinates in background image.
    This function support images of different sizes with formats: BGR background
    and BGRA foreground of Opencv or numpy images.

    :param x: x position in background.
    :param y: y position in background.
    :param back: background image (numpy array dim 3).
    :param fore: foreground image (numpy array dim 4).
    :return: back (with overlay)
    """
    if type(alfa).__module__ != np.__name__:
        overlay(*matrixIntercept(x, y, back, fore),
                alfainverted=alfainverted, under=under, flag=flag)
    else:
        overlay(*matrixIntercept(x, y, back, fore, alfa),
                alfainverted=alfainverted, under=under, flag=flag)
    return back


def pad_to_fit_H(shape1, shape2, H):
    """
    get boxPads to fit transformed shape1 in shape2.

    :param shape1: shape of array 1
    :param shape2: shape of array 2
    :param H: transformation matrix to use in shape1
    :return: [(left,top),(right,bottom)]
    """
    h, w = shape2[:2]
    bx = [[0, 0], [w, h]]
    corners = getTransformedCorners(shape1, H)
    return boxPads(bx, corners)


def superpose(back, fore, H, foreMask=None, grow=True):
    """
    Superpose foreground image to background image.

    :param back: background image
    :param fore: foreground image
    :param H: transformation matrix of fore to overlay in back
    :param foreMask: (None) foreground alpha mask, None or function.
            foreMask values are from 1 for solid to 0 for transparency.
            If a function is provided the new back,fore parameters are
            provided to produce the foreMask. If None is provided
            as foreMask then it is equivalent to a foreMask with all
            values to 1 where fore is True.
    :param grow: If True, im can be bigger than back and is calculated
            according to how fore is superposed in back; if False
            im is of the same shape as back.
    :return: im, H_back, H_fore
    """
    alpha_shape = fore.shape[:2]
    if grow:
        ((left, top), (right, bottom)) = pad_to_fit_H(fore.shape, back.shape, H)
        H_back = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        H_fore = H_back.dot(H)
        h2, w2 = back.shape[:2]
        w, h = int(left + right + w2), int(top + bottom + h2)
        back = cv2.warpPerspective(back.copy(), H_back, (w, h))
        fore = cv2.warpPerspective(fore.copy(), H_fore, (w, h))
    else:
        H_fore = H
        H_back = np.eye(3)
        h, w = back.shape[:2]
        fore = cv2.warpPerspective(fore.copy(), H_fore, (w, h))

    if foreMask is None:
        alpha = cv2.warpPerspective(np.ones(alpha_shape), H_fore, (w, h))
    elif callable(foreMask):
        alpha = foreMask(back, fore)
        if alpha is None:
            alpha = cv2.warpPerspective(np.ones(alpha_shape), H_fore, (w, h))
    else:
        alpha = cv2.warpPerspective(foreMask, H_fore, (w, h))
    im = overlay(back, fore, alpha)
    return im, H_back, H_fore


def multiple_superpose(base, fore, H, foremask=None):
    """
    Superpose multiple foreground images to a single base image.

    :param base: background, base or deepest level image (level -1)
    :param fore: foreground image list (in order of level i = 0, ... , N)
    :param H: transformation matrix of fore in level i to overlay in base
    :param foremask: foreground alfa mask in level i
    :return: generator of each overlay
    """
    foremask = foremask or (foremask,) * len(fore)
    for f, h, fm in zip(fore, H, foremask):
        base, H_back, H_fore = superpose(base, f, h, fm)
        yield base, H_back, H_fore


def recursiveMap(function, sequence):
    """
    Iterate recursively over a structure using a function.

    :param function: function to apply
    :param sequence: iterator
    :return:
    """
    def helper(seq):
        try:
            return list(map(helper, seq))
        except TypeError:
            return function(seq)
    return helper(sequence)

