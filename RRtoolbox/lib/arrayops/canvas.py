# -*- coding: utf-8 -*-
"""
Canvas layout, image padding, and dimension visualization operations.
"""

import cv2
import numpy as np


def isnumpy(arr):
    """
    Test whether an object is a numpy array.

    :param arr:
    :return: True if numpy array, else false.
    """
    return type(arr).__module__ == np.__name__


def im2shapeFormat(im, shape):
    """
    Tries to convert image to intuited format from shape.

    :param im: image.
    :param shape: shape to get format.

        shapes:
        * (None, None): converts to gray
        * (None, None, 2): converts to GR555
        * (None, None, 3): converts to BGR
        * (None, None, 4): converts to BGRA
    :return: reshaped image.
    """
    temp2 = im.shape
    if len(shape) == 2:  # GRAY, YUV_I420, YUV_IYUV or YUV_YV12
        if len(temp2) == 3 and temp2[2] == 2:  # BGR555 or BGR565
            im = cv2.cvtColor(im, cv2.COLOR_BGR5552GRAY)
        elif len(temp2) == 3 and temp2[2] == 3:  # BGR or RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        elif len(temp2) == 3 and temp2[2] == 4:  # BGRA or RGBA
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
    elif len(shape) == 3 and shape[2] == 2:  # BGR555 or BGR565
        if len(temp2) == 2:  # GRAY, YUV_I420, YUV_IYUV or YUV_YV12
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR555)
        elif len(temp2) == 3 and temp2[2] == 3:  # BGR or RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGR555)
        elif len(temp2) == 3 and temp2[2] == 4:  # BGRA or RGBA
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR555)
    elif len(shape) == 3 and shape[2] == 3:  # BGR or RGB
        if len(temp2) == 2:  # GRAY, YUV_I420, YUV_IYUV or YUV_YV12
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif len(temp2) == 3 and temp2[2] == 2:  # BGR555 or BGR565
            im = cv2.cvtColor(im, cv2.COLOR_BGR5552BGR)
        elif len(temp2) == 3 and temp2[2] == 4:  # BGRA or RGBA
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    elif len(shape) == 3 and shape[2] == 4:  # BGRA or RGBA
        if len(temp2) == 2:  # GRAY, YUV_I420, YUV_IYUV or YUV_YV12
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        elif len(temp2) == 3 and temp2[2] == 2:  # BGR555 or BGR565
            im = cv2.cvtColor(im, cv2.COLOR_BGR5552BGRA)
        elif len(temp2) == 3 and temp2[2] == 3:  # BGR or RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return im


def im2imFormat(src, dst):
    """
    Tries to convert source image to destine image format.

    :param src: source image.
    :param dst: destine image.
    :return: reshaped source image.
    """
    return im2shapeFormat(src, dst.shape)


def boxPads(bx, points):
    """
    Get box pads to fit all.

    :param bx: box coordinates or previous boxPads [left_top, right_bottom]
    :param points: array of points
    :return: [(left,top),(right,bottom)] where bx and points fit.
    """
    points = points
    minX, minY = np.min(points, 0)  # left_top
    maxX, maxY = np.max(points, 0)  # right_bottom
    x0, y0 = bx[0]  # left_top
    x1, y1 = bx[1]  # right_bottom
    top, bottom, left, right = 0.0, 0.0, 0.0, 0.0
    if minX < x0:
        left = x0 - minX
    if minY < y0:
        top = y0 - minY
    if maxX > x1:
        right = maxX - x1
    if maxY > y1:
        bottom = maxY - y1
    return [(left, top), (right, bottom)]


def getdataVH(array, ypad=0, xpad=0, bgrcolor=None, alfa=None):
    """
    Get data from array according to padding (Helper function for :func:`padVH`).

    :param array: list of arrays to get data
    :param ypad: how much to pad in y axis
    :param xpad: how much to pad in x axis
    :return: matrix_shapes, grid_div, row_grid, row_gridpad, globalgrid
    """
    # array = [V1,..,VN] donde VN = [H1,..,HM]
    matrix_shapes = []  # get image shapes
    grid_div = []  # get grid divisions
    row_grid = []  # get grid pixel dimensions
    row_gridpad = []  # get grid pixel dimensions with pad
    gy, gx, gc = 0, 0, 0  # for global shape
    for i in range(len(array)):  # rows
        matrix_shapes.append([])
        grid_div.append((1, len(array[i])))
        row_grid.append([])
        row_gridpad.append([])
        x = 0
        xp = 0
        for j in range(len(array[i])):  # columns
            if not isnumpy(array[i][j]):
                array[i][j] = padVH(array[i], ypad, xpad, bgrcolor, alfa)[0]
            if len(array[i][j].shape) > 2:  # if more than 1 channel
                matrix_shapes[i].append(array[i][j].shape)
            else:  # if 1 channel
                matrix_shapes[i].append(
                    (array[i][j].shape[0], array[i][j].shape[1]))
            x += matrix_shapes[i][j][1]  # add x
            xp += xpad * 2  # add xpad
        # get max column dimensions of row
        yxc = np.max(matrix_shapes[i], axis=0)
        if len(yxc) > 2:
            row_grid[i] = (yxc[0], x, yxc[2])  # assign row dimension
            # assign row dimension with pad
            row_gridpad[i] = (yxc[0] + ypad * 2, x + xp, yxc[2])
            if yxc[2] > gc:
                gc = yxc[2]
        else:
            row_grid[i] = (yxc[0], x)  # assign row dimension
            # assign row dimension with pad
            row_gridpad[i] = (yxc[0] + ypad * 2, x + xp)
        gy += row_gridpad[i][0]  # add y with ypad
        if x > gx:
            gx = x
    if gc > 0:
        globalgrid = (gy, gx, gc)  # assign global dimension of grid with pad
    else:
        globalgrid = (gy, gx)  # assign global dimension of grid with pad
    return matrix_shapes, grid_div, row_grid, row_gridpad, globalgrid


def makeVis(globalgrid, bgrcolor=None):
    """
    Make visualization (Helper function for :func:`padVH`)

    :param globalgrid: shape
    :param bgrcolor: color of visualization
    :return: array of shape globalgrid
    """
    if bgrcolor is not None:
        if type(bgrcolor) is np.ndarray:
            if bgrcolor.shape == globalgrid:
                graph = bgrcolor
            else:
                graph = cv2.resize(bgrcolor, (globalgrid[1], globalgrid[0]))
        else:
            # making visualization image
            graph = np.zeros(globalgrid, np.uint8)
            graph[:, :] = bgrcolor
    else:
        graph = np.zeros(globalgrid, np.uint8)  # making visualization image
    return graph


def padVH(imgs, ypad=0, xpad=0, bgrcolor=None, alfa=None):
    """
    Pad Vertically and Horizontally image or group of images into an array.

    :param imgs: image to pad or list of horizontal images (i.e. piled
                 up horizontally as [V1,..,VN] where each can be a list
                 of vertical piling VN = [H1,..,HM]. It can be successive
                 like horizontals, verticals, horizontals, etc.
    :param ypad: padding in axis y
    :param xpad: padding in axis x
    :param bgrcolor: color of spaces
    :param alfa: transparency of imgs over background of bgrcolor color.
    :return: visualization of paded and piled images in imgs.
    """
    # imgs = [V1,..,VN] donde VN = [H1,..,HM] V=visualization
    shapes, div, grid, gridpad, globalgrid = getdataVH(
        imgs, ypad, xpad, bgrcolor, alfa)
    graph = makeVis(globalgrid, bgrcolor)
    shape = graph.shape
    mask = makeVis(globalgrid[0:2], 0)
    Ha = 0  # accumulated high
    for i in range(len(div)):  # vertical or rows
        # WT = W1+...+WN + W_space*(N+1) where N = div[i][1],
        # W1+...+WN = grid[i][1], W_space*(N+1) = globalgrid[1]-grid[i][1]
        Ws = (globalgrid[1] - grid[i][1]) / (div[i][1] + 1)
        Wa = 0  # accumulated Wide
        for j in range(div[i][1]):
            # HT = HN+H_space*2 where HT = gridpad[i][0]
            Hn = int(shapes[i][j][0])
            Hs = int(Ha + (gridpad[i][0] - Hn) / 2)
            Wa = int(Wa + Ws)
            Wn = int(shapes[i][j][1])
            if alfa is None:
                graph[Hs:Hs + Hn, Wa:Wa +
                      Wn] = im2shapeFormat(imgs[i][j], shape)
            else:
                from .overlay import overlay
                if type(alfa) is list:
                    overlay(graph[Hs:Hs + Hn, Wa:Wa + Wn],
                            imgs[i][j], alfa[i][j])
                else:
                    overlay(graph[Hs:Hs + Hn, Wa:Wa + Wn], imgs[i][j], alfa)
            mask[Hs:Hs + Hn, Wa:Wa + Wn] = 1
            Wa += Wn
        Ha += gridpad[i][0]
    return graph, mask

