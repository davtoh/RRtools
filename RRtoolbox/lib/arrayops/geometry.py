# -*- coding: utf-8 -*-
"""
Coordinate frames, point transformations, and spatial geometry operations.
"""

import cv2
import numpy as np

from .canvas import isnumpy


def axesIntercept(coorSM, maxS, maxM):
    """
    Intercept static axis (S) and mobile axis (M) with a coordinate connecting
    both axes from minS to minM.

                    S1                     S2
                  S0 |<---------------------|-----------> maxS
    coorSM <---------|                      |
                    M1                     M2
        M0 <---------|--------------------->| maxM

    :param coorSM: coordinate of vector from S=0 to M=0.
    :param maxS: value representing end of estatic axis.
    :param maxM: value representing end of mobile axis.
    :return: S1,S2,M1,M2.
    """
    if coorSM < 0:  # left of the static axis
        S1 = 0
        S2 = np.min((np.max((0, maxM + coorSM)), maxS))
        M1 = np.min((maxM, -coorSM))
        M2 = np.min((maxM, S2 - coorSM))
    else:  # right of the static axis
        S1 = np.min((coorSM, maxS))
        S2 = np.min((maxS, coorSM + maxM))
        M1 = 0
        M2 = np.min((maxS - S1, maxM))
    return S1, S2, M1, M2


def matrixIntercept(x, y, staticm, *mobilem):
    """
    Intercepts planes x and y of a static matrix (staticm) with N mobile matrices (mobilem)
    translated from the origin to x,y coordinates.

    :param x: x coordinate.
    :param y: y coordinate.
    :param staticm: static matrix.
    :param mobilem: mobile matrices.
    :return: ROI of intercepted matrices [staticm,*mobilem].
    """
    foreshape = np.min([i.shape[0:2] for i in mobilem], axis=0)
    BminY, BmaxY, FminY, FmaxY = axesIntercept(
        int(y), staticm.shape[0], foreshape[0])
    BminX, BmaxX, FminX, FmaxX = axesIntercept(
        int(x), staticm.shape[1], foreshape[1])
    ROIs = [staticm[BminY:BmaxY, BminX:BmaxX]]
    mobilem = [i[FminY:FmaxY, FminX:FmaxX] for i in mobilem]
    ROIs.extend(mobilem)
    return ROIs


def centerSM(coorSM, maxS, maxM):
    """
    Center vector coorSM in both S and M axes.

    :param coorSM: coordinate of vector from S to M centers.
    :param maxS: value representing end of estatic axis.
    :param maxM: value representing end of mobile axis.
    :return: SM centered coordinate.
    """
    return int((maxS) / 2.0 + coorSM - (maxM) / 2.0)


def invertSM(coorSM, maxS, maxM):
    """
    Invert S and M axes.

    :param coorSM: coordinate of vector for SM inverted axes.
    :param maxS: value representing end of estatic axis.
    :param maxM: value representing end of mobile axis.
    :return: SM coordinate on inverted SM axes.
    """
    return int(maxS + coorSM - maxM)


def quadrant(coorX, coorY, maxX, maxY, quadrant=0):
    """
    Moves a point to a quadrant

    :param coorX: point in x coordinate
    :param coorY: point in y coordinate
    :param maxX: max value in x axis
    :param maxY: max value in y axis
    :param quadrant: Cartesian quadrant, if 0 or False it leaves coorX and coorY unprocessed.
    :return:
    """
    if not quadrant:
        return coorX, coorY
    elif quadrant == 4:
        coorX = coorX + maxX / 2
        coorY = coorY + maxY / 2
    elif quadrant == 3:
        coorX = coorX - maxX / 2
        coorY = coorY + maxY / 2
    elif quadrant == 2:
        coorX = coorX - maxX / 2
        coorY = coorY - maxY / 2
    elif quadrant == 1:
        coorX = coorX + maxX / 2
        coorY = coorY - maxY / 2

    return coorX, coorY


def angleXY(coorX, coorY, angle):
    """
    Rotate coordinate.

    :param coorX: x coordinate.
    :param coorY: y coordinate.
    :param angle: radian angle.
    :return: rotated x,y
    """
    magnitude = np.sqrt(coorX * coorY)
    coorX = int(magnitude * np.cos(angle))
    coorY = int(-magnitude * np.sin(angle))  # y axis is downward
    return coorX, coorY


def invertM(coorSM, maxM):
    """
    Invert M axis.

    :param coorSM: coordinate of vector for M inverted axes.
    :param maxM: value representing end of mobile axis.
    :return: SM coordinate on S axis and inverted M axis.
    """
    return int(coorSM - maxM)


def centerS(coor, maxS):
    """
    Center vector coor in S axis.

    :param coor: coordinate of vector from S center to M=0
    :param maxS: value representing end of estatic axis
    :return: S centered coordinate
    """
    return int(maxS / 2.0 + coor)


def centerM(coor, maxM):
    """
    Center vector coor in M axis.

    :param coor: coordinate of vector from S=0 to M center
    :param maxM: value representing end of mobile axis
    :return: M centered coordinate
    """
    return int(coor - maxM / 2.0)


def convertXY(x, y, backshape, foreshape, flag=0, quartile=0, angle=None):
    """
    Convert absolute XY 0,0 coordinates to new system WZ.

    :param x: x coordinate.
    :param y: y coordinate.
    :param backshape: shape of background image.
    :param foreshape:shape of foreground image.
    :param flag: flag for position (default=0).

        * flag==0 : foreground to left up.
        * flag==1 : foreground to left down.
        * flag==2 : foreground to right up.
        * flag==3 : foreground to right down.
        * flag==4 : foreground at center of background.
        * flag==5 : XY 0,0 is at center of background.
        * flag==6 : XY 0,0 is at center of foreground.
        * flag==7 : XY 0,0 is at right down of foreground.
    :param quartile: place Mobile image at quartile 1,2,3,4.
            if left quartile=0 image won't be moved.
    :param angle: angle in radians (defalut=None). if None it does not apply.
    :return: W,Z
    """
    if flag == 0:  # left up
        pass
    elif flag == 1:
        # vector coor to the end of both axes # left down
        y = invertSM(y, backshape[0], foreshape[0])
    elif flag == 2:
        # vector coor to the end of both axes # right up
        x = invertSM(x, backshape[1], foreshape[1])
    elif flag == 3:
        # vector coor to the end of both axes # right down
        x = invertSM(x, backshape[1], foreshape[1])
        y = invertSM(y, backshape[0], foreshape[0])
    elif flag == 4:
        # vector coor centered in both axes
        x = centerSM(x, backshape[1], foreshape[1])
        y = centerSM(y, backshape[0], foreshape[0])
    elif flag == 5:
        # vector coor centered in static axis
        x = centerS(x, backshape[1])
        y = centerS(y, backshape[0])
    elif flag == 6:
        # vector coor centered in mobile axis
        x = centerM(x, foreshape[1])
        y = centerM(y, foreshape[0])
    elif flag == 7:
        # vector coor to the S axis and inverted M axis
        x = invertM(x, foreshape[1])
        y = invertM(y, foreshape[0])

    x, y = quadrant(x, y, foreshape[1], foreshape[0], quartile)
    if angle is not None:
        x, y = angleXY(x, y, angle)

    return x, y


def transformPoint(p, H):
    """
    Transform individual x,y point with Transformation Matrix.

    :param p: x,y point
    :param H: transformation matrix
    :return: transformed x,y point
    """
    # METHOD 1
    # x,y = p
    # amp = np.array(H).dot(np.array([x,y,1]))
    # return (amp/amp[-1])[:-1]
    # METHOD 2
    return cv2.perspectiveTransform(np.float32([[p]]), H)[0, 0]


def transformPoints(p, H):
    """
    Transform x,y points in array with Transformation Matrix.

    :param p: array of points
    :param H: transformation matrix
    :return: transformed array of x,y point
    """
    return cv2.perspectiveTransform(np.float32([p]), H).reshape(-1, 2)


def getTransformedCorners(shape, H):
    """
    from shape gets transformed corners of array.

    :param shape: H,W array shape
    :param H: transformation matrix
    :return: upper_left, upper_right, lower_right, lower_left transformed corners.
    """
    h, w = shape[:2]
    corners = [[0, 0], [w, 0], [w, h], [0, h]]  # get list of image corners
    # get perspective of corners with transformation matrix
    projection = transformPoints(corners, H)
    return projection  # return projection points


def separePointsByAxis(pts, ptaxis=(1, 0), origin=(0, 0)):
    """
    Separate scattered points with respect to axis (splitting line).

    :param pts: points to separate.
    :param ptaxis: point to form axis from origin
    :param origin: origin
    :return: left, right points from axis.
    """
    from .signal import vectorsAngles
    angles = np.nan_to_num(vectorsAngles(pts, ptaxis, origin, absolute=False))
    return pts[angles >= 0], pts[angles < 0]


def splitPoints(pts, aslist=None):
    """
    from points get x,y columns

    :param pts: array of points
    :param aslist: True to return lists instead of arrays
    :return: x, y columns
    """
    if aslist is None:  # select automatically if not specified
        if isnumpy(pts):
            aslist = False
        else:
            aslist = True

    pts = standarizePoints(pts)  # standard points

    if pts.size == 0:
        if aslist:
            return [], []
        else:
            return np.array([], pts.dtype), np.array([], pts.dtype)
    else:
        if aslist:  # as lists
            return list(pts[:, 0]), list(pts[:, 1])
        else:  # as array
            return pts[:, 0], pts[:, 1]


def standarizePoints(pts, aslist=False):
    """
    converts points to a standard form
    :param pts: list or array of points
    :param aslist: True to return list instead of array
    :return: standard points
    """
    pts = np.array(pts)
    shape = pts.shape
    if len(shape) == 1 and pts.size == 2:
        pts = np.array([pts])
        shape = pts.shape

    half = pts.size // 2

    if pts.size != 0 and (2 not in shape or half not in shape):
        raise Exception(
            "array of shape {} does not seems to correspond to points".format(shape))

    if pts.size == 0:
        if aslist:
            return []
        return pts

    if shape.index(2) < shape.index(half):
        pts = pts.T

    pts = pts.reshape(half, 2)
    if aslist:  # as list
        return list(map(tuple, pts))
    return pts


def relativeQuadrants(points):
    """
    Get quadrants of relative vectors obtained from points.

    :param points: array of points.
    :return: quadrants.
    """
    from .signal import relativeVectors
    vecs = relativeVectors(points)
    return np.sign(vecs)


def vectorsQuadrants(vecs):
    """
    Get quadrants of vectors.

    :param vecs: array of vectors.
    :return: quadrants.
    """
    return np.sign(vecs)


def random_points(axes_range=((-50, 50),), nopoints=4, complete=False):
    """
    Get random points.

    :param axes_range: [x_points_range, y_points_range] where points_range is (min,max) range in axis.
    :param nopoints: number of points.
    :param complete: last point is the first point (adds an additional point i.e. nopoints+1).
    :return: numpy array.
    """
    def getRange(axes_range):
        xr = yr = axes_range[0]
        if len(axes_range) > 1:
            yr = axes_range[1]
        return xr, yr

    def getRandom(rmin, rmax):
        return np.random.random() * (rmax - rmin) + rmin

    xr, yr = getRange(axes_range)
    arr = [(getRandom(*xr), getRandom(*yr)) for _ in range(nopoints)]
    if complete:
        arr.append(arr[0])
    return np.array(arr)


def points_generator(shape=(10, 10), nopoints=None, convex=False, erratic=False, complete=False):
    """
    generate points.

    :param shape: enclosed frame (width, height)
    :param nopoints: number of points
    :param convex: if True make points convex,
            else points follow a circular pattern.
    :return:
    """
    h, w = shape[:2]
    if nopoints is None:
        nopoints = np.min(shape)
    random = np.random.random
    if convex and erratic:
        pts = [(w * random(), h * random()) for _ in range(nopoints)]
    else:
        h2, w2 = h / 2., w / 2.  # center
        div = 2 * np.pi / float(nopoints)  # circle angle division
        subdiv = 0.9
        pts = []
        for i in range(nopoints):
            if convex and i % 2:
                r = np.sqrt((h2 - (h2 * subdiv) * random())**2 +
                            (w2 - (w2 * subdiv) * random())**2)  # radius
            else:
                r = np.sqrt((h2)**2 + (w2)**2)  # radius
            pts.append((r * np.cos(i * div - div * random()),
                        r * np.sin(i * div - div * random())))
    if complete:
        pts.append(pts[0])
    return pts

