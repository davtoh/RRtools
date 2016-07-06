# -*- coding: utf-8 -*-
"""
    This module unlike common and basic array operations classifies just the from-to-conversions methods
"""
from __future__ import division
from RRtoolbox.lib.config import FLOAT
import numpy as np
import cv2
__author__ = 'Davtoh'
"""
#im1: object image, im2: scenery image
Tpt = cv2.perspectiveTransform(FLOAT([[point]]), H) # point: [col,row] -> [x,y]
TM = cv2.getPerspectiveTransform(bx1,bx2) # box points: FLOAT([Top_left,Top_right,Bottom_left,Bottom_right])
Tim = cv2.warpPerspective(im1,TM,(w,h)) # h,w = im2.shape
#Tpt: transformed point, TM: transformation matrix, Tim: transformed image
"""


def getSOpointRelation(source_shape, destine_shape, asMatrix = False):
    """
    Return parameters to change scaled point to original point.

        # destine_domain = relation*source_domain

    :param source_shape: image shape for source domain
    :param destine_shape: image shape for destine domain
    :param asMatrix: if true returns a Transformation Matrix H
    :return: x, y coordinate relations or H if asMatrix is True

    .. note:: Used to get relations to convert scaled points to original points of an Image.
    """
    rH = destine_shape[0]
    rW = destine_shape[1]
    sH = source_shape[0]
    sW = source_shape[1]
    if asMatrix:
        return np.array([[rW/sW,0,0],[0,rH/sH,0],[0,0,1]])
    return rW/sW, rH/sH

def spoint2opointfunc(source_shape,destine_shape):
    """
    Return function with parameters to change scaled point to original point.

    :param source_shape:
    :param destine_shape: shape of
    :return:

    Example::

        forefunc = scaled2realfunc(imgf.shape,bgr.shape)
        backfunc = scaled2realfunc(imgb.shape,back.shape)
        p1fore = np.array([forefunc(i) for i in p1])
        p2back = np.array([backfunc(i) for i in p2])
    """
    x,y = getSOpointRelation(source_shape,destine_shape)
    op = np.array([x,y],dtype=FLOAT)
    def scaled2original(p):
        #rx = sx*rW/sW
        #ry = sy*rH/sH
        return p*op
    return scaled2original

def sh2oh(sH,osrc_sh,sscr_sh,odst_sh,sdst_sh):
    """
    Convert scaled transformation matrix (sH) to original (oH).

    :param sH: scaled transformation matrix
    :param osrc_sh: original source's shape
    :param sscr_sh: scaled source's shape
    :param odst_sh: original destine's shape
    :param sdst_sh: scaled destine's shape
    :return:
    """
    oH = sH.copy()
    #height, Width
    Hos,Wos = (osrc_sh[0]),(osrc_sh[1]) # original source
    Hss,Wss = (sscr_sh[0]),(sscr_sh[1]) # scaled source
    Hod,Wod = (odst_sh[0]),(odst_sh[1]) # original destine
    Hsd,Wsd = (sdst_sh[0]),(sdst_sh[1]) # scaled destine
    oH[:,0] = oH[:,0]*Wss/Wos # first row
    oH[:,1] = oH[:,1]*Hss/Hos # second row
    oH[0] = oH[0]*Wod/Wsd # first column
    oH[1] = oH[1]*Hod/Hsd # second column
    return oH

def invertH(H):
    """
    Invert Transformation Matrix.

    :param H:
    :return:
    """
    # inverse perspective
    return np.linalg.inv(H)

def conv3H4H(M):
    """
    Convert a 3D transformation matrix (TM) to 4D TM.

    :param M: Matrix
    :return: 4D Matrix
    """
    M = np.append(M.copy(),[[0,0,1]],0) # add row
    return np.append(M,[[0],[0],[0],[0]],1) # add column

def apply2kp_pairs(kp_pairs, kp1_rel, kp2_rel, func=None):
    """
    Apply to kp_pairs.

    :param kp_pairs: list of (kp1,kp2) pairs
    :param kp1_rel: x,y relation or function to apply to kp1
    :param kp2_rel: x,y relation or function to apply to kp2
    :param func: function to build new copy of keypoint
    :return: transformed kp_pairs
    """
    def withtupple(keypoint,kp_op):
        if func:
            keypoint = func(keypoint)
        try:
            keypoint = keypoint.copy()
            keypoint["pt"] = np.multiply(keypoint["pt"],kp_op) # transform pt with kp_op
        except:
            x,y = keypoint.pt
            rx,ry = kp_op
            keypoint.pt = (x*rx,y*ry)
        return keypoint

    def withfunc(keypoint,kp_op):
        if func:
            keypoint = func(keypoint)
        try:
            keypoint = keypoint.copy()
            keypoint["pt"] = kp_op(*keypoint["pt"]) # transform pt with kp_op
        except:
            x,y = keypoint.pt
            keypoint.pt = kp_op(x,y)
        return keypoint

    if type(kp1_rel) is tuple: # expected tuple operands
        kp1_func = withtupple
    else:
        kp1_func = withfunc

    if type(kp2_rel) is tuple: # expected tuple operands
        kp2_func = withtupple
    else:
        kp2_func = withfunc

    return [(kp1_func(i, kp1_rel), kp2_func(j, kp2_rel)) for i, j in kp_pairs]

def spairs2opairs(kp_pairs,osrc_sh,sscr_sh,odst_sh,sdst_sh,func=None):
    """
    Convert scaled kp_pairs to original kp_pairs.

    :param kp_pairs: list of kp_pairs
    :param osrc_sh: original source's shape
    :param sscr_sh: scaled source's shape
    :param odst_sh: original destine's shape
    :param sdst_sh: scaled destine's shape
    :param func: function to build new copy of keypoint
    :return:
    """
    kp1_pair = getSOpointRelation(sscr_sh,osrc_sh) # fore
    kp2_pair = getSOpointRelation(sdst_sh,odst_sh) # back
    return apply2kp_pairs(kp_pairs,kp1_pair,kp2_pair,func=func)

def keyPoint2tuple(keypoint):
    """ obj.angle, obj.class_id, obj.octave, obj.pt, obj.response, obj.size"""
    return (keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id)

def tuple2keyPoint(points, func = cv2.KeyPoint):
    """ KeyPoint([x, y, _size[, _angle[, _response[, _octave[, _class_id]]]]]) -> <KeyPoint object> """
    return func(*(points[0][0],points[0][1],points[1],points[2], points[3],points[4], points[5]))

def dict2keyPoint(d, func = cv2.KeyPoint):
    """ KeyPoint([x, y, _size[, _angle[, _response[, _octave[, _class_id]]]]]) -> <KeyPoint object> """
    return func(*(d["pt"][0],d["pt"][1],d["size"],d["angle"], d["response"],d["octave"], d["class_id"]))

class SimKeyPoint:
    """
    Simulates opencv keypoint (it allows manipulation, conversion and serialization of keypoints).

    .. note:: Used for conversions and data persistence.
    """
    # FIXME: correct for memoizer: some warning are created if the script is run as __main__
    # it would be great if cv2.KeyPoint did not have pickling incompatibilities
    def __init__(self,*args):
        if len(args)==1:
            obj = args[0]
            if isinstance(obj,dict): # it got a dictionary
                getattr(self,"__dict__").update(obj)
                return
            elif isinstance(obj,tuple):
                args = obj
            else: # it got cv2.Keypoint
                self.angle = obj.angle
                self.class_id = obj.class_id
                self.octave = obj.octave
                self.pt = obj.pt
                self.response = obj.response
                self.size = obj.size
                return
        # tupple is broadcasted as in cv2.KeyPoint
        self.pt =args[0]
        self.size=args[1]
        self.angle =args[2]
        self.response=args[3]
        self.octave=args[4]
        self.class_id=args[5]

def contour2points(contours):
    """
    Convert contours to points. (cnt2pts)

    :param contours: array of contours (cnt) ([[x,y]] only for openCV)
    :return:

    Example::

        contours = np.array([[[0, 0]], [[1, 0]]]) # contours
        points = contour2points(contours)
        print points # np.array([[0, 0], [1, 0]])
    """
    return contours.reshape(-1,2)

cnt2pts = contour2points # compatibility reasons

def points2contour(points):
    """
    Convert points to contours. (pts2cnt)

    :param points: array of points ([x,y] for openCV, [y,x] for numpy)
    :return:

    Example::

        points = np.array([[0, 0], [1, 0]]) # points
        contours = points2contour(points)
        print contours # np.array([[[0, 0]], [[1, 0]]])
    """
    return points.reshape(-1,1,2)

pts2cnt = points2contour # compatibility reasons

def toTupple(obj):
    """
    Converts recursively to tuple

    :param obj: numpy array, list structure, iterators, etc.
    :return: tuple representation obj.
    """
    try:
        return tuple(map(toTupple, obj))
    except TypeError:
        return obj

def points2vectos(pts, origin = None):
    """
    Convert points to vectors with respect to origin.

    :param pts: array of points.
    :param origin: point of origin.
    :return: vectors.
    """
    pts = np.array(pts)
    return pts - (origin or np.zeros_like(pts))

def vectos2points(vecs, origin = None):
    """
    Convert points to vectors with respect to origin.

    :param vecs: array of vectors.
    :param origin: point of origin.
    :return: points.
    """
    vecs = np.array(vecs)
    return vecs + (origin or np.zeros_like(vecs))

quadrantmap = {(0,0):"origin",(1,0):"right",(1,1):"right-up",(0,1):"up",(-1,1):"left-up",
               (-1,0):"left",(-1,-1):"left-down",(0,-1):"down",(1,-1):"right-down"}

def translateQuadrants(quadrants, quadrantmap = quadrantmap):
    """
    Convert quadrants into human readable data.

    :param quadrants: array of quadrants.
    :param quadrantmap: dictionary map to translate quadrants. it is of the form::

            {(0,0):"origin",(1,0):"right",(1,1):"top-right",(0,1):"top",(-1,1):"top-left",
               (-1,0):"left",(-1,-1):"bottom-left",(0,-1):"bottom",(1,-1):"bottom-right"}
    :return: list of translated quadrants.
    """
    return [quadrantmap[i] for i in toTupple(quadrants)]