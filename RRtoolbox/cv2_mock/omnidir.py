# encoding: utf-8
# module cv2.omnidir
# from /home/davtoh/anaconda3/envs/rrtools/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
# by generator 1.144
# no doc
# no imports

# Variables with simple values

CALIB_FIX_CENTER = 256
CALIB_FIX_GAMMA = 128
CALIB_FIX_K1 = 4
CALIB_FIX_K2 = 8
CALIB_FIX_P1 = 16
CALIB_FIX_P2 = 32
CALIB_FIX_SKEW = 2
CALIB_FIX_XI = 64

CALIB_USE_GUESS = 1

RECTIFY_CYLINDRICAL = 2
RECTIFY_LONGLATI = 3
RECTIFY_PERSPECTIVE = 1
RECTIFY_STEREOGRAPHIC = 4

XYZ = 2
XYZRGB = 1

__loader__ = None

__spec__ = None

# functions


# real signature unknown; restored from __doc__
def calibrate(objectPoints, imagePoints, size, K, xi, D, flags, criteria, rvecs=None, tvecs=None, idx=None):
    """ calibrate(objectPoints, imagePoints, size, K, xi, D, flags, criteria[, rvecs[, tvecs[, idx]]]) -> retval, K, xi, D, rvecs, tvecs, idx """
    pass


# real signature unknown; restored from __doc__
def initUndistortRectifyMap(K, D, xi, R, P, size, mltype, flags, map1=None, map2=None):
    """ initUndistortRectifyMap(K, D, xi, R, P, size, mltype, flags[, map1[, map2]]) -> map1, map2 """
    pass


# real signature unknown; restored from __doc__
def projectPoints(objectPoints, rvec, tvec, K, xi, D, imagePoints=None, jacobian=None):
    """ projectPoints(objectPoints, rvec, tvec, K, xi, D[, imagePoints[, jacobian]]) -> imagePoints, jacobian """
    pass


# real signature unknown; restored from __doc__
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, flags, criteria, rvec=None, tvec=None, rvecsL=None, tvecsL=None, idx=None):
    """ stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, flags, criteria[, rvec[, tvec[, rvecsL[, tvecsL[, idx]]]]]) -> retval, objectPoints, imagePoints1, imagePoints2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecsL, tvecsL, idx """
    pass


# real signature unknown; restored from __doc__
def stereoReconstruct(image1, image2, K1, D1, xi1, K2, D2, xi2, R, T, flag, numDisparities, SADWindowSize, disparity=None, image1Rec=None, image2Rec=None, newSize=None, Knew=None, pointCloud=None, pointType=None):
    """ stereoReconstruct(image1, image2, K1, D1, xi1, K2, D2, xi2, R, T, flag, numDisparities, SADWindowSize[, disparity[, image1Rec[, image2Rec[, newSize[, Knew[, pointCloud[, pointType]]]]]]]) -> disparity, image1Rec, image2Rec, pointCloud """
    pass


# real signature unknown; restored from __doc__
def stereoRectify(R, T, R1=None, R2=None):
    """ stereoRectify(R, T[, R1[, R2]]) -> R1, R2 """
    pass


# real signature unknown; restored from __doc__
def undistortImage(distorted, K, D, xi, flags, undistorted=None, Knew=None, new_size=None, R=None):
    """ undistortImage(distorted, K, D, xi, flags[, undistorted[, Knew[, new_size[, R]]]]) -> undistorted """
    pass


# real signature unknown; restored from __doc__
def undistortPoints(distorted, K, D, xi, R, undistorted=None):
    """ undistortPoints(distorted, K, D, xi, R[, undistorted]) -> undistorted """
    pass

# no classes
