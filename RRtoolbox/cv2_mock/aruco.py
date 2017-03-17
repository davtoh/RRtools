# encoding: utf-8
# module cv2.aruco
# from /home/davtoh/anaconda3/envs/rrtools/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
# by generator 1.144
# no doc
# no imports

# Variables with simple values

DICT_4X4_100 = 1
DICT_4X4_1000 = 3
DICT_4X4_250 = 2
DICT_4X4_50 = 0

DICT_5X5_100 = 5
DICT_5X5_1000 = 7
DICT_5X5_250 = 6
DICT_5X5_50 = 4

DICT_6X6_100 = 9
DICT_6X6_1000 = 11
DICT_6X6_250 = 10
DICT_6X6_50 = 8

DICT_7X7_100 = 13
DICT_7X7_1000 = 15
DICT_7X7_250 = 14
DICT_7X7_50 = 12

DICT_ARUCO_ORIGINAL = 16

__loader__ = None

__spec__ = None

# functions


# real signature unknown; restored from __doc__
def Board_create(objPoints, dictionary, ids):
    """ Board_create(objPoints, dictionary, ids) -> retval """
    pass


# real signature unknown; restored from __doc__
def calibrateCameraAruco(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, flags=None, criteria=None):
    """ calibrateCameraAruco(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs """
    pass


# real signature unknown; restored from __doc__
def calibrateCameraArucoExtended(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, stdDeviationsIntrinsics=None, stdDeviationsExtrinsics=None, perViewErrors=None, flags=None, criteria=None):
    """ calibrateCameraArucoExtended(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, perViewErrors[, flags[, criteria]]]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors """
    pass


# real signature unknown; restored from __doc__
def calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, flags=None, criteria=None):
    """ calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs """
    pass


# real signature unknown; restored from __doc__
def calibrateCameraCharucoExtended(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, stdDeviationsIntrinsics=None, stdDeviationsExtrinsics=None, perViewErrors=None, flags=None, criteria=None):
    """ calibrateCameraCharucoExtended(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, perViewErrors[, flags[, criteria]]]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors """
    pass


# real signature unknown; restored from __doc__
def CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary):
    """ CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary) -> retval """
    pass


# real signature unknown; restored from __doc__
def custom_dictionary(nMarkers, markerSize):
    """ custom_dictionary(nMarkers, markerSize) -> retval """
    pass


# real signature unknown; restored from __doc__
def custom_dictionary_from(nMarkers, markerSize, baseDictionary):
    """ custom_dictionary_from(nMarkers, markerSize, baseDictionary) -> retval """
    pass


# real signature unknown; restored from __doc__
def detectCharucoDiamond(image, markerCorners, markerIds, squareMarkerLengthRate, diamondCorners=None, diamondIds=None, cameraMatrix=None, distCoeffs=None):
    """ detectCharucoDiamond(image, markerCorners, markerIds, squareMarkerLengthRate[, diamondCorners[, diamondIds[, cameraMatrix[, distCoeffs]]]]) -> diamondCorners, diamondIds """
    pass


# real signature unknown; restored from __doc__
def detectMarkers(image, dictionary, corners=None, ids=None, parameters=None, rejectedImgPoints=None):
    """ detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedImgPoints]]]]) -> corners, ids, rejectedImgPoints """
    pass


def DetectorParameters_create():  # real signature unknown; restored from __doc__
    """ DetectorParameters_create() -> retval """
    pass


# real signature unknown; restored from __doc__
def Dictionary_create(nMarkers, markerSize):
    """ Dictionary_create(nMarkers, markerSize) -> retval """
    pass


# real signature unknown; restored from __doc__
def Dictionary_create_from(nMarkers, markerSize, baseDictionary):
    """ Dictionary_create_from(nMarkers, markerSize, baseDictionary) -> retval """
    pass


def Dictionary_get(dict):  # real signature unknown; restored from __doc__
    """ Dictionary_get(dict) -> retval """
    pass


# real signature unknown; restored from __doc__
def drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length):
    """ drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length) -> image """
    pass


# real signature unknown; restored from __doc__
def drawDetectedCornersCharuco(image, charucoCorners, charucoIds=None, cornerColor=None):
    """ drawDetectedCornersCharuco(image, charucoCorners[, charucoIds[, cornerColor]]) -> image """
    pass


# real signature unknown; restored from __doc__
def drawDetectedDiamonds(image, diamondCorners, diamondIds=None, borderColor=None):
    """ drawDetectedDiamonds(image, diamondCorners[, diamondIds[, borderColor]]) -> image """
    pass


# real signature unknown; restored from __doc__
def drawDetectedMarkers(image, corners, ids=None, borderColor=None):
    """ drawDetectedMarkers(image, corners[, ids[, borderColor]]) -> image """
    pass


# real signature unknown; restored from __doc__
def drawMarker(dictionary, id, sidePixels, img=None, borderBits=None):
    """ drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img """
    pass


# real signature unknown; restored from __doc__
def drawPlanarBoard(board, outSize, img=None, marginSize=None, borderBits=None):
    """ drawPlanarBoard(board, outSize[, img[, marginSize[, borderBits]]]) -> img """
    pass


# real signature unknown; restored from __doc__
def estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec=None, tvec=None):
    """ estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs[, rvec[, tvec]]) -> retval, rvec, tvec """
    pass


# real signature unknown; restored from __doc__
def estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec=None, tvec=None):
    """ estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs[, rvec[, tvec]]) -> retval, rvec, tvec """
    pass


# real signature unknown; restored from __doc__
def estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs=None, tvecs=None):
    """ estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs]]) -> rvecs, tvecs """
    pass


# real signature unknown; restored from __doc__
def getPredefinedDictionary(dict):
    """ getPredefinedDictionary(dict) -> retval """
    pass


# real signature unknown; restored from __doc__
def GridBoard_create(markersX, markersY, markerLength, markerSeparation, dictionary, firstMarker=None):
    """ GridBoard_create(markersX, markersY, markerLength, markerSeparation, dictionary[, firstMarker]) -> retval """
    pass


# real signature unknown; restored from __doc__
def interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners=None, charucoIds=None, cameraMatrix=None, distCoeffs=None, minMarkers=None):
    """ interpolateCornersCharuco(markerCorners, markerIds, image, board[, charucoCorners[, charucoIds[, cameraMatrix[, distCoeffs[, minMarkers]]]]]) -> retval, charucoCorners, charucoIds """
    pass


# real signature unknown; restored from __doc__
def refineDetectedMarkers(image, board, detectedCorners, detectedIds, rejectedCorners, cameraMatrix=None, distCoeffs=None, minRepDistance=None, errorCorrectionRate=None, checkAllOrders=None, recoveredIdxs=None, parameters=None):
    """ refineDetectedMarkers(image, board, detectedCorners, detectedIds, rejectedCorners[, cameraMatrix[, distCoeffs[, minRepDistance[, errorCorrectionRate[, checkAllOrders[, recoveredIdxs[, parameters]]]]]]]) -> detectedCorners, detectedIds, rejectedCorners, recoveredIdxs """
    pass

# no classes
