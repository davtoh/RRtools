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

def Board_create(objPoints, dictionary, ids): # real signature unknown; restored from __doc__
    """ Board_create(objPoints, dictionary, ids) -> retval """
    pass

def calibrateCameraAruco(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, flags=None, criteria=None): # real signature unknown; restored from __doc__
    """ calibrateCameraAruco(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs """
    pass

def calibrateCameraArucoExtended(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, stdDeviationsIntrinsics=None, stdDeviationsExtrinsics=None, perViewErrors=None, flags=None, criteria=None): # real signature unknown; restored from __doc__
    """ calibrateCameraArucoExtended(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, perViewErrors[, flags[, criteria]]]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors """
    pass

def calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, flags=None, criteria=None): # real signature unknown; restored from __doc__
    """ calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs """
    pass

def calibrateCameraCharucoExtended(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, stdDeviationsIntrinsics=None, stdDeviationsExtrinsics=None, perViewErrors=None, flags=None, criteria=None): # real signature unknown; restored from __doc__
    """ calibrateCameraCharucoExtended(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, perViewErrors[, flags[, criteria]]]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors """
    pass

def CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary): # real signature unknown; restored from __doc__
    """ CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary) -> retval """
    pass

def custom_dictionary(nMarkers, markerSize): # real signature unknown; restored from __doc__
    """ custom_dictionary(nMarkers, markerSize) -> retval """
    pass

def custom_dictionary_from(nMarkers, markerSize, baseDictionary): # real signature unknown; restored from __doc__
    """ custom_dictionary_from(nMarkers, markerSize, baseDictionary) -> retval """
    pass

def detectCharucoDiamond(image, markerCorners, markerIds, squareMarkerLengthRate, diamondCorners=None, diamondIds=None, cameraMatrix=None, distCoeffs=None): # real signature unknown; restored from __doc__
    """ detectCharucoDiamond(image, markerCorners, markerIds, squareMarkerLengthRate[, diamondCorners[, diamondIds[, cameraMatrix[, distCoeffs]]]]) -> diamondCorners, diamondIds """
    pass

def detectMarkers(image, dictionary, corners=None, ids=None, parameters=None, rejectedImgPoints=None): # real signature unknown; restored from __doc__
    """ detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedImgPoints]]]]) -> corners, ids, rejectedImgPoints """
    pass

def DetectorParameters_create(): # real signature unknown; restored from __doc__
    """ DetectorParameters_create() -> retval """
    pass

def Dictionary_create(nMarkers, markerSize): # real signature unknown; restored from __doc__
    """ Dictionary_create(nMarkers, markerSize) -> retval """
    pass

def Dictionary_create_from(nMarkers, markerSize, baseDictionary): # real signature unknown; restored from __doc__
    """ Dictionary_create_from(nMarkers, markerSize, baseDictionary) -> retval """
    pass

def Dictionary_get(dict): # real signature unknown; restored from __doc__
    """ Dictionary_get(dict) -> retval """
    pass

def drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length): # real signature unknown; restored from __doc__
    """ drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length) -> image """
    pass

def drawDetectedCornersCharuco(image, charucoCorners, charucoIds=None, cornerColor=None): # real signature unknown; restored from __doc__
    """ drawDetectedCornersCharuco(image, charucoCorners[, charucoIds[, cornerColor]]) -> image """
    pass

def drawDetectedDiamonds(image, diamondCorners, diamondIds=None, borderColor=None): # real signature unknown; restored from __doc__
    """ drawDetectedDiamonds(image, diamondCorners[, diamondIds[, borderColor]]) -> image """
    pass

def drawDetectedMarkers(image, corners, ids=None, borderColor=None): # real signature unknown; restored from __doc__
    """ drawDetectedMarkers(image, corners[, ids[, borderColor]]) -> image """
    pass

def drawMarker(dictionary, id, sidePixels, img=None, borderBits=None): # real signature unknown; restored from __doc__
    """ drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img """
    pass

def drawPlanarBoard(board, outSize, img=None, marginSize=None, borderBits=None): # real signature unknown; restored from __doc__
    """ drawPlanarBoard(board, outSize[, img[, marginSize[, borderBits]]]) -> img """
    pass

def estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec=None, tvec=None): # real signature unknown; restored from __doc__
    """ estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs[, rvec[, tvec]]) -> retval, rvec, tvec """
    pass

def estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec=None, tvec=None): # real signature unknown; restored from __doc__
    """ estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs[, rvec[, tvec]]) -> retval, rvec, tvec """
    pass

def estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs=None, tvecs=None): # real signature unknown; restored from __doc__
    """ estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs]]) -> rvecs, tvecs """
    pass

def getPredefinedDictionary(dict): # real signature unknown; restored from __doc__
    """ getPredefinedDictionary(dict) -> retval """
    pass

def GridBoard_create(markersX, markersY, markerLength, markerSeparation, dictionary, firstMarker=None): # real signature unknown; restored from __doc__
    """ GridBoard_create(markersX, markersY, markerLength, markerSeparation, dictionary[, firstMarker]) -> retval """
    pass

def interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners=None, charucoIds=None, cameraMatrix=None, distCoeffs=None, minMarkers=None): # real signature unknown; restored from __doc__
    """ interpolateCornersCharuco(markerCorners, markerIds, image, board[, charucoCorners[, charucoIds[, cameraMatrix[, distCoeffs[, minMarkers]]]]]) -> retval, charucoCorners, charucoIds """
    pass

def refineDetectedMarkers(image, board, detectedCorners, detectedIds, rejectedCorners, cameraMatrix=None, distCoeffs=None, minRepDistance=None, errorCorrectionRate=None, checkAllOrders=None, recoveredIdxs=None, parameters=None): # real signature unknown; restored from __doc__
    """ refineDetectedMarkers(image, board, detectedCorners, detectedIds, rejectedCorners[, cameraMatrix[, distCoeffs[, minRepDistance[, errorCorrectionRate[, checkAllOrders[, recoveredIdxs[, parameters]]]]]]]) -> detectedCorners, detectedIds, rejectedCorners, recoveredIdxs """
    pass

# no classes
