# -*- coding: utf-8 -*-
"""
    This module contains simple array operation methods
"""
from __future__ import division
from __future__ import absolute_import

from builtins import zip
from builtins import map
from builtins import range
import cv2
import numpy as np
try:
    #from skimage.util import view_as_windows, view_as_blocks
    raise
except:
    from numpy.lib.stride_tricks import as_strided as _ast

    def view_as_blocks(arr_in, block_shape= (3, 3)):
        """
        Provide a 2D block_shape view to 2D array. No error checking made.
        Therefore meaningful (as implemented) only for blocks strictly
        compatible with the shape of arr_in.

        :param arr_in:
        :param block_shape:
        :return:
        """
        # simple shape and strides computations may seem at first strange
        # unless one is able to recognize the 'tuple additions' involved ;-)
        # http://stackoverflow.com/a/5078155/5288758
        shape= (int(arr_in.shape[0] / block_shape[0]), int(arr_in.shape[1] / block_shape[1])) + block_shape
        strides= (int(block_shape[0] * arr_in.strides[0]), int(block_shape[1] * arr_in.strides[1])) + arr_in.strides
        return _ast(arr_in, shape= shape, strides= strides)

    def view_as_windows(arr_in, window_shape, step=1):
        """
        Provide a 2D block_shape rolling view to 2D array. No error checking made.
        Therefore meaningful (as implemented) only for blocks strictly
        compatible with the shape of arr_in.

        :param arr_in:
        :param window_shape:
        :param step:
        :return:
        """
        # taken from /skimage/util/shape.py
        arr_shape = np.array(arr_in.shape)
        arr_in = np.ascontiguousarray(arr_in)

        new_shape = (tuple((arr_shape - window_shape) // step + 1) +
                    tuple(window_shape))

        arr_strides = np.array(arr_in.strides)
        new_strides = np.concatenate((arr_strides * step, arr_strides))

        arr_out = _ast(arr_in, shape=new_shape, strides=new_strides)

        return arr_out



################## OVERLAY


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
    if coorSM<0: # left of the static axis
        S1 = 0
        S2 = np.min((np.max((0,maxM+coorSM)), maxS))
        M1 = np.min((maxM,-coorSM))
        M2 = np.min((maxM,S2-coorSM))
    else: # right of the static axis
        S1 = np.min((coorSM,maxS))
        S2 = np.min((maxS,coorSM+maxM))
        M1 = 0
        M2 = np.min((maxS-S1,maxM))
    return S1,S2,M1,M2

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
    foreshape = np.min([i.shape[0:2] for i in mobilem],axis=0)
    BminY,BmaxY,FminY,FmaxY = axesIntercept(int(y), staticm.shape[0], foreshape[0])
    BminX,BmaxX,FminX,FmaxX = axesIntercept(int(x), staticm.shape[1], foreshape[1])
    ROIs = [staticm[BminY:BmaxY,BminX:BmaxX]]
    mobilem = [i[FminY:FmaxY,FminX:FmaxX] for i in mobilem]
    ROIs.extend(mobilem)
    return ROIs

def centerSM(coorSM,maxS,maxM):
    """
    Center vector coorSM in both S and M axes.

    :param coorSM: coordinate of vector from S to M centers.
    :param maxS: value representing end of estatic axis.
    :param maxM: value representing end of mobile axis.
    :return: SM centered coordinate.
    """
    return int((maxS)/2.0+coorSM-(maxM)/2.0)

def invertSM(coorSM,maxS,maxM):
    """
    Invert S and M axes.

    :param coorSM: coordinate of vector for SM inverted axes.
    :param maxS: value representing end of estatic axis.
    :param maxM: value representing end of mobile axis.
    :return: SM coordinate on inverted SM axes.
    """
    return int(maxS+coorSM-maxM)

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
        return coorX,coorY
    elif quadrant==4:
        coorX = coorX+maxX/2
        coorY = coorY+maxY/2
    elif quadrant==3:
        coorX = coorX-maxX/2
        coorY = coorY+maxY/2
    elif quadrant==2:
        coorX = coorX-maxX/2
        coorY = coorY-maxY/2
    elif quadrant==1:
        coorX = coorX+maxX/2
        coorY = coorY-maxY/2

    return coorX,coorY

def angleXY(coorX,coorY,angle):
    """
    Rotate coordinate.

    :param coorX: x coordinate.
    :param coorY: y coordinate.
    :param angle: radian angle.
    :return: rotated x,y
    """
    magnitude = np.sqrt(coorX*coorY)
    coorX = int(magnitude*np.cos(angle))
    coorY = int(-magnitude*np.sin(angle)) # y axis is downward
    return coorX,coorY

def invertM(coorSM,maxM):
    """
    Invert M axis.

    :param coorSM: coordinate of vector for M inverted axes.
    :param maxS: value representing end of estatic axis.
    :param maxM: value representing end of mobile axis.
    :return: SM coordinate on S axis and inverted M axis.
    """
    return int(coorSM-maxM)

def centerS(coor,maxS):
    """
    Center vector coor in S axis.

    :param coor: coordinate of vector from S center to M=0
    :param maxS: value representing end of estatic axis
    :return: S centered coordinate
    """
    return int(maxS/2.0+coor)

def centerM(coor,maxM):
    """
    Center vector coor in M axis.

    :param coor: coordinate of vector from S=0 to M center
    :param maxM: value representing end of mobile axis
    :return: M centered coordinate
    """
    return int(coor-maxM/2.0)

def convertXY(x,y,backshape,foreshape,flag=0,quartile=0,angle=None):
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
    if flag==0: # left up
        pass
    elif flag==1:
        # vector coor to the end of both axes # left down
        y = invertSM(y,backshape[0],foreshape[0])
    elif flag==2:
        # vector coor to the end of both axes # right up
        x = invertSM(x,backshape[1],foreshape[1])
    elif flag==3:
        # vector coor to the end of both axes # right down
        x = invertSM(x,backshape[1],foreshape[1])
        y = invertSM(y,backshape[0],foreshape[0])
    elif flag==4:
        # vector coor centered in both axes
        x = centerSM(x,backshape[1],foreshape[1])
        y = centerSM(y,backshape[0],foreshape[0])
    elif flag==5:
        # vector coor centered in static axis
        x = centerS(x,backshape[1])
        y = centerS(y,backshape[0])
    elif flag==6:
        # vector coor centered in mobile axis
        x = centerM(x,foreshape[1])
        y = centerM(y,foreshape[0])
    elif flag==7:
        # vector coor to the S axis and inverted M axis
        x = invertM(x,foreshape[1])
        y = invertM(y,foreshape[0])

    x,y=quadrant(x, y, foreshape[1], foreshape[0], quartile)
    if angle is not None:
        x,y=angleXY(x,y,angle)

    return x,y

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

def im2imFormat(src,dst):
    """
    Tries to convert source image to destine image format.

    :param src: source image.
    :param dst: destine image.
    :return: reshaped source image.
    """
    return im2shapeFormat(src, dst.shape)

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
            trans = np.ones(temp[0:2],dtype=np.uint8)
            #trans = 1
    except:
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
    if np.max(alpha)>1:
        alpha = alpha / 255.0

    def png(back,fore,dst):
        for chanel in range(3):
            dst[:,:,chanel] = fore[:,:,chanel]*(alpha) + back[:, :, chanel] * (1 - alpha) #(2) without correction
        temp = dst.shape
        if not under and len(temp) == 3 and temp[2]>3:
            if flag==0: dst[:,:,3] = fore[:,:,3] + back[:,:,3]*(1 - alpha) # normally replace inverted transparency of alpha in back (N); (2) superpose alpha in back (V)
            elif flag==1: dst[:,:,3] = fore[:,:,3]*(alpha) + back[:, :, 3] * (1 - alpha) # bloat and replace inverted transparency of alpha in back; (2) superpose bgr in back (V)
            elif flag==2: dst[:,:,3] = back[:,:,3]*(1 - alpha) # superpose inverted transparent of alpha in back; (2) superpose inverted transparent COLOR of alpha in back
            elif flag==3: dst[:,:,3] = fore[:,:,3]*(1 - alpha) # superpose inverted transparent of alpha in back; (2) superpose inverted transparent COLOR of alpha in back
            elif flag==4: dst[:,:,3] = back[:,:,3]*(alpha) # superpose transparent of alpha in back; (2) superpose transparent COLOR of alpha in back
            elif flag==5: dst[:,:,3] = fore[:,:,3]*(alpha) # superpose transparent of alpha in back; (2) superpose transparent COLOR of alpha in back
            #if under: (1) do nothing; (2) superpose bgr in visible parts of back
    if alfainverted:
        png(fore,back,back)
    else:
        png(back,fore,back)
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

    .. seealso:: :func:`overlay2`
    """
    if alpha is None:
        alpha = getTransparency(fore)
    if np.max(alpha)>1:
        alpha = alpha / 255.0
    # addWeighted operation
    temp = back.shape
    temp2 = fore.shape
    if len(temp) == 2:  # 1 channel
        fore = im2imFormat(fore, back)
        if alfainverted:
            back[:,:] = fore*(1 - alpha) + back * (alpha)
        else:
            back[:,:] = fore*(alpha) + back * (1 - alpha)
    elif len(temp) == 3 and temp[2]>=3 and len(temp2) == 3 and temp2[2] > 3: # 4 channels
        overlaypng(back, fore, alpha, alfainverted, under, flag)
    elif len(temp) == 3: # more than one channel but 4
        fore = im2imFormat(fore, back)
        if alfainverted:
            for chanel in range(temp[2]):
                back[:,:,chanel] = fore[:,:,chanel]*(1 - alpha) + back[:, :, chanel] * (alpha)
        else:
            for chanel in range(temp[2]):
                back[:,:,chanel] = fore[:,:,chanel]*(alpha) + back[:, :, chanel] * (1 - alpha)
    return back

def overlay2(back,fore):
    """
    Overlay foreground to x,y coordinates in background image.

    :param back: background image (numpy array dim 3).
    :param fore: foreground image (numpy array dim 4). the fourth
                dimension is used for transparency.
    :return: back (with overlay).

    #Example::

        import cv2
        import numpy as np
        import time
        a= time.time()
        back = cv2.imread("t1.jpg")
        temp = back.shape
        bgr = np.zeros((temp[0],temp[1],4), np.uint8)
        points = [(86, 162), (1219, 1112), (2219, 2112), (1277,3000),(86, 162)]
        col_in = (0, 0, 0,255)
        thickness = 10
        for i in range(len(points)-1):
            pt1 = (points[i][0], points[i][1])
            pt2 = (points[i+1][0], points[i+1][1])
            cv2.line(bgr, pt1, pt2, col_in, thickness)

        overlay(back,bgr)

        win = "overlay"
        cv2.namedWindow(win,cv2.WINDOW_NORMAL)
        cv2.imshow(win, back)
        print time.time()-a
        cv2.waitKey()
        cv2.destroyAllWindows()

    .. seealso:: :func:`overlay`
    """
    # addWeighted operation
    data = fore[:,:,3]/255.0
    for chanel in (0,1,2):
        back[:,:,chanel] = fore[:,:,chanel]*data + back[:,:,chanel]*(1-data)
    return back

def overlayXY(x,y,back,fore,alfa=None,alfainverted=False,under=False,flag=0):
    """
    Overlay foreground image to x,y coordinates in background image.
    This function support images of different sizes with formats: BGR background
    and BGRA foreground of Opencv or numpy images.

    :param x: x position in background.
    :param y: y position in background.
    :param back: background image (numpy array dim 3).
    :param fore: foreground image (numpy array dim 4). the fourth
        dimension is used for transparency.
    :return: back (with overlay)

    Example::

        import cv2
        back = cv2.imread("t1.jpg")
        bgr = cv2.imread("mustache.png",-1)
        x,y=convertXY(0,0,back.shape,bgr.shape,flag=1)
        overlayXY(x,y,back,bgr)
        win = "overlay"
        cv2.namedWindow(win,cv2.WINDOW_NORMAL)
        cv2.imshow(win, back)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """
    if type(alfa).__module__ != np.__name__:
        overlay(*matrixIntercept(x, y, back, fore), alfainverted=alfainverted, under=under, flag=flag)
    else:
        overlay(*matrixIntercept(x, y, back, fore, alfa), alfainverted=alfainverted, under=under, flag=flag)
    return back

################## STITCH

def getdataVH(array, ypad=0, xpad=0, bgrcolor = None, alfa = None):
    """
    Get data from array according to padding (Helper function for :func:`padVH`).

    :param array: list of arrays to get data
    :param ypad: how much to pad in y axis
    :param xpad: how much to pad in x axis
    :return: matrix_shapes, grid_div, row_grid, row_gridpad, globalgrid
    """
    #array = [V1,..,VN] donde VN = [H1,..,HM]
    matrix_shapes = [] # get image shapes
    grid_div = [] # get grid divisions
    row_grid = [] # get grid pixel dimensions
    row_gridpad = [] # get grid pixel dimensions with pad
    gy,gx,gc=0,0,0 # for global shape
    for i in range(len(array)): # rows
        matrix_shapes.append([])
        grid_div.append((1,len(array[i])))
        row_grid.append([])
        row_gridpad.append([])
        x=0
        xp=0
        for j in range(len(array[i])): # columns
            if not isnumpy(array[i][j]):
                array[i][j] = padVH(array[i], ypad, xpad, bgrcolor, alfa)[0]
            if len(array[i][j].shape)>2: # if more than 1 channel
                matrix_shapes[i].append(array[i][j].shape)
            else: # if 1 channel
                matrix_shapes[i].append((array[i][j].shape[0], array[i][j].shape[1]))
            x+=matrix_shapes[i][j][1] # add x
            xp+=xpad*2 # add xpad
        yxc = np.max(matrix_shapes[i],axis=0) # get max column dimensions of row
        if len(yxc)>2:
            row_grid[i] = (yxc[0],x,yxc[2]) # assign row dimension
            row_gridpad[i] = (yxc[0]+ypad*2,x+xp,yxc[2]) # assign row dimension with pad
            if yxc[2]>gc: gc = yxc[2]
        else:
            row_grid[i] = (yxc[0],x) # assign row dimension
            row_gridpad[i] = (yxc[0]+ypad*2,x+xp) # assign row dimension with pad
        gy+= row_gridpad[i][0] # add y with ypad
        if x>gx: gx=x
    if gc>0:
        globalgrid = (gy,gx,gc) # assign global dimension of grid with pad
    else:
        globalgrid = (gy,gx) # assign global dimension of grid with pad
    return matrix_shapes,grid_div,row_grid,row_gridpad,globalgrid

def makeVis(globalgrid, bgrcolor = None):
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
                graph = cv2.resize(bgrcolor,(globalgrid[1],globalgrid[0]))
        else:
            graph = np.zeros(globalgrid, np.uint8)  # making visualization image
            graph[:,:] =  bgrcolor
    else:
        graph = np.zeros(globalgrid, np.uint8)  # making visualization image
    return graph

def padVH(imgs, ypad=0, xpad=0, bgrcolor = None, alfa = None):
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
    #imgs = [V1,..,VN] donde VN = [H1,..,HM] V=visualization
    shapes,div,grid,gridpad,globalgrid = getdataVH(imgs, ypad, xpad, bgrcolor, alfa)
    graph = makeVis(globalgrid, bgrcolor)
    shape = graph.shape
    mask = makeVis(globalgrid[0:2], 0)
    Ha = 0 # accumulated high
    for i in range(len(div)): # vertical or rows
        # WT = W1+...+WN + W_space*(N+1) where N = div[i][1],
        # W1+...+WN = grid[i][1], W_space*(N+1) = globalgrid[1]-grid[i][1]
        Ws = (globalgrid[1]-grid[i][1])/(div[i][1]+1)
        Wa = 0 # accumulated Wide
        for j in range(div[i][1]):
            # HT = HN+H_space*2 where HT = gridpad[i][0]
            Hn = int(shapes[i][j][0])
            Hs = int(Ha+(gridpad[i][0]-Hn)/2)
            Wa = int(Wa+Ws)
            Wn = int(shapes[i][j][1])
            if alfa is None:
                graph[Hs:Hs+Hn,Wa:Wa+Wn] = im2shapeFormat(imgs[i][j], shape)
            else:
                if type(alfa) is list:
                    overlay(graph[Hs:Hs+Hn,Wa:Wa+Wn], imgs[i][j], alfa[i][j])
                else:
                    overlay(graph[Hs:Hs+Hn,Wa:Wa+Wn], imgs[i][j], alfa)
            mask[Hs:Hs+Hn,Wa:Wa+Wn] = 1
            Wa+=Wn
        Ha += gridpad[i][0]
    return graph,mask

### OTHERS

def findContours(*args,**kwargs):
    """
    Compatibility wrapper around cv2.findContour to support both openCV 2 and
    openCV 3.

    findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
    """
    try:
        contours, hierarchy = cv2.findContours(*args,**kwargs)
    except ValueError: # opencv >= 3.2
        image, contours, hierarchy = cv2.findContours(*args,**kwargs)
    return contours, hierarchy

def anorm2(a):
    """
    Summation of squares (helper function for :func:`anorm`)

    :param a:
    :return:
    """
    return (np.square(a)).sum(-1)

def anorm(a):
    """
    norm in array.

    :param a:
    :return:
    """
    return np.sqrt(anorm2(a))

def relativeVectors(pts, all = True):
    """
    Form vectors from points.

    :param pts: array of points [p0, ... ,(x,y)].
    :param all: (True) if True adds last vector from last and first point.
    :return: array of vectors [V0, ... , (V[n] = x[n+1]-x[n],y[n+1]-y[n])].
    """
    pts = np.array(pts)
    if all: pts = np.append(pts,[pts[0]],axis=0)
    return np.stack([np.diff(pts[:, 0]), np.diff(pts[:, 1])], 1)

def vertexesAngles(pts, dtype= None, deg = False):
    """
    Relative angle of vectors formed by vertexes (where vectors cross).

    i.e. angle between vectors "v01" formed by points "p0-p1" and "v12"
        formed by points "p1-p2" where "p1" is seen as a vertex (where vectors cross).

    :param pts: points seen as vertexes (vectors are recreated from point to point).
    :param dtype: return array of type supported by numpy.
    :param deg: if True angle is in Degrees, else radians.
    :return: angles.
    """
    vs = relativeVectors(pts, all =True) # get all vectors from points.
    vs = np.roll(np.append(vs,[vs[-1]],axis=0),2) # add last vector to first position
    return np.array([angle(vs[i-1],vs[i],deg) for i in range(1,len(vs))],dtype) # caculate angles

def vectorsAngles(pts, ptaxis=(1, 0), origin=(0, 0), dtype= None, deg = False, absolute = None):
    """
    Angle of formed vectors in Cartesian plane with respect to formed axis vector.

    i.e. angle between vector "Vn" (formed by point "Pn" and the "origin")
            and vector "Vaxis" formed by "ptaxis" and the "origin".
        where pts-origin = (P0-origin ... Pn-origin) = V0 ... Vn

    :param pts: points to form vectors from origin
    :param ptaxis: point to form axis from origin
    :param origin: origin
    :param dtype: return array of type supported by numpy.
    :param deg: if True angle is in Degrees, else radians.
    :param absolute: if None returns angles (0 yo 180(pi)) between  pts-origin (V0 .. Vn) and Vaxis.
                    if True returns any Vn absolute angle (0 to 360(2pi)) from Vaxis as axis to Vn.
                    if False returns any Vn angle (0 to 180 or 0 to -180) from Vaxis as axis to Vn,
                    where any Vn angle is positive or negative if counter-clock or clock wise from Vaxis.
    :return:
    """
    if np.sum(origin)==0: # speed up calculations
        return np.array([angle2D(v1=ptaxis,v2=i,deg=deg,absolute=absolute) for i in np.float32(pts)], dtype) # caculate angles
    ptaxis = np.float32(ptaxis) - origin
    return np.array([angle2D(v1=ptaxis,v2=i-origin,deg=deg,absolute=absolute) for i in np.float32(pts)], dtype) # caculate angles

def separePointsByAxis(pts, ptaxis=(1, 0), origin = (0,0)):
    """
    Separate scattered points with respect to axis (splitting line).

    :param pts: points to separate.
    :param ptaxis: point to form axis from origin
    :param origin: origin
    :return: left, right points from axis.
    """
    angles = np.nan_to_num(vectorsAngles(pts, ptaxis, origin, absolute=False))
    # compare this angles[angles>=0].size+angles[angles<0].size == angles.size but it fails, i suspect it is nan values
    # update, yup it was Nan values, i used np.nan_to_num to convert to numbers.
    return pts[angles >= 0], pts[angles < 0]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle(v1, v2, deg = False):
    """
    Angle between two N dimmensional vectors.

    :param v1: vector 1.
    :param v2: vector 2.
    :param deg: if True angle is in Degrees, else radians.
    :return: angle in radians.

    Example::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

    .. note:: obtained from http://stackoverflow.com/a/13849249/5288758
            and tested in http://onlinemschool.com/math/assistance/vector/angl/
    """
    # http://stackoverflow.com/a/13849249/5288758
    # test against http://onlinemschool.com/math/assistance/vector/angl/
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    a = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if deg: return np.rad2deg(a) # *180.0/np.pi
    return a


def angle2D(v1,v2, deg = False, absolute= None):
    """
    Angle between two 2 dimensional vectors.

    :param v1: vector 1.
    :param v2: vector 2.
    :param deg: if True angle is in Degrees, else radians.
    :param absolute: if None returns the angle (0 yo 180(pi)) between v1 and v2.
                    if True returns the absolute angle (0 to 360(2pi)) from v1 as axis to v2.
                    if False returns the angle (0 to 180 or 0 to -180) from v1 as axis to v2,
                    where v2 angle relative to v1 is positive or negative if counter-clock or clock wise.
    :return: angle in radians.

    .. note:: implemented according to http://math.stackexchange.com/a/747992
            and tested in http://onlinemschool.com/math/assistance/vector/angl/
    """
    # implemented according to http://math.stackexchange.com/a/747992
    cross = np.cross(v1,v2)
    if cross.size !=1:
        raise Exception("angle2D receives only cartesian coordinates")
    mag = 1
    if absolute is not None:
        if cross == 0:
            for i in np.array(v1)+v2:
                if i<0:
                    mag = -1
                    break
        elif cross < 0:
            mag = -1
    if absolute and mag<0:
        return 360.0 - angle(v1=v1,v2=v2,deg=deg)
    return mag*angle(v1=v1,v2=v2,deg=deg)

def entroyTest(arr):
    """
    Entropy test of intensity arrays. (Helper function for :func:`entropy`)

    :param arr: array MxN of dim 2.
    :return: entropy.
    """
    N = arr.shape[0] * arr.shape[1] # gray image contains N pixels
    E = 0.0
    for i in range(256):
        Ni = np.count_nonzero(arr == i) # number of pixels having intensity i
        if Ni: # if Ni different to 0
            pi = Ni/N # probability that an arbitrary pixel has intensity i in the image
            E += -pi*np.log(pi) # calculate entropy
    return E

def transformPoint(p,H):
    """
    Transform individual x,y point with Transformation Matrix.

    :param p: x,y point
    :param H: transformation matrix
    :return: transformed x,y point
    """
    #METHOD 1
    #x,y = p
    #amp = np.array(H).dot(np.array([x,y,1]))
    #return (amp/amp[-1])[:-1]
    #METHOD 2
    return cv2.perspectiveTransform(np.float32([[p]]), H)[0,0]

def transformPoints(p,H):
    """
    Transform x,y points in array with Transformation Matrix.

    :param p: array of points
    :param H: transformation matrix
    :return: transformed array of x,y point
    """
    return cv2.perspectiveTransform(np.float32([p]), H).reshape(-1,2)

def getTransformedCorners(shape,H):
    """
    from shape gets transformed corners of array.

    :param shape: H,W array shape
    :param H: transformation matrix
    :return: upper_left, upper_right, lower_right, lower_lef transformed corners.
    """
    h,w = shape[:2]
    corners = [[0, 0], [w, 0], [w, h], [0, h]] # get list of image corners
    projection = transformPoints(corners, H) # get perspective of corners with transformation matrix
    return projection # return projection points

def boxPads(bx, points):
    """
    Get box pads to fit all.

    :param bx: box coordinates or previous boxPads [left_top, right_bottom]
    :param points: array of points
    :return: [(left,top),(right,bottom)] where bx and points fit.
    """
    points = points
    minX,minY = np.min(points,0) # left_top
    maxX,maxY = np.max(points,0) # right_bottom
    x0,y0 = bx[0] # left_top
    x1,y1 = bx[1] # right_bottom
    top,bottom,left,right = 0.0,0.0,0.0,0.0
    if minX<x0: left = x0-minX
    if minY<y0: top = y0-minY
    if maxX>x1: right = maxX-x1
    if maxY>y1: bottom = maxY-y1
    return [(left,top),(right,bottom)]

def pad_to_fit_H(shape1, shape2, H):
    """
    get boxPads to fit transformed shape1 in shape2.

    :param shape1: shape of array 1
    :param shape2: shape of array 2
    :param H: transformation matrix to use in shape1
    :return: [(left,top),(right,bottom)]
    """
    h,w = shape2[:2] # get hight,width of image
    bx = [[0,0],[w,h]] # make box
    corners = getTransformedCorners(shape1,H) # get corners from image
    return boxPads(bx, corners)

def superpose(back, fore, H, foreMask= None, grow = True):
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
    # fore on top of back
    alpha_shape = fore.shape[:2]
    if grow: # this makes the images bigger if possible
        # fore(x,y)*H = fore(u,v) -> fore(u,v) + back(u,v)
        ((left,top),(right,bottom)) = pad_to_fit_H(fore.shape, back.shape, H)
        H_back = np.float32([[1,0,left],[0,1,top],[0,0,1]]) # moved transformation matrix with pad
        H_fore = H_back.dot(H) # moved transformation matrix with pad
        # need: top_left, bottom_left, top_right,bottom_right
        h2,w2 = back.shape[:2]
        w,h = int(left + right + w2),int(top + bottom + h2)
        # this memory inefficient, image is copied to prevent cross-references
        back = cv2.warpPerspective(back.copy(), H_back, (w, h))
        fore = cv2.warpPerspective(fore.copy(), H_fore, (w, h))
    else: # this keeps back shape
        H_fore = H
        H_back = np.eye(3)
        h, w = back.shape[:2]
        fore = cv2.warpPerspective(fore.copy(), H_fore, (w, h))

    if foreMask is None: # create valid mask for stitching
        alpha = cv2.warpPerspective(np.ones(alpha_shape), H_fore, (w, h))
    elif callable(foreMask): # get alpha with custom function
        alpha = foreMask(back,fore)
        if alpha is None:
            alpha = cv2.warpPerspective(np.ones(alpha_shape), H_fore, (w, h))
    else: # update foreMask to new position
        alpha = cv2.warpPerspective(foreMask, H_fore, (w, h))
    im = overlay(back, fore, alpha) # overlay fore on top of back
    return im, H_back, H_fore

def multiple_superpose(base,fore,H,foremask=None):
    """
    Superpose multiple foreground images to a single base image.

    :param base: backgraound, base or dipest level image (level -1)
    :param fore: foreground image list (in order of level i = 0, ... , N)
    :param H: transformation matrix of fore in level i to overlay in base
    :param foremask: foreground alfa mask in level i
    :return: generator of each overlay
    """
    # TODO: not finished?
    foremask = foremask or (foremask,)*len(fore) # if foremask is None create tuple to match fore
    for f,h,fm in zip(fore,H,foremask): # all must match otherwise raise error
        base,H_back,H_fore = superpose(base,f,h,fm) # do not consume more memory than actual process
        yield base,H_back,H_fore # keep references keptAlive in memory if user wants

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


def contoursArea(contours):
    """
    Accumulates areas from list of contours.

    :param contours: list of contours or binary array.
    :return: area.
    """
    if isnumpy(contours):
        contours, _ = findContours(contours.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0 # accumulate area
    for cnt in contours:
        area += cv2.contourArea(cnt.astype(np.int32)) # get each contour area
    return area

def points2mask(pts, shape = None, astype = np.bool):
    """
    Creates an array with the filled polygon formed by points.

    :param pts: points.
    :param shape: (None) shape of array. If None it creates an array fitted to points.
    :param astype: ("bool") numpy type
    :return: array.

    Example::

        pts = random_points([(-100, 100), (-100, 100)])
        img = points2mask(pts)
        Plotim("filled",img).show()
    """
    if shape is None:
        pts = pts-pts.min(0) # shift
        xmax,ymax = pts.max(0)
        array = np.zeros((ymax,xmax),astype)
    else:
        array = np.zeros(shape[:2],astype)
    #cv2.drawContours(array,[pts.astype(np.int32)],0,1,-1)
    cv2.fillConvexPoly(array,pts.astype(np.int32),1,0)
    return array

def contours2mask(contours, shape = None, astype = np.bool):
    """
    Creates an array with filled polygons formed by contours.

    :param contours: list of contour or points forming objects
    :param shape: (None) shape of array. If None it creates an array fitted to contours.
    :param astype: ("bool") numpy type
    :return:
    """
    ncoors = [] # adequate contours
    # calculate variables
    minx,miny = np.inf,np.inf
    maxx,maxy = -np.inf,-np.inf
    for coors in contours:
        coors = standarizePoints(coors,aslist=False).astype(np.int32)
        if coors.size != 0:
            ncoors.append(coors)
            # calculate variables for shape
            if shape is None:
                pminx,pminy = coors.min(0)
                pmaxx,pmaxy = coors.max(0)
                if pminx < minx:
                    minx = pminx
                if pminy < miny:
                    miny = pminy
                if pmaxx > maxx:
                    maxx = pmaxx
                if pmaxy > maxy:
                    maxy = pmaxy
    if shape is None:
        minv = np.array([(minx,miny)])
        ncoors = [c-minv for c in ncoors] # shift
        mask = np.zeros((maxy,maxx))
    else:
        mask = np.zeros(shape[:2])
    #for coors in contours:
    #    mask = np.logical_or(mask, points2mask(coors, shape))
    cv2.drawContours(mask,ncoors,-1,1,-1)
    return mask.astype(astype)

def polygonArea_contour(pts):
    """
    Area of points using contours.

    :param pts: points.
    :return: area value.

    ..note:: if polygon is incomplete (last is not first point) it completes the array.
    """
    return contoursArea([pts])

def polygonArea_fill(pts):
    """
    Area of points using filled polygon and pixel count.

    :param pts: points.
    :return: area value.

    ..note:: if polygon is incomplete (last is not first point) it completes the array.
    """
    return points2mask(pts).sum()

def splitPoints(pts, aslist = None):
    """
    from points get x,y columns

    :param pts: array of points
    :param aslist: True to return lists instead of arrays
    :return: x, y columns

    example::

        splitPoints((1,2))
        >>> ([1], [2])
        splitPoints([[[[(1,2),(1,2),(2,3)]]]])
        >>> ([1, 1, 2], [2, 2, 3])
        splitPoints(np.array([[[[(1,2),(1,2),(2,3)]]]]))
        >>> (array([1, 1, 2]), array([2, 2, 3]))
        splitPoints(np.array([[[[(1,2),(1,2),(2,3)]]]]), True)
        >>> ([1, 1, 2], [2, 2, 3])
        splitPoints([(1,2,3,4,5),(1,2,3,4,5)]) # it is processed anyways
        >>> ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    """
    if aslist is None: # select automatically if not specified
        if isnumpy(pts):
            # it is a numpy so it should be array
            aslist = False
        else:
            # it is something else so it should be list
            aslist = True

    pts = standarizePoints(pts) # standard points

    if pts.size == 0:
        # return empty x,y columns
        if aslist:
            return [],[]
        else:
            return np.array([],pts.dtype),np.array([],pts.dtype)
    else:
        if aslist: # as lists
            return list(pts[:, 0]), list(pts[:, 1])
        else: # as array
            return pts[:, 0], pts[:, 1]

def standarizePoints(pts, aslist = False):
    """
    converts points to a standard form
    :param pts: list or array of points
    :param aslist: True to return list instead of array
    :return: standard points

    example::

        standarizePoints((1,2))
        >>> array([[1, 2]])
        standarizePoints([[[[(1,2)]]]])
        >>> array([[1, 2]])
        standarizePoints([[[[(1,2),(1,2),(2,3)]]]],True)
        >>> [(1, 2), (1, 2), (2, 3)]
        standarizePoints([[[[(1,2,3),(1,2,3)]]]],True)
        >>> [(1, 2), (1, 2), (2, 3)]
        standarizePoints([(1,2,3,4,5),(1,2,3,4,5)],True)
        >>> [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    """
    #return zip(*splitPoints(pts))
    pts = np.array(pts)
    shape = pts.shape
    if len(shape)==1 and pts.size == 2:
        pts = np.array([pts])
        shape = pts.shape

    half = pts.size//2

    if pts.size != 0 and (2 not in shape or half not in shape):
    #if pts.size != 0 and (2 != shape[0] and 2 != shape[-1]):
        raise Exception("array of shape {} does not seems to correspond to points".format(shape))

    # return empty points
    if pts.size == 0:
        if aslist:
            return []
        return pts

    # convert to shape (None,2)
    if shape.index(2) < shape.index(half):
        pts = pts.T

    pts =  pts.reshape(half,2)
    if aslist: # as list
         return list(map(tuple,pts))
    return pts


def polygonArea_calcule(pts):
    """
    Area of points calculating polygon Area.

    :param pts: points.
    :return: area value.

    ..note::
        - If polygon is incomplete (last is not first point) it completes the array.
        - If the polygon crosses over itself the algorithm will fail.
        - Based on http://www.mathopenref.com/coordpolygonarea.html
    """
    pts = standarizePoints(pts)
    if (pts[0]==pts[-1]).all(): # test if last is the first (closed points)
        x,y= pts[:, 0], pts[:, 1]
    else: # complete the points (last point must be the first)
        x = pts[list(range(len(pts)))+[0],0]
        y = pts[list(range(len(pts)))+[0],1]
    xmul = x[:]*np.roll(y, -1) # shifted multiplication in axis x
    ymul = y[:]*np.roll(x, -1) # shifted multiplication in axis y
    return np.abs(np.sum(xmul)-np.sum(ymul))/2. # summation and absolute area

polygonArea = polygonArea_calcule

def normalize(arr):
    """Normalize array to ensure range [0,1]"""
    arr = np.float32(arr-np.min(arr)) # shift array to 0
    return arr / np.max(arr) # scales to 1

def normalize2(arr):
    """
    Normalize with factor of absolute maximum value.

    .. notes:: it does not ensures range [0,1], array must be shifted to 0 to achieve this.
    """
    return arr / np.max([np.max(arr), -1.0 * np.min(arr)])

def rescale(arr,max=1,min=0):
    """
    Rescales array values to range [min,max].

    :param arr: array.
    :param max: maximum value in range.
    :param min: minimum value in range.
    :return: rescaled array.
    """
    return (max-min)*normalize(arr)+min

def normalizeCustom(arr, by = np.max, axis = None):
    """
    Normalize array with custom operations.

    :param arr: array (it does not correct negative values, use preferable NxM).
    :param by: np,max, np.sum or any function that gets an array to obtain factor.
    :param axis: if None it normalizes in all axes else in the selected axis.
    :return: normalized to with factor.

    .. notes:: it does not ensures range [0,1], array must be shifted to 0 to achieve this.
    """
    # max, sum,
    factor = by(arr,axis=axis)
    if axis is None:
        return np.float32(arr) / factor
    return np.float32(arr) / np.expand_dims(factor, axis=axis)

def relativeQuadrants(points):
    """
    Get quadrants of relative vectors obtained from points.

    :param points: array of points.
    :return: quadrants.
    """
    vecs = relativeVectors(points)
    return np.sign(vecs) #recursiveMap(np.sign, vecs)

def vectorsQuadrants(vecs):
    """
    Get quadrants of vectors.

    :param vecs: array of vectors.
    :return: quadrants.
    """
    return np.sign(vecs) # recursiveMap(np.sign, vecs)

def random_points(axes_range = ((-50,50),), nopoints = 4, complete = False):
    """
    Get random points.

    :param axes_range: [x_points_range, y_points_range] where points_range is (min,max) range in axis.
    :param nopoints: number of points.
    :param complete: last point is the first point (adds an additional point i.e. nopoints+1).
    :return: numpy array.
    """
    def getRange(axes_range):
        xr = yr = axes_range[0]
        if len(axes_range)>1:
            yr = axes_range[1]
        return xr,yr
    def getRandom(rmin,rmax):
        return np.random.random()*(rmax-rmin)+rmin
    xr,yr = getRange(axes_range)
    arr = [(getRandom(*xr),getRandom(*yr)) for _ in range(nopoints)]
    if complete: arr.append(arr[0])
    return np.array(arr)

def points_generator(shape = (10,10), nopoints = None, convex = False, erratic = False, complete = False):
    """
    generate points.

    :param shape: enclosed frame (width, height)
    :param nopoints: number of points
    :param convex: if True make points convex,
            else points follow a circular pattern.
    :return:
    """
    h,w = shape[:2]
    if nopoints is None:
        nopoints = np.min(shape)
    random = np.random.random
    if convex and erratic:
        pts = [(w*random(), h*random()) for _ in range(nopoints)]
    else:
        h2,w2 = h/2.,w/2. # center
        div = 2*np.pi/float(nopoints) # circle angle division
        subdiv = 0.9
        pts = []
        for i in range(nopoints):
            if convex and i%2:
                r = np.sqrt((h2-(h2*subdiv)*random())**2+
                            (w2-(w2*subdiv)*random())**2)# radius
            else:
                r = np.sqrt((h2)**2+(w2)**2)# radius
            pts.append((r*np.cos(i*div-div*random()),r*np.sin(i*div-div*random())))
    if complete: pts.append(pts[0])
    return pts

def isnumpy(arr):
    """
    Test whether an object is a numpy array.

    :param arr:
    :return: True if numpy array, else false.
    """
    return type(arr).__module__ == np.__name__

def instability_bf(funcs, step = 10, maximum = 300, guess = 0, tolerance=0.01):
    """
    Find the instability of function approaching value by brute force,

    :param funcs: list of functions
    :param step: (10) step to close guess to maximum
    :param maximum: (300) maximum value, if guess surpass this value then calculations are stopped.
    :param guess: (0) initial guess
    :param tolerance: (0.01) tolerance with last step to check instability.
    :return: (state, updated guess). state is True if successful, else False.
    """
    if guess < maximum:
        s = 1 # to increase
    else:
        s = -1 # to decrease
    step = s*abs(step) # correct step
    while s*(maximum-guess)>0: # check approximation to maximum
        val = np.linspace(start=guess,stop=maximum)
        val = np.sum(np.abs(np.diff(np.array([f(val) for f in funcs]))))
        if val<=tolerance: # check minimization
            return True, guess
        guess += step # update step
    return False, guess # not found or limit reached

def get_x_space(funcs, step = 10, xleft = -300, xright = 300):
    """
    get X axis space by brute force. This can be used to find the x points
    where the points in the y axis of any number of functions become stable.

    :param funcs: list of functions
    :param step: (10) step to close guess to maximum
    :param xleft: maximum left limit
    :param xright: maximum right limit
    :return: linspace
    """
    assert xleft < xright # check data consistency
    l1 = instability_bf(funcs, step = step, maximum= xleft, guess= 0)[1] # left limit
    l2 = instability_bf(funcs, step = step, maximum= xright, guess= 0)[1] # right limit
    return np.linspace(l1,l2,l2-l1) # get array

def histogram(img):
    """
    Get image histogram.

    :param img: gray or image with any bands
    :return: histogram of every band
    """
    sz = img.shape
    if len(sz)==2: # it is gray
        histr = [np.histogram(img.flatten(),256,[0,256])[0]]
        #histr = [cv2.calcHist([img],[0],None,[256],[0,256])]
    else: # it has colors
        histr = [np.histogram(img[:,:,i].flatten(),256,[0,256])[0] for i in range(sz[2])]
        #histr = [cv2.calcHist([img],[i],None,[256],[0,256]) for i in xrange(sz[2])]
    return histr

def find_near(m, thresh = None, side = None):
    """
    helper function for findminima and findmaxima
    :param m: minima or maxima points
    :param thresh: guess or seed point
    :param side: left or right
    :return: value
    """
    if thresh is None: return m
    if side == "left":
        if thresh> m[0]:
            m = m[m <= thresh]
        else:
            return m[0]
    if side == "right":
        if thresh < m[-1]:
            m = m[m >= thresh]
        else:
            return m[-1]
    distances = np.sqrt((m - thresh) ** 2)
    idx = np.where(np.min(distances)==distances)[0]  # get index
    return m[idx][0]

def findminima(hist,thresh=None, side = None):
    """
    Get nearest valley value to a thresh point from a histogram.

    :param hist: histogram
    :param thresh: initial seed
    :param side: find valley from left or right of thresh
    :return:
    """
    # http://stackoverflow.com/a/9667121/5288758
    mn = (np.diff(np.sign(np.diff(hist))) > 0).nonzero()[0] + 1 # local min
    return find_near(mn, thresh, side)

def findmaxima(hist,thresh=None, side = None):
    """
    Get nearest peak value to a thresh point from a histogram.

    :param hist: histogram
    :param thresh: initial seed
    :param side: find valley from left or right of thresh
    :return:
    """
    # http://stackoverflow.com/a/9667121/5288758
    mn = (np.diff(np.sign(np.diff(hist))) < 0).nonzero()[0] + 1 # local max
    return find_near(mn, thresh, side)

def getOtsuThresh(hist):
    """
    From histogram calculate Otsu threshold value.

    :param hist: histogram
    :return: otsu threshold value
    """
    #http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    # find normalized_histogram, and its cumulative distribution function
    hist_norm = hist.astype(np.float32).ravel()/hist.max()
    Q = hist_norm.cumsum() # cumulative distribution function
    bins = np.arange(len(hist_norm))
    fn_min = np.inf # begin from infinity
    thresh = -1 # start thresh from invalid value
    for i in range(1,len(hist_norm)):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[len(hist_norm)-1]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights
        if q1 == 0 or q2 == 0:
            # explicedly continue when there are Nan values
            # it gives the same result even if this block is not evalauted
            # it is used to not create Invalid Division warnings
            continue
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh


def convexityRatio(cnt,hull=None):
    """
    Ratio to test if contours are irregular

    :param cnt: contour
    :param hull:(None) convex hull
    :return: ratio
    """
    if hull is None:
        hull = cv2.convexHull(cnt) # get convex hull
    if len(hull.shape)==2: # convert back from indexes
        from .convert import points2contour
        hull = points2contour(cnt[hull])
    ahull = cv2.contourArea(hull)
    acnt = cv2.contourArea(cnt)
    if ahull == 0: # solves issue dividing by 0
        ahull = 0.0000001
    return acnt/ahull # contour area / hull area


def noisy(arr, mode):
    """
    Add noise to arrays

    :param arr: Input ndarray data (it will be converted to float).
    :param mode: noise method:

        * 'gauss' - Gaussian-distributed additive noise.
        * 'poisson' - Poisson-distributed noise generated from the data.
        * 's&p' - Replaces random pixels with 0 or 1.
        * 'speckle' - Multiplicative noise using out = arr + n*arr,where
                    n is uniform noise with specified mean & variance.
    :return noisy arr

    .. notes:: code was based from http://stackoverflow.com/a/30609854/5288758
    """
    if mode == "gauss":
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, arr.shape)
        gauss = gauss.reshape(*arr.shape)
        return arr + gauss
    elif mode == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        # Salt mode
        num_salt = np.ceil(amount * arr.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in arr.shape]
        arr[coords] = 1 # insert salt

        # Pepper mode
        num_pepper = np.ceil(amount * arr.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in arr.shape]
        arr[coords] = 0 # insert pepper
        return arr
    elif mode == "poisson":
        vals = len(np.unique(arr))
        vals = 2 ** np.ceil(np.log2(vals))
        return np.random.poisson(arr * vals) / float(vals)
    elif mode == "speckle":
        gauss = np.random.randn(*arr.shape)
        gauss = gauss.reshape(*arr.shape)
        return arr + arr * gauss
    else:
        raise Exception("mode {} not recognized".format(mode))

def process_as_blocks(arr, func, block_shape = (3, 3), mask = None, asWindows = False):
    """
    process with function over an array using blocks (using re-striding).

    :param arr: array to process
    :param func: function to feed blocks
    :param block_shape: (3,3) shape of blocks
    :param mask: (None) mask to process arr
    :param asWindows: (False) if True all blocks overlap each other to give
            a result for each position of arr, if False the results are
            given in blocks equivalent for each processed blocks of arr (faster).
    :return: processed array.
    """
    result = np.zeros_like(arr, dtype=np.float32)
    if asWindows:
        blocks_in = view_as_windows(arr, block_shape)
        if mask is not None:
            blocks_mask = view_as_windows(mask, block_shape)
        sr,sc = blocks_in.shape[:2]
        for row in range(sr):
            for col in range(sc):
                b_in = blocks_in[row,col]
                if mask is not None:
                    b_mask = blocks_mask[row,col]
                    indexes = b_mask==1
                    if np.any(indexes):
                        nrow,ncol = row + block_shape[0] // 2, col + block_shape[1] // 2
                        result[nrow,ncol] = func(b_in[indexes])
                else:
                    nrow,ncol = row + block_shape[0] // 2, col + block_shape[1] // 2
                    result[nrow,ncol] = func(b_in)
    else:
        blocks_in = view_as_blocks(arr, block_shape)
        blocks_result = view_as_blocks(result, block_shape)
        if mask is not None:
            blocks_mask = view_as_blocks(mask, block_shape)
        sr,sc = blocks_in.shape[:2]
        for row in range(sr):
            for col in range(sc):
                b_in = blocks_in[row,col]
                b_result = blocks_result[row,col]
                if mask is not None:
                    b_mask = blocks_mask[row,col]
                    indexes = b_mask==1
                    if np.any(indexes):
                        b_result[indexes] = func(b_in[indexes])
                else:
                    b_result[:,:] = func(b_in)
    return result