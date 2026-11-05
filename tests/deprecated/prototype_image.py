# -*- coding: utf-8 -*-
"""
    Bundle of methods for handling images. Rather than manipulating specialized operations in images methods
    in this module are used for loading, outputting and format-converting methods, as well as color manipulation.

    SUPPORTED FORMATS

    see http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#imread

    Windows bitmaps - *.bmp, *.dib (always supported)
    JPEG files - *.jpeg, *.jpg, *.jpe (see the Notes section)
    JPEG 2000 files - *.jp2 (see the Notes section)
    Portable Network Graphics - *.png (see the Notes section)
    Portable image format - *.pbm, *.pgm, *.ppm (always supported)
    Sun rasters - *.sr, *.ras (always supported)
    TIFF files - *.tiff, *.tif (see the Notes section)
"""
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
from config import FLOAT,INT
import glob
import cv2
import numpy as np
from arrayops.basic import anorm, polygonArea, im2shapeFormat, angle, vectorsAngles
from pyqtgraph import QtGui
from cache import cache, resourceManager
from collections import MutableSequence
from directory import getData,strdifference,changedir, checkFile, getFileHandle
import matplotlib.axes
import matplotlib.figure


supported_formats = ("bmp","dib","jpeg","jpg","jpe","jp2","png","pbm","pgm","ppm","sr","ras","tiff","tif")

from plotter import Plotim, limitaxis


def transposeIm(im):
    if len(im.shape) == 2:
        return im.transpose(1, 0)
    else:
        return im.transpose(1, 0, 2)

#from matplotlib import colors

# colors to use
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)
orange = (51, 103, 236)
black = (0, 0, 0)
blue = (255, 0, 0)

# dictionary of colors to use
colors = {
"blue":blue,
"green":green,
"red":red,
"white":white,
"orange":orange,
"black":black}

# look for these as <numpy array>.dtype.names
bgra_dtype = np.dtype({'b': (np.uint8, 0), 'g': (np.uint8, 1), 'r': (np.uint8, 2), 'a': (np.uint8, 3)})

def plt2bgr(image):
    if isinstance(image, matplotlib.axes.SubplotBase):
        image = fig2bgr(image.figure)
    elif isinstance(image, matplotlib.figure.Figure):
        image = fig2bgr(image)
    return image

def plt2bgra(image):
    if isinstance(image, matplotlib.axes.SubplotBase):
        image = fig2bgra(image.figure)
    elif isinstance(image, matplotlib.figure.Figure):
        image = fig2bgra(image)
    return image

def fig2bgr(fig):
    """
    Convert a Matplotlib figure to a RGB image.

    :param fig: a matplotlib figure
    :return: RGB image.
    """
    fig.canvas.draw()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='') # get bgr
    return buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

def np2str(arr):
    return arr.tostring()

def str2np(string,shape):
    buf = np.fromstring(string, dtype=np.uint8, sep='') # get bgr
    return buf.reshape(shape)

def fig2bgra(fig):
    """
    Convert a Matplotlib figure to a RGBA image.

    :param fig: a matplotlib figure
    :return: RGBA image.
    """
    #http://www.icare.univ-lille1.fr/drupal/node/1141
    #http://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8 ) # get bgra
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,)) # reshape to h,w,c
    return np.roll(buf,3,axis = 2) # correct channels

def qi2np(qimage, dtype ='array'):
    """
    Convert QImage to numpy.ndarray.  The dtype defaults to uint8
    for QImage.Format_Indexed8 or `bgra_dtype` (i.e. a record array)
    for 32bit color images.  You can pass a different dtype to use, or
    'array' to get a 3D uint8 array for color images.

    source from: https://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py
    """
    if qimage.isNull():
        raise IOError("Image is Null")
    result_shape = (qimage.height(), qimage.width())
    temp_shape = (qimage.height(), qimage.bytesPerLine() * 8 / qimage.depth())
    if qimage.format() in (QtGui.QImage.Format_ARGB32_Premultiplied,
                            QtGui.QImage.Format_ARGB32,
                            QtGui.QImage.Format_RGB32):
        if dtype == 'rec':
            dtype = bgra_dtype
        elif dtype == 'array':
            dtype = np.uint8
            result_shape += (4, )
            temp_shape += (4, )
    elif qimage.format() == QtGui.QImage.Format_Indexed8:
        dtype = np.uint8
    else:
        raise ValueError("qi2np only supports 32bit and 8bit images")
    # FIXME: raise error if alignment does not match
    buf = qimage.bits().asstring(qimage.numBytes())
    result = np.frombuffer(buf, dtype).reshape(result_shape)
    if result_shape != temp_shape:
        result = result[:,:result_shape[1]]
    if qimage.format() == QtGui.QImage.Format_RGB32 and dtype == np.uint8:
        result = result[...,:3]
    return result

def np2qi(array):
    """
    Convert numpy array to Qt Image.

    source from: https://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py

    :param array:
    :return:
    """
    if np.ndim(array) == 2:
        return gray2qi(array)
    elif np.ndim(array) == 3:
        return rgb2qi(array)
    raise ValueError("can only convert 2D or 3D arrays")

def gray2qi(gray):
    """
    Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!).

    source from: https://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py
    """
    if len(gray.shape) != 2:
        raise ValueError("gray2QImage can only convert 2D arrays")

    gray = np.require(gray, np.uint8, 'C')

    h, w = gray.shape

    result = QtGui.QImage(gray.data, w, h, QtGui.QImage.Format_Indexed8)
    result.ndarray = gray # let object live to avoid garbage collection
    """
    for i in xrange(256):
        result.setColor(i, QtGui.QColor(i, i, i).rgb())"""
    return result

def rgb2qi(rgb):
    """
    Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
    have three dimensions with the vertical, horizontal and RGB image axes.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!).

    source from: https://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py
    """
    if len(rgb.shape) != 3:
        raise ValueError("rgb2QImage can only convert 3D arrays")
    if rgb.shape[2] not in (3, 4):
        raise ValueError("rgb2QImage can expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")

    h, w, channels = rgb.shape
    # Qt expects 32bit BGRA data for color images:
    bgra = np.empty((h, w, 4), np.uint8, 'C')
    bgra[:,:,2] = rgb[:,:,2]
    bgra[:,:,1] = rgb[:,:,1]
    bgra[:,:,0] = rgb[:,:,0]
    # dstack, dsplit, stack
    if rgb.shape[2] == 3:
        bgra[...,3].fill(255)
        fmt = QtGui.QImage.Format_RGB32
    else:
        bgra[...,3] = rgb[...,3]
        fmt = QtGui.QImage.Format_ARGB32
    result = QtGui.QImage(bgra.data, w, h, fmt)
    result.ndarray = bgra  # let object live to avoid garbage collection
    return result

# STABLE FUNCTIONS
def bgra2bgr(im,bgrcolor = colors["white"]):
    """
    Convert BGR to BGRA image.

    :param im: image
    :param bgrcolor: BGR color representing transparency. (information is lost when
                    converting BGRA to BGR) e.g. [200,200,200].
    :return:
    """
    # back[chanel] = bgr[chanel]*(bgr[3]/255.0) + back[chanel]*(1-bgr[3]/255.0)
    temp=im.shape
    im2 = np.zeros((temp[0],temp[1],3), np.uint8)
    im2[:,:,:] = bgrcolor
    for c in range(0,3): #looping over channels
        im2[:,:,c] = im[:,:,c]*(im[:,:,3]/255.0) + im2[:,:,c]*(1.0-im[:,:,3]/255.0)
    return im2

def saveAs(fn, base = "", ext = "jpg"):
    """
    Reads a file and save as other file based in a pattern.

    :param fn: file name or glob name.
    :param base: replacing string for base name.
    :param ext: replacing string for extension.
    :return: True if successful save.
    """
    filelist = glob.glob(fn) # list of files
    mainparts = getData(fn) # get main path parts
    if not ext.startswith("."): ext = "."+ext # correct extension
    for file in filelist:
        replaceparts = getData(base) # from base get parts
        parts = getData(file) # file parts
        if replaceparts[0]: parts[0] = replaceparts[0] # replace drive
        if replaceparts[1]: parts[1] = replaceparts[1] # replace root
        if replaceparts[2]: # to replace basic name
            diff1,diff2,index = strdifference(mainparts[2],parts[2])
            for i in range(len(diff2)):
                if i not in index:
                    diff2[i] = replaceparts[2]
                    break
            parts[2] = "".join(diff2)
        parts[3] = ext # replace extension
        im = cv2.imread(file)
        cv2.imwrite("".join(parts),im)
    return True

def checkLoaded(obj, fn="", raiseError = False):
    """
    Simple function to determine if variable is valid.

    :param obj: loaded object
    :param fn: path of file
    :param raiseError: if True and obj is None, raise
    :return: None
    """
    if obj is not None:
        print(fn, " Loaded...")
    else:
        print(fn, " Could not be loaded...")
        if raiseError: raise

def loadcv(path, flags=-1, shape=None):
    """
    Simple function to load using opencv.

    :param path: path to image.
    :param flag: openCV flags:

                +-------+------------------------------+--------+
                | value | openCV flag                  | output |
                +=======+==============================+========+
                | (1)   | cv2.CV_LOAD_IMAGE_COLOR      | BGR    |
                +-------+------------------------------+--------+
                | (0)   | cv2.CV_LOAD_IMAGE_GRAYSCALE  | GRAY   |
                +-------+------------------------------+--------+
                | (-1)  | cv2.CV_LOAD_IMAGE_UNCHANGED  | format |
                +-------+------------------------------+--------+
    :param shape: shape to resize image.
    :return: loaded image
    """
    im = cv2.imread(path, flags)
    if shape:
        im = cv2.resize(im,shape)
    return im

def interpretImage(toparse, flags):
    """

    Interprets to get image.

    :param toparse: string to parse or array. It can interpret:

        *connection to server (i.e. host:port)
        *path to file (e.g. /path_to_image/image_name.ext)
        *URL to image (e.g. http://domain.com/path_to_image/image_name.ext)
        *image as string (i.g. numpy converted to string)
        *image itself (i.e. numpy array)
    :param flags: openCV flags:

                +-------+------------------------------+--------+
                | value | openCV flag                  | output |
                +=======+==============================+========+
                | (1)   | cv2.CV_LOAD_IMAGE_COLOR      | BGR    |
                +-------+------------------------------+--------+
                | (0)   | cv2.CV_LOAD_IMAGE_GRAYSCALE  | GRAY   |
                +-------+------------------------------+--------+
                | (-1)  | cv2.CV_LOAD_IMAGE_UNCHANGED  | format |
                +-------+------------------------------+--------+
    :return: image or None if not successfull
    """
    # test it is from server
    if ":" in toparse: #process request to server
        from serverServices import parseString
        toparse = parseString(toparse,5)
    # test is object itself
    if type(toparse).__module__ == np.__name__: # test numpy array
        if flags == 1:
            return im2shapeFormat(toparse,(0,0,3))
        if flags == 0:
            return im2shapeFormat(toparse,(0,0))
        return toparse
    # test image in string
    try:
        return cv2.imdecode(toparse,flags)
    except TypeError:
        # test path to file or URL
        return loadsfrom(toparse,flags)

def loadsfrom(path, flags=cv2.IMREAD_COLOR):
    """
    Loads Image from URL or file.

    :param path: filepath or url
    :param flags: openCV flags:

                +-------+------------------------------+--------+
                | value | openCV flag                  | output |
                +=======+==============================+========+
                | (1)   | cv2.CV_LOAD_IMAGE_COLOR      | BGR    |
                +-------+------------------------------+--------+
                | (0)   | cv2.CV_LOAD_IMAGE_GRAYSCALE  | GRAY   |
                +-------+------------------------------+--------+
                | (-1)  | cv2.CV_LOAD_IMAGE_UNCHANGED  | format |
                +-------+------------------------------+--------+
    :return:
    """
    resp = getFileHandle(path) # download the image
    #nparr = np.asarray(bytearray(resp.read()), dtype=dtype) # convert it to a NumPy array
    nparr = np.fromstring(resp.read(), dtype=np.uint8)
    image = cv2.imdecode(nparr, flags=flags) # decode using OpenCV format
    return image

class imFactory(object):
    """
    image factory for RRToolbox to create scripts to standardize loading images and
    provide lazy loading (it can load images from disk with the customized options
    and/or create mapping images to load when needed) to conserve memory.

    .. warning:: In development.
    """
    _interpolations = {"nearest": 0, "bilinear":1, "bicubic":2, "area":3, "lanczos":4}
    _convertions = {}

    def __init__(self, **kwargs):
        """
        :param kwargs:
        :return:

        An image can be represented as a matrix of width "W" and height "H" with elements
        called pixels,each pixel is a representation of a color in one point of a plane
        (2D dimension). In the case of openCV and many other libraries for image manipulation,
        the use of numpy arrays as base for image representation is becoming the standard
        (numpy is a fast and powerful library for array manipulation and one of the main modules
        for scientific development in python). A numpy matrix with n rows and m columns has a
        shape (n,m), that in an Image is H,W which in a Cartesian plane would be y,x.

        if image is W,H = 100,100 then
        dsize = (W,H) = (300,100) would be the same as fsize = (fx,fy) = (3,1)
        after the image is loaded in a numpy array the image would have shape
        (n,m) = (rows,cols) = (H,W) = im.shape
        """
        self.path = None # path to use to load image
        self.mmap_mode = None # mapping file modes
        self.mmap_path = None # path to create numpy file; None, do not create mapping file
        self.w = None
        self.h = None
        self.fx = None
        self.fy = None
        self.convert = None
        self.interpolation = None
        self.throw = True
        self.update(**kwargs)

    def update(self, **kwargs):
        for key,value in kwargs.items():
            if hasattr(self,key):
                setattr(self,key,value)
            else:
                raise Exception("Not attribute '{}'".format(key))

    def get_Func(self):
        """
        gets the loading function
        """
        pass

    def get_code(self):
        """
        get the script code
        """
        pass

    def get_errorFunc(self, path=None, throw=None):
        def errorFunc(im):
            if throw and im is None:
                if checkFile(path):
                    if getData(path)[-1] in supported_formats:
                        raise Exception("Not enough permissions to load {}".format(path))
                    else:
                        raise Exception("Failed to load {}. Format not supported".format(path))
                else:
                    raise Exception("Missing file {}".format(path))
        return {None:errorFunc}

    def get_loadFunc(self, flag=None):
        def loadFunc(path):
            return cv2.imread(path, flag)
        return {"im":loadFunc}

    def get_resizeFunc(self, dsize= None, dst=None, fx=None, fy=None, interpolation=None):
        # see http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
        fx, fy, interpolation= fx or 0, fy or 0, interpolation or 0
        def resizeFunc(im):
            return cv2.resize(im, dsize, dst, fx, fy, interpolation)
        return {"im":resizeFunc}

    def get_mapFunc(self, flag = None, RGB = None, mpath=None,mode=None,func=None,dsize= None, dst=None, fx=None, fy=None, interpolation=None):
        def mapFunc(path):
            if mpath == "*": # save mmap in working directory
                drive,dirname,(filename,ext) = "","",getData(path)[-2:]
            elif mpath:# save mmap in mpath
                drive,dirname,filename,ext = getData(changedir(path,mpath))
            else: # save mmap in image path
                drive,dirname,filename,ext = getData(path)
            # THIS CREATES ONE HASHED FILE
            hashed = hash("{}{}{}{}{}{}".format(flag,RGB,dsize,fx,fy,interpolation))
            savepath = "{}{}{}{}.{}.npy".format(drive,dirname,filename,ext,hashed)
            try: # load from map
                return np.lib.load(savepath,mode) # mapper(savepath,None,mode,True)[0]#
            except IOError: # create object and map
                im = func(path)
                if im is None: # this is regardless of throw flag
                    raise Exception("Failed to load image to map")
                np.save(savepath,im)
                return np.lib.load(savepath,mode) # mapper(savepath,im,mode,True)[0]#
        return {"im":mapFunc}

    def get_transposeFunc(self):
        def transposeFunc(im):
            if len(im.shape) == 2:
                return im.transpose(1,0)
            else:
                return im.transpose(1,0,2) # np.ascontiguousarray? http://stackoverflow.com/a/27601130/5288758
        return {"im":transposeFunc}

    def get_convertionFunc(self, code):
        def convertionFunc(im):
            return cv2.cvtColor(im,code)
        return {"im":convertionFunc}

    def get_np2qi(self):
        return {"im":np2qi}


class imLoader(resourceManager):
    """
    Class to standardize loading images and manage memory efficiently.
    """
    def __init__(self, loader = None, maxMemory = None, margin = 0.8, unit = "MB", all = True, config = None):
        """

        :param loader:
        :param maxMemory:
        :param margin:
        :param unit:
        :param all:
        :param config:
        :return:
        """
        super(imLoader,self).__init__(maxMemory,margin,unit,all)
        # create factory functions
        #if config is None: from config import MANAGER as config
        #self._config = config
        self._default_loader = loader or loadFunc()

    def register(self, key, path = None, method=None):
        if method:
            def func(): return method(func.path)
        else:
            def func(): return self._default_loader(func.path)
        func.path = path
        super(imLoader,self).register(key=key,method=func)

def loadFunc(flag = 0, dsize= None, dst=None, fx=None, fy=None, interpolation=None, mmode = None, mpath = None, throw = True):
    """
    Creates a function that loads from path or url an image or file (supports databases)

    :param flag: (default: 0) 0 to read as gray, 1 to read as BGR, -1 to
                read as BGRA, 2 to read as RGB, -2 to read as RGBA.

                It supports openCV flags:
                    * cv2.CV_LOAD_IMAGE_COLOR
                    * cv2.CV_LOAD_IMAGE_GRAYSCALE
                    * cv2.CV_LOAD_IMAGE_UNCHANGED

                +-------+-------------------------------+--------+
                | value | openCV flag                   | output |
                +=======+===============================+========+
                | (2)   | N/A                           | RGB    |
                +-------+-------------------------------+--------+
                | (1)   | cv2.CV_LOAD_IMAGE_COLOR       | BGR    |
                +-------+-------------------------------+--------+
                | (0)   | cv2.CV_LOAD_IMAGE_GRAYSCALE   | GRAY   |
                +-------+-------------------------------+--------+
                | (-1)  | cv2.CV_LOAD_IMAGE_UNCHANGED   | BGRA   |
                +-------+-------------------------------+--------+
                | (-2)  | N/A                           | RGBA   |
                +-------+-------------------------------+--------+

    :param dsize: (None) output image size; if it equals zero, it is computed as:

                \texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}

    :param dst: (None) output image; it has the size dsize (when it is non-zero) or the
                size computed from src.size(), fx, and fy; the type of dst is uint8.
    :param fx: scale factor along the horizontal axis; when it equals 0, it is computed as

                \texttt{(double)dsize.width/src.cols}

    :param fy: scale factor along the vertical axis; when it equals 0, it is computed as

                \texttt{(double)dsize.height/src.rows}

    :param interpolation: interpolation method compliant with opencv:

                +-----+-----------------+-------------------------------------------------------+
                |flag | Operation       | Description                                           |
                +=====+=================+=======================================================+
                |(0)  | INTER_NEAREST   | nearest-neighbor interpolation                        |
                +-----+-----------------+-------------------------------------------------------+
                |(1)  | INTER_LINEAR    | bilinear interpolation (used by default)              |
                +-----+-----------------+-------------------------------------------------------+
                |(2)  | INTER_CUBIC     | bicubic interpolation over 4x4 pixel neighborhood     |
                +-----+-----------------+-------------------------------------------------------+
                |(3)  | INTER_AREA      | resampling using pixel area relation.                 |
                |     |                 | It may be a preferred method for image decimation,    |
                |     |                 | as it gives moire’-free results. But when the image   |
                |     |                 | is zoomed, it is similar to the INTER_NEAREST method. |
                +-----+-----------------+-------------------------------------------------------+
                |(4)  | INTER_LANCZOS4  | Lanczos interpolation over 8x8 pixel neighborhood     |
                +-----+-----------------+-------------------------------------------------------+

    :param mmode: (None) mmode to create mapped file. if mpath is specified loads image, converts
                to mapped file and then loads mapping file with mode {None, 'r+', 'r', 'w+', 'c'}
                (it is slow for big images). If None, loads mapping file to memory (useful to keep
                image copy for session even if original image is deleted or modified).
    :param mpath: (None) path to create mapped file.
                None, do not create mapping file
                "", uses path directory;
                "*", uses working directory;
                else, uses specified directory.

    .. note:: If mmode is None and mpath is given it creates mmap file but loads from it to memory.
             It is useful to create physical copy of data to keep loading from (data can be reloaded
             even if original file is moved or deleted).
    :return loader function
    """
    # create factory functions
    def errorFunc(im,path):
        if im is None:
            if checkFile(path):
                if getData(path)[-1] in supported_formats:
                    raise Exception("Not enough permissions to load {}".format(path))
                else:
                    raise Exception("Failed to load {}. Format not supported".format(path))
            else:
                raise Exception("Missing file {}".format(path))

    RGB = False
    if abs(flag)==2: # determine if needs to do conversion from BGR to RGB
        flag = flag//2 # get normal flag
        RGB = True

    def loadfunc(path):
        im = loadsfrom(path, flag) # load func
        if throw: errorFunc(im,path) # if not loaded throw error
        if flag<0 and len(im.shape)!=4:
            if RGB:
                return cv2.cvtColor(im,cv2.COLOR_BGR2RGBA)
            return cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)
        if RGB:
            if flag < 0:
                return cv2.cvtColor(im,cv2.COLOR_BGRA2RGBA)
            else:
                return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        return im

    if dsize or dst or fx or fy:
        fx, fy, interpolation= fx or 0, fy or 0, interpolation or 0
        def resizefunc(path):
            return cv2.resize(loadfunc(path), dsize, dst, fx, fy, interpolation)
        func = resizefunc
    else:
        func = loadfunc

    if mmode or mpath is not None: # if there is a mmode, or mpath is string

        def mapfunc(path):
            if mpath == "*": # save mmap in working directory
                drive,dirname,(filename,ext) = "","",getData(path)[-2:]
            elif mpath:# save mmap in mpath
                drive,dirname,filename,ext = getData(changedir(path,mpath))
            else: # save mmap in image path
                drive,dirname,filename,ext = getData(path)
            """
            # THIS CREATES A FOLDER TO MEMOIZE
            def dummy(path,flag=0,dsize=0,fx=0,fy=0,interpolation=0):
                return func(path)
            savepath = "{}{}{}{}".format(drive,dirname,filename,ext)
            return memoize(savepath,mmap_mode=mmode)(dummy)(path,flag,dsize,fx,fy,interpolation)"""
            """
            # THIS CREATES TWO FILES BUT ONLY MEMOIZE ONE STATE OF IM ARGUMENTS
            savepath = "{}{}{}{}.{}".format(drive,dirname,filename,ext,"memoized")
            comps = ("flag","dsize","fx","fy","interpolation")
            try: # load from map
                data = mapper(savepath, mmode=mmode)[0]
                bad = [i for i in comps if data[i] != locals()[i]]
                if bad:
                    raise IOError
                else:
                    return data["im"]
            except IOError: # create object and map
                im = func(path)
                if im is None: # this is regardless of throw flag
                    raise Exception("Failed to load image to map")
                data = dict(im=im,flag=flag,dsize=dsize,fx=fx,fy=fy,interpolation=interpolation)
                return mapper(savepath,data,mmode)[0]["im"]"""
            """
            # THIS CREATES TWO HASHED FILES
            hashed = hash("{}{}{}{}{}{}".format(flag,RGB,dsize,fx,fy,interpolation))
            savepath = "{}{}{}{}.{}".format(drive,dirname,filename,ext,hashed)
            try: # load from map
                im = mapper(savepath, mmode=mmode)[0]
                return im
            except IOError: # create object and map
                im = func(path)
                if im is None: # this is regardless of throw flag
                    raise Exception("Failed to load image to map")
                return mapper(savepath,im,mmode)[0]"""
            # THIS CREATES ONE HASHED FILE
            hashed = hash("{}{}{}{}{}{}".format(flag,RGB,dsize,fx,fy,interpolation))
            savepath = "{}{}{}{}.{}.npy".format(drive,dirname,filename,ext,hashed)
            try: # load from map
                return np.lib.load(savepath, mmode) # mapper(savepath,None,mmode,True)[0]#
            except IOError: # create object and map
                im = func(path)
                if im is None: # this is regardless of throw flag
                    raise Exception("Failed to load image to map")
                np.save(savepath,im)
                return np.lib.load(savepath, mmode) # mapper(savepath,im,mmode,True)[0]#
        return mapfunc # factory function
    return func # factory function

class pathLoader(MutableSequence):
    """
    Class to standardize loading images from list of paths and offer lazy evaluations.
    """
    def __init__(self, fns = None, loader = None):
        """

        :param fns: list of paths
        :param loader: path loader (loadcv,loadsfrom, or function from loadFunc)

        .. olsosee:: :func:`loadcv`, :func:`loadsfrom`, :func:`loadFunc`

        Example::

            fns = ["/path to/image 1.ext","/path to/image 2.ext"]
            imgs = pathLoader(fns)
            print imgs[0] # loads image in path 0
            print imgs[1] # loads image in path 1
        """
        # create factory functions
        self._fns = fns or []
        self._loader = loader or loadFunc()

    def __call__(self):
        """
        if called returns the list of paths
        """
        return self._fns

    def __getitem__(self, key):
        return self._loader(self._fns[key])

    def __setitem__(self, key, value):
        self._fns[key] = value

    def __delitem__(self, key):
        del self._fns[key]

    def __len__(self):
        return len(self._fns)

    def insert(self, index, value):
        self._fns.insert(index,value)

############################# GETCOORS ############################################
# http://docs.opencv.org/master/db/d5b/tutorial_py_mouse_handling.html
# http://docs.opencv.org/modules/highgui/doc/qt_new_functions.html

class imcoors(object):
    """
    Image's coordinates class.
    Example::

        a = ImCoors(np.array([(116, 161), (295, 96), (122, 336), (291, 286)]))
        print a.__dict__
        print "mean depend on min and max: ", a.mean
        print a.__dict__
        print "after mean max has been already been calculated: ", a.max
        a.data = np.array([(116, 161), (295, 96)])
        print a.__dict__
        print "mean and all its dependencies are processed again: ", a.mean
    """
    def __init__(self, pts, dtype=FLOAT, deg=False):
        """
        Initiliazes ImCoors.

        :param pts: list of points
        :param dtype: return data as dtype. Default is config.FLOAT
        """
        self._pts = pts # supports bigger numbers
        self._dtype = dtype
        self._deg = deg
    @property
    def pts(self):
        return self._pts
    @pts.setter
    def pts(self, value):
        getattr(self,"__dict__").clear()
        self._pts = value
    @pts.deleter
    def pts(self):
        raise Exception("Cannot delete attribute")
    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self,value):
        getattr(self,"__dict__").clear()
        self._dtype = value
    @dtype.deleter
    def dtype(self):
        raise Exception("Cannot delete attribute")
    # DATA METHODS
    def __len__(self):
        return len(self._pts)
    @cache
    def max(self):
        """
        Maximum in each axis.

        :return: x_max, y_max
        """
        #self.max_x, self.max_y = np.max(self.data,0)
        return tuple(np.max(self._pts, 0))
    @cache
    def min(self):
        """
        Minimum in each axis.

        :return: x_min, y_min
        """
        #self.min_x, self.min_y = np.min(self.data,0)
        return tuple(np.min(self._pts, 0))
    @cache
    def rectbox(self):
        """
        Rectangular box enclosing points (origin and end point or rectangle).

        :return: (x0,y0),(x,y)
        """
        return (self.min,self.max)
    @cache
    def boundingRect(self):
        """
        Rectangular box dimensions enclosing points.

        :return: x0,y0,w,h
        """
        return cv2.boundingRect(self._pts)
    @cache
    def minAreaRect(self):
        return cv2.minAreaRect(self._pts)
    @cache
    def rotatedBox(self):
        """
        Rotated rectangular box enclosing points.

        :return: 4 points.
        """
        return self._dtype(cv2.cv.BoxPoints(self.minAreaRect))
    @cache
    def boxCenter(self):
        """
        Mean in each axis.

        :return: x_mean, y_mean
        """
        #self.mean_x = (self.max_x+self.min_x)/2
        #self.mean_y = (self.max_y+self.min_y)/2
        xX,xY = self.max
        nX,nY = self.min
        return tuple(self._dtype((xX + nX, xY + nY)) / 2)
    @cache
    def mean(self):
        """
        Center or mean.
        :return: x,y
        """
        # http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
        # https://www.grc.nasa.gov/www/K-12/airplane/cg.html
        #self.center_x, self.center_y = np.sum(self.data,axis=0)/len(self.data)
        #map(int,np.mean(self.data,0))
        #tuple(np.sum(self.data,axis=0)/len(self.data))
        return tuple(np.mean(self._pts, 0, dtype = self._dtype))
    center = mean
    @cache
    def area(self):
        """
        Area of points.

        :return: area number
        """
        return polygonArea(self._pts)
    @cache
    def rectangularArea(self):
        """
        Area of rectangle enclosing points aligned with x,y axes.

        :return: area number
        """
        #method 1, it is not precise in rotation
        (x0,y0),(x,y) = self.rectbox
        return np.abs((x-x0)*(y-y0),dtype=self.dtype)
    @cache
    def rotatedRectangularArea(self):
        """
        Area of Rotated rectangle enclosing points.

        :return: area number
        """
        return polygonArea(self.rotatedBox)
    @cache
    def rectangularity(self):
        """
        Ratio that represent a perfect square aligned with x,y axes.

        :return: ratio from 1 to 0, 1 representing a perfect rectangle.
        """
        #method 1
        #cx,cy = self.center
        #bcx,bcy=self.boxCenter
        #return (cx)/bcx,(cy)/bcy # x_ratio, y_ratio
        #method 2
        return self.dtype(self.area/self.rectangularArea)
    @cache
    def rotatedRectangularity(self):
        """
        Ratio that represent a perfect rotated square fitting points.

        :return: ratio from 1 to 0, 1 representing a perfect rotated rectangle.
        """
        return self.dtype(self.area/self.rotatedRectangularArea)
    @cache
    def regularity(self):
        """
        Ratio of rectangular forms. e.g. squares and rectangles have rect
        angles so they are regular and regularity must give 1.

        :return:
        """
        pi = angle((1,0),(0,1),deg=self._deg) # get pi value in radian or degrees
        av = self.vertexesAngles
        return pi*(len(av))/np.sum(av) # pi*number_agles/sum_angles
    @cache
    def relativeVectors(self):
        """
        Form vectors from points.

        :return: array of vectors [V0, ... , (V[n] = x[n+1]-x[n],y[n+1]-y[n])].
        """
        pts = np.array(self._pts)
        pts = np.append(pts,[pts[0]],axis=0) # adds last vector from last and first point.
        return np.stack([np.diff(pts[:, 0]), np.diff(pts[:, 1])], 1)
    @cache
    def vertexesAngles(self):
        """
        Relative angle of vectors formed by vertexes.

        i.e. angle between vectors "v01" formed by points "p0-p1" and "v12"
        formed by points "p1-p2" where "p1" is seen as a vertex (where vectors cross).

        :return: angles.
        """
        vs = self.relativeVectors # get all vectors from points.
        vs = np.roll(np.append(vs,[vs[-1]],axis=0),2) # add last vector to first position
        return np.array([angle(vs[i-1],vs[i],deg=self._deg) for i in range(1,len(vs))],self._dtype) # caculate angles
    @cache
    def pointsAngles(self):
        """
        Angle of vectors formed by points in Cartesian plane with respect to x axis.

        i.e. angle between vector "v01" (formed by points "p0-p1") and vector unity in axis x.

        :return: angles.
        """
        vs = self.relativeVectors # get all vectors from points.
        return vectorsAngles(vec=vs, dtype=self._dtype, deg=self._deg)
    @cache
    def vectorsAngles(self):
        """
        Angle of vectors in Cartesian plane with respect to x axis.

        i.e. angle between vector "v0" (formed by point "p0" and the origin) and vector unity in axis x.
        :return: angles.
        """
        return np.array([angle((1,0),i,deg=self._deg) for i in self._pts],self._dtype) # caculate angles with respect to x axis

def drawcoorpoints(vis,points,col_out=black,col_in=red,radius=2):
    """
    Funtion to draw points.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """
    points = np.array(points,INT)
    radius_in = radius-1
    for x,y in points:
        cv2.circle(vis, (x,y), radius, col_out, -1)
        cv2.circle(vis, (x,y), radius_in, col_in, -1)
    return vis

def myline(img, pt1, pt2, color, thickness=None):
    """
    Funtion to draw points (experimental).

    :param img:
    :param pt1:
    :param pt2:
    :param color:
    :param thickness:
    :return:
    """
    # y=m*x+b
    x1,y1=np.array(pt1,dtype=FLOAT)
    x2,y2=np.array(pt2,dtype=FLOAT)
    m = (y2-y1)/(x2-x1)
    xmin,xmax = np.sort([x1,x2])
    xvect = np.arange(xmin,xmax+1).astype('int')
    yvect = np.array(xvect*m+int(y1-x1*m),dtype=np.int)
    for i in zip(yvect,xvect):
        #img.itemset(i,color)
        img[i]=color

def drawcooraxes(vis,points,col_out=black,col_in=green,radius=2):
    """
    Function to draw axes instead of points.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """
    points = np.array(points,INT)
    thickness = radius-1
    h1, w1 = vis.shape[:2]  # obtaining image dimensions
    for i in points:
        h1pt1 = (0,i[1])
        h1pt2 = (w1,i[1])
        w2pt1 = (i[0],0)
        w2pt2 = (i[0],h1)
        cv2.line(vis, h1pt1, h1pt2, col_in, thickness)
        cv2.line(vis, w2pt1, w2pt2, col_in, thickness)
        vis = drawcoorpoints(vis,points,col_out,col_in,radius)
    return vis

def drawcoorpolyline(vis,points,col_out=black,col_in=red,radius=2):
    """
    Function to draw interaction with points to obtain polygonal.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """

    thickness = radius-1
    if len(points)>1:
        points = np.array(points,INT)
        cv2.polylines(vis,[points],False, col_in, thickness)
        """
        for i in range(len(points)-1):
            pt1 = (points[i][0], points[i][1])
            pt2 = (points[i+1][0], points[i+1][1])
            cv2.line(vis, pt1, pt2, col_in, thickness)"""
    else:
        vis = drawcoorpoints(vis,points,col_out,col_in,radius)
    return vis

def drawcoorperspective(vis,points,col_out=black,col_in=red,radius=2):
    """
    Function to draw interaction with points to obtain perspective.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """
    points = np.array(points,INT)
    thickness = radius-1
    if len(points)>1 and len(points)<5:
        for i in range(len(points)-1):
            if i%2:
                for j in range(i+1,min(len(points),i+3)):
                    if j%2:
                        #print "i=",i," j=",j
                        pt1 = (points[i][0], points[i][1])
                        pt2 = (points[j][0], points[j][1])
                        cv2.arrowedLine(vis, pt1, pt2, col_in, thickness)
            else:
                for j in range(i+1,min(len(points),i+3)):
                    #print "i=",i," j=",j
                    pt1 = (points[i][0], points[i][1])
                    pt2 = (points[j][0], points[j][1])
                    cv2.arrowedLine(vis, pt1, pt2, col_in, thickness)
        vis = drawcoorpoints(vis,points,col_out,col_in,radius)
    else:
        vis = drawcoorpoints(vis,points,col_out,col_in,radius)
    return vis

def limitaxispoints(c,maxc,minc=0):
    """
    Limit a point in axis.

    :param c: list of points..
    :param maxc: maximum value of point.
    :param minc: minimum value of point.
    :return: return limited points.
    """
    x = np.zeros(len(c),dtype=np.int)
    for i,j in enumerate(c):
        x[i] = limitaxis(j,maxc,minc)
    return tuple(x)

def getcoor(im, win = "get coordinates", updatefunc=drawcoorpoints, prox=8, radius = 3, unique=True, col_out=black,col_in=red):
    """
    Create window to select points from image.

    :param im: image to get points.
    :param win: window name.
    :param updatefunc: function to draw interaction with points.(e.g. limitaxispoints, drawcoorperspective, etc.).
    :param prox: proximity to identify point.
    :param radius: radius of drawn points.
    :param unique: If True no point can be repeated, else selected points can be repeated.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :return:
    """
    # functions
    def getcoor_drawstats(self,points,col_out=black,col_in=green,radius=2):
        """

        :param self:
        :param points:
        :param col_out:
        :param col_in:
        :param radius:
        :return:
        """
        vis = self.rimg
        p = imcoors(points)
        self.data2 = np.zeros((vis.shape[0],vis.shape[1],1),dtype=np.uint8)
        drawcooraxes(vis,[p.boxCenter],col_out,col_in,radius)
        drawcooraxes(self.data2,[p.boxCenter],1,1,self.prox)
        drawcooraxes(vis,[p.mean],col_in,col_out,radius)
        drawcooraxes(self.data2,[p.mean],2,2,self.prox)
        p1 = imcoors(self.coors)
        self.mapdata2 = [None,"center at "+str(p1.boxCenter),"mean at "+str(p1.mean)]

    def getcoor_updatecoors(self):
        """

        :param self:
        :return:
        """
        self.coorlen = len(self.coors)
        self.updaterenderer()
        if self.coors != []:
            self.rcoors = self.coors[:]
            newc = self.coors[:]
            for j,i in enumerate(self.coors):
                newc[j] = self.real2render(i[0],i[1])
                self.rcoors[j] = limitaxispoints(newc[j],10000,-10000)
            if self.showstats:
                getcoor_drawstats(self,newc,radius=self.radius)
            self.rimg = self.updatefunc(self.rimg,newc,self.col_out,self.col_in,self.radius)
        else:
            self.data2[:] = 0
            self.coordinateText = [["xy({self.x},{self.y})"]]

    def getcoor_mouse(self):
        """

        :param self:
        :return:
        """
        # control system
        controlled = self.builtincontrol()
        drawed = False

        # get nearest coordinate to pointer
        isnear = False
        if self.coors != [] and self.rx is not None and self.ry is not None:
            # vals = anorm(np.int32(self.coors) - (self.x, self.y))  # relative to real coordinates
            vals = anorm(np.int32(self.rcoors) - (self.rx, self.ry))  # relative to rendered coordinates
            near_point = np.logical_and(vals < self.prox, vals == np.min(vals))
            if np.any(near_point): # if near point
                idx = np.where(near_point)[0]  # get index
                isnear = True
                val = self.coors[idx[0]]
                count = self.coors.count(val)
                self.coordinateText = [["point "+str(idx[0])+" at "+str(val)+"x"+str(count)]]
            else:
                self.coordinateText = [["xy({self.x},{self.y})"]]

        # coordinate system
        if not controlled and bool(self.flags):
            if self.event== cv2.EVENT_RBUTTONDBLCLK:  # if middle button DELETE ALL COORDINATES
                self.coors = []
                self.img = self.data.copy()
                drawed = True
            elif isnear and self.event== cv2.EVENT_RBUTTONDOWN:  # if right button DELETE NEAREST COORDINATE
                self.coors.pop(idx[0])  # if more than one point delete first
                drawed = True
            elif self.event== cv2.EVENT_LBUTTONDOWN:  # if left button ADD COORDINATE
                val = (self.x,self.y)
                if not self.coors.count(val) or not self.unique:
                    self.coors.append(val)
                drawed = True

        # update renderer
        if (controlled or drawed):
            getcoor_updatecoors(self)

        if self.y is not None and self.x is not None:
            if self.showstats:
                data = self.mapdata2[self.data2[self.ry,self.rx]]
                if not isnear and data is not None:
                    self.coordinateText = [[data]]
            self.builtinplot(self.data[self.y,self.x])

    if type(im) is Plotim:
        self = im
    else:
        self = Plotim(win, im)
    # assign functions
    self.mousefunc = getcoor_mouse
    self.updatefunc = updatefunc
    self.userupdatefunc = updatefunc
    self.prox = prox  # proximity to keypoint
    # initialize user variables
    self.radius = radius
    self.unique = unique
    self.col_out = col_out
    self.col_in = col_in
    # initialize control variables
    self.interpolation=cv2.INTER_AREA
    self.coors = []
    self.rcoors = []
    self.coorlen = 0
    self.showstats = False
    self.mapdata2 = [None,None,None]
    self.data2 = np.zeros((self.rH,self.rW,1),dtype=np.uint8)
    self.getcoor_updatecoors = getcoor_updatecoors
    self.drawcooraxes = drawcooraxes
    self.drawcoorperspective = drawcoorperspective
    self.drawcoorpolyline = drawcoorpolyline
    self.drawcoorpoints = drawcoorpoints
    self.controlText[0].extend([" No. coordinates: {self.coorlen}. "])
    self.cmdeval.update({"points":"self.updatefunc = self.drawcoorpoints",
                  "polyline":"self.updatefunc = self.drawcoorpolyline",
                  "perspective":"self.updatefunc = self.drawcoorperspective",
                  "axes":"self.updatefunc = self.drawcooraxes",
                  "user":"self.updatefunc = self.userupdatefunc",
                  "end":["self.getcoor_updatecoors(self)","self.mousefunc(self)"]})
    self.cmdlist.extend(["unique","showstats","user","points","polyline","perspective","axes"])
    # show window
    self.show()
    return self.coors # return coordinates

def separe(values, sep, axis=0):
    """
    Separate values from separator or threshold.

    :param values: list of values
    :param sep: peparator value
    :param axis: axis in each value
    :return:lists of greater values, list of lesser values
    """
    greater,lesser = [],[]
    for i in values:
        if i[axis]>sep:
            greater.append(i)
        else:
            lesser.append(i)
    return greater,lesser


def getrectcoors(*data):
    """
     Get ordered points.

    :param data: list of points
    :return: [Top_left,Top_right,Bottom_left,Bottom_right]
    """
    #[Top_left,Top_right,Bottom_left,Bottom_right]
    #img, win = "get pixel coordinates", updatefunc = drawpoint
    if len(data)==1:  # points
        points = data[0]
    else:  # img, win
        points = getcoor(*data)

    p = imcoors(points)
    min_x,min_y = p.min
    max_x,max_y = p.max
    Top_left = (min_x,min_y)
    Top_right = (max_x,min_y)
    Bottom_left = (min_x,max_y)
    Bottom_right = (max_x,max_y)
    return [Top_left,Top_right,Bottom_left,Bottom_right]

def quadrants(points):
    """
    Separate points respect to center of gravity point.

    :param points: list of points
    :return: [[Top_left],[Top_right],[Bottom_left],[Bottom_right]]
    """
    # group points on 4 quadrants
    # [Top_left,Top_right,Bottom_left,Bottom_right]
    p = imcoors(points)  # points data x,y -> (width,height)
    mean_x, mean_y = p.mean
    Bottom,Top = separe(points,mean_y,axis=1)
    Top_right,Top_left = separe(Top,mean_x,axis=0)
    Bottom_right,Bottom_left = separe(Bottom,mean_x,axis=0)
    return [Top_left,Top_right,Bottom_left,Bottom_right]

def getgeometrycoors(*data):
    """
    Get filled object coordinates. (function in progress)
    """
    #[Top_left,Top_right,Bottom_left,Bottom_right]
    #img, win = "get pixel coordinates", updatefunc = drawpoint
    if len(data)==1:  # points
        points = data[0]
    else:  # img, win
        points = getcoor(*data)
    return points