# -*- coding: utf-8 -*-
r"""
Dynamic image loader factories, lazy path loaders, and resource-managed loader dictionaries.
"""
from __future__ import division
import cv2
import numpy as np
from collections.abc import MutableSequence
from ..cache import ResourceManager
from ..directory import getData, checkFile, changedir
from ..config import MANAGER
from .conversions import supported_formats, np2qi
from .io import interpretImage


class ImFactory(object):
    """
    Image factory for RRToolbox to create scripts to standardize loading images and
    provide lazy loading (it can load images from disk with the customized options
    and/or create mapping images to load when needed) to conserve memory.

    .. warning:: In development.
    """
    _interpolations = {"nearest": 0, "bilinear": 1,
                       "bicubic": 2, "area": 3, "lanczos": 4}
    _convertions = {}

    def __init__(self, **kwargs):
        self.path = None  # path to use to load image
        self.mmap_mode = None  # mapping file modes
        self.mmap_path = None  # path to create numpy file; None, do not create mapping file
        self.w = None
        self.h = None
        self.fx = None
        self.fy = None
        self.convert = None
        self.interpolation = None
        self.throw = True
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception("Not attribute '{}'".format(key))

    def get_Func(self):
        """gets the loading function"""
        pass

    def get_code(self):
        """get the script code"""
        pass

    def get_errorFunc(self, path=None, throw=None):
        def errorFunc(im):
            if throw and im is None:
                if checkFile(path):
                    if getData(path)[-1] in supported_formats:
                        raise Exception(
                            "Not enough permissions to load '{}'".format(path))
                    else:
                        raise Exception(
                            "Failed to load '{}'. Format not supported".format(path))
                else:
                    raise Exception("Missing file '{}'".format(path))
        return {None: errorFunc}

    def get_loadFunc(self, flag=None):
        def loadFunc(path):
            return cv2.imread(path, flag)
        return {"im": loadFunc}

    def get_resizeFunc(self, dsize=None, dst=None, fx=None, fy=None, interpolation=None):
        fx, fy, interpolation = fx or 0, fy or 0, interpolation or 0

        def resizeFunc(im):
            return cv2.resize(im, dsize, dst, fx, fy, interpolation)
        return {"im": resizeFunc}

    def get_mapFunc(self, flag=None, RGB=None, mpath=None, mode=None,
                    func=None, dsize=None, dst=None, fx=None, fy=None,
                    interpolation=None):
        def mapFunc(path):
            if mpath == "*":  # save mmap in working directory
                drive, dirname, (filename, ext) = "", "", getData(path)[-2:]
            elif mpath:  # save mmap in mpath
                drive, dirname, filename, ext = getData(changedir(path, mpath))
            else:  # save mmap in image path
                drive, dirname, filename, ext = getData(path)
            # THIS CREATES ONE HASHED FILE
            hashed = hash("{}{}{}{}{}{}".format(
                flag, RGB, dsize, fx, fy, interpolation))
            savepath = "{}{}{}{}.{}.npy".format(
                drive, dirname, filename, ext, hashed)
            try:  # load from map
                return np.lib.load(savepath, mode)
            except IOError:  # create object and map
                im = func(path)
                if im is None:  # this is regardless of throw flag
                    raise Exception("Failed to load image to map")
                np.save(savepath, im)
                return np.lib.load(savepath, mode)
        return {"im": mapFunc}

    def get_transposeFunc(self):
        def transposeFunc(im):
            if len(im.shape) == 2:
                return im.transpose(1, 0)
            else:
                return im.transpose(1, 0, 2)
        return {"im": transposeFunc}

    def get_convertionFunc(self, code):
        def convertionFunc(im):
            return cv2.cvtColor(im, code)
        return {"im": convertionFunc}

    def get_np2qi(self):
        return {"im": np2qi}


def _calc_dsize_keepratio(shape, dsize=None):
    x, y = dsize
    sy, sx = shape[:2]
    if x is not None and y is None:
        return x, int(sy * (x * 1.0 / sx))
    elif y is not None and x is None:
        return int(sx * (y * 1.0 / sy)), y
    else:
        return sx, sy


def _calc_dsize_nokeepratio(shape, dsize=None):
    dsize = list(dsize)
    ndsize = shape[:2][::-1]
    for i, val in enumerate(dsize):
        if val is None:
            dsize[i] = ndsize[i]
    return tuple(dsize)


def loadFunc(flag=0, dsize=None, dst=None, fx=None, fy=None, interpolation=None,
              mmode=None, mpath=None, throw=True, keepratio=True):
    """
    Creates a function that loads image array from path, url,
    server, string or directly from numpy array (supports databases).

    :param flag: (default: 0) 0 to read as gray, 1 to read as BGR, -1 to
                read as BGRA, 2 to read as RGB, -2 to read as RGBA.
    :param dsize: (None) output image size.
    :param dst: (None) output image destination.
    :param fx: scale factor along horizontal axis.
    :param fy: scale factor along vertical axis.
    :param interpolation: interpolation method compliant with opencv.
    :param mmode: mmode to create mapped file.
    :param mpath: path to create mapped file.
    :param keepratio: True to keep image ratio.
    :return: loader function
    """
    def errorFunc(im, path):
        if im is None:
            if checkFile(path):
                if getData(path)[-1][1:] in supported_formats:
                    raise Exception(
                        "Not enough permissions to load '{}'".format(path))
                else:
                    raise Exception(
                        "Failed to load '{}'. Format not supported".format(path))
            else:
                raise Exception("Missing file '{}'".format(path))

    RGB = False
    if abs(flag) == 2:  # determine if needs to do conversion from BGR to RGB
        flag = flag // 2  # get normal flag
        RGB = True

    def loadfunc(path):
        im = interpretImage(path, flag)  # load func
        if throw:
            errorFunc(im, path)  # if not loaded throw error
        if flag < 0 and im.shape[2] != 4:
            if RGB:
                return cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
            return cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        if RGB:
            if flag < 0:
                return cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
            else:
                return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    if dsize or dst or fx or fy:
        if fx is None and fy is None:
            fx = fy = 1.0
        elif keepratio:
            if fx is not None and fy is None:
                fy = fx
            elif fy is not None and fx is None:
                fx = fy
        else:
            if fx is not None and fy is None:
                fy = 1
            elif fy is not None and fx is None:
                fx = 1
        interpolation = interpolation or 0

        calc_func = _calc_dsize_keepratio if keepratio else _calc_dsize_nokeepratio

        if dsize is not None and None in dsize:
            def resizefunc(path):
                img = loadfunc(path)
                return cv2.resize(img, calc_func(img.shape, dsize), dst, fx, fy, interpolation)
        else:
            def resizefunc(path):
                return cv2.resize(loadfunc(path), dsize, dst, fx, fy, interpolation)
        func = resizefunc
    else:
        func = loadfunc

    if mmode or mpath is not None:
        def mapfunc(path):
            if mpath == "*":
                drive, dirname, (filename, ext) = "", "", getData(path)[-2:]
            elif mpath:
                drive, dirname, filename, ext = getData(changedir(path, mpath))
            else:
                drive, dirname, filename, ext = getData(path)

            hashed = hash("{}{}{}{}{}{}".format(
                flag, RGB, dsize, fx, fy, interpolation))
            savepath = "{}{}{}{}.{}.npy".format(
                drive, dirname, filename, ext, hashed)
            try:
                return np.lib.load(savepath, mmode)
            except IOError:
                im = func(path)
                if im is None:
                    raise Exception("Failed to load image to map")
                np.save(savepath, im)
                return np.lib.load(savepath, mmode)
        func = mapfunc

    def loader(path):
        try:
            return func(path)
        except Exception:
            if throw:
                raise
    return loader


class ImLoader(object):
    """
    Class to load image array from path, url,
    server, string or directly from numpy array (supports databases).
    """

    def __init__(self, path, flag=0, dsize=None, dst=None, fx=None, fy=None,
                 interpolation=None, mmode=None, mpath=None, throw=True):
        self.path = path
        self._flag = flag
        self._dsize = dsize
        self._dst = dst
        self._fx = fx
        self._fy = fy
        self._interpolation = interpolation
        self._mmode = mmode
        self._mpath = mpath
        self._throw = throw
        self.load = loadFunc(flag, dsize, dst, fx, fy,
                             interpolation, mmode, mpath, throw)

    def __call__(self):
        return self.load(self.path)

    def getConfiguration(self, **kwargs):
        """
        Get Custom configuration from default configuration.
        """
        temp = {"flag": self._flag, "dsize": self._dsize, "dst": self._dst, "fx": self._fx, "fy": self._fy,
                "interpolation": self._interpolation, "mmode": self._mmode, "mpath": self._mpath, "throw": self._throw}
        if kwargs:
            temp.update(kwargs)
        return temp

    def temp(self, **kwargs):
        """
        loads from temporal loader created with customized and default parameters.
        """
        if len(kwargs) == 1 and "path" in kwargs:
            return self.load(kwargs["path"])
        path = kwargs.get("path", self.path)
        return loadFunc(**self.getConfiguration(**kwargs))(path)


class PathLoader(MutableSequence):
    """
    Class to standardize loading images from list of paths and offer lazy evaluations.
    """

    def __init__(self, fns=None, loader=None):
        self._fns = fns or []
        self._loader = loader or loadFunc()

    def __call__(self):
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
        self._fns.insert(index, value)


class LoaderDict(ResourceManager):
    """
    Class to standardize loading objects and manage memory efficiently.
    """

    def __init__(self, loader=None, maxMemory=None, margin=0.8,
                 unit="MB", all=True, config=None):
        super(LoaderDict, self).__init__(maxMemory, margin, unit, all)
        self._default_loader = loader or loadFunc()

    def register(self, key, path=None, method=None):
        if method is not None:
            def func():
                return method(func.path)
        else:
            def func():
                return self._default_loader(func.path)
        func.path = path
        super(LoaderDict, self).register(key=key, method=func)


def try_loads(fns, func=cv2.imread, paths=None, debug=False, addpath=False):
    """
    Try to load images from paths.
    """
    default = ("", str(MANAGER["TESTPATH"]))
    if paths is None:
        paths = []
    if isinstance(paths, (str, bytes)):
        paths = [paths]
    paths = list(paths)
    for i in default:
        if i not in paths:
            paths.append(i)

    for fn in fns:
        for path in paths:
            try:
                if path[-1] not in ("/", "\\"):
                    path += "/"
            except Exception:
                pass
            path += fn
            im = func(path)
            if im is not None:
                if debug:
                    print(path, " Loaded...")
                if addpath:
                    return im, path
                return im

