# -*- coding: utf-8 -*-
r"""
High-level Image wrapper providing lazy multi-channel color access and lifecycle management.
"""
from __future__ import division
import cv2
from ..directory import getData, mkPath, getPath, increment_if_exits
from .loaders import loadFunc, try_loads


class Image(object):
    """
    Structure to load and save images.
    """

    def __init__(self, name=None, ext=None, path=None, shape=None, verbosity=False):
        self._loader = loadFunc(-1, dsize=None, throw=False)
        self._shape = None
        self.shape = shape  # it is the inverted of dsize
        self.ext = ext
        self.name = name
        self.path = path
        self._RGB = None
        self._RGBA = None
        self._gray = None
        self._BGRA = None
        self._BGR = None
        self.overwrite = False
        self.verbosity = verbosity
        self.log_saved = None
        self.log_loaded = None
        self.last_loaded = None

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if value != self._shape:
            if value is not None:
                # invert, for value is shape and we need dsize
                value = value[1], value[0]
            self._loader = loadFunc(-1, dsize=value, throw=False)
        self._shape = value

    @shape.deleter
    def shape(self):
        del self._shape

    @property
    def ext(self):
        if self._ext is None:
            return ""
        return self._ext

    @ext.setter
    def ext(self, value):
        try:
            if not value.startswith("."):  # ensures path
                value = "." + value
        except Exception:
            pass
        self._ext = value

    @ext.deleter
    def ext(self):
        del self._ext

    @property
    def path(self):
        if self._path is None:
            return ""
        return self._path

    @path.setter
    def path(self, value):
        try:
            if value[-1] not in ("/", "\\"):  # ensures path
                value += "/"
        except Exception:
            pass
        self._path = value

    @path.deleter
    def path(self):
        del self._path

    @property
    def BGRA(self):
        if self._BGRA is None:
            self.load()
        return self._BGRA

    @BGRA.setter
    def BGRA(self, value):
        self._BGRA = value

    @BGRA.deleter
    def BGRA(self):
        self._BGRA = None

    @property
    def BGR(self):
        if self._BGR is None:
            self.load()
        return self._BGR

    @BGR.setter
    def BGR(self, value):
        self._BGR = value

    @BGR.deleter
    def BGR(self):
        self._BGR = None

    @property
    def RGB(self):
        if self._RGB is None:
            self._RGB = cv2.cvtColor(self.BGR, cv2.COLOR_BGR2RGB)
        return self._RGB

    @RGB.setter
    def RGB(self, value):
        self._RGB = value

    @RGB.deleter
    def RGB(self):
        self._RGB = None

    @property
    def RGBA(self):
        if self._RGBA is None:
            self._RGBA = cv2.cvtColor(self.BGRA, cv2.COLOR_BGRA2RGBA)
        return self._RGBA

    @RGBA.setter
    def RGBA(self, value):
        self._RGBA = value

    @RGBA.deleter
    def RGBA(self):
        self._RGBA = None

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.BGR, cv2.COLOR_BGR2GRAY)
        return self._gray

    @gray.setter
    def gray(self, value):
        self._gray = value

    @gray.deleter
    def gray(self):
        self._gray = None

    def save(self, name=None, image=None, overwrite=None):
        """
        save restored image in path.

        :param name: filename, string to format or path to save image.
                if path is not a string it would be replaced with the string
                "{path}restored_{name}{ext}" to format with the formatting
                "{path}", "{name}" and "{ext}" from the baseImage variable.
        :param image: (self.BGRA)
        :param overwrite: If True and the destine filename for saving already
            exists then it is replaced, else a new filename is generated
            with an index "{filename}_{index}.{extension}"
        :return: saved path, status (True for success and False for fail)
        """
        if name is None:
            name = self.name
        if name is None:
            raise Exception("name parameter needed")

        if image is None:
            image = self.BGRA

        if overwrite is None:
            overwrite = self.overwrite

        bbase, bpath, bname = getData(self.path)
        bext = self.ext
        # format path if user has specified so
        data = getData(name.format(path="".join((bbase, bpath)),
                                   name=bname, ext=bext))
        # complete any data lacking in path
        for i, (n, b) in enumerate(zip(data, (bbase, bpath, bname, bext))):
            if not n:
                data[i] = b
        # joint parts to get string
        fn = "".join(data)
        mkPath(getPath(fn))

        if not overwrite:
            fn = increment_if_exits(fn)

        if cv2.imwrite(fn, image):
            if self.verbosity:
                print("Saved: {}".format(fn))
            if self.log_saved is not None:
                self.log_saved.append(fn)
            return fn, True
        else:
            if self.verbosity:
                print("{} could not be saved".format(fn))
            return fn, False

    def load(self, name=None, path=None, shape=None):
        if name is None:
            name = self.name
        if path is None:
            path = self.path
        if path is None:
            path = ""
        if shape is not None:
            self.shape = shape

        data = try_loads([name, name + self.ext], paths=path,
                         func=self._loader, addpath=True)
        if data is None:
            raise Exception("Image not Loaded")

        img, last_loaded = data
        if self.log_loaded is not None:
            self.log_loaded.append(last_loaded)
        if self.verbosity:
            print("loaded: {}".format(last_loaded))
        self.last_loaded = last_loaded

        self._RGB = None
        self._RGBA = None
        self._gray = None

        if img.shape[2] == 3:
            self.BGR = img
            self.BGRA = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        else:
            self.BGRA = img
            self.BGR = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return self

