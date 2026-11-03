# -*- coding: utf-8 -*-
"""
Directory representation objects and referenced path management.
Designed to support :mod:`config` and be used with :mod:`session`.
"""
import os
from functools import wraps

from .path_utils import (
    resource_path,
    getFileHandle,
    getFileSize,
    checkPath,
    checkDir,
    checkURL,
    checkFile,
    getPath,
    getData,
    increment_if_exits,
    strdifference,
    getSplitted,
    correctSep,
    mkPath,
    rmFile,
    rmPath,
    getSep,
    quickOps,
    getShortenedPath,
    decoratePath,
    correctPath,
    joinPath,
)

__all__ = [
    "resource_path",
    "getFileHandle",
    "getFileSize",
    "checkPath",
    "checkDir",
    "checkURL",
    "checkFile",
    "getPath",
    "getData",
    "increment_if_exits",
    "changedir",
    "strdifference",
    "getSplitted",
    "correctSep",
    "mkPath",
    "rmFile",
    "rmPath",
    "getSep",
    "quickOps",
    "getShortenedPath",
    "decoratePath",
    "correctPath",
    "joinPath",
    "Directory",
    "FileDirectory",
]


def changedir(filepath, dirname, ext=True):
    """
    Change path to file with dirname.

    :param filepath: path to file.
    :param dirname: new path to replace in filepath.
    :param ext: True to keep extension of file if any.
    :return: directory object of changed path.
    """
    if ext:
        data = getData(filepath)[2:]
    else:
        data = getData(filepath)[2:3]
    return Directory(dirname) + Directory(data, False)


class Directory(str):
    """
    Semi-mutable string representation of an immutable string with support for path representations.

    :param data: list, directory instance, dictionary or string.
    :param ispath: True to add support for paths.
    :param copy: when data is a directory if copy is True then this instance data is independent.
    :param kwargs: additional data to add in directory.
    """

    def __new__(cls, data, ispath=None, copy=False, **kwargs):
        data = Directory.filterdata(data, ispath, kwargs)
        string = Directory.repr2str(data["repr"], data["ispath"])
        self = super(Directory, cls).__new__(cls, string)
        if copy:
            for k, v in data.items():
                setattr(self, k, v)
            setattr(self, "repr", string)
        else:
            setattr(self, "__dict__", data)
        self.__name__ = "directory"
        return self

    @staticmethod
    def filterdata(data, ispath=None, kwargs=None):
        """
        Adequate data for dictionary creation.

        :param data: any supported object.
        :param ispath: True to add support for paths.
        :param kwargs: additional data to add in directory.
        :return: dictionary
        """
        if isinstance(data, Directory):
            data = getattr(data, "__dict__")
        elif not isinstance(data, dict):
            if isinstance(data, list):
                data = {"repr": data}
            else:
                data = {"repr": [data]}
        if kwargs:
            data.update(kwargs)
        if ispath is not None:
            data["ispath"] = ispath
        if "ispath" not in data:
            data["ispath"] = True
        if "repr" in data:
            data["repr"] = Directory.repr2list(data["repr"])
        else:
            data["repr"] = [""]
        return data

    @staticmethod
    def repr2list(data, level=0):
        """
        Converts the representation of a directory.repr to pickleable.

        :param data: directory.repr of the form ["string", directory, ..., directory.repr].
        :return: pickleable list.
        """
        if isinstance(data, list):
            if level == 0:
                for i, value in enumerate(data):
                    data[i] = Directory.repr2list(value, level + 1)
                return data
            return Directory(data)
        elif isinstance(data, str):
            return data
        return Directory(data)

    @staticmethod
    def repr2str(data, ispath=True):
        """
        Converts the representation of a directory.repr to string.

        :param data: directory.repr of the form ["string", directory, ..., directory.repr].
        :return: converted string.
        """
        if isinstance(data, list):
            if len(data) > 1:
                string = str(Directory.repr2str(data[0]))
                for i in data[1:]:
                    if ispath:
                        string = joinPath(string, Directory.repr2str(i))
                    else:
                        string += Directory.repr2str(i, ispath)
                return string
            elif data:
                return str(data[0])
            return ""
        elif isinstance(data, Directory):
            return Directory.repr2str(data.repr, data.ispath)
        return str(data)

    def correctSTRBuiltin(self):
        """Decorate all built-in functions of class directory."""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                return getattr(str(self), f.__name__, f.__name__)(*args, **kwargs)
            return wrapper
        return decorator

    def update_right(self, other):
        """Updates representation at the right."""
        self.repr.append(other)
        return Directory(self)

    def update_left(self, other):
        """Updates representation at the left."""
        self.repr.insert(0, other)
        return Directory(self)

    def update(self, data=None):
        """Return an updated copy with provided data."""
        if isinstance(data, list):
            i = 0
            if len(data) > len(self.repr):
                for i in range(len(self.repr)):
                    self.repr[i] = data[i]
                for j in range(i + 1, len(data)):
                    self.repr.append(data[j])
            else:
                for i in range(len(data)):
                    self.repr[i] = data[i]
                del self.repr[i + 1:]
            return Directory(self)
        elif isinstance(data, dict):
            for k, v in data.items():
                setattr(self, k, v)
            return Directory(self)
        elif data:
            return self.update([data])
        return Directory(self)

    def copy(self):
        """Creates copy of itself."""
        return Directory(self, copy=True)

    def __str__(self):
        return Directory.repr2str(self.repr, self.ispath)

    def __repr__(self):
        return str(self.repr)

    def __add__(self, other):
        if type(other) is str:
            return joinPath(self, other)
        return self.update_right(other)

    __iadd__ = __add__

    def __sub__(self, other):
        if type(other) is str:
            return joinPath(other, self)
        return self.update_left(other)


class FileDirectory(Directory):
    """
    Saves contents of a file as with directories.

    :param data: list, directory instance, dictionary or string.
    :param filename: name of file.
    :param path: path to folder where file is.
    :param notes: optional description string.
    """

    def __new__(cls, data, filename, path, notes=None):
        self = super(FileDirectory, cls).__new__(cls, data, False, False)
        self.filename = filename
        self.path = path
        self.notes = notes
        return self

    def makeFile(self):
        """Makes a file with its contents to path/filename."""
        path = mkPath(self.path)
        initfile = joinPath(path, self.filename)
        with open(initfile, 'wb') as logger:
            logger.write(str(self).encode('utf-8'))
            return True
