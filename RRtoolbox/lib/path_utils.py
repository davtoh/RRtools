# -*- coding: utf-8 -*-
"""
Path manipulation, normalization, filesystem inspection, and path arithmetic.
"""
import os
import sys
import shutil
from functools import wraps
from glob import glob
from urllib.request import urlopen
from urllib.error import URLError

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
]


def resource_path(relative_path=""):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = correctPath(__file__, -2)
    return os.path.join(base_path, relative_path)


def getFileHandle(path):
    """
    Gets a file handle from url or disk file.

    :param path: filepath or url
    :return: file object
    """
    try:
        return urlopen(path)
    except (ValueError, URLError):
        return open(path, 'rb')


def getFileSize(path):
    """
    Gets a size from url or disk file.

    :param path: filepath or url
    :return: size in bytes
    """
    try:
        return urlopen(path).info()["Content-Length"]
    except (ValueError, URLError):
        return os.stat(path).st_size


def checkPath(path):
    """Checks if path exists."""
    return os.path.exists(path)


def checkDir(dirname):
    """Checks if dirname exists."""
    return os.path.isdir(dirname)


def checkURL(url):
    """Checks if url exists."""
    try:
        urlopen(url)
        return True
    except (ValueError, URLError):
        return False


def checkFile(path):
    """Checks if filepath or filename exists."""
    return os.path.isfile(path)


def getPath(path=__file__):
    """
    Get standard path from path. Supports ~ as home directory.

    :param path: folder or file path. Default is __file__.
    :return: dirname (path to a folder)
    """
    if path.startswith("~"):
        path = os.path.expanduser("~") + path[1:]
    if "." in path:
        return os.path.dirname(os.path.abspath(path))
    return os.path.abspath(path)


def getData(path=__file__):
    """
    Get standard path components from path.

    :param path: folder or file path. Default is __file__.
    :return: [drive, dirname, filename, ext].
    """
    if os.path.isdir(path):
        drive, dirname = os.path.splitdrive(path)
        return [drive, dirname, "", ""]
    dirname, ext = os.path.splitext(path)
    base, filename = os.path.split(dirname)
    drive, dirname = os.path.splitdrive(dirname[:len(dirname) - len(filename)])
    return [drive, dirname, filename, ext]


def increment_if_exits(path, add="_{num}", force=None):
    """
    Generates new filename/path if target exists.

    :param path: absolute path or filename
    :param add: pattern to insert if path exists
    :param force: force existent path even if non-existent on disk
    :return: non-existent path
    """
    path = os.path.abspath(path)
    listing = []
    if force is True:
        listing.append(path)
    elif force is not None:
        listing.extend(os.path.abspath(f) for f in force)

    if path in listing or os.path.exists(path):
        data = getData(path)
        pattern = "".join(data[:-1]) + "{}" + data[-1]
        listing.extend(glob(pattern.format("*")))
        num = 0
        while True:
            num += 1
            new_path = pattern.format(add.format(num=num))
            if new_path not in listing and not os.path.exists(new_path):
                return new_path
    return path


def _flush_diff(ls1, ls2, state, diff1, diff2):
    """Helper for strdifference to flush different character chunks."""
    if diff1 or diff2:
        ls1.append("".join(diff1))
        ls2.append("".join(diff2))
        state.append(len(ls1) - 1)
        return True
    return False


def _flush_equal(ls1, ls2, equal):
    """Helper for strdifference to flush equal character chunks."""
    if equal:
        streq = "".join(equal)
        ls1.append(streq)
        ls2.append(streq)
        return True
    return False


def strdifference(s1, s2):
    """
    Get string differences.

    :param s1: string 1
    :param s2: string 2
    :return: (split string 1, split string 2, index).
    """
    ls1, ls2, state, equal, diff1, diff2 = [], [], [], [], [], []
    i = -1
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            equal.append(s1[i])
            if _flush_diff(ls1, ls2, state, diff1, diff2):
                diff1, diff2 = [], []
        else:
            diff1.append(s1[i])
            diff2.append(s2[i])
            if _flush_equal(ls1, ls2, equal):
                equal = []
    _flush_diff(ls1, ls2, state, diff1, diff2)
    _flush_equal(ls1, ls2, equal)
    diff1, diff2 = s1[i + 1:], s2[i + 1:]
    _flush_diff(ls1, ls2, state, diff1, diff2)
    return ls1, ls2, state


def getSep(path, pattern='/\\'):
    """
    Get path separator or indicator.

    :param path: relative or absolute path.
    :param pattern: separator characters.
    :return: separator character.
    """
    i = len(path)
    while i and path[i - 1] not in pattern:
        i -= 1
    head = path[:i]
    head2 = head
    count = 0
    while head2 and head2[-1] in pattern:
        head2 = head2[:-1]
        count += 1
    return head[len(head) - count:len(head)]


def getSplitted(path=__file__):
    """Splits a file path by its separators."""
    return path.split(getSep(path) or None)


def correctSep(path=__file__, separator=os.path.sep):
    """Replaces path separators by custom or OS standard separator."""
    return separator.join(getSplitted(path))


def mkPath(path):
    """Make path directory for filepath."""
    path = os.path.abspath(str(path))
    try:
        os.makedirs(path)
    except OSError as exception:
        import errno
        if exception.errno == errno.EEXIST:
            if os.path.isfile(path):
                raise Exception(
                    "Folder not created. There is a file with the same name at {}".format(path))
        else:
            raise exception
    return path


def rmFile(filepath):
    """Remove file safely."""
    try:
        os.remove(filepath)
    except OSError:
        pass


def rmPath(path, ignore_errors=False, onerror=None):
    """Remove path and contents."""
    path = getPath(path)
    shutil.rmtree(path, ignore_errors, onerror)


def _find_quick_ops(toCorrect, toCompare):
    """Helper to detect operations for quickOps."""
    size = len(toCompare)
    test = ("..",)
    where = []
    current = 0
    for i, cp in enumerate(toCompare):
        if cp not in test:
            tfrom = i + 1 < size and toCompare[i + 1] in test
            tto = i > 0 and toCompare[i - 1] in test
            isin = cp in toCorrect[current:]
            if tfrom and tto:
                if isin:
                    where.append((i, cp, 'del'))
            elif tfrom:
                where.append((i, cp, 'from' if isin else 'lput'))
            elif tto:
                where.append((i, cp, 'to' if isin else 'rput'))
    return where


def quickOps(path, comp):
    """
    Make quick matching operations in path.

    :param path: path to folder
    :param comp: pattern
    """
    toCorrect = getSplitted(path)
    toCompare = getSplitted(comp)
    where = _find_quick_ops(toCorrect, toCompare)
    if where:
        for order, (i, cp, op) in enumerate(where):
            if cp in toCorrect:
                del toCorrect[toCorrect.index(cp)]
                idx = toCompare.index(cp)
                del toCompare[idx:idx + 2]


def getShortenedPath(path, comp):
    """
    Give absolute path from relative path or integer level.

    :param path: absolute path string.
    :param comp: pattern, relative path string, or integer folder level.
    :return: shortened path string.
    """
    indicator = getSep(path)
    if not isinstance(comp, str):
        return indicator.join(path.split(indicator)[:comp])
    toCorrect = getSplitted(path)
    toCompare = getSplitted(comp)
    isrel = False
    for i, cp in enumerate(toCompare):
        if cp == "..":
            if toCorrect:
                del toCorrect[-1]
            isrel = True
        elif cp == ".":
            pass
        elif isrel or cp not in toCorrect:
            toCorrect.extend(toCompare[i:])
            break
        else:
            if cp in toCorrect:
                toCorrect = toCorrect[:toCorrect.index(cp)] + toCompare[i:]
            else:
                toCorrect.extend(toCompare[i:])
            break
    return indicator.join(toCorrect)


def decoratePath(relative, sep=os.path.sep):
    """Decorated path is controlled to give absolute path from relative path."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            path = f(*args, **kwargs)
            return getShortenedPath(correctSep(path, sep), relative)
        return wrapper
    return decorator


def correctPath(path, relative):
    """Get path corrected from its relative path or level index."""
    return decoratePath(relative)(getPath)(path)


def joinPath(absolute, relative):
    """
    Joins an absolute path to a relative path.

    :param absolute: directory or path string.
    :param relative: directory or path string.
    :return: joined path string.
    """
    absolute = str(absolute)
    relative = str(relative)
    if relative.startswith("\\") or relative.startswith("/"):
        relative = relative[1:]
    return os.path.join(absolute, relative)

