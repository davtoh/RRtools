# -*- coding: utf-8 -*-
"""
Core utility methods, system primitives, decorators, locking, and pattern filtering.
Re-exports profiling, logging, and unit conversion components for backward compatibility.
"""
import sys
import os
import errno
import io
import inspect
import types
from fnmatch import fnmatch, fnmatchcase
from collections.abc import Iterable
from string import Formatter
import multiprocessing

try:
    import portalocker
except ImportError:
    portalocker = None

# Re-exported submodules for 100% backward compatibility
from .unit_conversion import FactorConvert, Magnitude
from .logging_util import StdoutSIM, StdoutMULTI, StdoutLOG, Controlstdout, stdout
from .profiling import TimeCode, Profiler

formater = Formatter()

__all__ = [
    # Re-exported from unit_conversion
    "FactorConvert",
    "Magnitude",
    # Re-exported from logging_util
    "StdoutSIM",
    "StdoutMULTI",
    "StdoutLOG",
    "Controlstdout",
    "stdout",
    # Re-exported from profiling
    "TimeCode",
    "Profiler",
    # Core primitives & file locking
    "NamedLock",
    "secure_open",
    "load_module",
    # Decorators & inspection
    "decorateInstanceMethods",
    "addto",
    # String formatting
    "formatOnly",
    "formatConsume",
    # Globbing & search
    "glob",
    "ensureList",
    "globFilter",
    "lookinglob",
    # Exceptions
    "TimeOutException",
    "TransferExeption",
    "VariableNotSettable",
    "VariableNotDeletable",
    "VariableNotGettable",
    "VariableNotAvailable",
    "NotConvertibleToInt",
    "ClassNotAllowed",
    "NotCreatable",
    "NotCallable",
    "NoParserFound",
    "CorruptPersistent",
    # Globals & environment
    "NameSpace",
    "NO_CPUs",
]


class NamedLock:
    """File-descriptor based named lock using portalocker."""

    def __init__(self, name):
        self.name = name
        self.descriptor = os.open(name, os.O_CREAT)
        self.file = os.fdopen(self.descriptor, 'w')

    def acquire(self):
        if portalocker is not None:
            portalocker.lock(self.file, portalocker.LOCK_EX)

    def release(self):
        if portalocker is not None:
            portalocker.unlock(self.file)

    def close(self):
        os.close(self.descriptor)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        self.close()

    def __del__(self):
        try:
            self.release()
            self.close()
        except Exception:
            pass


class secure_open:
    """Compatible text file reading context."""

    def __init__(self, file, mode='r', buffering=-1, encoding=None,
                 errors=None, newline=None, closefd=True):
        self.file = io.open(file, mode=mode, buffering=buffering,
                            encoding=encoding, errors=errors,
                            newline=newline, closefd=closefd)

    def __enter__(self):
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


def load_module(name, code=None, name_path=""):
    """Dynamically load or write-and-load a module."""
    try:
        import importlib
    except ImportError:
        import imp as importlib

    if code is not None:
        try:
            fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            if portalocker is not None:
                portalocker.lock(fd, portalocker.LOCK_EX)
            with os.fdopen(fd, 'w') as f:
                f.write(code)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    with open(name, "r") as f:
        if portalocker is not None:
            portalocker.lock(f, portalocker.LOCK_EX)

    return importlib.import_module(name, name_path)


def decorateInstanceMethods(self, decorator, excludeMth=("__init__",), includeMth=None):
    """
    Decorate methods in an instance at initialization.

    :param self: class instance.
    :param decorator: decorator function to apply.
    :param excludeMth: list/tuple of methods to exclude.
    :param includeMth: list/tuple of methods to include if not in exclude.
    :return: self
    """
    classmethods = dict(inspect.getmembers(self, predicate=inspect.ismethod))
    for key, method in classmethods.items():
        if excludeMth and key in excludeMth:
            continue
        if includeMth is None or key in includeMth:
            setattr(self, key, decorator(method))
    return self


def addto(instance, funcname=None):
    """
    Decorator: Add function as bound method to instance.

    :param instance: class instance.
    :param funcname: optional name to register in instance.
    """
    def decorator(fn):
        bound_fn = types.MethodType(fn, instance)
        target_name = funcname or fn.__name__
        setattr(instance, target_name, bound_fn)
        return bound_fn
    return decorator


def formatOnly(format_string, **kwargs):
    """
    Format string only with provided keys, keeping unprovided keys intact.

    :param format_string: string to format
    :param kwargs: format keys
    :return: formatted string
    """
    keys = [i[1] for i in formater.parse(format_string) if i[1] is not None]
    for key in keys:
        if key not in kwargs:
            kwargs[key] = "{{{}}}".format(key)
    return format_string.format(**kwargs)


def formatConsume(format_string, kwargs, formatter=None, handle=None):
    """
    Format with dictionary and consume keys from dictionary.

    :param format_string: string to format
    :param kwargs: dictionary containing format keys and values
    :param formatter: optional custom formatter function
    :param handle: optional custom key removal handler
    :return: formatted string
    """
    keys_in_str = [i[1] for i in formater.parse(format_string) if i[1] is not None]
    if formatter is None:
        formatted_string = format_string.format(**kwargs)
    else:
        formatted_string = formatter(format_string, **kwargs)
    for k in keys_in_str:
        if handle is not None:
            handle(kwargs, k)
        else:
            kwargs.pop(k, None)
    return formatted_string


def glob(path, contents="*", check=os.path.isfile):
    """
    Return a list of paths matching a pathname pattern with valid files.

    :param path: path to process in glob filter
    :param contents: If path is a folder then looks for contents
    :param check: predicate to filter matching paths
    :return: list of matching files
    """
    from glob import glob as pyglob
    fns = pyglob(path)
    if len(fns) == 1 and not os.path.isfile(fns[0]):
        fns = pyglob(os.path.join(fns[0], contents))
    return list(filter(check, fns))


def ensureList(obj):
    """Ensures that object is wrapped in a list if not already iterable."""
    if isinstance(obj, list):
        return obj
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return list(obj)
    return [obj]


class globFilter:
    """Glob filter for patterns with case and exclusion support."""

    def __init__(self, include=None, exclude=None, case=False):
        self.include = ensureList(include) if include is not None else []
        self.exclude = ensureList(exclude) if exclude is not None else []
        self.cmpfunc = fnmatchcase if case else fnmatch

    def __call__(self, cmp=None):
        if not isinstance(cmp, (str, bytes)) and isinstance(cmp, Iterable):
            return [self(i) for i in cmp]
        cmpfunc = self.cmpfunc
        for pattern in self.exclude:
            if cmpfunc(cmp, pattern):
                return False
        if self.include:
            for pattern in self.include:
                if cmpfunc(cmp, pattern):
                    return True
            return False
        return True


def _build_lookinglob_patterns(pattern, path, ext, forward):
    """Helper to generate glob test patterns for lookinglob."""
    if path is None and ext is None and forward is None:
        return {pattern}
    effective_path = path or ""
    tests = [pattern, "*{pattern}", "{path}{pattern}", "{path}*{pattern}"]
    if forward:
        tests.extend(["{pattern}*", "*{pattern}*", "{path}{pattern}*", "{path}*{pattern}*"])
    if ext is not None and not pattern.endswith(ext):
        dot_ext = ext if ext.startswith(".") else "." + ext
        tests = [i + dot_ext for i in tests]
    return {test.format(path=effective_path, pattern=pattern) for test in tests}


def _execute_lookinglob(tests, filelist, aslist):
    """Helper to query matching files across test patterns."""
    if filelist is not None:
        return list(filter(globFilter(case=True, include=list(tests)), filelist))
    ress = []
    for test in tests:
        res = glob(test)
        if not res:
            continue
        if not aslist and len(res) == 1:
            return res[0]
        ress.extend(res)
    return ress


def lookinglob(pattern, path=None, ext=None, forward=None,
               filelist=None, aslist=False, raiseErr=False):
    """
    Look for patterns in path or simulated filelist.

    :param pattern: string to look for pattern.
    :param path: directory path to look for pattern
    :param ext: extension of pattern
    :param forward: look changes after pattern and before ext parameter
    :param filelist: simulated list of files
    :param aslist: return match as list
    :param raiseErr: raise exception if pattern not found or ambiguous
    :return: matched path string or list of matched paths
    """
    tests = _build_lookinglob_patterns(pattern, path, ext, forward)
    ress = _execute_lookinglob(tests, filelist, aslist)

    if isinstance(ress, str):
        return [ress] if aslist else ress
    if len(ress) == 1:
        return ress if aslist else ress[0]
    if ress and raiseErr:
        raise Exception("More than one file with pattern '{}'".format(pattern))
    if not ress and raiseErr:
        err_path = path if path is not None else ""
        raise Exception("pattern '{}' not found in {}".format(pattern, err_path))
    if aslist:
        return list(set(ress))
    return None


# ---------------------------- EXCEPTIONS ---------------------------- #

class TimeOutException(Exception):
    """Raise when a process surpasses the timeout."""


class TransferExeption(Exception):
    """Raise when transferred data is corrupt."""


class VariableNotSettable(Exception):
    """Exception for property not settable."""


class VariableNotDeletable(Exception):
    """Exception for property not deletable."""


class VariableNotGettable(Exception):
    """Exception for property not gettable."""


class VariableNotAvailable(Exception):
    """Exception for variable that is not available."""


class NotConvertibleToInt(ValueError):
    """Exception to denote that value cannot be represented as int."""


class ClassNotAllowed(Exception):
    """Exception to denote that given class is not allowed."""


class NotCreatable(Exception):
    """ObjectGetter error: cannot create new object."""


class NotCallable(Exception):
    """ObjectGetter error: given object is not callable."""


class NoParserFound(Exception):
    """Raise when no parser is found to interpret user input."""


class CorruptPersistent(EOFError, OSError):
    """Denotes persistent data read from disk has been corrupted."""


# ---------------------------- GLOBAL VARIABLES ---------------------------- #

class NameSpace:
    """Used to store arbitrary namespace variables."""


NO_CPUs = multiprocessing.cpu_count()
