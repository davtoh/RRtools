# -*- coding: utf-8 -*-
"""
.. moduleauthor:: David Toro <davsamirtor@gmail.com>

:platform: Unix, Windows
:synopsis: Function memoization, caching descriptors, and disk-backed memory decorators.

Provides function and method decorators to cache computation results in memory
or persist them to disk using joblib.

*@cache* is used as replacement of *@property* to compute a class method once.
It is computed only one time after which an attribute of the same name is generated in its place.

*@cachedProperty* is used as replacement of *@property* to compute
a class method depending on changes in its watched variables.

*@memoize* used as a general memoizer decorator for functions
where metadata is generated to disk for persistence.

Made by Davtoh, powered by joblib.
Dependent project: https://github.com/joblib/joblib
"""

# READ: https://wiki.python.org/moin/PythonDecoratorLibrary

__license__ = """joblib is BSD-licenced (3 clause):
    Copyright (c) 2009-2011, joblib developpers All rights reserved.
    OSI Certified Open Source Software (http://www.opensource.org/licenses/bsd-license.php)"""

from functools import wraps
from weakref import ref
import joblib

__all__ = [
    "NotMemorizedFunc",
    "MemorizedFunc",
    "DynamicMemoizedFunc",
    "Memory",
    "Memoizer",
    "memoize",
    "cache",
    "cachedProperty",
    "Memo",
]


class NotMemorizedFunc(joblib.memory.NotMemorizedFunc):
    pass


class MemorizedFunc(joblib.memory.MemorizedFunc):
    pass


class DynamicMemoizedFunc(object):
    def __init__(self, func, cachedir=None, ignore=None, mmap_mode=None,
                 compress=False, verbose=1, timestamp=None, banned=False):
        self._func = func  # the only one that should not be able to change
        self._mmap_mode = mmap_mode
        self._ignore = ignore
        self._verbose = verbose
        self._cachedir = cachedir
        self._compress = compress
        self._timestamp = timestamp
        self._enabled = banned
        self._use = None
        self._build()

    def _build(self):
        if self._cachedir is None or not self._enabled:
            self._use = NotMemorizedFunc(self._func)
        else:
            self._use = MemorizedFunc(func=self._func, cachedir=self._cachedir,
                                      ignore=self._ignore, mmap_mode=self._mmap_mode,
                                      compress=self._compress, verbose=self._verbose,
                                      timestamp=self._timestamp)
            self.__doc__ = self._use.__doc__

    def _set_and_rebuild(self, attr_name, value):
        if getattr(self, attr_name) != value:
            setattr(self, attr_name, value)
            self._build()

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, value):
        if value != self._func:
            if isinstance(value, (MemorizedFunc, NotMemorizedFunc, DynamicMemoizedFunc)):
                value = value.func
            self._func = value
            self._build()

    @property
    def mmap_mode(self):
        return self._mmap_mode
    @mmap_mode.setter
    def mmap_mode(self, value):
        self._set_and_rebuild("_mmap_mode", value)

    @property
    def ignore(self):
        return self._ignore
    @ignore.setter
    def ignore(self, value):
        self._set_and_rebuild("_ignore", value)

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, value):
        self._set_and_rebuild("_verbose", value)

    @property
    def cachedir(self):
        return self._cachedir
    @cachedir.setter
    def cachedir(self, value):
        self._set_and_rebuild("_cachedir", value)

    @property
    def compress(self):
        return self._compress
    @compress.setter
    def compress(self, value):
        self._set_and_rebuild("_compress", value)

    @property
    def enabled(self):
        return self._enabled
    @enabled.setter
    def enabled(self, value):
        self._set_and_rebuild("_enabled", value)

    def __call__(self, *args, **kwargs):
        return self._use(*args, **kwargs)

    def call_and_shelve(self, *args, **kwargs):
        return self._use.call_and_shelve(*args, **kwargs)

    def __reduce__(self):
        return self._use.__reduce__()

    def __repr__(self):
        return self._use.__repr__()

    def clear(self, warn=True):
        return self._use.clear(warn=warn)


class Memory(joblib.Memory):
    """
    A wrapper to joblib Memory to have better control.
    """

    def __init__(self, cachedir, mmap_mode=None, compress=False, verbose=1):
        super(Memory, self).__init__(None, mmap_mode, compress, verbose)
        if cachedir is None:
            self.cachedir = None
        else:
            self.cachedir = cachedir
            joblib.memory.mkdirp(self.cachedir)

    __call__ = joblib.Memory.cache


class Memoizer(object):
    memoizers = {}

    def __init__(self, ignore=(), ignoreAll=False):
        self._ignore = None
        self.ignoreAll = ignoreAll
        self.ignore = ignore
        # FOR CLEANING UP handle # flag = True is safe to remove, flag = False
        # is unsafe to remove
        self.memoized = {}
        Memoizer.memoizers[id(self)] = ref(self)

    @property
    def ignore(self):
        return self._ignore

    @ignore.setter
    def ignore(self, value):
        temp = set()
        for f in value:
            if not callable(f):
                raise Exception("{} must be callable".format(f))
            temp.add(id(f))
        self._ignore = temp

    @ignore.deleter
    def ignore(self):
        self._ignore = set()

    def makememory(self, cachedir=None, mmap_mode=None, compress=False, verbose=0):
        """
        Make memory for :func:`memoize` decorator.

        :param cachedir: path to save metadata, if left None function is not cached.
        :param mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional.
                    The memmapping mode used when loading from cache
                    numpy arrays. See numpy.load for the meaning of the
                    arguments.
        :param compress: (boolean or integer)
                    Whether to zip the stored data on disk. If an integer is
                    given, it should be between 1 and 9, and sets the amount
                    of compression. Note that compressed arrays cannot be
                    read by memmapping.
        :param verbose: (int, optional)
                    Verbosity flag, controls the debug messages that are issued
                    as functions are evaluated.
        :return:
        """
        return Memory(cachedir, mmap_mode, compress, verbose)

    def memoize(self, memory=None, ignore=None, verbose=0, mmap_mode=False):
        """
        Decorated functions are faster by trading memory for time, only hashable values can be memoized.

        :param memory: (Memory or path to folder) if left None function is not cached.
        :param ignore: (list of strings) A list of arguments name to ignore in the hashing.
        :param verbose: (integer) Verbosity flag, controls the debug messages that are issued as functions are evaluated.
        :param mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional. The memmapping mode used when loading from cache
                    numpy arrays. See numpy.load for the meaning of the arguments.
        :return: decorator
        """
        def decorator(fn):
            if isinstance(memory, joblib.Memory):  # use provided memory
                memoizedfn = memory.cache(fn, ignore, verbose, mmap_mode)
            else:
                memoizedfn = self.makememory(cachedir=str(memory)).cache(
                    func=fn, ignore=ignore, verbose=verbose, mmap_mode=mmap_mode)

            @wraps(fn)
            def wrapper(*args, **kwargs):
                if not self.ignoreAll and id(fn) not in self._ignore and id(wrapper) not in self._ignore:
                    return memoizedfn(*args, **kwargs)
                else:
                    return fn(*args, **kwargs)
            self.memoized[id(fn)] = ref(memoizedfn)  # safe to remove
            return wrapper
        return decorator

    __call__ = memoize


memoize = Memoizer()  # make memoizer manager


class cache(object):
    """
    Descriptor (non-data) for building an attribute on-demand at first use.
    @cache decorator is used for class methods without inputs (only self reference to the object)
    and it caches on first compute. ex::

        class x(object):
            @cache
            def method_x(self):
                return self.data

    .. note:: Cached data can be deleted in the decorated object to recalculate its value.
    """
    __slots__ = ('func',)

    def __init__(self, func):
        """
        Initialize cache with a property function.
        """
        self.func = func  # function handle

    def __get__(self, instance, owner):  # if trying to get attribute
        if instance is not None:  # Class not instantiated if instance is None
            # Only for instances of Class
            # Build the attribute.
            cached = self.func(instance)  # evaluate function over instance
            setattr(instance, self.func.__name__, cached)
            return cached
        else:  # return method signature
            return self.func


class Memo(object):
    """Memo function for cache."""

    def __init__(self):
        self._cache = {}
        self._input_cache = {}

    def clear(self):
        self._cache = {}
        self._input_cache = {}


def _make_cached_property_noargs(f, memo_holder):
    """Helper for cachedProperty without watched arguments."""
    @wraps(f)
    def get(self):
        try:
            return memo_holder._cache[f]
        except AttributeError:
            memo_holder._cache = {}
        except KeyError:
            pass
        ret = memo_holder._cache[f] = f(self)
        return ret
    return property(get)


def _make_cached_property_withargs(f, memo_holder, watch):
    """Helper for cachedProperty with watched arguments."""
    @wraps(f)
    def get(self):
        input_values = dict((key, getattr(self, key)) for key in watch)
        try:
            x = memo_holder._cache[f]
            if input_values == memo_holder._input_cache[f]:
                return x
        except AttributeError:
            memo_holder._cache = {}
            memo_holder._input_cache = {}
        except KeyError:
            pass
        x = memo_holder._cache[f] = f(self)
        memo_holder._input_cache[f] = input_values
        return x
    return property(get)


def cachedProperty(watch=[], handle=[]):
    """
    A memoize decorator of @property decorator specifying what to trigger caching.

    :param watch: (list of strings) A list of arguments name to watch in the hashing.
    :param handle: (list of handles or empty list) Provided list is appended with the Memo
                handle were data is stored for the method and where a clear() function is provided.
    :return:
    """
    this = Memo()
    handle.append(this)

    if not isinstance(watch, list):  # if not arguments, decorating directly
        return _make_cached_property_noargs(watch, this)
    elif len(watch) == 0:
        return lambda f: _make_cached_property_noargs(f, this)
    return lambda f: _make_cached_property_withargs(f, this, watch)
