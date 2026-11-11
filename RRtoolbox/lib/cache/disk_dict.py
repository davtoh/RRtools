# -*- coding: utf-8 -*-
"""
.. moduleauthor:: David Toro <davsamirtor@gmail.com>

:platform: Unix, Windows
:synopsis: Disk-persisted dictionary and memory-mapped object persistence.

Provides file-backed key-value mapping (MemoizedDict) and disk-mapping (mapper)
for caching structures and large arrays across sessions.
"""

from collections.abc import MutableMapping
import os
import pickle
import joblib
from numpy import load as numpyLoad, save as numpySave

try:
    from ..root import (VariableNotSettable, VariableNotDeletable,
                       CorruptPersistent, secure_open)
    from ..directory import mkPath
except (ImportError, ValueError):
    from RRtoolbox.lib.root import (VariableNotSettable, VariableNotDeletable,
                                   CorruptPersistent, secure_open)
    from RRtoolbox.lib.directory import mkPath

__all__ = ["mapper", "MemoizedDict"]


def mapper(path, obj=None, mode=None, onlynumpy=False):
    """
    Save and load or map live objects to disk to free RAM memory.

    :param path: path to save mapped file.
    :param obj: the object to map, if None it tries to load obj from path if exist
    :param mode: {None, 'r+', 'r', 'w+', 'c'}.
    :param onlynumpy: if True, it saves a numpy mapper from obj.
    :return: mmap image, names of mmap files
    """
    names = None
    if onlynumpy:
        if not path.endswith(".npy"):
            path += ".npy"
        if obj is not None:
            numpySave(path, obj)
            names = [path]
        return numpyLoad(path, mode), names
    else:
        if obj is not None:
            names = joblib.dump(obj, path)
        return joblib.load(path, mmap_mode=mode), names


class MemoizedDict(MutableMapping):
    """
    Memoized dictionary with keys and values persisted to files.

    :param path: path to save memo file
    :param mode: loading mode from memo file {None, 'r+', 'r', 'w+', 'c'}

    .. note::

        If saveAtKey is True it will attempt to memoize each time a keyword is added
        and throw an error if not successful. But if saveAtKey is False this process
        will be carried out when the MemoizedDict instance is being destroyed in
        a proper deletion, that is, if the program ends unexpectedly all data will be
        lost or if data cannot be saved it will be lost without warning.

    .. warning::

        Some data structures cannot be memoized, so this structure is not safe yet.
        Use at your own risk.
    """

    def __init__(self, path, mode=None):
        self._map_serializer = pickle
        self._path = mkPath(path)
        self._map_file = os.path.join(self._path, "metadata")
        self._map_old = None
        self._mode = mode
        self._loader = joblib.load
        self._saver = joblib.dump
        self._hasher = hash
        self._secure = False

    @property
    def _map(self):
        self._map_old = self._load_map()
        if self._map_old is None:
            self._map_old = {}
        return self._map_old

    @_map.setter
    def _map(self, value):
        raise VariableNotSettable("_map cannot be set")

    @_map.deleter
    def _map(self):
        raise VariableNotDeletable("_map cannot be deleted")

    def _getHash(self, key):
        """
        Hasher function to use in persistent files.

        :param key: key to hash
        """
        return '{}'.format(self._hasher(key))

    def _load(self, key):
        """
        Loads key from persistent file using map.

        :param key: hashable key.
        :return: value stored in key.
        """
        hashed, files = self._map[key]
        filename = os.path.join(self._path, hashed)
        if os.path.isfile(filename):
            try:
                return self._loader(filename, self._mode)
            except (EOFError, IOError):
                raise CorruptPersistent("Persistent data is corrupt")
        else:
            raise KeyError

    def _save(self, key, value):
        """
        saves key to persistent files.

        :param key: hashable key.
        :param value: serializable object.
        :return: None
        """
        try:
            hashed, files = self._map[key]
            for file in files:
                try:
                    os.remove(file)
                except OSError:
                    pass
            del self._map_old[key]
        except KeyError:
            hashed = self._getHash(key)

        filename = os.path.join(self._path, hashed)
        try:
            self._map_old[key] = (hashed, self._saver(value, filename))
            self._save_map()
        except OSError:
            print(" Race condition in the creation of the directory ")

    def _save_map(self):
        """
        persist dictionary map to file (metadata).
        """
        if self._map_old is not None:
            with secure_open(self._map_file, 'wb') as f:
                return self._map_serializer.dump(self._map_old, f)

    def _load_map(self):
        """
        loads metadata of dictionary map from persisted file.
        """
        try:
            with secure_open(self._map_file, 'rb') as f:
                return self._map_serializer.load(f)
        except (IOError, OSError):
            return None
        except EOFError as e:
            if os.path.exists(self._map_file) and os.stat(self._map_file).st_size == 0:
                return None
            raise CorruptPersistent("{}. Memoized data appears to be corrupt.".format(e)) from e

    def clear(self):
        """
        Remove all items from D.
        """
        for key in list(self._map.keys()):
            try:
                del self[key]
            except KeyError:
                pass

    def __setitem__(self, key, value):
        self._save(key, value)

    def __getitem__(self, key):
        return self._load(key)

    def __delitem__(self, key):
        hashed, files = self._map[key]
        for file in files:
            try:
                os.remove(file)
            except OSError:
                pass
        del self._map_old[key]
        self._save_map()

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __contains__(self, key):
        try:
            hashed, files = self._map[key]
            for file in files:
                if not os.path.exists(file):
                    raise KeyError
        except KeyError:
            return False
        else:
            return True

