# -*- coding: utf-8 -*-
"""
.. moduleauthor:: David Toro <davsamirtor@gmail.com>

:platform: Unix, Windows
:synopsis: Looking for a reference? look here!.

This module contains all config data to the package.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import object
from .directory import Directory as _directory
from .directory import FileDirectory as _FileDirectory
from .directory import correctPath as _correctPath
from .directory import resource_path as _resource_path
from .session import serializer
from .session import updateSession as _updateSession
from .session import readSession as _readSession
from .session import saveSession as _saveSession
import pkgutil
from numpy import float32, int32
import sys
if sys.version_info[0] < 3:  # Python 2?
    # using exec avoids a SyntaxError in Python 3
    exec("""def reraise(exc_type, exc_value, exc_traceback=None):
                raise exc_type, exc_value, exc_traceback""")
else:
    def reraise(exc_type, exc_value, exc_traceback=None):
        if exc_value is None:
            exc_value = exc_type()
        if exc_value.__traceback__ is not exc_traceback:
            raise exc_value.with_traceback(exc_traceback)
        raise exc_value


# ----------------------------GLOBAL VARIABLES---------------------------- #
# FLAGS
FLAG_MEMOIZE = True
FLAG_DEBUG = False
# global variables
FLOAT = float32
INT = int32
PY3 = sys.version_info[0] == 3

# ----------------------------ConfigTool FUNCTIONS---------------------------- #


def getPackagePath(package):
    """
    Get the path of a package object.

    :param package: package object or path (str).
    :return: path to the package.
    """
    if isinstance(package, str):  # if path
        return package
    else:  # if imported package
        return package.__path__[0]


def findModules(package, exclude=None):
    """
    Find modules from a package.

    :param package: imported packaged or path (str).
    :param exclude: list of modules to exclude.
    :return: dictionary containing importer, ispkg
    """
    path = getPackagePath(package)
    modules = {}
    for importer, modname, ispkg in pkgutil.walk_packages([path]):
        if exclude and modname in exclude:
            continue
        modules[modname] = importer, ispkg
        if FLAG_DEBUG:
            print("Found submodule {0} (is a package: {1})".format(
                modname, ispkg))
    if FLAG_DEBUG and modules == {}:
        print("No modules have been found")
    return modules


def getModules(package, exclude=None):
    """
    Import modules from a package.

    :param package: imported packaged or path (str).
    :return: dictionary containing imported modules.
    """
    path = getPackagePath(package)
    modules = {}
    for importer, modname, ispkg in pkgutil.walk_packages([path]):
        if exclude and modname in exclude:
            continue
        modules[modname] = importer.find_module(modname).load_module(modname)
        if FLAG_DEBUG:
            print("Imported submodule {0}...".format(modname, ispkg))
    return modules

# TODO> make a function to update fields in the config file
# TODO> make a function just to update the MAINPATH relative to the config
# file for the other direcoties to update too

# ----------------------------CLASSES---------------------------- #


class DirectoryManager(object):
    """
    Manage the configured variables, paths and files.

    :param path: (None) path to configuration file. If None uses default path.
    :param raiseError: True to raise when not attribute in ConfigFile.
    :param autosave: (True) if True saves at each change.

    .. note:: Any attribute that is not in ConfigFile returns None.
              Use raiseError to control this behaviour.
    """

    def __init__(self, path=None, raiseError=True, autosave=False):
        """
        Initializes ConfigFile.
        """
        if path is None:
            path = str(_directory('ConfigFile.pkl') - self.default["MAINPATH"])

        self._path = str(path)
        self._raiseError = raiseError  # raise error if not directory found
        self.autosave = autosave  # flag to save at each change
        self._loaded = {}  # directories
        self.load()

    @property
    def default(self):
        """
        get directories from dictionary representing environment variables.

        :return: dictionary of directories.

        .. note:: Only directories in the scope of the module are detected.
        """

        # PATHS
        # where package is located, if package is installed this can be converted
        # to a virtual folder which does not exists physically and cannot be found
        # in disk
        SOURCEPATH = _directory(_correctPath(
            __file__, -1), notes="don't change!")
        # lib path is inside package
        LIBPATH = _directory("lib") - SOURCEPATH
        # where additional folders will be located, what holds SOURCEPATH
        MAINPATH = _directory(_resource_path(), notes="don't change!")
        # add more tools to this folder, iniside package is a tools folder too
        TOOLPATH = _directory("tools") - MAINPATH
        # place temporal files in this folder
        TEMPPATH = _directory("temp") - MAINPATH
        # tests should be found in tests folder
        TESTPATH = _directory("tests") - MAINPATH
        # save path is where results are placed
        SAVEPATH = TESTPATH.copy()
        # return {key:val for key,val in locals().iteritems() if
        # isinstance(val, _directory)}
        return {key: val for key, val in locals().items() if key != "self"}

    def reset(self):
        """
        Returns the configuration file to default variables.

        :return: False, if error. Dictionary of new data, if successful.

        .. warning:: All custom data is lost in configuration file.

        .. warning:: ConfigFile is purposely not updated. Call manually method load()
        """
        if FLAG_DEBUG:
            print("Creating default '{}' file...".format(self._path))
        try:
            data = _saveSession(str(self._path), self.default)
            if FLAG_DEBUG:
                print("'{}' has been reset successfully...".format(self._path))
            return data
        except Exception as e:
            if self._raiseError:
                raise e
            return False

    def load(self):
        """
        loads the configuration file and update.

        :return: loaded configuration file dictionary.

        .. warning:: Unsaved instance variables will be replaced by configuration file variables.
        """
        if FLAG_DEBUG:
            print("Loading '{}'...".format(self._path))
        error = False
        try:
            vars = _readSession(str(self._path))
        except IOError:
            if FLAG_DEBUG:
                print("'{}' not found...".format(self._path))
            error = True
        except AttributeError:
            if FLAG_DEBUG:
                print("'{}' structure is old...".format(self._path))
            error = True

        if error:
            # not tolerable as error and should not be caught
            data = self.reset()
            try:
                vars = _readSession(str(self._path))
            except Exception as e:
                # http://stackoverflow.com/questions/6062576/adding-information-to-an-exception
                reraise(type(e), type(e)(str(e) + "Trying to read '{}' with writted "
                                         "session as '{}'".format(self._path, data)), sys.exc_info()[2])

        if FLAG_DEBUG:
            print("Default '{}' read successfully...".format(self._path))
        self._loaded.update(vars)
        return vars

    def save(self, mode=0):
        """
        saves configuration file.

        :param mode: 0- delete and save, 1- update without replace,
                2- update replacing variables.
        :return: False, if error. Dictionary of new data, if successful.
        """
        try:
            if mode:
                _updateSession(str(self._path), self._loaded, replace=mode - 1)
                return True
            else:  # to delete and replace
                _saveSession(str(self._path), self._loaded)
                return True
        except IOError as e:
            if self._raiseError:
                raise e
            return False

    def _retrieve(self, item):
        """
        retrieves value from default or loaded
        :param item:
        :return:
        """
        try:
            return self._loaded[item]
        except:
            return self.default[item]

    def __getitem__(self, item):
        try:
            return self._retrieve(item)
        except Exception as e:
            if self._raiseError:
                raise e
            return None

    def __setitem__(self, key, value):
        self._loaded[key] = value
        if self.autosave:
            self.save()

    def __delitem__(self, key):
        del self._loaded[key]
        if self.autosave:
            self.save()

    '''
    def __getattr__(self, item): # this is called as last resort
        """
        Allows to load self.object instead of self["object"]
        """
        return self[item]'''


MANAGER = DirectoryManager()  # configure directories


class ConfigTool(object):
    """
    Manage the configured Tools.
    """

    # CONTENTS
    _init_tool = _FileDirectory(["""

    import sys
    sys.path.append('""", MANAGER["LIBPATH"], """')
    sys.path.append('""", MANAGER["MAINPATH"], """')"""], notes="contents of __init__.py file in TOOLPATH directory", filename="__init__.py", path=MANAGER["TOOLPATH"])

    _init_box = _FileDirectory(["""

    import sys
    sys.path.append('""", MANAGER["LIBPATH"], """')
    sys.path.append('""", MANAGER["TOOLPATH"], """')"""], notes="contents of __init__.py file in MAINPATH directory", filename="__init__.py", path=MANAGER["MAINPATH"])

    @staticmethod
    def getTools(package):
        """
        Obtains the tools of a directory for the RRtoolbox.

        :param package: path to the directory or package object.
        :return: a dictionary of imported modules.
        """
        path = getPackagePath(package)
        # sys.path.insert(0,path)
        modname = "__init__"
        try:
            pkgutil.get_importer(path).find_module(
                modname).load_module(modname)
        except AttributeError as e:
            if FLAG_DEBUG:
                print("No " + modname + " file found at " + path)
            raise e
        except Exception as e:
            if FLAG_DEBUG:
                print(modname + " could not be loaded from " + path)
            raise e
        return getModules(path, exclude=[modname])


if __name__ == '__main__':
    #MANAGER._default = {}
    # print MANAGER.TEMPPATH
    MANAGER.reset()
    #tools = ConfigTool()
    # tools._init_tool.makeFile()
    # tools._init_box.makeFile()
