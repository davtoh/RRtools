# -*- coding: utf-8 -*-
"""
    This module holds all path manipulation methods and a string concept called directory (referenced paths and strings)
    designed to support :mod:`config` and be used with :mod:`session`.

    keywords:
    ----------
    *path*: it can be to a folder or file or url if specified
    *filename*: the file name without its path
    *filepath*: the path to a file
    *dirname*: the path to a folder
    *url*: Universal Resource Locator
"""
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range


import os, sys
from functools import wraps
import shutil
from glob import glob
#from urlparse import urlparse
#import urllib
#import urllib2
#import urllib3
#import requests

try:
    from urllib.request import urlopen, URLError # urllib.urlopen disappears in python 3
except ImportError:
    from urllib.request import urlopen
    from urllib.error import URLError

def resource_path(relative_path=""):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    # based on http://stackoverflow.com/a/37920111/5288758
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        #base_path = os.path.abspath(".") # working dir
        base_path = correctPath(__file__, -2)
    return os.path.join(base_path, relative_path)

def getFileHandle(path):
    """
    Gets a file handle from url or disk file.

    :param path: filepath or url
    :return: file object
    """
    # urllib.urlopen does the same but is deprecated in python 3
    # this function intents to overcome this limitation
    try:
        f = urlopen(path)
    except (ValueError,URLError):  # invalid URL
        f = open(path,'rb')
    return f

def getFileSize(path):
    """
    Gets a size from url or disk file.

    :param path: filepath or url
    :return: size in bytes
    """
    # urllib.urlopen does the same but is deprecated in python 3
    # this function intents to overcome this limitation
    try:
        return urlopen(path).info()["Content-Length"]
    except ValueError:  # invalid URL
        return os.stat(path).st_size

def checkPath(path):
    """
    checks if path exists.

    :param path: path to folder or file.
    :return: True if exits, False if not
    """
    return os.path.exists(path)

def checkDir(dirname):
    """
    checks if dirname exists.

    :param dirname: path to folder
    :return: True if exits, False if not
    """
    return os.path.isdir(dirname)

def checkURL(url):
    """
    checks if url exists.
    :param url: path to url
    :return: True if exits, False if not
    """
    try:
        urlopen(url)
        return True
    except ValueError:  # invalid URL
        return False

def checkFile(path):
    """
    checks if filepath or filename exists.

    :param path: filepath or filename
    :return: True if exits, False if not
    """
    return os.path.isfile(path)

def getPath(path=__file__):
    """
    Get standard path from path. It supports ~ as home directory.

    :param path: it can be to a folder or file. Default is __file__ or module's path.
                If file exists it selects its folder.
    :return: dirname (path to a folder)

    .. note:: It is the same as os.path.dirname(os.path.abspath(path)).
    """
    if path.startswith("~"):
        path = os.path.expanduser("~") + path[1:]
    if "." in path: # check extension
        return os.path.dirname(os.path.abspath(path)) # just use os
    else:
        return os.path.abspath(path)

def getData(path=__file__): # FIXME not working windows syntax under linux
    """
    Get standard path from path.

    :param path: it can be to a folder or file. Default is __file__ or module's path.
    :return: [drive,dirname,filename,ext].
            1. drive or UNC (Universal Naming Convention)
            2. dirname is path to folder.
            3. filename is name of file.
            4. ext is extension of file.
    """
    if os.path.isdir(path): # is path to folder
        drive,dirname = os.path.splitdrive(path)
        return [drive,dirname,"",""]
    else: # anything else is treated as file
        dirname,ext = os.path.splitext(path)
        base,filename = os.path.split(dirname) # base,filename = os.path.dirname(dirname), os.path.basename(dirname)
        drive,dirname = os.path.splitdrive(dirname[:len(dirname)-len(filename)])
        return [drive,dirname,filename,ext]

def increment_if_exits(path, add ="_{num}", force=None):
    """
    Generates new name if it exits.

    :param path: absolute path or filename
    :param add: if fn exists add pattern
    :param force: (None) force existent files even if they don't. if True
            treats fn as existent or if it is a list it treats names from
            the list as existent names.
    :return: un-existent fn
    """
    # normalize
    path = os.path.abspath(path)

    # list of exceptions
    listing = []
    if force is True:
        listing.append(path) # append itself
    elif force is not None:
        for f in force: # normalize all paths
            listing.append(os.path.abspath(f))

    # check existence
    if path in listing or os.path.exists(path):
        # get parts
        data = getData(path)
        # make pattern from parts
        pattern = "".join(data[:-1])+"{}"+data[-1]
        # get a priory list of files
        listing.extend(glob(pattern.format("*")))
        num = 0
        while True:
            num += 1
            path = pattern.format(add.format(num=num))
            if path not in listing and not os.path.exists(path):
                return path
        #return increment_if_exits(fn=fn, add=add, force=force) # double check
    else:
        return path

def changedir(filepath, dirname, ext=True):
    """
    Change path to file with dirname.

    :param filepath: path to file.
    :param dirname: new path to replace in filepath.
    :param ext: True to keep extension of file if any.
    :return: directory object of changed path.
    """
    if ext: data = getData(filepath)[2:]
    else: data = getData(filepath)[2:3]
    return Directory(dirname) + Directory(data, False)

def strdifference(s1,s2):
    """
    Get string differences.

    :param s1: string 1
    :param s2: string 2
    :return: (splitted string 1, splitted string 2, index).
        A splitted string is a list with the string parts.
        Index is a list containing the indexes of different parts of the two splitted strings.
    """
    def forequal():
        if equal:
            streq = "".join(equal)
            ls1.append(streq)
            ls2.append(streq)
            return True
        return False

    def fordiff():
        if diff1 or diff2:
            ls1.append("".join(diff1))
            ls2.append("".join(diff2))
            state.append(len(ls1)-1)
            return True
        return False

    ls1,ls2,state,equal,diff1,diff2= [],[],[],[],[],[]
    i=0
    for i in range(min(len(s1),len(s2))):
        if s1[i]==s2[i]:
            equal.append(s1[i])
            if fordiff(): diff1,diff2= [],[]
        else:
            diff1.append(s1[i])
            diff2.append(s2[i])
            if forequal(): equal = []
    fordiff() # append for different
    forequal() # append for equal
    diff1,diff2= s1[i+1:],s2[i+1:]
    fordiff() #append remaining
    return ls1,ls2,state

def getSplitted(path=__file__):
    """
    Splits a file path by its separators.

    :param path: it can be to a folder or file. Default is __file__ or module's path.
    :return: splitted path.
    """
    return path.split(getSep(path) or None)

def correctSep(path=__file__, separator = os.path.sep):
    """
    Replaces the path separators by custom or OS standard separator.

    :param path: relative or absolute path (str). Default is __file__ or module's path.
    :param separator: desired separators, By default uses system separator (os.path.sep).
    :return: path with corrected separator.
    """
    return separator.join(getSplitted(path))

def mkPath(path): # FIXME: not sure if this is correct
    """
    Make path (i.e. creating folder) for filepath.

    :param path: path to nonexistent folder or file.
    :return: created path.
    """
    path = os.path.abspath(str(path))
    try:
        #if not os.path.exists(path): os.makedirs(path) # not os.path.isdir(path)
        os.makedirs(path)
    except OSError as exception:
        import errno
        if exception.errno == errno.EEXIST:
            if os.path.isfile(path):
                raise Exception("Folder not created. There is a file with the same name at {}".format(path))
        else: #exception.errno == errno.EACCES:
            raise exception #Exception("Permission Denied")
    return path # returns created path

def rmFile(filepath):
    """
    Remove file.

    :param filepath: path to file.
    :return: None
    """
    try:
        os.remove(filepath) # enclose in try/except, who knows files could be deleted
    except OSError:
        pass # file could have been deleted

def rmPath(path, ignore_errors=False, onerror=None):
    """
    Remove path from path.

    :param path: path to nonexistent folder or file.
    :return: None

    .. seealso:: shutil.rmtree
    """
    path = getPath(path)
    shutil.rmtree(path, ignore_errors, onerror)

def getSep(path, pattern='/\\'):
    """
    Get path separator or indicator.

    :param path: relative or absolute path (str).
    :param pattern: guess characters to compare path (str).
    :return: sep (str).

    .. note:: It is equivalent to os.path.sep but obtained from the given path and patterns.
    """
    i = len(path)
    while i and path[i-1] not in pattern: i -= 1
    head, tail = path[:i], path[i:]  # now tail has no slashes
    # remove trailing slashes from head, unless it's all slashes
    head2 = head
    count = 0
    while head2 and head2[-1] in pattern:
        head2 = head2[:-1]
        count += 1
    character = head[len(head)-count:len(head)] # get the slash character
    return character

def quickOps(path,comp):
    """
    (IN DEVELOPMENT) make quick matching operations in path.

    :param path: path to folder
    :param comp: pattern
    :return:

    Requirements::

        path = 'LEVEL1/LEVEL2/LEVEL3/LEVEL4/LEVEL5'
        print quickOps(path,'../ROOT/../LEVEL1/../LEVEL2/LEVEL3/../../LEVEL4')
        'LEVEL4'
        print quickOps(path,'ROOT/../LEVEL1/LEVEL2/../../LEVEL4')
        'LEVEL3/LEVEL4'
        print quickOps(path,'../LEVEL2/../')
        'LEVEL1/LEVEL3/LEVEL4/LEVEL5'
        print quickOps(path,'../LEVEL2/..')
        'LEVEL1/LEVEL3/LEVEL4/LEVEL5/'
        print quickOps(path,'LEVEL2/../../LEVEL4/')
        'LEVEL2/LEVEL3/LEVEL4/'
        print quickOps(path,'ROOT/../LEVEL2/../../LEVEL4')
        'ROOT/LEVEL3/LEVEL4'
        print quickOps(path,'LEVEL-1/../NEW7/LEVEL8')
        'LEVEL-1/LEVEL1/LEVEL2/LEVEL3/LEVEL4/LEVEL5/NEW7/LEVEL8'
        print
    """
    toCorrect = getSplitted(path) # what has to be corrected
    toCompare = getSplitted(comp) # what is used to correct
    size = len(toCompare)
    test = ("..",)
    where = []
    current = 0
    for i,cp in enumerate(toCompare):
        if cp not in test: # is a level
            tfrom = i+1<size and toCompare[i+1] in test
            tto = i>0 and toCompare[i-1] in test
            isin = cp in toCorrect[current:]
             # TODO: instead of using index i from 'toCompare', find a way to do it directly in 'toCorrect'
            if tfrom and tto:
                if isin:
                    where.append((i,cp,'del'))
            elif tfrom:
                if isin:
                    where.append((i,cp,'from'))
                else:
                    where.append((i,cp,'lput'))
            elif tto:
                if isin:
                    where.append((i,cp,'to'))
                else:
                    where.append((i,cp,'rput'))

    if where:
        for order,(i,cp,op) in enumerate(where): # index, comparison, operation
            if cp in toCorrect:
                del toCorrect[toCorrect.index(cp)]
                i = toCompare.index(cp)
                del toCompare[i:i+2]

def getShortenedPath(path, comp):
    """
    Path is controlled to give absolute path from relative path or integer.

    :param path: absolute path (str).
    :param comp: pattern or relative path (str) or integer representing level of folder
                determined by the separator Ex. "/level 1/level 2/.../level N or -1".
    :return: path before matched to comp  Ex: "C://level 1//comp --> C://level 1"

    Example::

        >>> path = 'LEVEL1/LEVEL2/LEVEL3/LEVEL4/LEVEL5'
        >>> print getShortenedPath(path,-2) # minus two levels
        LEVEL1/LEVEL2/LEVEL3
        >>> print getShortenedPath(path,2) # until three levels
        LEVEL1/LEVEL2
        >>> print getShortenedPath(path,'LEVEL1/LEVEL2/LEVEL3/')
        LEVEL1/LEVEL2/LEVEL3/
        >>> print getShortenedPath(path,'LEVEL4/REPLACE5/NEWLEVEL')
        LEVEL1/LEVEL2/LEVEL3/LEVEL4/REPLACE5/NEWLEVEL
        >>> print getShortenedPath(path,'../../SHOULD_BE_LEVEL4')
        LEVEL1/LEVEL2/LEVEL3/SHOULD_BE_LEVEL4

    """
    indicator = getSep(path)
    if isinstance(comp,str):
        toCorrect = getSplitted(path) # what has to be corrected
        toCompare = getSplitted(comp) # what is used to correct
        #isrel = not where and ".." in toCompare # isRelative
        isrel = False
        for i,cp in enumerate(toCompare):
            if cp == "..": # go to previous folder
                del toCorrect[-1] # eliminate last
                isrel = True
            elif cp == ".": # stay in folder
                pass #do nothing
            elif isrel or cp not in toCorrect: # place new folders in position
                toCorrect.extend(toCompare[i:]) # add folders
                break
            else: # look for folder and place new folders there
                if cp and cp in toCorrect:
                    toCorrect = toCorrect[:toCorrect.index(cp)] + toCompare[i:]
                else:
                    toCorrect.extend(toCompare[i:]) # add folders
                break
        path2 = indicator.join(toCorrect)
        return path2 # path.split(comp.split(getSep(comp))[0])[0]+comp
    else:
        return indicator.join(path.split(indicator)[:comp])

def decoratePath(relative, sep=os.path.sep):
    """
    Decorated path is controlled to give absolute path from relative path.

    :param relative: int or path.
    :param sep: path separator
    :return: decorator
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args,**kwargs):
            path = f(*args,**kwargs)
            return getShortenedPath(correctSep(path,sep), relative) # relative is corrected internally
            #return sep.join(getShortenedPath(path, relative).split(getSep(path)))
        return wrapper
    return decorator

def correctPath(path, relative):
    """
    Get path corrected from its relative path or level index.

    :param path: path or file name.
    :param relative: pattern or level in directory.
    :return: corrected path.
    """
    return decoratePath(relative)(getPath)(path)

def joinPath(absolute,relative):
    """
    Joins an absolute path to a relative path.

    :param absolute: directory or path.
    :param relative: directory or path.
    :return: joined path.

    .. note:: It is equivalent to os.path.join but works with directories.
    """
    absolute = str(absolute)
    relative = str(relative)
    if relative.startswith("\\") or relative.startswith("/"):
        relative = relative[1:]
    return os.path.join(absolute, relative) # str ensures updated version is processed

class Directory(str):
    """
    semi-mutable string representation of a inmutable string with support for path representations.

    :param data: list, directory instance, dictionary or string.
    :param ispath: True to add support for paths.
    :param copy: when data is a directory if copy is True then this instance data is independent
            of the passed directory otherwise both directories are a reference to the same
            dictionary data but they are not the same object.
    :param kwargs: additional data to add in directory.
    """
    def __new__(cls, data, ispath = None, copy = False, **kwargs):
        """
        Creates and initializes directory.
        """
        # TODO overcome limitation of string not being mutable. use basestring?
        # TODO keep string inheritance, keep string methods, but enable mutabilite
        # data can be list, str, directory or dictionary
        data = Directory.filterdata(data, ispath, kwargs)
        string = Directory.repr2str(data["repr"], data["ispath"])
        self = super(Directory, cls).__new__(cls, string)
        if copy:
            # do not use __dict__ to not overlook custom setters
            for k, v in data.items(): # self.__dict__.update(data)
                setattr(self, k, v)
            setattr(self, "repr", string) # self.__dict__["repr"] = [string]
        else:
            setattr(self,"__dict__",data) # self.__dict__ = data
        self.__name__ = "directory"
        return self

    @staticmethod
    def filterdata(data, ispath = None, kwargs=None):
        """
        Adequate data for dictionary creation.

        :param data: any supported object.
        :param ispath: True to add support for paths.
        :param kwargs: additional data to add in directory.
        :return: dictionary
        """
        if isinstance(data, Directory):
            data = getattr(data,"__dict__") # data.__dict__
        elif not isinstance(data,dict):
            if isinstance(data,list):
                data = {"repr":data}
            else:
                data = {"repr":[data]}
        if kwargs: data.update(kwargs) # update data with kwargs
        if ispath is not None: data["ispath"]= ispath
        if "ispath" not in data: data["ispath"] = True # ispath default
        if "repr" in data:
            data["repr"] = Directory.repr2list(data["repr"]) # ensure repr is maintained
        else:
            data["repr"] = [""] # repr default
        return data

    @staticmethod
    def repr2list(data,level=0):
        """
        Converts the representation of a directory.repr to pickleable.

        :param data: directory.repr of the form ["string",directory,...,directory.repr].
        :return: pickleable list.
        """
        if isinstance(data,list): # list defines levels of directories
            if level == 0: # [level 0, ..., [level 1, [...[level N]...]], level 0]
                for i,value in enumerate(data): # process several objects in the list
                    data[i] = Directory.repr2list(value, level + 1)
                return data
            else: # convert anything to directory if not in level 0
                return Directory(data)
        elif isinstance(data,str): # if string or directory
            return data
        else: # try to convert to directory
            return Directory(data)

    @staticmethod
    def repr2str(data, ispath = True):
        """
        Converts the representation of a directory.repr to string.

        :param data: directory.repr of the form ["string",directory,...,directory.repr].
        :return: converted string.
        """
        if isinstance(data,list):
            if len(data)>1:
                string = str(Directory.repr2str(data[0]))
                for i in data[1:]:
                    if ispath:
                        string = joinPath(string, Directory.repr2str(i)) # join paths
                    else:
                        string += Directory.repr2str(i, ispath)
                return string
            else:
                return str(data[0]) # get single path
        elif isinstance(data, Directory):
            return Directory.repr2str(data.repr, data.ispath)
        else:
            return str(data) # object must be string

    def correctSTRBuiltin(self):
        """
        Decorate all the built-in functions of class directory.

        :return: built-in decorated function.
        """
        def decorator(f):
            @wraps(f)
            def wrapper(*args,**kwargs):
                return getattr(str(self),f.__name__,f.__name__)(*args,**kwargs)
            return wrapper
        return decorator

    def update_right(self, other):
        """
        Updates representation a the right.

        :param other: any supported object.
        :return: new directory referenced to itself.

        .. note:: Equivalent to self + other e.g. directory([self, other])
        """
        self.repr.append(other) # it will be parsed at directory creation
        return Directory(self) # string is immutable and must be renewed

    def update_left(self, other):
        """
        Updates representation a the left.

        :param other: any supported object.
        :return: new directory referenced to itself.

        .. note:: Equivalent to self - other e.g. directory([other, self])
        """
        self.repr.insert(0,other) # it will be parsed at directory creation
        return Directory(self) # string is immutable and must be renewed

    def update(self, data = None):
        """
        Return an updated copy with provided data.

        :param data: any supported object.
                    If None return updated and referenced copy of itself.
        :return: new directory referenced to itself.
        """
        if isinstance(data,list): # if list
            if len(data)>len(self.repr):
                for i in range(len(self.repr)):
                    self.repr[i] = data[i]
                for j in range(i+1,len(data)):
                    self.repr.append(data[j])
            else:
                for i in range(len(data)):
                    self.repr[i] = data[i]
                del self.repr[i+1:]
            return Directory(self) # string is immutable and must be renewed
        elif isinstance(data,dict): # if dictionary
            for k, v in data.items(): # self.__dict__.update(data)
                setattr(self, k, v)
            return Directory(self)
        elif data: # if not list or dict
            return self.update([data])
        else: # if None return updated version
            return Directory(self)

    def copy(self):
        """
        Creates copy of itself.

        :return: non-referenced directory copy.
        """
        return Directory(self, copy=True)

    def __str__(self):
        return Directory.repr2str(self.repr, self.ispath)

    def __repr__(self):
        return str(self.repr)

    def __add__(self, other):
        if type(other)==str: # it's exactly str an not directory
            return joinPath(self,other)

        return self.update_right(other)
    __iadd__ = __add__

    def __sub__(self, other):
        if type(other)==str: # it's exactly str an not directory
            return joinPath(other,self)

        return self.update_left(other)
    # TODO: implement 2 versions: one with directory-like functionality and other basic for str support
    """
    ### MAGIC FUNCTIONS
    def __nonzero__(self):
        return self == self.update()
    def __getitem__(self, item):
        return self.repr[item]
    def __len__(self):
        return len(self.repr)"""

class FileDirectory(Directory):
    """
    Saves contents of a file as with directories.

    :param data: list, directory instance, dictionary or string.
    :param filename: name of file.
    :param path: path to folder where file is (it must finish in /).
    :param notes: optional description string
    :param kwargs: additional data to add in directory.
    """
    def __new__(cls, data, filename, path, notes = None):
        self = super(FileDirectory,cls).__new__(cls,data,False,False)
        self.filename=filename,
        self.path = path,
        self.notes = notes
        return self

    def makeFile(self):
        """
        Makes a file with its contents to path/filename.

        :return: True if successful
        """
        path = mkPath(self.path)
        initfile = joinPath(path, self.filename)
        with open(initfile, 'wb') as logger:
            logger.write(str(self))
            return True

if __name__=="__main__":

    from . import session as sn
    ## TESTS
    a = Directory("string1", sapo ="mamo")
    b = a.update(["string2",["string4","string 5"]])
    result = bool(a) # False
    result = bool(b) # True
    c = Directory(["string3"]) - a
    sn.saveSession("test.pkl",{"d",a})
    c = sn.readSession("test.pkl")
    print(type(a) == Directory)
    print(type(a) is Directory)
    print(a is Directory)
    print(a == Directory)
    print("with str")
    print(type(a)==str)
    print(type(a) is str)
    print(a is str)
    print(a == str)
    print(isinstance(a,str))
    print(type(a))
    print(os.path.splitext(os.path.basename(__file__))) ### look here ###

    path = Directory(["path1", "path2", "path3"])
    #path += "path4"
    print(path)
    print(os.path.join(path,"new path"))