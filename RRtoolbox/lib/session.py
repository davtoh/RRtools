# -*- coding: utf-8 -*-
"""
    This module have serializing methods for data persistence so to let the package "save" custom objects

    session module made by Davtoh and powered by dill
    Dependency project: https://github.com/uqfoundation/dill

"""

__author__ = 'Davtoh'

try:
    # for security reason read this: http://www.benfrederickson.com/dont-pickle-your-data/
    # download: https://pypi.python.org/pypi/dill#downloads
    # see print dill.license() https://github.com/uqfoundation
    #import jsonpickle as serializer # http://jsonpickle.github.io/
    import cpickle as serializer
    #import dill as serializer # dill must be >= 0.2.4
    #__license__ = serializer.__license__
    #dill.detect.trace(True)
except:
    import pickle as serializer

import types
import os
__excludeType = [types.FunctionType,types.ModuleType,types.NoneType,types.ClassType,types.TypeType]
__excludeVar = []
__excludePattern = ['__']


def getEnviromentSession(enviroment = None):
    """
    Gets the filtered session from the global variables.

    :return: dictionary containing filtered session.
    """
    enviroment = enviroment or globals()
    #globals(), dir(), [type(enviroment[keys]) for keys in enviroment]
    session = {}
    for keys in enviroment.keys():
        if __excludePattern != [] and keys.startswith(*__excludePattern):
            continue
        if not (type(enviroment[keys]) in __excludeType or keys in __excludeVar):
            session[keys] = enviroment[keys]
    return session

def saveSession(filepath, session, helper = None):
    """
    Saves dictionary session to file.

    :param filepath: path to save session file.
    :param session: dictionary
    :param helper: function to pre-process session
    :return: filename of saved session
    """
    # safely save session file
    #with os.fdopen(os.open(filepath, os.O_WRONLY | os.O_CREAT, 0600), 'wb') as logger: # http://stackoverflow.com/a/5624691/5288758
    with open(filepath, 'wb') as logger:
        if helper:
            serializer.dump(helper(session), logger, serializer.HIGHEST_PROTOCOL) # save dictionary
        else:
            serializer.dump(session, logger, serializer.HIGHEST_PROTOCOL) # save dictionary
        return logger.name

def readSession(filepath, helper=None):
    """
    Loads a dictionary session from file.

    :param filepath: path to load session file.
    :param helper: function to pos-process session file
    :return: session
    """
    # safely read session file
    with open(filepath, 'rb') as logger:
        session = serializer.load(logger) # get session
    if helper:
        return helper(session)
    else:
        return session

def updateSession(filepath, session, replace=True, rdhelper=None, svhelper=None):
    """
    Updates a dictionary session in file.

    :param filepath: path to session file.
    :param session: dictionary.
    :param replace: if True key values are replaced else old key values ar kept.
    :param rdhelper: read helper.
    :param svhelper: save helper.
    :return: None
    """
    current = readSession(filepath,rdhelper)
    if replace: # update by replacing existing values
        current.update(session)
    else: # update without replacing existing values
        for key in session:
            if not current.has_key(key):
                current[key] = session[key]
    saveSession(filepath, current, svhelper) # save updated session

def flushSession(filepath):
    """
    Empty session in file.

    :param filepath: path to session file.
    :return:
    """
    readSession(filepath)
    saveSession(filepath, {}) # save updated session

def checkFromSession(filepath, varlist):
    """
    Check that variables exits in session file.

    :param filepath: path to session file.
    :param varlist: list of variables to checkLoaded.
    :return: list checkLoaded results
    """
    current = readSession(filepath)
    return [current.has_key(var) for var in varlist] # checking variables

def deleteFromSession(filepath, varlist):
    """
    Delete variables from session file.

    :param filepath: path to session file.
    :param varlist: list of variables to delete.
    :return: None
    """
    current = readSession(filepath)
    for var in varlist: # deleting variables
        del(current[var])
    saveSession(filepath, current) # save updated session
