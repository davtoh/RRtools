from __future__ import print_function


import ast # https://docs.python.org/3.2/library/ast.html#module-ast
import imp # see https://docs.python.org/2/library/imp.html
import os

#from info.infoOld import info

import RRtoolbox
from RRtoolbox.lib import config

#import knee # see http://svn.python.org/projects/python/trunk/Demo/imputil/knee.py

MODULE_EXTENSIONS = [ext[0] for ext in imp.get_suffixes()]
def package_contents(package_name,ext=MODULE_EXTENSIONS):
    # http://stackoverflow.com/questions/487971/is-there-a-standard-way-to-list-names-of-python-modules-in-a-package
    ext = tuple(ext)
    file, pathname, description = imp.find_module(package_name)
    if file:
        raise ImportError('Not a package: %r', package_name)
    # Use a set because some may be both source and compiled.
    return set([os.path.splitext(module)[0]
        for module in os.listdir(pathname)
        if module.endswith(ext)])

def top_level_functions(body):
    return (f for f in body if isinstance(f, ast.FunctionDef))

def parse_ast(filename):
    with open(filename, "rt") as file:
        r = file.read()
        return ast.parse(r, filename=filename)

def listfunctions(filename):
    # http://stackoverflow.com/questions/139180/listing-all-functions-in-a-python-module/31005891#31005891
    tree = parse_ast(filename)
    return [func.name for func in top_level_functions(tree.body)]

d0 = listfunctions(__file__)
d = package_contents("RRtoolbox")
data = config.getModules(config.MANAGER["MAINPATH"])
#information = info(RRtoolbox)
print(data)