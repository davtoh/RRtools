"""
    This module contains core-like, too-much-used and too-much-referenced modules
"""


#__all__ = ['cache','overlay','root','session']
# add relativity to script
import os,sys
# add relative path so that serializing methods know where to find modules
_lib_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_lib_path)
"""
from arrayops import *
from cache import *
from config import *
from directory import *
from image import *
from inspector import *
from plotter import *
from root import *
from session import *"""