# http://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/

# import the necessary packages
import numpy as np

import requests
import cv2
from RRtoolbox.lib.plotter import fastplt
import sys, os

PY3 = sys.version_info >= (3,0)

try:
    from urllib.request import urlopen # urllib.urlopen disappears in python 3
except ImportError:
    from urllib2 import urlopen

def getFileHandle(path):
    # urllib.urlopen does the same but is deprecated in python 3
    # this function intents to overcome this limitation
    try:
        f = urlopen(path)
    except ValueError:  # invalid URL
        f = open(path)
    return f

def getFileSize(path):
    # urllib.urlopen does the same but is deprecated in python 3
    # this function intents to overcome this limitation
    try:
        return urlopen(path).info()["Content-Length"]
    except ValueError:  # invalid URL
        return os.stat(path).st_size

def loadsFromURL(url, flags=cv2.IMREAD_COLOR):
    """
    Only loads images from urls.

    :param url: url to image
    :param flags: openCV supported flags
    :param dtype:
    :return:
    """
    resp = urlopen(url) # download the image
    #nparr = np.asarray(bytearray(resp.read()), dtype=dtype) # convert it to a NumPy array
    nparr = np.fromstring(resp.read(), dtype=np.uint8)
    image = cv2.imdecode(nparr, flags=flags) # decode using OpenCV format
    return image

def loadsfrom(url, flags=cv2.IMREAD_COLOR):
    resp = getFileHandle(url) # download the image
    #nparr = np.asarray(bytearray(resp.read()), dtype=dtype) # convert it to a NumPy array
    nparr = np.fromstring(resp.read(), dtype=np.uint8)
    image = cv2.imdecode(nparr, flags=flags) # decode using OpenCV format
    return image

r = requests.get('https://github.com/davtoh', auth=('user', 'pass'))
#link = "/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/tests/_tmp0000000.png"#"http://python.org"
link = "https://raw.githubusercontent.com/davtoh/RRtoolbox/master/tests/_tmp0000000.png"
im = loadsfrom(link)
fastplt(im,title=link)