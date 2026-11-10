# -*- coding: utf-8 -*-
r"""
Image I/O, format interpretation, network/file loading, and batch conversions.
"""
from __future__ import division
import os
import cv2
import numpy as np
from ..directory import getData, mkPath, increment_if_exits, getFileHandle
from ..arrayops.basic import im2shapeFormat
from ..root import glob
from ..serverServices import parseString, string_is_socket_address
from .conversions import supported_formats


def checkLoaded(obj, fn="", raiseError=False):
    """
    Simple function to determine if variable is valid.

    :param obj: loaded object
    :param fn: path of file
    :param raiseError: if True and obj is None, raise
    :return: None
    """
    if obj is not None:
        print(fn, " Loaded...")
    else:
        print(fn, " Could not be loaded...")
        if raiseError:
            raise


def loadcv(path, flags=-1, shape=None):
    """
    Simple function to load using opencv.

    :param path: path to image.
    :param flags: openCV flags:

                +-------+------------------------------+--------+
                | value | openCV flag                  | output |
                +=======+==============================+========+
                | (1)   | cv2.CV_LOAD_IMAGE_COLOR      | BGR    |
                +-------+------------------------------+--------+
                | (0)   | cv2.CV_LOAD_IMAGE_GRAYSCALE  | GRAY   |
                +-------+------------------------------+--------+
                | (-1)  | cv2.CV_LOAD_IMAGE_UNCHANGED  | format |
                +-------+------------------------------+--------+
    :param shape: shape to resize image.
    :return: loaded image
    """
    im = cv2.imread(path, flags)
    if shape:
        im = cv2.resize(im, shape)
    return im


def loadsfrom(path, flags=cv2.IMREAD_COLOR):
    """
    Loads Image from URL or file.

    :param path: filepath or url
    :param flags: openCV flags:

                +-------+------------------------------+--------+
                | value | openCV flag                  | output |
                +=======+==============================+========+
                | (1)   | cv2.CV_LOAD_IMAGE_COLOR      | BGR    |
                +-------+------------------------------+--------+
                | (0)   | cv2.CV_LOAD_IMAGE_GRAYSCALE  | GRAY   |
                +-------+------------------------------+--------+
                | (-1)  | cv2.CV_LOAD_IMAGE_UNCHANGED  | format |
                +-------+------------------------------+--------+
    :return:
    """
    if isinstance(path, (str, bytes)):
        if isinstance(path, str) and path.endswith(".npy"):  # reads numpy arrays
            return np.lib.load(path, None)
        resp = getFileHandle(path)  # download the image
    else:
        resp = path  # assume path is a file-like object ie. cStringIO or file
    content = resp.read()
    if hasattr(np, "frombuffer"):
        nparr = np.frombuffer(content, dtype=np.uint8)
    else:
        nparr = np.fromstring(content, dtype=np.uint8)
    image = cv2.imdecode(nparr, flags=flags)  # decode using OpenCV format
    return image


def interpretImage(toparse, flags):
    """
    Interprets to get image.

    :param toparse: string to parse or array. It can interpret:

        * connection to server (i.e. host:port)
        * path to file (e.g. /path_to_image/image_name.ext)
        * URL to image (e.g. http://domain.com/path_to_image/image_name.ext)
        * image as string (i.g. numpy converted to string)
        * image itself (i.e. numpy array)

    :param flags: openCV flags:

                +-------+------------------------------+--------+
                | value | openCV flag                  | output |
                +=======+==============================+========+
                | (1)   | cv2.CV_LOAD_IMAGE_COLOR      | BGR    |
                +-------+------------------------------+--------+
                | (0)   | cv2.CV_LOAD_IMAGE_GRAYSCALE  | GRAY   |
                +-------+------------------------------+--------+
                | (-1)  | cv2.CV_LOAD_IMAGE_UNCHANGED  | format |
                +-------+------------------------------+--------+

    :return: image or None if not successfull
    """
    # test it is from server
    if isinstance(toparse, (str, bytes)) and string_is_socket_address(toparse):  # process request to server
        toparse = parseString(toparse, 5)
    # test is object itself
    if type(toparse).__module__ == np.__name__:  # test numpy array
        if flags == 1:
            return im2shapeFormat(toparse, (0, 0, 3))
        if flags == 0:
            return im2shapeFormat(toparse, (0, 0))
        return toparse
    # test image in string
    try:
        return cv2.imdecode(toparse, flags)
    except (TypeError, cv2.error):
        # test path to file or URL
        return loadsfrom(toparse, flags)


def convertAs(fns, base=None, folder=None, name=None, ext=None,
              overwrite=False, loader=None, simulate=False):
    """
    Reads a file and save as other file based in a pattern.

    :param fns: file name or list of file names. It supports glob operations.
            By default glob operations ignore folders.
    :param base: path to place images.
    :param folder: (None) folder to place images in base's path.
            If True it uses the folder in which image was loaded.
            If None, not folder is used.
    :param name: string for formatting new name of image with the {name} tag.
            Ex: if name is 'new_{name}' and image is called 'img001' then the
            formatted new image's name is 'new_img001'
    :param ext: (None) extension to save all images. If None uses the same extension
            as the loaded image.
    :param overwrite: (False) If True and the destine filename for saving already
            exists then it is replaced, else a new filename is generated
            with an index "{name}_{index}.{extension}"
    :param loader: (None) loader for the image file to change image attributes.
            If None reads the original images untouched.
    :param simulate: (False) if True, no saving is performed but the status is returned
            to confirm what images where adequately processed.
    :return: list of statuses (0 - no error, 1 - image not loaded,
            2 - image not saved, 3 - error in processing image)
    """
    if loader is None:
        from .loaders import loadFunc
        loader = loadFunc(1)
    if isinstance(fns, (str, bytes)):
        filelist = glob(fns)  # list
    else:  # is an iterator
        filelist = []
        for f in fns:
            filelist.extend(glob(f))
    if base is None:
        base = ''
    # ensures that path from base ends with separator
    if base:
        base = os.path.join(base, "")
    replaceparts = getData(base)  # from base get parts
    # ensures that extension starts with point "."
    if isinstance(ext, (str, bytes)) and not ext.startswith("."):
        ext = "." + ext  # correct extension

    status = []
    for file in filelist:
        parts = getData(file)  # file parts
        # replace drive
        if replaceparts[0]:
            parts[0] = replaceparts[0]
        # replace root
        if replaceparts[1]:
            if folder is True:
                parts[1] = os.path.join(replaceparts[1],
                                        os.path.split(os.path.split(parts[1])[0])[1], "")
            elif isinstance(folder, (str, bytes)):
                parts[1] = os.path.join(replaceparts[1], folder, "")
            else:
                parts[1] = replaceparts[1]
        # to replace basic name
        if isinstance(name, (str, bytes)):
            parts[2] = name.format(name=parts[2])
        if isinstance(ext, (str, bytes)):
            parts[3] = ext  # replace extension
        newfile = "".join(parts)
        if not overwrite:
            newfile = increment_if_exits(newfile)

        try:
            im = loader(file)

            # image not loaded
            if im is None:
                status.append((file, 1, newfile))
                continue

            # image successfully saved
            if simulate:
                status.append((file, 0, newfile))
                continue
            else:
                mkPath("".join(parts[:2]))
                if cv2.imwrite(newfile, im):
                    status.append((file, 0, newfile))
                    continue

            # image not saved
            status.append((file, 2, newfile))
        except:
            # an error in the process
            status.append((file, 3, newfile))
    return status

