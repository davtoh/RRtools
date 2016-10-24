"""
crop input images and save them
"""

from RRtoolFC.GUI.forms import getROI
from RRtoolbox.lib.directory import getData, increment_if_exits, getPath, mkPath
from RRtoolbox.lib.image import loadFunc
from RRtoolbox.lib.root import glob
import numpy as np
import os
import cv2

#app = QtGui.QApplication([]) # make a quick application

def crops(fn, outpath = None, loader=None, preview=None,
          form = None, startfrom = None, append = None):
    """
    Crop input image and save ROIs

    :param fn: file name
    :param outpath: (None)
    :param loader: (loadFunc(1))
    :param preview: (rect)
    :param form: crop shape type supported by :func:`getROI`
    :param startfrom: start from an specific pattern in path
    :param append: append transformation to list of transformations
            which includes (ImageItem,save_path) items
    :return: ROI object, list of transformations
    """
    imsets = glob(fn) # only folders
    if preview is None:
        preview = True
    if form is None:
        form = "rect"
    if loader is None:
        loader = loadFunc(1)

    start = False
    ops = [] # operations
    roi = None
    for impath in imsets:
        if startfrom is None or startfrom in impath:
            start = True
        if start:
            image = loader(impath).astype(np.float32)
            print "loaded",impath
            cropping = True
            a,b,c,d = getData(impath)

            # get path to save ROIs
            if isinstance(outpath,basestring): # custom sting path
                outpath2 = outpath
            elif outpath is True: # create default path
                outpath2 = a+b+c
            else:
                outpath2 = None
                append = True

            # make path if it does not exists
            mkPath(getPath(outpath2))

            # get ROIs
            while cropping:
                # get ROI
                roi, imgla = getROI(image, preview=preview, form= form, crop=False)
                fn = None
                if outpath2 is not None:
                    # save ROI
                    fn = increment_if_exits(os.path.join(outpath2,"{}{}".format(c,d)),force=True)
                    if cv2.imwrite(fn, roi.getArrayRegion(image, imgla)):
                        print "Saved: {}".format(fn)
                    else:
                        "{} could not be saved".format(fn)
                        fn = None

                if append:
                    ops.append((imgla,fn))

                if raw_input("continue?(y,n)").lower() in ("n","not","no"):
                    cropping = False

    return roi,ops

if __name__ == "__main__": # for a folder with many sets
    """
    Example using crop function
    """
    fn = "./../tests/im1_1.jpg"
    outpath = None
    startfrom = None
    loader=None
    preview=True
    form = "rect"

    crops(fn,outpath,loader,preview,form,startfrom)
