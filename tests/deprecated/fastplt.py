# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
from multiprocessing import Process
from RRtoolbox.lib.config import FLAG_DEBUG
wins = [0] # keeps track of image number through different processes

def fastplt(image, cmap = None, title ="visualazor", win = None, block = False, daemon = False):
    """
    Fast plot.

    :param image: image to show
    :param cmap: "gray" or None
    :param title: title of subplot
    :param win: title of window
    :param block: if True it wait for window close, else it detaches
    :param daemon: if True window closes if main thread ends, else windows must be closed to main thread to end
    :return: plt
    """
    if FLAG_DEBUG: print("fastplt received image type: ",type(image))
    def myplot():
        if isinstance(image, matplotlib.axes.SubplotBase):
            f = image.figure
        elif isinstance(image, matplotlib.figure.Figure):
            f = image
        else:
            f = plt.figure()
            # Normally this will always be "Figure 1" since it's the first
            # figure created by this process. So do something about it.
            plt.imshow(image,cmap)
            if title: plt.title(title)
            plt.xticks([]), plt.yticks([])
            #plt.colorbar()
        wins[0]+=1
        if win: f.canvas.set_window_title(win)
        else:f.canvas.set_window_title("Figure {}".format(wins[-1]))
        if FLAG_DEBUG: print("showing now...")
        #plt.ion()
        plt.show()
        if FLAG_DEBUG: print("showed...")
    if block:
        myplot()
    else:
        if FLAG_DEBUG: print("multiprocessing...")
        p = Process(target=myplot) # FIXME i shoud call a miniprogram
        p.daemon = daemon
        p.start()
    if FLAG_DEBUG: print("left fastplt...")

if __name__ == "__main__":
    import argparse
    from RRtoolbox.lib.serverServices import parseString as _parseString
    #import sys
    #if FLAG_DEBUG: print sys.argv
    parser = argparse.ArgumentParser(description='fast plot of images.')
    parser.add_argument('image', metavar='N', #action='append',
                        help='path to image or numpy string',nargs="+")
    parser.add_argument('-m','--cmap', dest='cmap', action='store',
                       help='map to use in matplotlib')
    parser.add_argument('-t','--title', dest='title', action='store',default="visualazor",
                       help='title of subplot')
    parser.add_argument('-w','--win', dest='win', action='store',
                       help='title of window')
    parser.add_argument('-n','--num', dest='num', action='store',type = int, default=0,
                       help='number of Figure')
    parser.add_argument('-b','--block', dest='block', action='store_true', default=False,
                       help='number of Figure')
    parser.add_argument('-d','--daemon', dest='daemon', action='store_true', default=False,
                       help='number of Figure')
    v = vars(parser.parse_args())
    images = _parseString(v.pop("image"))
    wins[-1] = v.pop("num")
    if FLAG_DEBUG: print("properties:", v)
    for image in images: fastplt(image,**v)
    if FLAG_DEBUG: print("leaving fastplt module...")
