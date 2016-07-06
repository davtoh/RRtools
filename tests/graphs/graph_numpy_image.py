"""
shows a numpy image sample (NOT STANDARD)
"""
from RRtoolbox.lib.plotter import fastplt,plotim
from RRtoolbox.lib.image import loadFunc
import numpy as np

if __name__ == "__main__":

    im = loadFunc(flag=1,dsize=(300,100))("../lena.png")
    print im.shape, im[99,299]

    rows = 3
    cols = 1
    a = np.ones((rows,cols)) # max index a[2,0]
    assert rows,cols == a.shape
    #fastplt(im)
    plotim("image",im).show()
