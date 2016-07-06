# snippet from http://stackoverflow.com/a/9667121/5288758
# http://scipy.github.io/devdocs/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter
# http://stackoverflow.com/a/28857249/5288758

from tesisfunctions import histogram, brightness,plotim, graphmath, graphHistogram, \
    overlay, findmaxima, findminima, smooth, graph_filter, getOtsuThresh, SAVETO, correctString,stdoutLOG,printParams
from RRtoolbox.lib.directory import getData,mkPath
from RRtoolbox.lib.arrayops import bandpass
import cv2
import numpy as np
from scipy.signal import savgol_filter
import pylab as plt

name_script = getData(__file__)[-2]

def decompose(points):
    #points = np.array([x[a], hist_PS[a]])
    U, s, V = np.linalg.svd(points, full_matrices=False)
    #print U.shape, V.shape, s.shape
    #s[-1] = 0
    points2 = np.dot(U, np.dot(np.diag(s), V))
    #points2.sort(0) # sort with respect to axis 1, that is to sort each column
    return points2[0,:],points2[1,:]

def demo():
    fn1 = r'im5_1.jpg'
    #fn1 = r'im1_1.jpg'
    fn1 = r'im3_1.jpg'
    name = fn1.split('\\')[-1].split(".")[0]

    windows = 'flat', 'hanning', 'hamming', 'bartlett', 'blackman','savgol'
    #windows = ("savgol",)
    window_len=11
    save = False
    show = True
    correct = False
    format= "svg"
    saveTo_root = SAVETO+name_script+"/"
    config = locals().copy()
    mkPath(saveTo_root) # make directory
    stdoutLOG(saveTo_root+ "log") # save file output
    # now everything printed will be logged
    printParams(config) # this is logged too

    # read image
    fore = cv2.imread(fn1)
    fore = cv2.resize(fore,(300,300)) # resize image
    #plotim(name,fore).show() # show image

    # get intensity
    P = brightness(fore)
    #P = (255-P)


    hist_P, bins = np.histogram(P.flatten(),256,[0,256])

    for w in windows:
        # filter histogram to smooth it.
        title = "{}({})".format(w,window_len)
        if w == "savgol":
            polyorder = 5
            add = ": polyorder {}".format(polyorder)
            hist_PS = savgol_filter(hist_P, window_len, polyorder) # window size 51, polynomial order 3
        else:
            if correct:
                add = ": corrected convolution"
            else:
                add = ": normal convolution"
            hist_PS = smooth(hist_P,window_len, window=w, correct=correct) # it smooths and expands the histogram
        title = (title+add).title()
        x = np.arange(len(hist_PS))#bins[:-1] #np.indices(hist_P.shape)[0] # get indexes
        #x2,hist_PD = decompose(np.array([x,hist_P]))
        #U, s, V = np.linalg.svd(np.array([x,hist_P]), full_matrices=False)
        #U2, s2, V2 = np.linalg.svd(np.array([x,hist_PS]), full_matrices=False)
        #points2 = np.dot(U2, np.dot(np.diag(s2), V))
        #x2,hist_PD = points2[0,:],points2[1,:]
        fig2 = plt.figure(title)
        plt.title(title)
        plt.plot(bins[:-1],hist_P, label= "normal", color="black")
        plt.plot(x,hist_PS, label= "smooth")
        #plt.plot(x2,hist_PD, label= "decomposed")

        a = getOtsuThresh(hist_P)
        b = findminima(hist=hist_PS)
        c = findmaxima(hist=hist_PS)
        plt.plot(x[b], hist_PS[b], "o", label="min")
        plt.plot(x[c], hist_PS[c], "o", label="max")
        plt.plot(x[a], hist_PS[a], "o", label="Otsu")
        plt.legend()
        #plt.xlim(-30,256)
        if save:
            plt.savefig("{}{}_{}.{}".format(saveTo_root, name, correctString(title, True, "_"), format), bbox_inches='tight', format= format)
    if show: plt.show()

if __name__ == "__main__":
    demo()