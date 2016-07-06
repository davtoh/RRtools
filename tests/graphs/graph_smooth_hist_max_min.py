"""
applies different smoothing 1D-filters to histogram and finds their minima and maxima values
"""
# see more at tests/experimental_maxima_minima.py
from preamble import *
script_name = os.path.basename(__file__).split(".")[0]
from tests.tesisfunctions import brightness, findmaxima, findminima, smooth, getOtsuThresh
from scipy.signal import savgol_filter

def graph(pytex =None, name = script_name, fn = None, windows =  None, window_len=11, correct = False, useTitle = True):
    """
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param fn: (None) file name of figure to load
    :param windows: windows name 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman','savgol'
    :param window_len: and odd number
    :param correct: use corrected convolution
    :param useTitle: print title in subplots
    :return: locals()
    """
    # based on implementation http://stackoverflow.com/a/33047048/5288758
    gd = graph_data(pytex)
    if fn is None:
        fn = r'im1_1.jpg'
    gd.load(fn)

    if windows is None:
        windows = 'flat', 'hanning', 'hamming', 'bartlett', 'blackman','savgol'
        #windows = ("savgol",)

    # get intensity
    P = brightness(gd.BGR)
    #P = (255-P)

    hist_P, bins = np.histogram(P.flatten(),256,[0,256])
    titles = []
    for w in windows:
        # filter histogram to smooth it.
        titleStr = "{}({})".format(w,window_len)
        saveStr = "{}{}".format(w,window_len)
        if w.startswith("savgol"):
            if len(w)>6:
                polyorder = int(w[6:])
                w = w[:6]
            else:
                polyorder = 5
            add = ": polyorder {}".format(polyorder)
            hist_PS = savgol_filter(hist_P, window_len, polyorder) # window size 51, polynomial order 3
        else:
            if correct:
                add = ": corrected convolution"
                saveStr += "_c"
            else:
                add = ": normal convolution"
            hist_PS = smooth(hist_P,window_len, window=w, correct=correct) # it smooths and expands the histogram
        titleStr = (titleStr+add).title()
        x = np.arange(len(hist_PS))#bins[:-1] #np.indices(hist_P.shape)[0] # get indexes

        figure()
        if useTitle: title(titleStr)
        titles.append(titleStr)
        plot(bins[:-1],hist_P, label= "normal", color="black")
        plot(x, hist_PS, label= "smooth")
        #plot(x2,hist_PD, label= "decomposed")

        a = getOtsuThresh(hist_P)
        b = findminima(hist= hist_PS)
        c = findmaxima(hist= hist_PS)
        plot(x[b], hist_PS[b], "o", label="min")
        plot(x[c], hist_PS[c], "o", label="max")
        plot(x[a], hist_PS[a], "o", label="Otsu")
        legend()
        #xlim(-30,256)

        gd.output("{}_{}".format(name,saveStr),caption=titleStr)

    return locals()

if __name__ == "__main__":
    graph(windows =  None, window_len=11, correct = False)