"""
runs a histogram equalization example based from tests/Ex_histogram_equalization.py
"""

from preamble import *
from tests.tesisfunctions import hist_cdf, equalization

script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, fn = None, uselabels = False, useclahe = False, figsize=(20,10)):
    """
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param fn: (None) file name of figure to load
    :param uselabels: place labels at axes
    :param useclahe: add the CLAHE equalization
    :param figsize: figure shape (width,height) in inches
    :return: locals()
    """

    gd = graph_data(pytex)
    if fn is None:
        fn = "im1_1.jpg"
    gd.load(fn)

    def plothist(img):
        hst,bins = np.histogram(img.flatten(),256,[0,256])

        cdf = hst.cumsum()
        cdf_normalized = cdf * hst.max()/ cdf.max()

        plot(cdf_normalized, color = 'b')
        hist(img.flatten(),256,[0,256], color = 'r')
        xlim([0,256])
        legend(('cdf','histogram'), loc = 'upper left')

        if uselabels:
            xlabel("Levels")
            ylabel("Pixel Count")


    # histograms
    figure(figsize=figsize)
    img = gd.gray

    if useclahe:
        subplots = [231,234, 232, 235, 233, 236]
        name2 = "HE"
    else:
        subplots = [221,223,222,224]
        name2 = "Equalized"

    subplot(subplots[0]),imshow(img,cmap="gray")
    title(gd.wrap_title("Input"))
    xticks([]), yticks([])

    subplot(subplots[1])
    plothist(img)

    eqimg = equalization(gd.gray)[0]

    subplot(subplots[2]),imshow(eqimg,cmap="gray")
    title(gd.wrap_title(name2))
    xticks([]), yticks([])

    subplot(subplots[3])
    plothist(eqimg)

    if useclahe:
        # create a CLAHE object (Arguments are optional).
        if isinstance(useclahe,(list,tuple)):
            opts = useclahe
        else:
            opts = 2.0, (8,8)
        clahe = cv2.createCLAHE(*opts)
        eqimg2 = clahe.apply(gd.gray)

        subplot(subplots[4]),imshow(eqimg2,cmap="gray")
        title(gd.wrap_title("CLAHE clipLimit={}, tileGridSize={}".format(*opts)))
        xticks([]), yticks([])

        subplot(subplots[5])
        plothist(eqimg2)

    gd.output(name)
    return locals()


if __name__ == "__main__":
    graph(useclahe=False,uselabels = True)