"""
graph comparison of selection of parameters for bilateral filter
"""
from __future__ import division
from preamble import *
from RRtoolbox.lib.arrayops import getBilateralParameters, noisy
script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, fn = None,
          shapes = ((50,50),(100,100),(600,600),(1500,1500)),
          noise = 's&p', useTitle = True, split = False):
    """
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param fn: (None) file name of figure to load
    :param shapes: list of shapes of fn image
    :param noise: 'gauss', 'poisson', 's&p', 'speckle'
    :param useTitle: if True add titles to the figures.
    :param split: split images with subscripts 0,1,2 ... N=len(shapes)
            with division a and b as "{name}_{subscript}{division}"
    :return locals()
    """
    gd = graph_data(pytex)
    if fn is None: gd.load("lena.png")

    gd.RGB = noisy(gd.RGB,noise).astype(np.uint8) # add noise
    # graph filters' results
    gd.shape = None
    if not split: fig =figure()#figsize=(mm2inch(163,45))) # 163, 45 mm
    cols=2
    #num *= cols
    num = cols*len(shapes)
    gs = generateGrid(num,cols=cols)
    #for j in xrange(0,num,cols):
    for i,(j,b) in enumerate(zip(range(0,num,cols),shapes)):
        if split: # only two colmuns
            fig = figure() # create individual figures
        else:
            fig.add_subplot(gs[j]) # subplot axes
        #b = tuple([int(i+j/cols*increment) for i in base])
        img = cv2.resize(gd.RGB,b) # img size
        imshow(img) # plot filtered image
        t = 'Input shape = {}'.format(img.shape[:2]) # title
        n = "{}_{}a".format(name,i) # name of file name
        if useTitle: title(gd.wrap_title(t))
        xticks([]), yticks([])
        if split:
            gd.output(n, caption=t)


        if split: # only two colmuns
            fig = figure() # create individual figures
        else:
            fig.add_subplot(gs[j+1])
        params = getBilateralParameters(img.shape) # calculate bilateral parameters (21,82,57)
        imshow(cv2.bilateralFilter(img, *params)) # plot filtered image
        t = 'Filtered with d = {}, sigmaC = {}, sigmaS = {}'.format(*params) # title
        n = "{}_{}b".format(name,i) # name of file name
        if useTitle: title(gd.wrap_title(t))
        xticks([]), yticks([])
        if split:
            gd.output(n, caption=t)

    if not split: gd.output(name)
    return locals()

if __name__ == "__main__":
    graph(pytex.context,shapes = ((50,50),(100,100),(600,600),(1500,1500)),noise = 's&p', useTitle = True, split = True)