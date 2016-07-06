"""
graph the alfa mask created to merge two retinal images
"""
__author__ = 'Davtoh'

from preamble import *
script_name = os.path.basename(__file__).split(".")[0]
from RRtoolbox.lib.arrayops import  normalize, histogram, getOtsuThresh
from RRtoolbox.lib.arrayops import FilterBase,bandstop,bandpass, brightness
from RRtoolbox.lib.plotter import graph_filter
from RRtoolbox.tools.segmentation import get_beta_params
import numpy as np

beta_back = [None,None]
beta_fore = [None,None]

class back_filter(bandstop):
    name = "background filter"
    def __init__(self, alpha=3, beta1=None, beta2 = None):
        if beta1 is None: beta1 = beta_back[0]
        if beta2 is None: beta2 = beta_back[1]
        super(back_filter,self).__init__(alpha=alpha, beta1=beta1, beta2=beta2)

class fore_filter(bandpass):
    name = "foreground filter"
    def __init__(self, alpha=3, beta1=None, beta2 = None):
        if beta1 is None: beta1 = beta_fore[0]
        if beta2 is None: beta2 = beta_fore[1]
        super(fore_filter,self).__init__(alpha=alpha, beta1=beta1, beta2=beta2)

def getBrightAlfa(backgray, foregray, window = None):
    """
    Pseudo function of getBrightAlpha
    """
    backmask = back_filter()(backgray)
    foremask = fore_filter()(foregray)
    foremask = normalize(foremask * backmask)
    if window is not None: foremask *= normalize(window) # ensures that window is normilized to 1
    return foremask

class all_filter(FilterBase):
    name = "merged lateral view"
    def __call__(self, values):
        return getBrightAlfa(values,values)

def graph(pytex =None, name = script_name, fn1 = None, fn2 = None,figsize=(15,6)):
    """
    :param pytex:
    :param name:
    :param fn1:
    :param fn2:
    :param figsize:
    :return:
    """
    gd = graph_data(pytex)
    if fn1 is None: fn1 = 'im1_1.jpg'
    if fn2 is None: fn2 = 'im1_2.jpg'
    gd.load(fn1)
    P_fore = brightness(gd.BGR)
    gd.load(fn2)
    P_back = brightness(gd.BGR)
    for i,v in enumerate(get_beta_params(P_fore)):
        beta_fore[i]=v
    for i,v in enumerate(get_beta_params(P_back)):
        beta_back[i]=v

    filters = []
    filters.append(back_filter())
    filters.append(fore_filter())
    filters.append(all_filter())

    level = np.linspace(0, 256,256)

    fig = figure(figsize=figsize)
    graph_filter(filters,single=True,cols=3,
                 legend=True,annotate=True,
                 show=False,levels=level,
                 win=fig,titles = "",lxp=[(False,False),(True,True)],scale=0.03)

    hist_fore = histogram(P_fore)[0]
    hist_back = histogram(P_back)[0]
    fore_norm = normalize(hist_fore)
    back_norm = normalize(hist_back)
    plot(level,fore_norm,level,back_norm)
    Otsu_fore = int(getOtsuThresh(hist_fore))
    Otsu_back = int(getOtsuThresh(hist_back))
    data = {"Back":(Otsu_back,back_norm[Otsu_back]),"Fore":(Otsu_fore,fore_norm[Otsu_fore])}.items()
    for i,(key,(xp,yp)) in enumerate(data):
        val = 10
        if i%2: val = -10
        annotate(u'Normalized {}\nOtsu-thresh = {}'.format(key,xp).title(), xy=(xp,yp), textcoords='data',
                                        xytext=(xp+val,yp-0.12),
                                        arrowprops=dict(facecolor='black', shrink=0.05))
    gca().set_xlabel("Levels")

    gd.output(name+"_histAnalysis")

    #http://matplotlib.org/examples/pylab_examples/contour_label_demo.html
    # http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

    fig = figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(level,level)
    Z = getBrightAlfa(X,Y)
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

    ax.set_xlabel('Background')
    ax.set_xlim(0, 256)
    ax.set_ylabel('Foreground')
    ax.set_ylim(0, 256)
    ax.set_zlabel('Alpha')
    ax.set_zlim(0, 1)

    gd.output(name+"_3D")

    return locals()

if __name__ == "__main__":
    graph()