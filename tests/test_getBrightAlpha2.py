"""
test the alfa mask created to merge two retinal images
"""
from __future__ import absolute_import


from .tesisfunctions import filterFactory,normsigmoid,graph_filter, normalize, histogram,getOtsuThresh
from RRtoolbox.lib.arrayops import Bandpass, FilterBase,Bandstop,Bandpass
import numpy as np
import cv2

def getBrightAlfa(backgray, foregray, window = None):
    """
    Get alfa transparency for merging foreground to background gray image according to brightness.

    :param backgray: background image.
    :param foregray: foreground image.
    :param window: window used to customizing alfa. It can be a binary or alfa mask, values go from 0 for transparency
                    to any value where the maximum is visible i.e a window with all the same values does nothing.
                    A binary mask can be used, where 0 is transparent and 1 is visible.
                    If not window is given alfa is not altered and the intended alfa is returned.
    :return: alfa mask
    """
    # TODO: this method was obtained stoically, change to an automated one
    backmask = BackFilter()(backgray)
    foremask = ForeFilter()(foregray)
    foremask = normalize(foremask * backmask)
    foremask[foremask>0.9] = 2.0
    ksize = (21,21)
    foremask = normalize(cv2.blur(foremask,ksize))
    if window is not None: foremask *= normalize(window) # ensures that window is normilized to 1
    return foremask

class BackFilter(Bandstop):
    name = "background filter"
    def __init__(self, alpha=3, beta1=50, beta2 = 190):
        super(BackFilter, self).__init__(alpha=alpha, beta1=beta1, beta2=beta2)

class ForeFilter(Bandpass):
    name = "foreground filter"
    def __init__(self, alpha=3, beta1=50, beta2 = 220):
        super(ForeFilter, self).__init__(alpha=alpha, beta1=beta1, beta2=beta2)


class Brigh(FilterBase):
    name = "merged lateral view"
    def __init__(self, alpha=10, beta1=50, beta2 = 225):
        super(Brigh, self).__init__(alpha=alpha, beta1=beta1, beta2=beta2)
    def _test_beta1(self, value):
        assert self.beta2>value
    def _test_beta2(self, value):
        assert value>self._beta1
    def _test_alfa(self, value):
        assert value>0
    def __call__(self, backgray, foregray):
        A = normsigmoid(foregray, self._alfa, self._beta1) - normsigmoid(foregray, self._alfa, self._beta2)
        B = normsigmoid(backgray, self._alfa, self._beta1) - normsigmoid(backgray, self._alfa, self._beta2) + 1.0
        return normalize(A*B)

class AllFilters(FilterBase):
    name = "merged lateral view"
    def __call__(self, values):
        return getBrightAlfa(values,values)

filters = []
filters.append(BackFilter())
filters.append(ForeFilter())
filters.append(AllFilters())

level = np.linspace(0, 256,256)

import pylab as plt
figure = graph_filter(filters,single=True,cols=3,legend=True,annotate=True,show=False,levels=level,titles="Filters response")
fore = histogram(cv2.imread('im1_1.jpg',0))[0]
back = histogram(cv2.imread('im1_2.jpg',0))[0]
fore_norm = normalize(fore)
back_norm = normalize(back)
plt.plot(level,fore_norm,level,back_norm)
Otsu_fore = int(getOtsuThresh(fore))
Otsu_back = int(getOtsuThresh(back))
data = list({"Back":(Otsu_back,back_norm[Otsu_back]),"Fore":(Otsu_fore,fore_norm[Otsu_fore])}.items())
for i,(key,(xp,yp)) in enumerate(data):
    val = 10
    if i%2: val = -10
    plt.annotate(u'Normalized {}\nOtsu-thresh = {}'.format(key,xp).title(), xy=(xp,yp), textcoords='data',
                                    xytext=(xp+val,yp-0.12),
                                    arrowprops=dict(facecolor='black', shrink=0.05))
plt.gca().set_xlabel("Levels")
plt.show()


from mpl_toolkits.mplot3d import axes3d
#http://matplotlib.org/examples/pylab_examples/contour_label_demo.html
# http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
#X, Y, Z = axes3d.get_test_data(0.05)
X, Y = np.meshgrid(level,level)
#zs = np.array([getBrightAlfa2(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
#Z = np.nan_to_num(zs.reshape(X.shape))
Z = getBrightAlfa(X,Y)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#CS = ax.contour(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
#CS = ax.contour(X, Y, Z, zdir='y', offset=255, cmap=cm.coolwarm)
#CS = plt.contour(X, Y, Z)
#plt.clabel(CS, inline=1, fontsize=10)
"""
fmt = {}
strs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']
for l, s in zip(CS.levels, strs):
    fmt[l] = s
# Label every other level using strings
plt.clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=1)"""

ax.set_xlabel('Background')
ax.set_xlim(0, 256)
ax.set_ylabel('Foreground')
ax.set_ylim(0, 256)
ax.set_zlabel('Alpha')
ax.set_zlim(0, 1)

plt.show()
