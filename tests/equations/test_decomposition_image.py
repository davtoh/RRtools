# http://www.ams.org/samplings/feature-column/fcarc-svd
# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.svd.html
from RRtoolbox.lib.image import loadcv, cv2, np
from tests.tesisfunctions import getthresh

from RRtoolbox.lib.plotter import fastplt

W,H = 400,300
size = W,H # this is how functions receive size
rows,cols = H,W # This is how shape works H,W == im.shape = rows,cols
fn = "../im1_1.jpg"
#fn = '../paterns.png'
im = loadcv(fn,0,size)

U, s, V = np.linalg.svd(im, full_matrices=False)
print U.shape, V.shape, s.shape

class svdFamily:
    def __init__(self,gray):
        self._u, self._s, self._v = np.linalg.svd(gray, full_matrices=False)
    def u(self):
        return self._u.copy()
    def s(self):
        return self._s.copy()
    def v(self):
        return self._v.copy()
    def usv(self):
        return self.u(),self.s(),self.v()
    def decompose(self, gray):
        return np.linalg.svd(gray, full_matrices=False)
    def recompose(self,u=None,s=None,v=None):
        if u is None: u = self.u()
        if s is None: s = self.s()
        if v is None: v = self.v()
        return np.dot(u, np.dot(np.diag(s), v))
    def getUncommonMask(self):
        s = self.s()
        s[s==np.max(s)] = 0 # eliminate most common
        return self.recompose(s=s)>=0
    def getUncommonAlfa(self, times = 0, normalize = True):
        s = self.s()
        s[s==np.max(s)] = 0 # eliminate most common
        alfa = self.recompose(s=s)
        alfa[alfa<0] = 0
        if times:
            return svdFamily(alfa).getUncommonAlfa(times-1, normalize)
        elif normalize:
            return alfa*255.0/np.max(alfa)
        else:
            return alfa
    def getCommonMask(self):
        s = self.s()
        s[s<np.max(s)] = 0 # conserve most common
        return self.recompose(s=s)>0

testobj = svdFamily(im)

def test():
    def applyOp(slice, title = "Recomposed svd with s[{slice}]={fill}", fill=0):
        s_ = s.copy()
        slice = ":".join([str(i) for i in slice])
        try:
            exec "s2[{}] = fill".format(slice) in locals()
        except: # generalize error
            raise Exception("slice is of the form (from,to,step) to use as from:to:step")
        im = np.abs(np.dot(U, np.dot(np.diag(s_), V))).astype(np.uint8)
        fastplt(im,title=title.format(fill = fill, slice = slice,
                                                      min=np.min(im),max=np.max(im)),cmap="gray")
        return s_,im

    fastplt(im,title="normal: s.shape = {}".format(s.shape),cmap="gray")
    applyOp((50,"")) # eliminate least significands
    applyOp((0,)) # eliminates most significand

def test1():
    s[s>np.max(s)-1] = 0
    im2 = np.dot(U, np.dot(np.diag(s), V))
    #im2.sort(1) # sort with respect to axis 1, that is to sort each column
    fastplt(im,title="normal",cmap="gray")
    #th = threshold(im,getthresh(im))
    thresh,th = cv2.threshold(im,getthresh(im),255,cv2.THRESH_BINARY)
    fastplt(th,title="normal thresh")
    fastplt(im2,title="recomposed",cmap="gray")
    #th2 = threshold(im2.astype(np.uint8),getthresh(im2))
    thresh,th2 = cv2.threshold(im2.astype(np.uint8),getthresh(im2),255,cv2.THRESH_BINARY)
    fastplt(th2,title="recomposed thresh")

def test2():
    s[s==np.max(s)] = 0
    im2 = np.dot(U, np.dot(np.diag(s), V))
    #im2.sort(1) # sort with respect to axis 1, that is to sort each column
    fastplt(im,title="normal")
    #th = threshold(im,getthresh(im))
    #thresh,th = cv2.threshold(im,getthresh(im),255,cv2.THRESH_BINARY)
    #fastplt(th,title="normal thresh")
    fastplt(im2,title="recomposed")
    #th2 = threshold(im2.astype(np.uint8),getthresh(im2))
    th2 = im2.copy()
    th2[im2<0] = 0
    th2 = th2/np.max(th2)
    th2 = th2*255
    fastplt(th2,title="recomposed thresh")

def test3():
    im = testobj.recompose()
    #fastplt(im,title="recomposed")
    fastplt(testobj.getUncommonAlfa(0),title="0")
    fastplt(testobj.getUncommonMask(),title="mask")
    #assert np.allclose(testobj.getUncommonAlfa(5),testobj.getUncommonAlfa(5,False))

test()