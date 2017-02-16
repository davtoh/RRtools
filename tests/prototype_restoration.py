from __future__ import division
from __future__ import print_function

# ----------------------------    IMPORTS    ---------------------------- #

# multiprocessing
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import itertools as it
from multiprocessing.pool import ThreadPool as Pool
# three-party
import cv2
import numpy as np
# custom
from RRtool.RRtoolbox.lib import plotter, cache, config, image
from RRtool.RRtoolbox import filter,basic as ar
from RRtool.RRtoolbox.lib.arrayops import convert

# ----------------------------    GLOBALS    ---------------------------- #
cpc = cv2.getNumberOfCPUs()
print("configured to use {} cpus".format(cpc))
pool = Pool(processes = cpc) # DO NOT USE IT when module is imported and this runs with it. It creates a deadlock"
feature_name = 'sift-flann'
paths = config.ConfigFile()
# ----------------------------    DECORATORS    ---------------------------- #

def getalfa(foregray,backgray,window = None):
    """ get alfa transparency for merging to retinal images
    :param foregray: image on top
    :param backgray: image at bottom
    :param window: window used to customizing alfa, values go from 0 for transparency to any value
                    where the maximum is visible i.e a window with all the same values does nothing.
                    a binary image can be used, where 0 is transparent and 1 is visible.
                    If not window is given alfa is left as intended.
    :return: float window modified by alfa
    """
    normalize = filter.normalize
    normsigmoid = filter.normsigmoid
    backmask = normalize(normsigmoid(backgray,10,180)+normsigmoid(backgray,3.14,192)+normsigmoid(backgray,-3.14,45))
    foremask = normalize(normsigmoid(foregray,-1,242)*normsigmoid(foregray,3.14,50))
    foremask = normalize(foremask * backmask)
    foremask[foremask>0.9] = 2.0
    ksize = (21,21)
    foremask = normalize(cv2.blur(foremask,ksize))
    if window is not None: foremask *= normalize(window)
    return foremask

# ----------------------------VISUALIZER FUNCTIONS---------------------------- #
def matchExplorer(win, img1, img2, kp_pairs, status = None, H = None, show=True):
    # functions
    ## GET INITIAL VISUALIZATION
    if len(img1.shape)<3:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if len(img2.shape)<3:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]  # obtaining image1 dimensions
    h2, w2 = img2.shape[:2]  # obtaining image2 dimensions
    # imgf and imgb will be visualized horizontally (left-right)
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)  # making visualization image
    vis[:h1, :w1] = img1  # imgf at the left of vis
    vis[:h2, w1:w1+w2] = img2  # imgf at the right of vis

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)  # making sure every pair of keypoints is graphed

    kp_pairs = [(dict2keyPoint(i),dict2keyPoint(j)) for i,j in kp_pairs]
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])  # pair of coordinates for imgf
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0) # pair of coordinates for imgb

    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))

    def drawline(self):
        vis = self.rimg
        self.thick = int(filter.sigmoid(vis.shape[0] * vis.shape[1], 1723567, 8080000, 5, 1))
        if H is not None:  # enclosing object
            rcorners = np.array([self.real2render(corner[0],corner[1]) for corner in corners])
            cv2.polylines(vis, [rcorners], True, self.framecolor) # draw rendered TM encasing

        rp1 = []
        rp2 = []
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            rx1,ry1 = self.real2render(x1,y1) # real to render
            rx2,ry2 = self.real2render(x2,y2) # real to render
            rp1.append((rx1,ry1))
            rp2.append((rx2,ry2))
            r = self.thick
            if inlier and self.showgoods:  # drawing circles (good keypoints)
                col = self.goodcolor
                cv2.circle(vis, (rx1, ry1), r, col, -1)  # for left keypoint (imgf)
                cv2.circle(vis, (rx2, ry2), r, col, -1)  # for right keypoint (imgf)
            elif self.showbads:  # drawing x marks (wrong keypoints)
                col = self.badcolor
                thickness = r
                # for left keypoint (imgf)
                cv2.line(vis, (rx1-r, ry1-r), (rx1+r, ry1+r), col, thickness)
                cv2.line(vis, (rx1-r, ry1+r), (rx1+r, ry1-r), col, thickness)
                # for right keypoint (imgf)
                cv2.line(vis, (rx2-r, ry2-r), (rx2+r, ry2+r), col, thickness)
                cv2.line(vis, (rx2-r, ry2+r), (rx2+r, ry2-r), col, thickness)
            # drawing lines for non-onmouse event
        self.rp1 = np.int32(rp1)
        self.rp2 = np.int32(rp2)
        self.vis0 = vis.copy()  # saving state of the visualization for onmouse event
        # get rendered kp_pairs
        self.kp_pairs2 = apply2kp_pairs(kp_pairs,self.real2render,self.real2render)
        # drawing lines for non-onmouse event
        for (rx1, ry1), (rx2, ry2), inlier in zip(rp1, rp2, status):
            if inlier and self.showgoods:
                cv2.line(vis, (rx1, ry1), (rx2, ry2), self.goodcolor,r)
        self.vis = vis.copy() # visualization with all inliers

    def drawrelation(self):
        if self.flags & cv2.EVENT_FLAG_LBUTTON:
            x,y = self.rx, self.ry
            cur_vis = self.vis0.copy()  # actual visualization
            r = self.thick + 8  # proximity to keypoint
            m = (ar.anorm(self.rp1 - (x, y)) < r) | (ar.anorm(self.rp2 - (x, y)) < r)
            idxs = np.where(m)[0]  # get indexes near pointer
            kp1s, kp2s = [], []
            for i in idxs:  # for all keypints near pointer
                (rx1, ry1), (rx2, ry2) = self.rp1[i], self.rp2[i]  # my keypoint
                col = (self.badcolor, self.goodcolor)[status[i]]  # choosing False=red,True=green
                cv2.line(cur_vis, (rx1,ry1), (rx2,ry2), col, self.thick)  # drawing line
                # keypoints to show on event
                kp1, kp2 = self.kp_pairs2[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            # drawing keypoints near pointer for imgf and imgb
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=self.kpcolor)
            cur_vis = cv2.drawKeypoints(cur_vis, kp2s, flags=4, color=self.kpcolor)
            self.rimg = cur_vis
        else:
            self.rimg = self.vis

        if self.y is not None and self.x is not None:
            self.builtinplot(self.sample[self.y,self.x])

    def randomColor():
        return (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))

    def mousefunc(self):
        if self.builtincontrol():
            self.updaterenderer()
            drawline(self)

        if self.mousemoved:
            drawrelation(self)
    def keyfunc(self):
        if self.builtincmd():
            drawline(self)
            if self.y is not None and self.x is not None:
                self.builtinplot(self.img[self.y,self.x])
            else:
                self.builtinplot()

    self = plotter.plotim(win, vis)
    self.mousefunc = mousefunc
    self.keyfunc = keyfunc
    self.showgoods = True
    self.showbads = False
    self.__dict__.update(image.colors)
    self.randomColor = randomColor
    self.goodcolor = self.green
    self.badcolor = self.red
    self.kpcolor = self.orange
    self.framecolor = self.blue
    self.cmdlist.extend(["showgoods","showbads","framecolor","kpcolor","badcolor","goodcolor"])
    drawline(self)
    # show window
    if show: self.show()
    return self.rimg # return coordinates

def explore_match(win, img1, img2, kp_pairs, status = None, H = None, show=True):
    """
    This function draws a set of keypoint pairs obtained on a match method of a descriptor
    on two images imgf and imgb
    :param win: window's name (str)
    :param img1: image1 (numpy array)
    :param img2: image2 (numpy array)
    :param kp_pairs: zip(keypoint1, keypoint2)
    :param status: obtained from cv2.findHomography
    :param H: obtained from cv2.findHomography (default=None)
    :return: vis (image with matching result) (default=None)
    """
    # colors to use
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)

    if len(img1.shape)<3:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if len(img2.shape)<3:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]  # obtaining image1 dimensions
    h2, w2 = img2.shape[:2]  # obtaining image2 dimensions
    # imgf and imgb will be visualized horizontally (left-right)
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)  # making visualization image
    vis[:h1, :w1] = img1  # imgf at the left of vis
    vis[:h2, w1:w1+w2] = img2  # imgf at the right of vis
    #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  # changing color attribute to background image

    if H is not None:  # enclosing object
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, red)

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)  # making sure every pair of keypoints is graphed

    kp_pairs = [(dict2keyPoint(i),dict2keyPoint(j)) for i,j in kp_pairs]
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])  # pair of coordinates for imgf
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0) # pair of coordinates for imgb

    thick = int(filter.sigmoid(vis.shape[0] * vis.shape[1], 1723567, 8080000, 5, 1))

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:  # drawing circles (good keypoints)
            col = green
            cv2.circle(vis, (x1, y1), thick, col, -1)  # for left keypoint (imgf)
            cv2.circle(vis, (x2, y2), thick, col, -1)  # for right keypoint (imgf)
        else:  # drawing x marks (wrong keypoints)
            col = red
            r = thick
            thickness = thick
            # for left keypoint (imgf)
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            # for right keypoint (imgf)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()  # saving state of the visualization for onmouse event
    # drawing lines for non-onmouse event
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green,thick)

    if show:
        cv2.namedWindow(win,cv2.WINDOW_NORMAL) # Can be resized
        cv2.imshow(win, vis)  # show static image as visualization for non-onmouse event

        def onmouse(event, x, y, flags, param):
            cur_vis = vis  # actual visualization. lines drawed in it
            if flags & cv2.EVENT_FLAG_LBUTTON:  # if onmouse
                cur_vis = vis0.copy() # points and perspective drawed in it
                r = thick+8  # proximity to keypoint
                m = (ar.anorm(p1 - (x, y)) < r) | (ar.anorm(p2 - (x, y)) < r)
                idxs = np.where(m)[0]  # get indexes near pointer
                kp1s, kp2s = [], []
                for i in idxs:  # for all keypints near pointer
                     (x1, y1), (x2, y2) = p1[i], p2[i]  # my keypoint
                     col = (red, green)[status[i]]  # choosing False=red,True=green
                     cv2.line(cur_vis, (x1, y1), (x2, y2), col,thick)  # drawing line
                     # keypoints to show on event
                     kp1, kp2 = kp_pairs[i]
                     kp1s.append(kp1)
                     kp2s.append(kp2)
                # drawing keypoints near pointer for imgf and imgb
                cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=kp_color)
                cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, flags=4, color=kp_color)

            cv2.imshow(win, cur_vis)  # show visualization
        cv2.setMouseCallback(win, onmouse)
        cv2.waitKey()
        cv2.destroyWindow(win)
    return vis

# ----------------------------SPECIALIZED FUNCTIONS---------------------------- #
def init_feature(name,features = None):
    """
    This function takes parameters from a command to initialize a detector and matcher
    :param name: "<sift|surf|orb>[-flann]" (str) Ex: "sift-flann"
    :return: detector, matcher
    """
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    FLANN_INDEX_LSH    = 6
    if features is None: features = {}
    if name not in features: # if called with a different name
        chunks = name.split('-')
        if chunks[0] == 'sift':
            detector = cv2.SIFT()  # Scale-invariant feature transform
            norm = cv2.NORM_L2  # distance measurement to be used
        elif chunks[0] == 'surf':
            detector = cv2.SURF(800)  # Hessian Threshold to 800
            norm = cv2.NORM_L2  # distance measurement to be used
        elif chunks[0] == 'orb':
            detector = cv2.ORB(400)  # binary string based descriptors
            norm = cv2.NORM_HAMMING  # Hamming distance
        else:
            return None, None
        if 'flann' in chunks:  # FLANN based Matcher
            if norm == cv2.NORM_L2:  # for SIFT ans SURF
                flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            else:  # for ORB
                flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                   table_number = 6, # 12
                                   key_size = 12,     # 20
                                   multi_probe_level = 1) #2
            matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        else:  # brute force matcher
            matcher = cv2.BFMatcher(norm)
        features[name] = detector, matcher
    detector, matcher = features[name] # if possible get buffered detector and matcher
    return detector, matcher

detector, matcher = init_feature(feature_name) # global detector and matcher

def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=old_div(1.0,tilt), fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai

########################### CONVERSIONS #####################


def keyPoint2tuple(keypoint):
    """ obj.angle, obj.class_id, obj.octave, obj.pt, obj.response, obj.size"""
    return (keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id)

def tuple2keyPoint(points, func = cv2.KeyPoint):
    """ KeyPoint([x, y, _size[, _angle[, _response[, _octave[, _class_id]]]]]) -> <KeyPoint object> """
    return func(*(points[0][0],points[0][1],points[1],points[2], points[3],points[4], points[5]))

def dict2keyPoint(d, func = cv2.KeyPoint):
    """ KeyPoint([x, y, _size[, _angle[, _response[, _octave[, _class_id]]]]]) -> <KeyPoint object> """
    return func(*(d["pt"][0],d["pt"][1],d["size"],d["angle"], d["response"],d["octave"], d["class_id"]))

class SimKeyPoint(object):
    # FIXME: correct for memoizer: some warning are created if the script is run as __main__
    # it would be great if cv2.KeyPoint did not have pickling incompatibilities
    def __init__(self,*args):
        if len(args)==1:
            obj = args[0]
            if isinstance(obj,dict): # it got a dictionary
                self._dict.update(obj)
                return
            elif isinstance(obj,tuple):
                args = obj
            else: # it got cv2.Keypoint
                self.angle = obj.angle
                self.class_id = obj.class_id
                self.octave = obj.octave
                self.pt = obj.pt
                self.response = obj.response
                self.size = obj.size
                return
        # tupple is broadcasted as in cv2.KeyPoint
        self.pt =args[0]
        self.size=args[1]
        self.angle =args[2]
        self.response=args[3]
        self.octave=args[4]
        self.class_id=args[5]

def apply2kp_pairs(kp_pairs,kp1_pair,kp2_pair,func=None):
        """
        apply to kp_pairs
        :param kp_pairs:
        :param kp1_pair:
        :param kp2_pair:
        :param func: function to build new copy of keypoint
        :return:
        """
        def withtupple(keypoint,kp_op):
            if func:
                keypoint = func(keypoint)
            try:
                keypoint = keypoint.copy()
                keypoint["pt"] = np.multiply(keypoint["pt"],kp_op) # transform pt with kp_op
            except:
                x,y = keypoint.pt
                rx,ry = kp_op
                keypoint.pt = (x*rx,y*ry)
            return keypoint
        def withfunc(keypoint,kp_op):
            if func:
                keypoint = func(keypoint)
            try:
                keypoint = keypoint.copy()
                keypoint["pt"] = kp_op(*keypoint["pt"]) # transform pt with kp_op
            except:
                x,y = keypoint.pt
                keypoint.pt = kp_op(x,y)
            return keypoint
        if type(kp1_pair) is tuple: # expected tuple operands
            func1 = withtupple
        else:
            func1 = withfunc
        if type(kp2_pair) is tuple: # expected tuple operands
            func2 = withtupple
        else:
            func2 = withfunc
        return  [(func1(i,kp1_pair),func2(j,kp2_pair)) for i,j in kp_pairs]

def spairs2opairs(kp_pairs,osrc_sh,sscr_sh,odst_sh,sdst_sh,func=None):
        """
        convert scaled kp_pairs to original kp_pairs
        :param kp_pairs: list of kp_pairs
        :param osrc_sh: original source's shape
        :param sscr_sh: scaled source's shape
        :param odst_sh: original destine's shape
        :param sdst_sh: scaled destine's shape
        :param func: function to build new copy of keypoint
        :return:
        """
        kp1_pair = convert.getSOpointRelation(osrc_sh, sscr_sh) # fore
        kp2_pair = convert.getSOpointRelation(odst_sh, sdst_sh) # back
        return apply2kp_pairs(kp_pairs,kp1_pair,kp2_pair,func=func)

########################### END OF CONVERSIONS #####################

@cache.memoize(paths.TEMPPATH, ignore=["pool"])
def ASIFT(feature_name, img, mask=None, pool=pool):
    '''
    asift(feature_name, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    # bulding parameters of tilt and rotation variations
    detector = init_feature(feature_name)[0] # it must get detector object of cv2 here to prevent conflict with memoizers
    params = [(1.0, 0.0)]
    # phi rotations for t tilts of the image
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, old_div(72.0, t)):
            params.append((t, phi))

    def f(p):
        t, phi = p #tilt, phi (rotation)
        # computing the affine transform
        timg, tmask, Ai = affine_skew(t, phi, img)
        # Find keypoints and descriptors with the detector
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)
    keypoints, descrs = [], []
    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end=' ')
        keypoints.extend(k)
        descrs.extend(d)
    keypoints = [SimKeyPoint(obj)._dict for obj in keypoints]
    #return keyPoint2tuple(keypoints), np.array(descrs)
    return keypoints, np.array(descrs)

def multipleASIFT(imgs,feature_name=feature_name):
    """
    Affine-SIFT for N images
    :param imgs: images to apply asift
    :param feature_name: eg. SIFT SURF ORB
    :return: [(kp1,desc1),...,(kpN,descN)]
    """
    #print 'imgf - %d features, imgb - %d features' % (len(kp1), len(kp2))
    return [ASIFT(feature_name, img, pool=pool) for img in imgs]


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    """
    This function applies a ratio test
    :param kp1: raw keypoint 1
    :param kp2: raw keypoint 2
    :param matches: raw matches
    :param ratio: filtering ratio
    :return: filtered keypoint 1, filtered keypoint 2, keypoint pairs
    """
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )  # keypoint with Index of the descriptor in query descriptors
            mkp2.append( kp2[m.trainIdx] )  # keypoint with Index of the descriptor in train descriptors
    p1 = np.float32([kp["pt"] for kp in mkp1])
    p2 = np.float32([kp["pt"] for kp in mkp2])
    kp_pairs = list(zip(mkp1, mkp2))
    return p1, p2, kp_pairs

@cache.memoize(paths.TEMPPATH)
def MATCH(feature_name,kp1,desc1,kp2,desc2):
    """
    use matcher and asift output to obtain Transformation matrix (TM)
    :param feature_name: eg. BFMatcher, FlannBasedMatcher
    :param kp1: keypoints of source image
    :param desc1: descriptors of kp1
    :param kp2: keypoints of destine image
    :param desc2: descriptors of kp2
    :return: TM
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    """
    matcher = init_feature(feature_name)[1] # it must get matcher object of cv2 here to prevent conflict with memoizers
    # BFMatcher.knnMatch() returns k best matches where k is specified by the user
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    # If k=2, it will draw two match-lines for each keypoint.
    # So we have to pass a status if we want to selectively draw it.
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches) #ratio test of 0.75
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0) # status specifies the inlier and outlier points

        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        # do not draw outliers (there will be a lot of them)
        #kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag] # uncomment to give only good kp_pairs
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    return H, status, kp_pairs


def multipleMATCH(multipleASIFT, feature_name=feature_name):
    """
    :param multipleASIFT: from function that returns [(kp1,desc1),...,(kpN,descN)]
    :param feature_name:
    :return: [(H1, mask1, kp_pairs1),....(HN, maskN, kp_pairsN)]
    """
    kp1,desc1 = multipleASIFT[0]
    return [MATCH(feature_name,kp1,desc1,kpN,descN) for kpN,descN in multipleASIFT[1:]]

def MATCHto(cmp, multipleASIFT, feature_name=feature_name):
    """

    :param cmp: (kp0,desc0)
    :param multipleASIFT: from function that returns [(kp1,desc1),...,(kpN,descN)]
    :param feature_name:
    :return: [(H1, mask1, kp_pairs1),....(HN, maskN, kp_pairsN)]
    """
    kp1,desc1 = cmp
    return [MATCH(feature_name,kp1,desc1,kpN,descN) for kpN,descN in multipleASIFT]

def invertH(H):
    # inverse perspective
    return np.linalg.inv(H)

def boxpads(bx,points):
    points = points
    minX,minY = np.min(points,0) # left_top
    maxX,maxY = np.max(points,0) # right_bottom
    x0,y0 = bx[0] # left_top
    x1,y1 = bx[1] # right_bottom
    top,bottom,left,right = 0.0,0.0,0.0,0.0
    if minX<x0: left = x0-minX
    if minY<y0: top = y0-minY
    if maxX>x1: right = maxX-x1
    if maxY>y1: bottom = maxY-y1
    return [(left,top),(right,bottom)]

def transformPoint(p,H):
    return cv2.perspectiveTransform(np.float64([[p]]), H)

def transformPoints(p,H):
    return cv2.perspectiveTransform(np.float64([p]), H)

def getTransformedCorners(shape,H):
    h,w = shape[:2]
    corners = [[0, 0], [w, 0], [w, h], [0, h]] # get list of image corners
    projection = transformPoints(corners, H) # get perspective of corners with transformation matrix
    return projection.reshape(-1,2) # return projection points

def pads(shape1,shape2,H):
    h,w = shape2[:2] # get hight,width of image
    bx = [[0,0],[w,h]] # make box
    corners = getTransformedCorners(shape1,H) # get corners from image
    return boxpads(bx,corners)

def superpose(im1,im2,H):
    # im1 on top of im2
    # im1(x,y)*H = im1(u,v) -> im1(u,v) + im2(u,v)
    [(left,top),(right,bottom)] = pads(im1.shape,im2.shape,H)
    moveH = np.float64([[1,0,left],[0,1,top],[0,0,1]])
    movedH = moveH.dot(H)
    # need: top_left, bottom_left, top_right,bottom_right
    h2,w2 = im2.shape
    w,h = int(left + right + w2),int(top + bottom + h2)
    back = cv2.warpPerspective(im2,moveH,(w,h))
    fore = cv2.warpPerspective(im1,movedH,(w,h))
    alfa = cv2.warpPerspective(np.ones(im1.shape[:2]),movedH,(w,h))
    im = ar.overlay(back, fore, alfa)
    return im,movedH

@cache.memoize(paths.TEMPPATH) # convert cv2.bilateralfilter to memoized bilateral filter
def bilateralFilter(im,d,sigmaColor,sigmaSpace):
    return cv2.bilateralFilter(im,d,sigmaColor,sigmaSpace)

def asif_demo(**opts):

    flag_filter_scaled = opts.get("flag_filter_scaled",False)
    flag_filter_original = opts.get("flag_filter_original",False)
    flag_filter_out = opts.get("flag_filter_out",False)

    flag_invertH = opts.get("flag_invertH",False)

    flag_show_match = opts.get("flag_show_match",True)
    flag_show_result = opts.get("flag_show_result",True)

    flag_save_perspective = opts.get("flag_save_perspective",False)
    flag_save_result = opts.get("flag_save_result",False)

    #### LOADING
    #feature_name = opts.get('--feature', 'sift-flann') #default is 'sift-flann'
    #detector, matcher = init_feature(feature_name)
    original_fore = opts.get("original_fore",None)
    scaled_fore = opts.get("scaled_fore",None)

    try: fn1 = opts["fn1"]
    except:
        fn1 = paths.TESTPATH+'im1_2.jpg' # foreground is placed to background

    if not original_fore:
        original_fore = cv2.imread(fn1) # foreground
        print(fn1, " Loaded...")

    #### SCALING
    rzyf,rzxf = opts.get("fore_scale",(400,400)) # dimensions to scale foreground
    if not scaled_fore:
        scaled_fore = cv2.resize(cv2.imread(fn1, 0), (rzxf, rzyf))

    original_back = opts.get("original_back",None)
    scaled_back = opts.get("scaled_back",None)

    try: fn2 = opts["fn2"]
    except:
        fn2 = paths.TESTPATH+'im1_1.jpg' # background

    if not original_back:
        original_back = cv2.imread(fn2) # background
        print(fn2, " Loaded...")

    #### SCALING
    rzyb,rzxb = opts.get("back_scale",(400,400)) # dimensions to scale background
    if not scaled_back:
        scaled_back = cv2.resize(cv2.imread(fn2, 0), (rzxb, rzyb))

    #### PRE-PROCESSING
    if flag_filter_scaled:  # persistent by @root.memoize
        d,sigmaColor,sigmaSpace = 50,100,100
        scaled_fore = bilateralFilter(scaled_fore,d,sigmaColor,sigmaSpace)
        scaled_back = bilateralFilter(scaled_back,d,sigmaColor,sigmaSpace)
        print("merged image filtered with bilateral filter d={},sigmaColor={},sigmaSpace={}".format(d,sigmaColor,sigmaSpace))
    if flag_filter_original:  # persistent by @root.memoize
        d,sigmaColor,sigmaSpace = 50,100,100
        original_fore = bilateralFilter(original_fore,d,sigmaColor,sigmaSpace)
        original_back = bilateralFilter(original_back,d,sigmaColor,sigmaSpace)
        print("merged image filtered with bilateral filter d={},sigmaColor={},sigmaSpace={}".format(d,sigmaColor,sigmaSpace))

    #### FEATURE DETECTOR  # persistent by @root.memoize
    print("finding keypoints with its descriptos...")
    result = multipleASIFT([scaled_fore, scaled_back]) # OR use ASIFT for each image
    #kp1,desc1 = ASIFT(feature_name, scaled_fore, mask=None, pool=pool)
    #kp2,desc2 = ASIFT(feature_name, scaled_back, mask=None, pool=pool)
    #### MATCHING  # persistent by @root.memoize
    print("matching...")
    H, status, kp_pairs = multipleMATCH(result)[0] # OR use MATCH
    #H, status, kp_pairs = MATCH(feature_name,kp1,desc1,kp2,desc2)

    if H is not None:
        if flag_invertH:
            kp_pairs = [(j,i) for i,j in kp_pairs]
            H = convert.invertH(H)
            tmp1,tmp2,tmp3,tmp4 = original_fore,scaled_fore,original_back,scaled_back
            original_fore,scaled_fore,original_back,scaled_back = tmp3,tmp4,tmp1,tmp2

        shapes = original_fore.shape,scaled_fore.shape,original_back.shape,scaled_back.shape
        H2 = convert.sh2oh(H, *shapes) #### sTM to oTM

        if flag_show_match: # show matching
            win = 'matching result'
            kp_pairs2 = spairs2opairs(kp_pairs,*shapes)
            print("waiting to close match explorer...")
            vis = matchExplorer(win, original_fore, original_back, kp_pairs2, status, H2)
            #vis = MatchExplorer(win, scaled_fore, scaled_back, kp_pairs, status, H)

        # get perspective from the scaled to original Transformation matrix
        bgra_fore = cv2.cvtColor(original_fore,cv2.COLOR_BGR2BGRA) # convert BGR to BGRA
        fore_in_back = cv2.warpPerspective(bgra_fore,H2,(original_back.shape[1],original_back.shape[0])) # get perspective
        foregray = cv2.cvtColor(fore_in_back,cv2.COLOR_BGRA2GRAY).astype(float) # convert formats to float
        fore_in_back = fore_in_back.astype(float) # convert to float to make operations
        saveas = "perspective.png"
        if flag_save_perspective:
            cv2.imwrite(saveas,fore_in_back) # save perspective
            print("perspective saved as: "+saveas)
        # find alfa and do overlay
        alfa = fore_in_back[:,:,3].copy()
        for i in range(1): # testing damage by iteration
            backgray = cv2.cvtColor(original_back.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(float)
            fore_in_back[:,:,3]= getalfa(foregray,backgray,alfa) #### GET ALFA MASK
            original_back = ar.overlay(original_back, fore_in_back) #### MERGING
        original_back = original_back.astype(np.uint8) # convert back to uint8
        #### POS-PROCESSING
        if flag_filter_out: # filter  # persistent by @root.memoize
            # http://docs.opencv.org/modules/imgproc/doc/filtering.html
            d,sigmaColor,sigmaSpace =50,100,100 # best guess: (50,100,10), opencv: (9,75,75), d=-1 is filter distance until sigma
            original_back = bilateralFilter(original_back,d,sigmaColor,sigmaSpace)
            saveas = "merged_bilateralfilter_d_{}_sigmaColor_{}_sigmaSapace_{}.png".format(d,sigmaColor,sigmaSpace)
            title = "bilateral filtered d={},sigmaColor={},sigmaSpace={}".format(d,sigmaColor,sigmaSpace)
        else:
            saveas = "merged_nofilter.png"
            title = "merged image"
        print("image merged...")
        if flag_show_result: # plot result
            plt = plotter.plt
            plt.imshow(cv2.cvtColor(original_back,cv2.COLOR_BGR2RGB))
            plt.title(title), plt.xticks([]), plt.yticks([])
            plt.show()
        if flag_save_result:
            cv2.imwrite(saveas,original_back) # save result
            print("result saved as: "+saveas)
        print("process finished... ")

def asif_demo2(args=None):

    #### LOADING
    #feature_name = opts.get('--feature', 'sift-flann') #default is 'sift-flann'
    #detector, matcher = init_feature(feature_name)
    try: fn1, fn2 = args
    except:
        fn1 = paths.TESTPATH+'im1_2.jpg' # foreground is placed to background
        fn2 = paths.TESTPATH+'im1_1.jpg' # foreground is placed to background


    def check(im, fn):
        if im is not None:
            print(fn, " Loaded...")
        else:
            print(fn, " could not be loaded...")

    #original_fore = cv2.imread(fn1) # foreground
    #original_back = cv2.imread(fn2) # background
    #checkLoaded(original_fore, fn1)
    #checkLoaded(original_back, fn2)

    #### SCALING
    rzyf,rzxf = 400,400 # dimensions to scale foreground
    rzyb,rzxb = 400,400 # dimensions to scale background
    scaled_fore = cv2.resize(cv2.imread(fn1, 0), (rzxf, rzyf))
    scaled_back = cv2.resize(cv2.imread(fn2, 0), (rzxb, rzyb))

    check(scaled_fore, fn1)
    check(scaled_back, fn2)

    #### FEATURE DETECTOR  # persistent by @root.memoize
    print("finding keypoints with its descriptos...")
    result = multipleASIFT([scaled_fore, scaled_back]) # OR use ASIFT for each image
    #kp1,desc1 = ASIFT(feature_name, scaled_fore, mask=None, pool=pool)
    #kp2,desc2 = ASIFT(feature_name, scaled_back, mask=None, pool=pool)
    #### MATCHING  # persistent by @root.memoize
    print("matching...")
    H, status, kp_pairs = multipleMATCH(result)[0] # OR use MATCH
    #H, status, kp_pairs = MATCH(feature_name,kp1,desc1,kp2,desc2)

    if H is not None:
        from multiprocessing import Process
        #shapes = original_fore.shape,scaled_fore.shape,original_back.shape,scaled_back.shape
        #H2 = sh2oh(H,*shapes) #### sTM to oTM
        #kp_pairs2 = spairs2opairs(kp_pairs,*shapes)
        print("waiting to close match explorer...")
        win = "stitch"
        p = Process(target=superposeGraph,args= (win, scaled_fore, scaled_back, H))
        p.start()
        win = "inverted stitch"
        p2 = Process(target=superposeGraph,args= (win, scaled_back, scaled_fore, invertH(H)))
        p2.start()
        win = 'matching result'
        vis = explore_match(win, scaled_fore, scaled_back, kp_pairs, status, H)
        p.join()

def stich():
    from multiprocessing import Process
    from glob import glob
    from RRtool.RRtoolbox import imloader
    #### LOADING
    print("looking in path {}".format(paths.TESTPATH))
    fns = glob(paths.TESTPATH + "*.jpg")
    fns = fns[:3]
    print("found {} filtered files...".format(len(fns)))
    #### SCALING
    rzyf,rzxf = 400,400 # dimensions to scale foregrounds
    #ims = [cv2.resize(cv2.imread(i, 0), (rzxf, rzyf)) for i in fns] # normal list
    ims = imloader(fns,0, (rzxf, rzyf)) # load just when needed
    #img = [i for i in ims] # tests
    #ims = imloader(fns,0, (rzxf, rzyf),mmap=True,mpath=paths.TEMPPATH) # load just when needed
    #img = [i for i in ims] # tests
    #ims = [numpymapper(data, str(changedir(fns[i],paths.TEMPPATH))) for i,data in enumerate(imloader(fns))] # Too slow
    #nfns = [changedir(i,paths.TEMPPATH) for i in fns] # this get the temp files
    #### FEATURE DETECTOR  # persistent by @root.memoize

    print("finding keypoints with its descriptors...")
    descriptors = multipleASIFT(ims) # OR use ASIFT for each image
    print("total descriptors {}".format(len(descriptors)))
    #### MATCHING
    # H, status, kp_pairs
    threads,counter = [],0
    print("matching...")
    for i in range(len(descriptors)):
        for j in range(len(descriptors)):
            if j>i: # do not test itself and inverted tests
                counter +=1
                print("comparision No.{}".format(counter))
                # FIXME inefficient code ... just 44 descriptors generate 946 Homographies
                fore,back = ims[i], ims[j]
                (kp1,desc1),(kp2,desc2) = descriptors[i],descriptors[j]
                H, status, kp_pairs = MATCH(feature_name,kp1,desc1,kp2,desc2)
                inlines,lines = np.sum(status), len(status)
                pro = old_div(float(inlines),lines)
                test = pro>0.5 # do test to see if both match
                win = '{0}({2}) - {1}({3}) inliers({4})/matched({5}) rate({6}) pass({7})'.format(i,j,len(kp1),len(kp2), inlines,lines,pro,test)
                d = Process(target=explore_match,args = (win, fore, back, kp_pairs, status, H))
                d.start()
                threads.append(d)
                if test:
                    pass
    for t in threads:
        t.join()

if __name__ == "__main__":

    #asif_demo()
    asif_demo2()
    #stich()