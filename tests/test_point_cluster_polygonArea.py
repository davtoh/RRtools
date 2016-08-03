# -*- coding: utf-8 -*-
# http://stackoverflow.com/questions/5147112/matplotlib-how-to-put-individual-tags-for-a-scatter-plot
# http://stackoverflow.com/questions/22272081/label-python-data-points-on-plot
import pylab as plt
import numpy as np
import cv2
#from gtk import set_interactive
#set_interactive(0)
from RRtoolbox.lib.plotter import Plotim, plotPointsContour
from RRtoolbox.lib.image import getcoors, drawcoorperspective,quadrants, Imcoors
from RRtoolbox.lib.arrayops.basic import transformPoints, relativeVectors, vertexesAngles,\
    points2mask, polygonArea, relativeQuadrants, random_points
from RRtoolbox.lib.arrayops.convert import points2contour,contour2points, toTupple, translateQuadrants
from Equations.Eq_HomogeniousTransform import HZrotate, applyTransformations,getApplyCenter
random = np.random.random

def tests_groundTrue(funcs):
    """
    Ground True test for function that receives points and returns area.

    :param funcs: list of functions
    :return: None, raise an assertion error if test is not passed
    """
    if callable(funcs): funcs = (funcs,) # it is just one function
    pts60 = np.array([[-3,-2],[-1,4],[6,1],[3,10],[-4,9],[-3,-2]]) # expected area 60
    pts100 = np.array([[-5,5],[5,5],[5,-5],[-5,-5]]) # expected area 100
    for key,val in locals().iteritems():
        if key.startswith("pts"):
            ans = int(key[3:])
            for func in funcs:
                assert np.allclose(func(val),ans) # fifth decimal precision

def test_random(funcs, n_tests=1, n_points = 4, axes_range = ((-50, 50),), tolerance = 0.1, show = True, points = None):
    """
    Test and compare several area functions on the fly.

    :param funcs: list of functions to test.
    :param n_tests: number of tests.
    :param n_points: number of points per test.
    :param axes_range: parameter in :func:`random_points` to generate points.
    :param tolerance: decimal precision from variance
    :return: None
    """
    if callable(funcs): funcs = (funcs,) # it is just one function
    for ntest in xrange(1,n_tests+1):
        text =  "test No {}".format(ntest)
        print text
        if points:
            pts = points.pop()
        else:
            pts = random_points(axes_range, n_points)
        print "points {}".format(pts)
        ans = [func(pts) for func in funcs]
        mean = np.mean(ans)
        for i,val in enumerate(np.abs(mean-ans)): # similar to np.allclose
            if val > tolerance:
                print "{} failed to be in the mean {} answering {} and deviation {}".format(funcs[i].func_name,mean,ans[i],val)
            else:
                print "{} passed mean {} answering {} and deviation {}".format(funcs[i].func_name,mean,ans[i],val)
        if show:
            Plotim("filled", points2mask(pts)).show(block=False)
            f = plt.figure()
            f.suptitle(text)
            plotPointsContour(pts)
            f.show()

def test_fill_poly():
    # http://stackoverflow.com/a/17582850/5288758
    ymax,xmax = 100,100
    img = np.zeros((ymax,xmax,3))
    cv2.fillConvexPoly(img,random_points([(0,xmax),(0,ymax)]).astype(np.int32),(200,200,200))
    Plotim("filled", img).show()


def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    # https://newtonexcelbach.wordpress.com/2014/03/01/the-angle-between-two-vectors-python-version/
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def getRelativeAngles(pts):
    vs = relativeVectors(pts)
    angles = []
    for i in xrange(1,len(vs)):
        angles.append(py_ang(vs[i-1], vs[i]))
    return angles

h,w = 10,10
pts = np.array([[0, 0], [w, 0], [w, h], [0, h]],np.float32) # get list of image corners
#pts = np.array([[-3,-2],[-1,4],[6,1],[3,10],[-4,9],[-3,-2]])
#pts = random_points([(-100, 100), (-100, 100)])
# perspective:  top_left, top_right, bottom_left, bottom_right
# corners and areas: top_left, top_right, bottom_right, bottom_left

#test_random([polyArea0,polyArea1,polyArea2], axes_range = ((0, 300),),points=[pts])

H = np.array([[random(),random(),random()], [random(),random(),random()], [random(),random(),random()]]) # impredictable transformation
transformations = [HZrotate(3.14*random())]
sM_rel_ = applyTransformations(transformations,False,getApplyCenter(w,h)) # apply relative transformations # symbolic transformation matrix
H = np.array(sM_rel_)[0:3,0:3].astype(np.float)#*H # array transformation matrix
#H = np.array([[1,1,1], [random(),1,1], [random(),random(),1]])
H = None

if H is not None:
    projections = transformPoints(pts, H) # get perspective of corners with transformation matrix
else:
    pts1 = getcoors(np.ones((h,w)),"get pixel coordinates", updatefunc=drawcoorperspective)
    pts2 = pts[[0,1,3,2]] # np.float32([[0,0],[w,0],[0,h],[w,h]]) # top_left,top_right,bottom_left,bottom_right
    if pts1:
        pts1 = np.float32(pts1)
    else:
        pts1 = pts2
    H = cv2.getPerspectiveTransform(pts1,pts2)
    projections = pts1[[0,1,3,2]]

def getQuadrantComparison(pts1,pts2):
    hashp = toTupple(pts2)
    mapping = dict(zip(hashp,pts1))
    return quadrants(pts1),[[mapping[j] for j in i] for i in toTupple(quadrants(pts2))]


plotPointsContour(pts, deg=True)
plt.hold(True)
plotPointsContour(projections, lcor="b", deg=True)
print (polygonArea(pts), polygonArea(projections))
q1,q2 = getQuadrantComparison(pts,projections)
print "points quadrants" # [Top_left,Top_right,Bottom_left,Bottom_right]
print q1
print translateQuadrants(relativeQuadrants(pts))
print "points quadrants after projection"
print q2
print translateQuadrants(relativeQuadrants(projections))
U, s, V = np.linalg.svd(pts, full_matrices=False)
pU, ps, pV = np.linalg.svd(projections, full_matrices=False)
pass # U, pU, ps, s, pV, V
plt.show()
pass