"""
runs a convexity ratio example
"""
from preamble import *
script_name = os.path.basename(__file__).split(".")[0]
from RRtoolbox.lib.arrayops import convexityRatio, contour2points
from RRtoolbox.lib.plotter import plotPointsContour

def graph(pytex =None, name = script_name, points = (10,10), convex = True, width=0.002, draw = False):
    """
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param points: list of points or shape to place points eg. (height,width,nopoints),
            (height,width) with nopoints the minimum value of shape, [(x0,y0),..,(xN,yN)].
    :param convex: True to place convex points in case a shape is given, False to
            place un-convex points.
    :param width: line width. by default it is 0.002
    :param draw: True to draw the points, False to use shape with random points.
    :return: locals()
    """
    # based on implementation http://stackoverflow.com/a/33047048/5288758
    gd = graph_data(pytex)
    points = np.array(points)
    if points.ndim == 1 or draw: # generate points
        points = points_generator(points,draw=draw,convex=convex)
    hull = cv2.convexHull(points)
    figure()
    plotPointsContour(contour2points(hull), lcor="b", deg=True, annotate=False,width=width)
    hold(True)
    plotPointsContour(points, deg=True, lcor="gray",width=width)
    hold(False)
    title("ConvexityRatio: {}".format(convexityRatio(points,hull)))
    #xylim_points(box,labellen=20)
    #show()
    gd.output(name)
    return locals()

if __name__ == "__main__":
    graph(pytex.context)