"""
runs a convexity ratio example
"""
from preamble import *
script_name = os.path.basename(__file__).split(".")[0]
from RRtoolbox.lib.arrayops import polygonArea
from RRtoolbox.lib.plotter import plotPointsContour

def graph(pytex =None, name = script_name, points = (50,50,7), convex = True, width=0.002, draw = False):
    """
    :param pytex:
    :param name:
    :param points:
    :param convex:
    :param width:
    :param draw:
    :return:
    """
    # based on implementation http://stackoverflow.com/a/33047048/5288758
    gd = graph_data(pytex)
    points = np.array(points)
    if points.ndim == 1 or draw: # generate points
        points = points_generator(points,draw=draw, convex=convex)
    figure()
    plotPointsContour(points, deg=True, lcor="gray",width=width)
    title("polygonArea = {}".format(polygonArea(points)))
    #xylim_points(box,labellen=20)
    #show()
    gd.output(name)
    return locals()

if __name__ == "__main__":
    graph()