"""
runs the rectangularity ratio of a projection
"""
from preamble import *
script_name = os.path.basename(__file__).split(".")[0]
from RRtoolbox.lib.plotter import plotPointsContour
from RRtoolbox.lib.image import imcoors

def graph(pytex =None, name = script_name, points = (10,10), erratic = False,width=0.002, draw = False):
    """
    :param pytex:
    :param name:
    :param points:
    :param erratic:
    :param width:
    :param draw:
    :return:
    """
    # based on implementation http://stackoverflow.com/a/33047048/5288758
    gd = graph_data(pytex)
    rcParams['text.latex.unicode'] = True
    points = np.array(points)
    if points.ndim == 1 or draw: # generate points
        points = points_generator(points,draw=draw,nopoints=4, erratic= erratic)
    c = imcoors(points)
    box = c.rotatedBox
    figure()
    plotPointsContour(box, lcor="b", deg=True, annotate=False,width=width)
    hold(True)
    plotPointsContour(points, deg=True,width=width)
    hold(False)
    title("Rectangularity: {}".format(c.rectangularity))
    xylim_points(box,labellen=20)
    #show()
    gd.output(name)
    return locals()

if __name__ == "__main__":
    graph_data.shows = True
    graph_data.saves = False
    graph()