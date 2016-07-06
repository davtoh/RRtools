"""
graph comparison of selection of parameters for bilateral filter
"""
from __future__ import division
from preamble import *
from tests.tesisfunctions import graph_filter
from RRtoolbox.lib.arrayops import getBilateralParameters
script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, maxshape = 2000, useTitle = True):
    """
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param maxshape: maximum shape of any image passed to the bilateral filter
    :param useTitle: if True add titles to the figures.
    :return locals()
    """
    gd = graph_data(pytex)
    filters = getBilateralParameters.filters # get filters

    # graph filters' response
    xl = maxshape# np.max(shapes) #np.min(base)+increment*num
    fig = figure()
    graph_filter(filters,single=True,legend=True,
                 annotate=False,show=False,win=fig,
                 levels=np.linspace(0, xl,xl+1))
    ylim([0,100])
    ylabel("Parameter")
    xlabel("min(shape)")
    t = "Bilateral filter response"
    if useTitle:
        title(t)
    else:
        title("") # delete previous title
    gd.output(name,caption=t)
    return locals()

if __name__ == "__main__":
    graph(pytex.context, maxshape = 2000, useTitle = True)

