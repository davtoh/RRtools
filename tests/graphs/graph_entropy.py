"""
runs an entropy example
"""
from glob import glob

from RRtoolbox.lib.directory import getData
from RRtoolbox.tools.selectors import entropy
from preamble import *

script_name = os.path.basename(__file__).split(".")[0]


def graph(pytex =None, name = script_name, path = None):
    """
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param path: path to images to test entropy
    :return: locals()
    """
    gd = graph_data(pytex)
    if path is None:
        path = root_path+"/images_{}/*".format(script_name)
    fns = glob(path)
    if len(fns) < 2:
        raise Exception("not enough images")
    gs = generateGrid(len(fns),len(fns))

    ## raw images
    fig = figure()
    for i,k in enumerate(fns):
        kd = getData(k)
        gd.load(k)
        fig.add_subplot(gs[i])
        imshow(gd.RGB)
        title(scape_string(r'{}'.format(kd[-2])))
        xticks([]), yticks([])
    gd.output("{}_raw".format(name))

    ## comparison from entropy
    comparison = zip(*entropy(fns,invert=False)[:2])
    fig = figure()
    for i,(v,k) in enumerate(comparison):
        kd = getData(k)
        gd.load(k)
        fig.add_subplot(gs[i])
        imshow(gd.RGB)
        title(scape_string(r'{} {}'.format(kd[-2],str(v)[:5])))
        xticks([]), yticks([])
    #show()
    gd.output(name)
    return locals()

if __name__ == "__main__":
    graph_data.shows = True
    graph_data.saves = False
    graph()