"""
runs a histogram comparison with any or all OpenCV methods
"""
from glob import glob

from RRtoolbox.lib.directory import getData
from RRtoolbox.tools.selectors import hist_map,hist_comp
from preamble import *

script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, methods =  None, path = None, putTitle = False):
    """
    :param pytex:
    :param name:
    :param methods:
    :param path:
    :param putTitle:
    :return:
    """
    gd = graph_data(pytex)
    if path is None:
        path = root_path+"/images_{}/*".format(script_name)
    fns = glob(path)
    if len(fns) < 2:
        raise Exception("not enough images")
    if methods is None:
        methods = hist_map.keys()
    elif isinstance(methods,basestring):
        methods = [methods]
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

    ## comparison methods
    for m in methods:
        comparison = hist_comp(fns,method = m)
        fig = figure()
        if putTitle:
            val = 0.67
            if not isinstance(putTitle,bool): # sets user input
                val = putTitle
            suptitle(m.title(), fontsize =gd.context_data["size"] * 1.5).set_y(val)
        for i,(v,k) in enumerate(comparison):
            kd = getData(k)
            gd.load(k)
            fig.add_subplot(gs[i])
            imshow(gd.RGB)
            title(scape_string(r'{} {}'.format(kd[-2],str(v)[:5])))
            xticks([]), yticks([])
        #show()
        gd.output("{}_{}".format(name,m))

    return locals()

if __name__ == "__main__":
    graph()