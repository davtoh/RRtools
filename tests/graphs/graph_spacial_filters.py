"""
graph gauss, median and bilateral spacial filter comparisons
"""
from preamble import *
script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, core = 21, useTitle = True, split = False):
    """
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param core: kernel size
    :param useTitle: if True add titles to the figures.
    :param split: split images with subscripts 0, 1, 2 and 3 as:
            "{name}_{subscript}"
    :return: locals()
    """
    gd = graph_data(pytex)
    gauss = cv2.GaussianBlur(gd.RGB, (core, core), 0)
    median = cv2.medianBlur(gd.RGB, core)
    params = core,82,57 # # 21,75,75
    bilateral = cv2.bilateralFilter(gd.RGB, *params)

    figure()#figsize=(mm2inch(163,45))) # 163, 45 mm
    if not split: subplot(141)
    imshow(gd.RGB)
    t = 'Original'
    if useTitle: title(gd.wrap_title(t))
    xticks([]), yticks([])
    if split:
        gd.output(name+"_0", caption=t)
        figure()

    if not split: subplot(142)
    imshow(gauss)
    t = 'Gauss: kernel = {}'.format((core,core))
    if useTitle: title(gd.wrap_title(t))
    xticks([]), yticks([])
    if split:
        gd.output(name+"_1",caption=t)
        figure()

    if not split: subplot(143)
    imshow(median)
    t = 'Median: ksize = {}'.format(core)
    if useTitle: title(gd.wrap_title(t))
    xticks([]), yticks([])
    if split:
        gd.output(name+"_2",caption=t)
        figure()

    if not split: subplot(144)
    imshow(bilateral)
    t = 'Bilateral: d = {},\nsigmaC = {}, sigmaS = {}'.format(*params)
    if useTitle: title(gd.wrap_title(t))
    xticks([]), yticks([])
    if split:
        gd.output(name+"_3",caption=t)

    if not split: gd.output(name)

    return locals()

if __name__ == "__main__":
    graph(core = 21, useTitle= False, split = True)