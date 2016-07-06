"""
graph an overlay example
"""
from preamble import *
from RRtoolbox.lib.arrayops import overlay
script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, fore ="asift2fore.png", back ="im1_1.jpg", alpha = .5):
    """
    :param pytex:
    :param name:
    :param fore:
    :param back:
    :param alpha:
    :return:
    """
    gd = graph_data(pytex)

    figure()#figsize=(mm2inch(163,45))) # 163, 45 mm

    gd.load(fore)
    f = gd.RGB
    fa = gd.RGBA
    gd.load(back)
    b = gd.RGB
    ba = gd.RGBA

    subplot(131),imshow(ba)
    title(gd.wrap_title('background'))
    xticks([]), yticks([])

    subplot(132),imshow(fa)
    title(gd.wrap_title('foreground'))
    xticks([]), yticks([])

    subplot(133),imshow(overlay(b,fa,alpha=fa[:,:,3]*alpha))
    title(gd.wrap_title('Overlay: alpha = {}'.format(alpha)))
    xticks([]), yticks([])

    gd.output(name)
    return locals()

if __name__ == "__main__":
    graph()