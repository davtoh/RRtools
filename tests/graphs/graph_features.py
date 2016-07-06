"""
runs feature detector in images and shows the keypoints
"""
from preamble import *
from RRtoolbox.lib.descriptors import init_feature
script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, fn = None, feature_name = "sift", flags=4, color=None):
    """
    :param pytex:
    :param name:
    :param fn:
    :param feature_name:
    :param flags:
    :param color:
    :return:
    """
    gd = graph_data(pytex)
    detector, matcher = init_feature(feature_name)
    kp, des = detector.detectAndCompute(gd.gray,None)
    img=cv2.drawKeypoints(gd.gray,kp,flags=flags, color= color)
    figure()
    imshow(img)
    title(scape_string(r'{}'.format(feature_name.title())))
    xticks([]), yticks([])
    gd.output(name)
    return locals()

if __name__ == "__main__":
    graph_data.shows = True
    graph_data.saves = False
    graph()