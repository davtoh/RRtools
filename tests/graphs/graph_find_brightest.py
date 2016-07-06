from RRtoolbox.lib.arrayops import brightness, overlay
from RRtoolbox.lib.directory import getData
from RRtoolbox.tools.segmentation import retina_markers_thresh, find_optic_disc
from preamble import *
from tests.tesisfunctions import graphmath

script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, methods =  None, fn = None):
    """
    :param pytex:
    :param name:
    :param methods:
    :param fn:
    :return:
    """
    gd = graph_data(pytex)
    if fn is None:
        fn = "im1_2.jpg"
    gd.load(fn)

    im_name = scape_string(getData(fn)[-2])
    fore = gd.RGB
    # get intensity
    P = brightness(fore)
    #P = (255-P)

    # create color pallet: unknown, background, body, brightest, boundaries
    pallet = np.array([[255,0,0],[0,0,0],[0,0,255],[255,255,255],[255,0,255]],np.uint8)

    lines,comments = [],[]

    figure()
    title("Marker Thresholds")
    # calculate histogram
    hist_P, bins = np.histogram(P.flatten(),256,[0,256])
    lines.extend(graphmath(hist_P,show=False)[1]) # plots original histogram

    data_min,data_body_left,data_body,data_max_left = retina_markers_thresh(P)
    x = bins[:-1] #np.indices(hist_P.shape) #np.arange(len(hist_PS)) # get indexes
    annotate = [(x[data_min], hist_P[data_min], "Ends\nDarkest\nAreas"),
                (x[data_body_left], hist_P[data_body_left],"Begins\nRetina"),
                (x[data_body], hist_P[data_body],"Ends\nRetina"),
                (x[data_max_left], hist_P[data_max_left],"Begins\nBrightest\nareas"),
    ]
    arrowprops = dict(facecolor='black', shrink=0.05)

    ax = gca() # get current axes
    for xp,yp,ann in annotate:
        ax.annotate(ann.title(), xy=(xp,yp),
                    textcoords='data', xytext=(xp,yp+np.max(hist_P)*0.07),
                    arrowprops=arrowprops, fontsize=gd.context_data["size"])
        plot(xp,yp, "o", label=ann)

    xlabel("Levels")
    ylabel("No. pixels")
    xlim([0,255])
    gd.output(name+"_histAnalysis")
    #show()

    fig = figure()

    #t = suptitle("Find Flares and Optic disk", fontsize= gd.context_data["size"]*1.5)
    #t.set_y(0.8)

    fore = cv2.cvtColor(P,cv2.COLOR_GRAY2BGR)
    optic_disc, Crs, markers, water = find_optic_disc(fore,P)

    subplot(131), imshow(overlay(fore.copy(), pallet[markers], alpha=0.5))
    title(im_name +" markers"), xticks([]), yticks([])

    subplot(132), imshow(overlay(fore.copy(), pallet[water], alpha=0.3))
    title(im_name +" all segmentations"), xticks([]), yticks([])

    subplot(133), imshow(overlay(fore.copy(), optic_disc*255, alpha=0.3))
    title(im_name +" optic disc"), xticks([]), yticks([])

    fig.tight_layout()
    #fig.subplots_adjust(top=0.8)
    gd.output(name+"_opticDisk")
    #show()
    return locals()

if __name__ == "__main__":
    graph_data.shows = True
    graph_data.saves = False
    graph()