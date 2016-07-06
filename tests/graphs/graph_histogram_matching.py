"""
runs a histogram matching example based from
"""
# broader example at tests/Ex_histogram_matching.py
from preamble import *
from RRtoolbox.lib.image import hist_match
script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, fn = None):
    """
    :param pytex:
    :param name:
    :param fn:
    :return:
    """
    # based on implementation http://stackoverflow.com/a/33047048/5288758
    gd = graph_data(pytex)
    if fn is None:
        fn = "im1_1.jpg"
    gd.load(fn)
    source = cv2.cvtColor(gd.gray,cv2.COLOR_GRAY2RGB)
    template = gd.RGB

    matched = hist_match(source, template)

    def ecdf(x):
        """convenience function for computing the empirical CDF"""
        vals, counts = np.unique(x, return_counts=True)
        ecdf = np.cumsum(counts).astype(np.float64)
        ecdf /= ecdf[-1]
        return vals, ecdf

    x1, y1 = ecdf(source.ravel())
    x2, y2 = ecdf(template.ravel())
    x3, y3 = ecdf(matched.ravel())

    fig = figure()
    gs = GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, :])
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    n_source,n_template,n_match = 'Source','Template','Matched'
    ax1.imshow(source, cmap=cm.gray)
    ax1.set_title(n_source)
    ax2.imshow(template, cmap=cm.gray)
    ax2.set_title(n_template)
    ax3.imshow(matched, cmap=cm.gray)
    ax3.set_title(n_match)

    multi = 1
    lw=3
    ax4.plot(x1, y1 * multi, '-r', lw=lw, label=n_source)
    ax4.plot(x2, y2 * multi, '-k', lw=lw, label=n_template)
    ax4.plot(x3, y3 * multi, '--r', lw=lw, label=n_match)
    ax4.set_xlim(x1[0], x1[-1])
    ax4.set_xlabel('Pixel Levels')
    ax4.set_ylabel('Normalized CDF')
    ax4.legend(loc = 'upper left')

    #show()
    gd.output(name)
    return locals()

if __name__ == "__main__":
    graph()