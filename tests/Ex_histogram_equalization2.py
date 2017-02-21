
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
# http://opencvpython.blogspot.com.co/2013/03/histograms-2-histogram-equalization.html
# http://stackoverflow.com/a/31493356/5288758
import numpy as np

def hist_norm(x, bin_edges, quantiles, inplace=False):
    """
    Linearly transforms the histogram of an image such that the pixel values
    specified in `bin_edges` are mapped to the corresponding set of `quantiles`

    Arguments:
    -----------
        x: np.ndarray
            Input image; the histogram is computed over the flattened array
        bin_edges: array-like
            Pixel values; must be monotonically increasing
        quantiles: array-like
            Corresponding quantiles between 0 and 1. Must have same length as
            bin_edges, and must be monotonically increasing
        inplace: bool
            If True, x is modified in place (faster/more memory-efficient)

    Returns:
    -----------
        x_normed: np.ndarray
            The normalized array
    """

    bin_edges = np.atleast_1d(bin_edges)
    quantiles = np.atleast_1d(quantiles)

    if bin_edges.shape[0] != quantiles.shape[0]:
        raise ValueError('# bin edges does not match number of quantiles')

    if not inplace:
        x = x.copy()
    oldshape = x.shape
    pix = x.ravel()

    # get the set of unique pixel values, the corresponding indices for each
    # unique value, and the counts for each unique value
    pix_vals, bin_idx, counts = np.unique(pix, return_inverse=True,
                                          return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution function (which maps pixel
    # values to quantiles)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]

    # get the current pixel value corresponding to each quantile
    curr_edges = pix_vals[ecdf.searchsorted(quantiles)]

    # how much do we need to add/subtract to map the current values to the
    # desired values for each quantile?
    diff = bin_edges - curr_edges

    # interpolate linearly across the bin edges to get the delta for each pixel
    # value within each bin
    pix_delta = np.interp(pix_vals, curr_edges, diff)

    # add these deltas to the corresponding pixel values
    pix = pix+pix_delta[bin_idx]

    return pix.reshape(oldshape)

from scipy.misc import lena
import cv2

bin_edges = 0, 55, 200, 255
quantiles = 0, 0.2, 0.5, 1.0

fn1 = r'im5_1.jpg'
fn1 = r'im1_1.jpg'
fn1 = r'im3_1.jpg'
# read image
#img = lena()
img = cv2.resize(cv2.imread(fn1,0),(300,300)) # resize image
normed = hist_norm(img, bin_edges, quantiles)


from matplotlib import pyplot as plt

def ecdf(x):
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

x1, y1 = ecdf(img.ravel())
x2, y2 = ecdf(normed.ravel())

fig = plt.figure()
gs = plt.GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[1, :])
for aa in (ax1, ax2):
    aa.set_axis_off()

ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Original')
ax2.imshow(normed, cmap=plt.cm.gray)
ax2.set_title('Normalised')

ax3.plot(x1, y1 * 100, lw=2, label='Original')
ax3.plot(x2, y2 * 100, lw=2, label='Normalised')
for xx in bin_edges:
    ax3.axvline(xx, ls='--', c='k')
for yy in quantiles:
    ax3.axhline(yy * 100., ls='--', c='k')
ax3.set_xlim(bin_edges[0], bin_edges[-1])
ax3.set_xlabel('Pixel value')
ax3.set_ylabel('Cumulative %')
ax3.legend(loc=2)

plt.show()
