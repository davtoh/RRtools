"""
runs a matrix Singular Value Decomposition example result
"""
# broader example at tests/equations/test_decomposition_image.py
from preamble import *
script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, fn = None, least = None, most = None, fill = 0, useTitle = True, split = False):
    '''
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param fn: (None) file name of figure to load
    :param least: slice of form s[least:] = fill
    :param most: slice of form s[:most] = fill
    :param fill: fill values in s (singular values for every matrix)
    :param useTitle: if True add titles to the figures.
    :param split: split images with subscripts 0, 1 and 2 as:
            "{name}_{subscript}"
    :return: locals()

    U,s,V = Unitary matrices, singular values, Unitary matrices
    notes:
        * s.shape = min(img.shape)
        * if s[50:]= 0 for shape (200,200) then for shape (M,N) is almost the same
        * fill can be from - inf to + inf whereas abs(fill) proportional to noise
    # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.svd.html
    '''
    if least is None:
        least = (50,"")
    elif isinstance(least,int):
        least = (least,"")

    if most is None:
        most = ("",1)
    elif isinstance(most,int):
        most = (most,"")

    gd = graph_data(pytex)
    if fn is not None: gd.load(fn)

    U, s, V = np.linalg.svd(gd.gray, full_matrices=False)

    def applyOp(slice, fill=0):
        s_ = s.copy()
        slice = ":".join([str(i) for i in slice])
        try:
            exec "s_[{}] = fill".format(slice) in locals()
        except: # generalize error
            raise Exception("slice is of the form (from,to,step) to use as from:to:step")
        im = np.abs(np.dot(U, np.dot(np.diag(s_), V))).astype(np.uint8) # reduced SVD
        return dict(fill = fill, slice = slice, min=np.min(im),max=np.max(im),s=s_,im=im)

    figure()#figsize=(mm2inch(500,150)))
    if not split: subplot(131)
    imshow(gd.gray, "gray")
    t = "Normal: s.shape = {}".format(s.shape)
    if useTitle: title(gd.wrap_title(t))
    xticks([]), yticks([])
    if split:
        gd.output(name+"_0",caption=t)
        figure()

    title_str = "Recomposed svd with s[{slice}]={fill}"

    data = applyOp(least,fill=fill) # eliminate least significands
    if not split: subplot(132)
    imshow(data["im"],"gray")
    t = title_str.format(**data)
    if useTitle: title(gd.wrap_title(t))
    xticks([]), yticks([])
    if split:
        gd.output(name+"_1",caption=t)
        figure()

    data = applyOp(most,fill=fill) # eliminates most significand
    if not split: subplot(133)
    imshow(data["im"],"gray")
    t = title_str.format(**data)
    if useTitle: title(gd.wrap_title(t))
    xticks([]), yticks([])
    if split:
        gd.output(name+"_2",caption=t)

    if not split: gd.output(name)
    return locals()

if __name__ == "__main__":
    graph(pytex={"shape":"200/400"},fn="im1_2.jpg",fill=1, split=True, useTitle=False) # "im5_1.jpg"