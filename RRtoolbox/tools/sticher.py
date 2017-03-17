# -*- coding: utf-8 -*-
# ----------------------------    IMPORTS    ---------------------------- #
# three-party
from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
import numpy as np
# custom
from ..lib.config import MANAGER
from ..lib.plotter import MatchExplorer
from ..lib.descriptors import ASIFT, MATCH, ASIFT_multiple


def stich(**opts):
    from multiprocessing import Process
    from glob import glob
    from ..lib.image import loadFunc, PathLoader, LoaderDict
    feature_name = opts.get("feature", 'sift-flann')
    # LOADING
    print("looking in path {}".format(MANAGER["TESTPATH"]))
    fns = glob(MANAGER["TESTPATH"] + "*.jpg")
    fns = fns[:3]
    print("found {} filtered files...".format(len(fns)))
    # SCALING
    rzyf, rzxf = 400, 400  # dimensions to scale foregrounds
    # ims = [cv2.resize(cv2.imread(i, 0), (rzxf, rzyf)) for i in fns] # normal
    # list
    loader = loadFunc(0, dsize=(rzxf, rzyf))
    ims_dict = LoaderDict(loader)
    for i, val in enumerate(fns):
        ims_dict[i] = val
    ims = PathLoader(fns, loader)  # load just when needed
    # img = [i for i in ims] # tests
    # ims = imloader(fns,0, (rzxf, rzyf),mmap=True,mpath=MANAGER.TEMPPATH) # load just when needed
    # img = [i for i in ims] # tests
    # ims = [numpymapper(data, str(changedir(fns[i],MANAGER.TEMPPATH))) for i,data in enumerate(imloader(fns))] # Too slow
    # nfns = [changedir(i,MANAGER.TEMPPATH) for i in fns] # this get the temp files
    # FEATURE DETECTOR  # persistent by @root.memoize

    print("finding keypoints with its descriptors...")
    # OR use ASIFT for each image
    descriptors = ASIFT_multiple(ims, feature_name)
    print("total descriptors {}".format(len(descriptors)))
    # MATCHING
    # H, status, kp_pairs
    threads, counter = [], 0
    print("matching...")
    with open("stitch values", "a+") as f:
        for i in range(len(descriptors)):
            for j in range(len(descriptors)):
                if j > i:  # do not test itself and inverted tests
                    counter += 1
                    print("comparision No.{}".format(counter))
                    # FIXME inefficient code ... just 44 descriptors generate
                    # 946 Homographies
                    fore, back = ims[i], ims[j]
                    (kp1, desc1), (kp2, desc2) = descriptors[i], descriptors[j]
                    H, status, kp_pairs = MATCH(
                        feature_name, kp1, desc1, kp2, desc2)
                    inlines, lines = np.sum(status), len(status)
                    pro = old_div(float(inlines), lines)
                    test = pro > 0.5  # do test to see if both match
                    win = '{0}({2}) - {1}({3}) inliers({4})/matched({5}) rate({6}) pass({7})'.format(
                        i, j, len(kp1), len(kp2), inlines, lines, pro, test)
                    d = Process(target=MatchExplorer, args=(
                        win, fore, back, kp_pairs, status, H))
                    d.start()
                    threads.append(d)
                    if test:
                        pass
    for t in threads:
        t.join()


if __name__ == '__main__':
    stich()
