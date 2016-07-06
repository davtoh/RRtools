# -*- coding: utf-8 -*-
# ----------------------------    IMPORTS    ---------------------------- #
# three-party
import numpy as np
# custom
from RRtoolbox.lib.config import MANAGER
from RRtoolbox.lib.plotter import matchExplorer
from RRtoolbox.lib.descriptors import ASIFT,MATCH, ASIFT_multiple

def stich(**opts):
    from multiprocessing import Process
    from glob import glob
    from RRtoolbox.lib.image import loadFunc, pathLoader, loaderDict
    feature_name = opts.get("feature",'sift-flann')
    #### LOADING
    print "looking in path {}".format(MANAGER["TESTPATH"])
    fns = glob(MANAGER["TESTPATH"] + "*.jpg")
    fns = fns[:3]
    print "found {} filtered files...".format(len(fns))
    #### SCALING
    rzyf,rzxf = 400,400 # dimensions to scale foregrounds
    #ims = [cv2.resize(cv2.imread(i, 0), (rzxf, rzyf)) for i in fns] # normal list
    loader = loadFunc(0,dsize=(rzxf, rzyf))
    ims_dict = loaderDict(loader)
    for i, val in enumerate(fns):
        ims_dict[i] = val
    ims = pathLoader(fns,loader) # load just when needed
    #img = [i for i in ims] # tests
    #ims = imloader(fns,0, (rzxf, rzyf),mmap=True,mpath=MANAGER.TEMPPATH) # load just when needed
    #img = [i for i in ims] # tests
    #ims = [numpymapper(data, str(changedir(fns[i],MANAGER.TEMPPATH))) for i,data in enumerate(imloader(fns))] # Too slow
    #nfns = [changedir(i,MANAGER.TEMPPATH) for i in fns] # this get the temp files
    #### FEATURE DETECTOR  # persistent by @root.memoize

    print "finding keypoints with its descriptors..."
    descriptors = ASIFT_multiple(ims, feature_name) # OR use ASIFT for each image
    print "total descriptors {}".format(len(descriptors))
    #### MATCHING
    # H, status, kp_pairs
    threads,counter = [],0
    print "matching..."
    with open("stitch values","a+") as f:
        for i in xrange(len(descriptors)):
            for j in xrange(len(descriptors)):
                if j>i: # do not test itself and inverted tests
                    counter +=1
                    print "comparision No.{}".format(counter)
                    # FIXME inefficient code ... just 44 descriptors generate 946 Homographies
                    fore,back = ims[i], ims[j]
                    (kp1,desc1),(kp2,desc2) = descriptors[i],descriptors[j]
                    H, status, kp_pairs = MATCH(feature_name,kp1,desc1,kp2,desc2)
                    inlines,lines = np.sum(status), len(status)
                    pro = float(inlines)/lines
                    test = pro>0.5 # do test to see if both match
                    win = '{0}({2}) - {1}({3}) inliers({4})/matched({5}) rate({6}) pass({7})'.format(i,j,len(kp1),len(kp2), inlines,lines,pro,test)
                    d = Process(target=matchExplorer,args = (win, fore, back, kp_pairs, status, H))
                    d.start()
                    threads.append(d)
                    if test:
                        pass
    for t in threads:
        t.join()

if __name__ == '__main__':
    stich()