#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''
imrestore (oriented to retinal images):

Restore images by merging and stitching techniques.

Optimization techniques:
    resize to smaller versions*

    memoization*:
        -persistence
        -serialization and de-serialization
        -caching

    multitasking*:
        -multiprocessing
        -multithreading

    lazy evaluations:
        -load on demand
        -use of weak references

    Memory mapped files*

STEPS:

    (1) Local features: Key-points and descriptors:
        -(1.1) SIFT, SURF, ORB, etc
        -ASIFT*

    (2) Select main or base image from set for merging:
        -Raw, Sorting, User input

    (3) Matching (spacial):
        -filter 0.7 below Hamming distance
        -key points classification

    (4) selection in matching set: (pre selection of good matches)
        (4.1) Best matches: for general purpose
        (4.2) Entropy: used when set is ensured to be of the same object
            (The program ensures that, if it is not the case).
        (4.3) Histogram comparison: use if set contains unwanted
            perspectives or images that do not correspond to image.
        (4.4) Custom function

    (5) Calculate Homography

    (6) Probability tests: (ensures that the matches images
    correspond to each other)

    (7) Merging
        (7.1) Histogram matching* (color)
        (7.2) Segmentation*
        (7.3) Alpha mask calculation*
        (7.4) Stitching and Merging

    (8) Overall filtering*:
        Bilateral filtering

    (9) Lens simulation for retinal photos*

* optional

Notes:

    Optimization techniques:

        Resize to smaller versions: process smaller versions of the
        inputs and convert the result back to the original versions.
        This reduces processing times, standardize the way data is
        processed (with fixed sizes), lets limited memory to be used,
        allows to apply in big images without breaking down algorithms
        that cannot do that.

        Memoization:
            Persistence: save data to disk for later use.

            serialization and de-serialization: (serialization, in
            python is refereed as pickling) convert live objects into
            a format that can be recorded; (de-serialization, in python
            referred as unpickling) it is used to restore serialized
            data into "live" objects again as if the object was created
            in program conserving its data from precious sessions.

            Caching: saves the result of a function depending or not in
            the inputs to compute data once and keep retrieving it from
            the cached values if asked.

        Multitasking:
            Multiprocessing: pass tasks to several processes using the
                computer's cores to achieve concurrency.
            Multithreading: pass tasks to threads to use "clock-slicing"
                of a processor to achieve "concurrency".

        Lazy  evaluations:
            load on demand: if data is from an external local file, it is
            loaded only when it is needed to be computed otherwise it is
            deleted from memory or cached in cases where it is extensively
            used. For remotes images (e.g. a server, URL) or in a inadequate
            format, it is downloaded and converted to a numpy format in a
            temporal local place.

            Use of weak references: in cases where the data is cached or
            has not been garbage collected, data is retrieved through
            weak references and if it is needed but has been garbage
            collected it is load again and assigned to the weak reference.

        Memory mapped files:
        Instantiate an object and keep it not in memory but in a file and
        access it directly there. Used when memory is limited or data is
        too big to fit in memory. Slow downs are negligible for read only
        mmaped files (i.e. "r") considering the gain in free memory, but
        it is a real drawback for write operations (i.e. "w","r+","w+").

    Selection algorithms:
        Histogram comparison - used to quickly identify the images that
            most resemble a target
        Entropy - used to select the best enfoqued images of the same
            perspective of an object

    Local features: Key-points and descriptors:
        ASIFT: used to add a layer of robustness onto other local
        feature methods to cover all affine transformations. ASIFT
        was conceived to complete the invariance to transformations
        offered by SIFT which simulates zoom invariance using gaussian
        blurring and normalizes rotation and translation. This by
        simulating a set of views from the initial image, varying the
        two camera axis orientations: latitude and longitude angles,
        hence its acronym Affine-SIFT. Whereas SIFT stands for Scale
        Invariant Feature Transform.

    Matching (spacial):
        Calculate Homography: Used to find the transformation matrix
        to overlay a foreground onto a background image.

    Filtering:
        Bilateral filtering: used to filter noise and make the image
        colors more uniform (in some cases more cartoonist-like)

    Histogram matching (color): used to approximate the colors from the
        foreground to the background image.

    Segmentation: detect and individualize the target objects (e.g. optic
        disk, flares) to further process them or prevent them to be altered.

    Alfa mask calculation: It uses Alfa transparency obtained with sigmoid
        filters and binary masks from the segmentation to specify where an
        algorithm should have more effect or no effect at all
        (i.e. intensity driven).

    Stitching and Merging:
        This is an application point, where all previous algorithms are
        combined to stitch images so as to construct an scenery from the
        parts and merge them if overlapped or even take advantage of these
        to restore images by completing lacking information or enhancing
        poorly illuminated parts in the image. A drawback of this is that
        if not well processed and precise information if given or calculated
        the result could be if not equal worse than the initial images.

    Lens simulation for retinal photos: As its name implies, it is a
        post-processing method applied for better appeal of the image
        depending on the tastes of the user.
'''
from __future__ import division

# needed for installing
# program inputs
import os
import cv2
import warnings
import numpy as np
from RRtoolbox.tools.lens import simulateLens
from RRtoolbox.lib.config import MANAGER
from RRtoolbox.lib.image import hist_match
from RRtoolbox.lib.directory import getData, getPath, mkPath
from RRtoolbox.lib.cache import memoizedDict
from RRtoolbox.lib.image import loadFunc, imcoors
from RRtoolbox.lib.arrayops.mask import brightness
from multiprocessing.pool import ThreadPool as Pool
from RRtoolbox.tools.selectors import hist_map, hist_comp, entropy
from RRtoolbox.tools.segmentation import retinal_mask
from RRtoolbox.lib.root import TimeCode, glob, lookinglob
from RRtoolbox.lib.descriptors import Feature, inlineRatio
from RRtoolbox.tools.segmentation import getBrightAlpha, get_beta_params, bandpass, bandstop
from RRtoolbox.lib.plotter import matchExplorer, plotim, fastplt
from RRtoolbox.lib.arrayops.filters import getBilateralParameters
from RRtoolbox.lib.arrayops.convert import getSOpointRelation, dict2keyPoint
from RRtoolbox.lib.arrayops.basic import superpose, getTransformedCorners, transformPoint, \
    im2shapeFormat,normalize, getOtsuThresh

def imrestore(images, **opts):
    """
    Restore images by merging and stitching techniques.

    :param images: list of images or string to path which uses glob filter in path.
            Loads image array from path, url, server, string
            or directly from numpy array (supports databases)
    :param debug: (0) flag to print debug messages
            0 -> do not print messages.
            1 -> print messages.
            2 -> print messages and show results
                (consumes significantly more memory).
            3 -> print messages, show results and additional data
                (consumes significantly more memory).
    :param feature: (None) feature instance. It contains the configured
            detector and matcher.
    :param pool: (None) use pool Ex: 4 to use 4 CPUs.
    :param cachePath: (None) saves memoization to specified path and
            downloaded images.
    :param clearCache: (0) clear cache flag.
            * 0 do not clear.
            * 1 All CachePath is cleared before use.
            * 2 re-compute data but other cache data is left intact.
            Notes: using cache can result in unspected behaviour
                if some configurations does not match to the cached data.
    :param loader: (None) custom loader function used to load images
            to merge. If None it loads the original images in color.
    :param pshape: (400,400) process shape, used to load pseudo images
            to process features and then results are converted to the
            original images. If None it loads the original images to
            process the features but it can incur to performance penalties
            if images are too big and RAM memory is scarce.
    :param baseImage: (None) First image to merge to.
            * None -> takes first image from raw list.
            * True -> selects image with most features.
            * Image Name.
    :param selectMethod: (None) Method to sort images when matching. This
            way the merging order can be controlled.
            * (None) Best matches.
            * Histogram Comparison: Correlation, Chi-squared,
                Intersection, Hellinger or any method found in hist_map
            * Entropy.
            * custom function of the form: rating,fn <-- selectMethod(fns)
    :param distanceThresh: (0.75) filter matches by distance ratio.
    :param inlineThresh: (0.2) filter homography by inlineratio.
    :param rectangularityThresh: (0.5) filter homography by rectangularity.
    :param ransacReprojThreshold: (5.0) maximum allowed reprojection error
            to treat a point pair as an inlier.
    :param centric: (False) tries to attach as many images as possible to
            each matching. It is quicker since it does not have to process
            too many match computations.
    :param hist_match: (False) apply histogram matching to foreground
            image with merge image as template
    :param mergefunc: (None) function used to merge foreground with
            background image using the given transformation matrix.
            The structure is as follows:

            merged, H_back, H_fore= mergefunc(back,fore,H)

            ..where::

                back: background image
                fore: foreground image
                H: calculated Transformation Matrix
                merged: new image of fore in back image
                H_back: transformation matrix that modifies
                        background key-points
                H_fore: transformation matrix that modifies
                        foreground key-points

    :param postfunc: (None) function used for post processing
            the merging result. The function is called with the merging
            image and must return the processed image.
    :param save: (False)
            * True, saves in path with name restored_{base_image}
            * False, does not save
            * Image name used to save the restored image.
    :return: restored image
    """
    # for debug
    FLAG_DEBUG = opts.get("debug",1)

    # for multiprocessing
    pool = opts.get("pool",None)
    if pool is not None: # convert pool count to pool class
        NO_CPU = cv2.getNumberOfCPUs()
        if pool <= NO_CPU:
            pool = Pool(processes = pool)
        else:
            raise Exception("pool of {} exceeds the number of processors {}".format(pool,NO_CPU))
    # for features
    feature = opts.get("feature",None)
    if feature is None:
        feature = Feature(pool=pool,debug=FLAG_DEBUG)
        feature.config(name='a-sift-flann') # init detector and matcher to compute descriptors
    else:
        feature.pool = pool
        feature.debug = FLAG_DEBUG

    # select method to order images to feed in superposition
    selectMethod = opts.get("selectMethod",None)
    best_match_list = ("bestmatches", "best matches")
    entropy_list = ("entropy",)
    if callable(selectMethod):
        orderValue = 3
    elif selectMethod in hist_map:
        orderValue = 2
    elif selectMethod in entropy_list:
        orderValue = 1
    elif selectMethod in best_match_list or selectMethod is None:
        orderValue = 0
    else:
        raise Exception("selectMethod {} not recognized".format(selectMethod))

    # distance threshold to filter best matches
    distanceThresh = opts.get("distanceThresh",0.75) # filter ratio

    # threshold for inlineRatio
    inlineThresh = opts.get("inlineThresh",0.2) # filter ratio
    assert inlineThresh<=1 and inlineThresh>=0 # ensures adequate value [0,1]

    # threshold for rectangularity
    rectangularityThresh = opts.get("rectangularityThresh",0.5) # filter ratio
    assert rectangularityThresh<=1 and rectangularityThresh>=0 # ensures adequate value [0,1]

    # threshold to for RANSAC reprojection
    ransacReprojThreshold = opts.get("ransacReprojThreshold",5.0)

    centric = opts.get("centric",False) # tries to attach as many images as possible
    pshape = opts.get("pshape",(400,400))# it is not safe to compute descriptors from big images
    usepshape = False # output is as pshape if True, else process with pshape but output is as loader
    minKps = 3 # minimum len of key-points to find Homography
    histMatch = opts.get("hist_match",False)
    mergefunc = opts.get("mergefunc",None)
    if mergefunc is None:
        mergefunc = superpose
    assert callable(mergefunc)
    postfunc = opts.get("postfunc",None)
    assert postfunc is None or callable(postfunc)

    ############################## OPTIMIZATION MEMOIZEDIC #############################
    cachePath = opts.get("cachePath",None)
    if cachePath is not None:
        feature_dic = memoizedDict(cachePath+"descriptors")
        if FLAG_DEBUG: print "Cache path is in {}".format(feature_dic._path)
        clearCache = opts.get("clearCache",0)
        if clearCache==1:
            feature_dic.clear()
            if FLAG_DEBUG: print "Cache path cleared"
    else:
        feature_dic = {}

    expert = opts.get("expert",None)
    if expert is not None:
        expert = memoizedDict(expert) # convert path

    ################################## LOADING IMAGES ###################################
    if images is None or len(images)==0: # if images is empty use demonstration
        test = MANAGER.TESTPATH
        if FLAG_DEBUG: print "Looking in DEMO path {}".format(test)
        fns = glob(test + "*")
    elif isinstance(images,basestring):
        # if string assume it is a path
        if FLAG_DEBUG:print "Looking as {}".format(images)
        fns = glob(images)
    elif not isinstance(images,basestring) and len(images) == 1 and "*" in images[0]:
        images = images[0] # get string
        if FLAG_DEBUG:print "Looking as {}".format(images)
        fns = glob(images)
    else: # iterator containing data
        fns = images # list file names

    # check images
    if not len(fns)>1:
        raise Exception("list of images must be greater than 1, got {}".format(len(fns)))

    # to select base image ahead of any process
    baseImage = opts.get("baseImage",None)
    if isinstance(baseImage,basestring):
        base_old = baseImage
        try: # tries user input
            if baseImage not in fns:
                base, path, name, ext = getData(baseImage)
                if not path: # if name is incomplete look for it
                    base, path, _,_ = getData(fns[0])
                baseImage = lookinglob(baseImage, "".join((base, path)))
                # selected image must be in fns
                if baseImage is None:
                    raise IndexError
        except IndexError: # tries to find image based in user input
            # generate informative error for the user
            raise Exception("{} or {} is not in image list"
                            "\n A pattern is {}".format(
                            base_old,baseImage,fns[0]))

    if FLAG_DEBUG: print "No. images {}...".format(len(fns))

    # make loader
    loader = opts.get("loader",None) # BGR loader
    if loader is None: loader = loadFunc(1)

    ######################## Local features: Key-points and descriptors ####################
    with TimeCode("Computing features...\n",
                  endmsg="Computed feature time was {time}\n", enableMsg=FLAG_DEBUG):
        feature_list = [] # list of key points and descriptors
        index = 0
        tries = 0 # prevents infinite loop
        while index < len(fns):
            try:
                path = fns[index]
                try:
                    if cachePath is None or clearCache==2 and path in feature_dic:
                        raise KeyError # clears entry from cache
                    kps, desc, shape = feature_dic[path] # thread safe
                    # check memoized is the same
                    if pshape is None:
                        sh = loader(path).shape # and checks that image exists
                    else:
                        sh = pshape

                    # solves possible difference in object id instead of if sh != shape
                    for ii,jj in zip(sh,shape):
                        if ii!=jj: raise KeyError

                except (KeyError, ValueError) as e: # not memorized
                    if FLAG_DEBUG: print "Processing features for {}...".format(path)
                    img = loader(path) #cv2.imread(path)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    if pshape is None: # get features directly from original
                        kps, desc = feature.detectAndCompute(img) # get features
                    else: # optimize by getting features from scaled image and the rescaling
                        oshape = img.shape # cache original shape
                        img = cv2.resize(img, pshape) # pseudo image
                        kps, desc = feature.detectAndCompute(img) # get features
                        # re-scale keypoints to original image
                        if not usepshape:
                            # this necessarily does not produce the same result when not used
                            """
                            # METHOD 1: using Transformation Matrix
                            H = getSOpointRelation(pshape, oshape, True)
                            for kp in kps:
                                kp["pt"]=tuple(cv2.perspectiveTransform(
                                    np.array([[kp["pt"]]]), H).reshape(-1, 2)[0])
                            """
                            # METHOD 2:
                            rx,ry = getSOpointRelation(pshape, oshape)
                            for kp in kps:
                                x,y = kp["pt"]
                                kp["pt"] = x*rx,y*ry
                    shape = img.shape
                    feature_dic[path] = kps, desc, shape # to memoize

                # add paths to key-points
                for kp in kps:
                    kp["path"] = path

                # number of key-points, index, path, key-points, descriptors
                feature_list.append((len(kps),index,path,kps,desc))
                if FLAG_DEBUG: print "\rFeatures {}/{}...".format(index+1,len(fns)),
                index += 1
                tries = 0
            except Exception as e:
                tries += 1
                warnings.warn("caught error 139")
                if tries > 2:
                    raise e

    ############################## Pre-selection from a set ###############################
    # initialization and base image selection
    if baseImage is None:
       _,_,path,kps_base,desc_base = feature_list[0] # select first for most probable
    elif isinstance(baseImage,basestring):
        kps_base,desc_base = feature_dic[baseImage]
        path = baseImage
    elif baseImage is True: # sort images
        feature_list.sort(reverse=True) # descendant: from bigger to least
        _,_,path,kps_base,desc_base = feature_list[0] # select first for most probable
    else:
        raise Exception("baseImage must be None, True or String")

    if FLAG_DEBUG: print "baseImage is", path
    used = [path] # select first image path
    restored = loader(path) # load first image for merged image
    if usepshape:
        restored = cv2.resize(restored,pshape)
    failed = [] # registry for failed images

    ############################# Order set initialization ################################
    if orderValue: # obtain comparison with structure (value, path)
        if orderValue == 1: # entropy
            comparison = zip(*entropy(fns,loadfunc=loadFunc(1,pshape),invert=False)[:2])
            if FLAG_DEBUG: print "Configured to sort by entropy..."
        elif orderValue == 2: # histogram comparison
            comparison = hist_comp(fns,loadfunc=loadFunc(1,pshape),method=selectMethod)
            if FLAG_DEBUG: print "Configured to sort by {}...".format(selectMethod)
        elif orderValue == 3:
            comparison = selectMethod(fns)
            if FLAG_DEBUG: print "Configured to sort by Custom Function..."
        else:
            raise Exception("DEBUG: orderValue {} does "
                            +"not correspond to {}".format(orderValue,selectMethod))
    elif FLAG_DEBUG: print "Configured to sort by best matches"

    with TimeCode("Restoring ...\n",
                  endmsg= "Restoring overall time was {time}\n", enableMsg= FLAG_DEBUG):
        while True:
            with TimeCode("Matching ...\n",
                          endmsg= "Matching overall time was {time}\n", enableMsg= FLAG_DEBUG):
                ####################### remaining keypoints to match ##########################
                kps_remain,desc_remain = [],[] # initialize key-point and descriptor base list
                for _,_,path,kps,desc in feature_list:
                    if path not in used: # append only those which are not in the base image
                        kps_remain.extend(kps)
                        desc_remain.extend(desc)

                if not kps_remain: # if there is not image remaining to stitch break
                    if FLAG_DEBUG: print "All images used"
                    break

                desc_remain = np.array(desc_remain) # convert descriptors to array

                ################################## Matching ###################################
                # select only those with good distance (hamming, L1, L2)
                raw_matches = feature.matcher.knnMatch(desc_remain,
                                                       trainDescriptors = desc_base, k = 2) #2
                # If path=2, it will draw two match-lines for each key-point.
                classified = {}
                for m in raw_matches:
                    # filter by Hamming, L1 or L2 distance
                    if m[0].distance < m[1].distance * distanceThresh:
                        m = m[0]
                        kp1 = kps_remain[m.queryIdx]  # keypoint in query image
                        kp2 = kps_base[m.trainIdx]  # keypoint in train image

                        key = kp1["path"] # ensured that key is not in used
                        if key in classified:
                            classified[key].append((kp1,kp2))
                        else:
                            classified[key] = [(kp1,kp2)]

                ############################ Order set ########################################
                if orderValue: # use only those in classified of histogram or entropy comparison
                    ordered = [(val,path) for val,path in comparison if path in classified]
                else: # order with best matches
                    ordered = sorted([(len(kps),path) for path,kps in classified.items()],reverse=True)


            for rank, path in ordered: # feed key-points in order according to order set

                ########################### Calculate Homography ##########################
                mkp1,mkp2 = zip(*classified[path]) # probably good matches
                if len(mkp1)>minKps and len(mkp2)>minKps:
                    p1 = np.float32([kp["pt"] for kp in mkp1])
                    p2 = np.float32([kp["pt"] for kp in mkp2])
                    if FLAG_DEBUG > 1: print 'Calculating Homography for {}...'.format(path)
                    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, ransacReprojThreshold)
                else:
                    if FLAG_DEBUG > 1: print 'Not enough key-points for {}...'.format(path)
                    H = None

                if H is not None: #first test
                    fore = loader(path) # load fore image
                    if usepshape:
                        fore = cv2.resize(fore,pshape)
                    """
                    else:
                        # METHOD 3 for rescaling points. FIXME
                        #shapes = fore.shape,pshape,fore.shape,pshape
                        #H = sh2oh(H, *shapes) #### sTM to oTM
                        H = getSOpointRelation(pshape,fore.shape, True)*H
                        for kp in feature_dic[path][0]:
                            kp["pt"]=tuple(cv2.perspectiveTransform(
                                np.array([[kp["pt"]]]), H).reshape(-1, 2)[0])"""
                    if histMatch: # apply histogram matching
                        fore = hist_match(fore, restored)

                    h,w = fore.shape[:2] # image shape

                    # get corners of fore projection over back
                    projection = getTransformedCorners((h,w),H)
                    c = imcoors(projection) # class to calculate statistical data
                    lines, inlines = len(status), np.sum(status)

                    # ratio to determine how good fore is in back
                    inlineratio = inlineRatio(inlines,lines)

                    text = "inlines/lines: {}/{}={} and rectangularity {}".format(
                        inlines, lines, inlineratio, c.rotatedRectangularity)

                    if FLAG_DEBUG>1: print text

                    if FLAG_DEBUG > 2: # show matches
                        matchExplorer("Match " + text, fore,
                                      restored, classified[path], status, H)

                    ######################### probability test ############################
                    if inlineratio>inlineThresh \
                            and c.rotatedRectangularity>rectangularityThresh: # second test

                        if FLAG_DEBUG>1: print "Test succeeded..."
                        while path in failed: # clean path in fail registry
                            try: # race-conditions safe
                                failed.remove(path)
                            except ValueError:
                                pass

                        ###################### merging and stitching #######################
                        if FLAG_DEBUG > 1: print "Merging..."
                        restored, H_back, H_fore= mergefunc(restored,fore,H)
                        if H_fore is None: # H is not modified use itself
                            H_fore = H

                        if FLAG_DEBUG > 1: # show merging result
                            plotim("Last added with " + text, restored).show()

                        ####################### update base features #######################
                        # make projection to test key-points inside it
                        if FLAG_DEBUG > 1: print "Updating key-points..."
                        projection = getTransformedCorners((h,w),H_fore)
                        newkps, newdesc = [], []
                        for _,_,p,kps,desc in feature_list:
                            # append all points in the base image and update their position
                            if p in used: # transform points in back
                                for kp,dsc in zip(kps,desc): # kps,desc
                                    pt = kp["pt"] # get point
                                    if H_back is not None: # update point
                                        pt = tuple(transformPoint(pt,H_back))
                                        kp["pt"] = pt
                                    # include only those points outside foreground
                                    if cv2.pointPolygonTest(projection, pt, False) == -1:
                                        newkps.append(kp)
                                        newdesc.append(dsc)
                            elif p == path: # transform points in fore
                                # include only those points inside foreground
                                for kp,dsc in zip(kps,desc): # kps,desc
                                    kp["pt"] = tuple(transformPoint(kp["pt"],H_fore))
                                    newkps.append(kp)
                                    newdesc.append(dsc)
                        # update kps_base and desc_base
                        kps_base = newkps
                        desc_base = np.array(newdesc)

                        if FLAG_DEBUG > 2: # show keypints in merging
                            plotim("merged Key-points", # draw key-points in image
                                   cv2.drawKeypoints(
                                       im2shapeFormat(restored,restored.shape[:2]+(3,)),
                                              [dict2keyPoint(index) for index in kps_base],
                                              flags=4, color=(0,0,255))).show()
                        if FLAG_DEBUG: print "This image has been merged: {}...".format(path)
                        used.append(path) # update used
                        if not centric:
                            break
                    else:
                        failed.append(path)
                else:
                    failed.append(path)

            # if all classified have failed then end
            if set(classified.keys()) == set(failed):
                if FLAG_DEBUG:
                    print "Ended, these images do not fit: "
                    for index in classified.keys():
                        print index
                break

        if postfunc is not None:
            if FLAG_DEBUG: print "Applying post function..."
            restored = postfunc(restored)
        ####################################### Save image ####################################
        save = opts.get("save",False)
        if save:
            base, path, name, ext = getData(used[0])
            if isinstance(save,basestring):
                fn = save.format(path="".join((base, path)), ext=ext,)
            else:
                fn = "".join((base,path,"restored_"+ name,ext))

            mkPath(getPath(fn))
            r = cv2.imwrite(fn,restored)
            if FLAG_DEBUG and r:
                print "Saved: {}".format(fn)
            else:
                print "{} could not be saved".format(fn)

    return restored # return merged image

def retinalMerge(back,fore,H):
    # this window can be passed to getBrightAlpha to only add artifacts in fore
    #window = cv2.warpPerspective(np.ones(fore.shape[:2]),H,(back.shape[1],back.shape[0]))

    # find_optic_disc: can be passed as an array or as a function to superpose function
    # METHOD 1: function evaluated just after superposition and before overlay
    def _mask(back,fore): # TODO, not working for every scenario
        foregray = brightness(fore)
        thresh,w = cv2.threshold(foregray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        m = getBrightAlpha(brightness(back).astype(float), foregray.astype(float), window=w)
        return m

    def adjust_alpha(alpha, mask, invert = False):
        seg = alpha[mask.astype(bool)]
        smax,smin = np.max(seg),np.min(seg)
        alpha = alpha - smin
        alpha[alpha<0] = 0
        alpha = alpha / np.max(alpha)
        if invert: alpha = 1-alpha
        return alpha

    def alpha_betas(alpha, mask):
        seg = alpha[mask.astype(bool)]
        smax,smin = np.max(seg),np.min(seg)
        hist, bins = np.histogram(seg.flatten(),256)
        thresh = bins[getOtsuThresh(hist)]
        return smin,thresh

    def mask(back, fore):

        backgray = brightness(back).astype(float)
        foregray = brightness(fore).astype(float)

        mask_back, alpha_back = retinal_mask(back,biggest=True,addalpha=True)
        mask_fore, alpha_fore = retinal_mask(fore,biggest=True,addalpha=True)

        backmask = bandstop(3, *alpha_betas(alpha_back*255,mask_back))(backgray) # beta1 = 50, beta2 = 190
        foremask = bandpass(3, *alpha_betas(alpha_fore*255,mask_fore))(foregray) # beta1 = 50, beta2 = 220
        alphamask = normalize(foremask * backmask * alpha_back)

        #alpha_back[mask_back==0] = 0
        #alpha_back = adjust_alpha(alpha_back,mask_back)
        #alpha_fore = adjust_alpha(alpha_fore,mask_fore, invert=True)
        #alpha_fore[mask_fore==0] = 0

        #fastplt(np.hstack((backgray.astype(float)/255.,backmask)), title="alpha_back")
        #fastplt(np.hstack((foregray.astype(float)/255.,foremask)), title="alpha_fore")
        #alpha = normalize(alpha_back*alpha_fore)
        #fastplt(np.hstack((alpha,alphamask)),title="alpha")
        #raw_input()
        return alphamask

    # METHOD 2: Array before superposition
    #fore_in_back = cv2.warpPerspective(fore,H,(back.shape[1],back.shape[0]))
    #mask = getBrightAlpha(brightness(back).astype(float), brightness(fore_in_back).astype(float))

    #fastplt(alpha) # show alfa mask
    return superpose(back, fore, H, mask)

def createPostRetinal(heavynoise=False, lens=True, enclose = False):
    """
    Creates function to post-process a merged retinal image.

    :param heavynoise: True to process noisy images, False to
        process normal images
    :param lens: flag to determine if lens are applied. True
        to simulate lens, False to not apply lens
    :param enclose: flag to enclose and return only retinal area.
        True to return ROI, false to leave image "as is".
    :return:
    """
    def postRetinal(img):
        """
        post-process a merged retinal image.

        :param img: retinal image
        :return: filtered and with simulated lens
        """
        # detect how much noise to process?
        if heavynoise: # slower but interactive for heavy noise
            params = getBilateralParameters(img.shape)
        else:
            params = 15,82,57 # 21,75,75 # faster and for low noise
        img = cv2.bilateralFilter(img,*params)
        if lens:
            try:
                img = simulateLens(img)
            except Exception as e:
                warnings.warn("simulate lens failed with error {}".format(e))
        if enclose:
            pass # TODO
        return img
    return postRetinal

def retinalRestore(images, **opts):
    """
    Configure functions to use in imrestore to process retinal images.

    :param images:
    :param opts:
    :return:
    """
    if opts.get("mergefunc",None) is None:
        opts["mergefunc"] = retinalMerge
    if opts.get("postfunc",None) is None:
        opts["postfunc"] = createPostRetinal(opts.get("heavynoise",False),
                                             opts.get("lens",True))
    return imrestore(images,**opts)
retinalRestore.__doc__ = imrestore.__doc__

def True_or_String(string):
    if string == "":
        return True
    return string

def feature_creator(string):
    if string == "":
        return None
    return Feature().config(string)

def tuple_creator(string):
    if string == "":
        return None
    return tuple([int(i) for i in string.split(",")])

def loader_creator(string):
    return

def shell(args=None, namespace=None):
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Restore images by merging and stitching techniques.",
                                     epilog=__doc__)
    parser.add_argument('images', nargs='*',
                        help='List of images or path to images using * in folder. '
                             'It loads image array from path, url, server, string'
                             'or directly from numpy arrays (supports databases)'
                             'Because the shell process wildcards before it gets '
                             'to the parser it creates a list of filtered files in '
                             'the path. Use quotes in shell to prevent this behaviour '
                             'an let imrestore do it instead e.g. "/path/to/images/*.jpg"')
    parser.add_argument('-d','--debug',type=int,default=1,
                       help='''flag to print debug messages
                                0 -> do not print messages.
                                1 -> print messages.
                                2 -> print messages and show results
                                    (consumes significantly more memory).
                                3 -> print messages, show results and additional data
                                    (consumes significantly more memory).
                                "path" -> expert variables''')
    parser.add_argument('-f','--feature', type=feature_creator,
                       help='Configure detector and matcher')
    parser.add_argument('-u','--pool', action='store', type=int,
                       help='Use pool Ex: 4 to use 4 CPUs')
    parser.add_argument('-c','--cachePath',default=None,
                       help='Saves memoization to specified path and downloaded images')
    parser.add_argument('-e','--clearCache', type=int, default=0,
                       help='clear cache flag.'
                            '* 0 do not clear.'
                            '* 1 All CachePath is cleared before use.'
                            '* 2 re-compute data but other cache data is left intact.'
                            'Notes: using cache can result in unspected behaviour '
                            'if some configurations does not match to the cached data.')
    parser.add_argument('-l','--loader', type=loader_creator,
                       help='custom loader function used to load images '
                            'to merge. If None it loads the original images in color')
    parser.add_argument('-p','--pshape', default=(400,400), type=tuple_creator,
                       help='Process shape used to load pseudo images '
                            'to process features and then results are converted to the '
                            'original images. If None (e.g "") it loads the original images to '
                            'process the features but it can incur to performance penalties '
                            'if images are too big and RAM memory is scarce')
    parser.add_argument('-b','--baseImage', default=True, type=True_or_String,
                       help='First image to merge to. If None (e.g "") selects image with most features'
                            'or specify image Name to use from path')
    parser.add_argument('-m','--selectMethod',
                       help='Method to sort images when matching. This '
                            'way the merging order can be controlled.'
                            '* (None) Best matches'
                            '* Histogram Comparison: Correlation, Chi-squared,'
                            'Intersection, Hellinger or any method found in hist_map'
                            '* Entropy'
                            '* custom function of the form: rating,fn <-- selectMethod(fns)')
    parser.add_argument('-a','--distanceThresh', type = float, default=0.75,
                       help='Filter matches by distance ratio')
    parser.add_argument('-i','--inlineThresh', type = float, default=0.2,
                       help='Filter homography by inlineratio')
    parser.add_argument('-r','--rectangularityThresh', type = float, default=0.5,
                       help='Filter homography by rectangularity')
    parser.add_argument('-j','--ransacReprojThreshold', type = float, default=10.0,
                       help='Maximum allowed reprojection error '
                            'to treat a point pair as an inlier')
    parser.add_argument('-n','--centric', action='store_true',
                       help='Tries to attach as many images as possible to '
                            'each matching. It is quicker since it does not have to process '
                            'too many match computations')
    parser.add_argument('-t','--hist_match', action='store_true',
                       help='Apply histogram matching to foreground image with merge image as template')
    parser.add_argument('-s','--save', default=True,
                       help='Customize image name used to save the restored image.'
                            'By default it saves in path with name restored_{base_image}')
    parser.add_argument('--restorer',choices = ['retinalRestore','imrestore'], default='retinalRestore',
                       help='imrestore is for images in general but it can be parametrized. '
                            'By default it has the profile "retinalRestore" for retinal images '
                            'but its general behaviour can be restorerd by changing it to '
                            '"imrestore"')
    parser.add_argument('--expert', default=None,
                       help='path to expert variables')
    args = vars(parser.parse_args(args=args, namespace=namespace))
    # this is needed because the shell process wildcards before it gets to argparse creating a list in the path
    # thus it must be filtered. Use quotes in shell to prevent this behaviour
    args['images'] = [p for p in args['images'] if os.path.isfile(p) or "*" in p]

    ################################## TESTS ##################################
    #args['debug'] = 5
    #test = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/IMAGES/RETINA/*"
    #test = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/IMAGES/cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/*"
    #test = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/IMAGES/cellphone_retinal/ALCATEL ONE TOUCH IDOL X/right_DAVID/*"
    #test = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/IMAGES/cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_FAVIANI/*"
    #test = "/home/davtoh/Desktop/test1/*"
    #test = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/IMAGES/*"
    #args['images'] = test
    #args["cachePath"] = "/mnt/4E443F99443F82AF/restoration_data/"
    #args["baseImage"] = "IMG_20150730_125354"
    #args["loader"] = loadFunc(1,(600,600))
    #args["clearCache"] = 2
    ###########################################################################
    args["expert"] = "/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/tests/experimental_expert_cached"
    if args['debug']>2:
        print "Parsed Arguments\n",args
    r = args.pop("restorer")
    if r == 'retinalRestore':
        if args['debug']: print "Configured for retinal restoration..."
        retinalRestore(**args)
    elif r == 'imrestore':
        if args['debug']: print "Configured for general restoration..."
        imrestore(**args)

    return namespace

if __name__ == "__main__":
    shell()