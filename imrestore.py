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
# TODO install openCV 2.4.12 as described in http://stackoverflow.com/a/37283690/5288758
# to solve the error Process finished with exit code 139
# UPDATE: openCV 2.4.12 does not solve the error Process finished with exit code 139

__author__ = 'Davtoh'
# needed for installing executable
import six
import packaging
import packaging.specifiers
# program imports
import os
import cv2
import warnings
import numpy as np
from RRtoolbox.tools.lens import simulateLens
from RRtoolbox.lib.config import MANAGER, FLOAT
from RRtoolbox.lib.image import hist_match
from RRtoolbox.lib.directory import getData, getPath, mkPath, increment_if_exits
from RRtoolbox.lib.cache import memoizedDict
from RRtoolbox.lib.image import loadFunc, imcoors
from RRtoolbox.lib.arrayops.mask import brightness, foreground, thresh_biggestCnt
from multiprocessing.pool import ThreadPool as Pool
from RRtoolbox.tools.selectors import hist_map, hist_comp, entropy
from RRtoolbox.tools.segmentation import retinal_mask
from RRtoolbox.lib.root import TimeCode, glob, lookinglob
from RRtoolbox.lib.descriptors import Feature, inlineRatio
from RRtoolbox.tools.segmentation import getBrightAlpha, bandpass, bandstop
from RRtoolbox.lib.plotter import matchExplorer, plotim, fastplt
from RRtoolbox.lib.arrayops.filters import getBilateralParameters
from RRtoolbox.lib.arrayops.convert import getSOpointRelation, dict2keyPoint
from RRtoolbox.lib.arrayops.basic import superpose, getTransformedCorners, transformPoint, \
    im2shapeFormat,normalize, getOtsuThresh, contours2mask, pad_to_fit_H, overlay

class VariableNotSettable(Exception):
    pass

class VariableNotDeletable(Exception):
    pass

class ImRestore(object):
    """
    Restore images by merging and stitching techniques.

    :param filenames: list of images or string to path which uses glob filter in path.
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
    :param loadshape:
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
    :param grow_scene: If True, allow the restored image to grow in shape if
            necessary at the merging process.
    :param expert: Path to an expert database. If provided it will use this data
            to generate the mask used when merging to the restored image.
    :param maskforeground:
            * True, limit features area using foreground mask of input images.
                This mask is calculated to threshold a well defined object.
            * Callable, Custom function to produce the foreground image which
                receives the input gray image and must return the mask image
                where the keypoints will be processed.
    :param save: (False)
            * True, saves in path with name restored_{base_image}
            * False, does not save
            * Image name used to save the restored image.
    :param overwrite: If True and the destine filename for saving already
            exists then it is replaced, else a new filename is generated
            with an index "{filename}_{index}.{extenssion}"
    """

    def __init__(self, filenames, **opts):
        # for debug
        self.FLAG_DEBUG = opts.get("debug",1)

        # for multiprocessing
        self.pool = opts.get("pool",None)
        if self.pool is not None: # convert pool count to pool class
            NO_CPU = cv2.getNumberOfCPUs()
            if self.pool <= NO_CPU:
                self.pool = Pool(processes = self.pool)
            else:
                raise Exception("pool of {} exceeds the "
                                "number of processors {}".format(self.pool,NO_CPU))
        # for features
        self.feature = opts.get("feature",None)
        # init detector and matcher to compute descriptors
        if self.feature is None:
            self.feature = Feature(pool=self.pool,debug=self.FLAG_DEBUG)
            self.feature.config(name='a-sift-flann')
        else:
            self.feature.pool = self.pool
            self.feature.debug = self.FLAG_DEBUG

        # select method to order images to feed in superposition
        self.selectMethod = opts.get("selectMethod",None)
        best_match_list = ("bestmatches", "best matches")
        entropy_list = ("entropy",)
        if callable(self.selectMethod):
            self._orderValue = 3
        elif self.selectMethod in hist_map:
            self._orderValue = 2
        elif self.selectMethod in entropy_list:
            self._orderValue = 1
        elif self.selectMethod in best_match_list or self.selectMethod is None:
            self._orderValue = 0
        else:
            raise Exception("selectMethod {} not recognized".format(self.selectMethod))

        # distance threshold to filter best matches
        self.distanceThresh = opts.get("distanceThresh",0.75) # filter ratio

        # threshold for inlineRatio
        self.inlineThresh = opts.get("inlineThresh",0.2) # filter ratio
        # ensures adequate value [0,1]
        assert self.inlineThresh<=1 and self.inlineThresh>=0

        # threshold for rectangularity
        self.rectangularityThresh = opts.get("rectangularityThresh",0.5) # filter ratio
        # ensures adequate value [0,1]
        assert self.rectangularityThresh<=1 and self.rectangularityThresh>=0

        # threshold to for RANSAC reprojection
        self.ransacReprojThreshold = opts.get("ransacReprojThreshold",5.0)

        self.centric = opts.get("centric",False) # tries to attach as many images as possible
        # it is not memory efficient to compute descriptors from big images
        self.pshape = opts.get("pshape",(400,400)) # use processing shape
        self.loadshape = opts.get("loadshape",None) # shape to load images for merging
        self.minKps = 3 # minimum len of key-points to find Homography
        self.histMatch = opts.get("hist_match",False)

        ############################## OPTIMIZATION MEMOIZEDIC ###########################
        self.cachePath = opts.get("cachePath",None)
        if self.cachePath is not None:
            self.feature_dic = memoizedDict(self.cachePath+"descriptors")
            if self.FLAG_DEBUG: print "Cache path is in {}".format(self.feature_dic._path)
            self.clearCache = opts.get("clearCache",0)
            if self.clearCache==1:
                self.feature_dic.clear()
                if self.FLAG_DEBUG: print "Cache path cleared"
        else:
            self.feature_dic = {}

        self.expert = opts.get("expert",None)
        if self.expert is not None:
            self.expert = memoizedDict(self.expert) # convert path

        ################################## GET IMAGES ####################################
        if filenames is None or len(filenames)==0: # if images is empty use demonstration
            test = MANAGER["TESTPATH"]
            if self.FLAG_DEBUG: print "Looking in DEMO path {}".format(test)
            fns = glob(test + "*")
        elif isinstance(filenames, basestring):
            # if string assume it is a path
            if self.FLAG_DEBUG:print "Looking as {}".format(filenames)
            fns = glob(filenames)
        elif not isinstance(filenames, basestring) and \
                        len(filenames) == 1 and "*" in filenames[0]:
            filenames = filenames[0] # get string
            if self.FLAG_DEBUG:print "Looking as {}".format(filenames)
            fns = glob(filenames)
        else: # iterator containing data
            fns = filenames # list file names

        # check images
        if not len(fns)>1:
            raise Exception("list of images must be "
                            "greater than 1, got {}".format(len(fns)))

        self.filenames = fns

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

        self.baseImage = baseImage

        if self.FLAG_DEBUG: print "No. images {}...".format(len(fns))

        # make loader
        self.loader = opts.get("loader",None) # BGR loader
        if self.loader is None: self.loader = loadFunc(1)
        self._loader_cache = None # keeps last image reference
        self._loader_params = None # keeps last track of last loading options to reload

        self.save = opts.get("save",False)
        self.grow_scene = opts.get("grow_scene",True)
        self.maskforeground = opts.get("maskforeground",False)
        self.overwrite = opts.get("overwrite",False)

        # processing variables
        self._feature_list = None

    @property
    def feature_list(self):
        if self._feature_list is None:
            return self.compute_keypoints()
        return self._feature_list
    @feature_list.setter
    def feature_list(self,value):
        raise VariableNotSettable("feature_list is not settable")
    @feature_list.deleter
    def feature_list(self):
        self._feature_list = None

    def loadImage(self, path=None, shape=None):
        """
        load image from source

        :param path: filename, url, .npy, server, image in string
        :param shape: shape to convert image
        :return: BGR image
        """
        params = (path, shape)
        if self._loader_cache is None or params != self._loader_params:
            # load new image and cache it
            fore = self.loader(path) # load fore image
            if shape is not None:
                fore = cv2.resize(fore,shape)
            self._loader_cache = fore # this keeps a reference
            self._loader_params = params
            return fore
        else: # return cached image
            return self._loader_cache

    def compute_keypoints(self):
        """
        computes key-points from file names.

        :return: self.feature_list
        """
        #################### Local features: Key-points and descriptors #################
        fns = self.filenames
        def getMask(img):
            """
            helper function to get mask
            :param img: gray image
            :return:
            """
            if callable(self.maskforeground):
                mask = self.maskforeground(img)
            if self.maskforeground is True:
                mask = foreground(img)

            if self.FLAG_DEBUG > 2:
                fastplt(overlay(img.copy(),mask*255,alpha=mask*0.5),block=True,
                        title="{} mask to detect features".format(getData(path)[-2]))
            return mask

        with TimeCode("Computing features...\n",
                      endmsg="Computed feature time was {time}\n",
                      enableMsg=self.FLAG_DEBUG):

            self._feature_list = [] # list of key points and descriptors
            for index,path in enumerate(fns):
                try:
                    if self.cachePath is None or self.clearCache==2 \
                            and path in self.feature_dic:
                        raise KeyError # clears entry from cache
                    kps, desc, shape = self.feature_dic[path] # thread safe
                    # check memoized is the same
                    if self.loadshape != shape:
                        raise KeyError
                except (KeyError, ValueError) as e: # not memorized
                    if self.FLAG_DEBUG: print "Processing features for {}...".format(path)
                    img = cv2.cvtColor(self.loadImage(path),cv2.COLOR_BGR2GRAY)
                    if self.pshape is None: # get features directly from original
                        kps, desc = self.feature.detectAndCompute(img, getMask(img))
                    else: # optimize by getting features from scaled image and the rescaling
                        oshape = self.loadshape # original shape
                        if oshape is None:
                            oshape = img.shape # get original shape from image
                        img = cv2.resize(img, self.pshape) # pseudo image
                        # get features
                        kps, desc = self.feature.detectAndCompute(img, getMask(img))
                        # re-scale keypoints to original image
                        if oshape != self.pshape:
                            # this necessarily does not produce the same result
                            """
                            # METHOD 1: using Transformation Matrix
                            H = getSOpointRelation(self.pshape, oshape, True)
                            for kp in kps:
                                kp["pt"]=tuple(cv2.perspectiveTransform(
                                    np.array([[kp["pt"]]]), H).reshape(-1, 2)[0])
                            """
                            # METHOD 2:
                            rx,ry = getSOpointRelation(self.pshape, oshape)
                            for kp in kps:
                                x,y = kp["pt"]
                                kp["pt"] = x*rx,y*ry

                    self.feature_dic[path] = kps, desc, self.loadshape # to memoize

                # add paths to key-points
                for kp in kps: # be very carful, this should not appear in self.feature_dic
                    kp["path"] = path

                # number of key-points, index, path, key-points, descriptors
                self._feature_list.append((len(kps),index,path,kps,desc))
                if self.FLAG_DEBUG: print "\rFeatures {}/{}...".format(index+1,len(fns)),
        return self._feature_list

    def restore(self):
        """
        Restore using file names (self.file_names) with
        base image (self.baseImage) and configurations.

        :return: self.restored
        """

        fns = self.filenames # in this process fns should not be changed
        ########################### Pre-selection from a set ############################
        baseImage = self.baseImage # baseImage option should not be update
        # initialization and base image selection
        if baseImage is None: # select first image as baseImage
           _,_,baseImage,kps_base,desc_base = self.feature_list[0]
        elif isinstance(baseImage,basestring):
            kps_base,desc_base = self.feature_dic[baseImage]
        elif baseImage is True: # sort images
            self.feature_list.sort(reverse=True) # descendant: from bigger to least
            # select first for most probable
            _,_,baseImage,kps_base,desc_base = self.feature_list[0]
        else:
            raise Exception("baseImage must be None, True or String")

        if self.FLAG_DEBUG: print "baseImage is", baseImage
        self.used = [baseImage] # select first image path
        # load first image for merged image
        self.restored = self.loadImage(baseImage,self.loadshape)
        self.failed = [] # registry for failed images

        ########################## Order set initialization #############################
        if self._orderValue: # obtain comparison with structure (value, path)
            if self._orderValue == 1: # entropy
                comparison = zip(*entropy(fns,
                             loadfunc=loadFunc(1,self.pshape),invert=False)[:2])
                if self.FLAG_DEBUG: print "Configured to sort by entropy..."
            elif self._orderValue == 2: # histogram comparison
                comparison = hist_comp(fns,
                            loadfunc=loadFunc(1,self.pshape),method=self.selectMethod)
                if self.FLAG_DEBUG:
                    print "Configured to sort by {}...".format(self.selectMethod)
            elif self._orderValue == 3:
                comparison = self.selectMethod(fns)
                if self.FLAG_DEBUG: print "Configured to sort by Custom Function..."
            else:
                raise Exception("DEBUG: orderValue {} does "
                    "not correspond to {}".format(self._orderValue,self.selectMethod))
        elif self.FLAG_DEBUG: print "Configured to sort by best matches"

        with TimeCode("Restoring ...\n",
            endmsg= "Restoring overall time was {time}\n",
            enableMsg= self.FLAG_DEBUG):

            while True:
                with TimeCode("Matching ...\n",
                    endmsg= "Matching overall time was {time}\n",
                    enableMsg= self.FLAG_DEBUG):
                    ################### remaining keypoints to match ####################
                    # initialize key-point and descriptor base list
                    kps_remain,desc_remain = [],[]
                    for _,_,path,kps,desc in self.feature_list:
                        # append only those which are not in the base image
                        if path not in self.used:
                            kps_remain.extend(kps)
                            desc_remain.extend(desc)

                    if not kps_remain: # if there is not image remaining to stitch break
                        if self.FLAG_DEBUG: print "All images used"
                        break

                    desc_remain = np.array(desc_remain) # convert descriptors to array

                    ############################ Matching ###############################
                    # select only those with good distance (hamming, L1, L2)
                    raw_matches = self.feature.matcher.knnMatch(desc_remain,
                                    trainDescriptors = desc_base, k = 2) #2
                    # If path=2, it will draw two match-lines for each key-point.
                    classified = {}
                    for m in raw_matches:
                        # filter by Hamming, L1 or L2 distance
                        if m[0].distance < m[1].distance * self.distanceThresh:
                            m = m[0]
                            kp1 = kps_remain[m.queryIdx]  # keypoint in query image
                            kp2 = kps_base[m.trainIdx]  # keypoint in train image

                            key = kp1["path"] # ensured that key is not in used
                            if key in classified:
                                classified[key].append((kp1,kp2))
                            else:
                                classified[key] = [(kp1,kp2)]

                    ########################## Order set ################################
                    # use only those in classified of histogram or entropy comparison
                    if self._orderValue:
                        ordered = [(val,path) for val,path
                                   in comparison if path in classified]
                    else: # order with best matches
                        ordered = sorted([(len(kps),path)
                                    for path,kps in classified.items()],reverse=True)

                # feed key-points in order according to order set
                for rank, path in ordered:

                    ######################### Calculate Homography ######################
                    mkp1,mkp2 = zip(*classified[path]) # probably good matches
                    if len(mkp1)>self.minKps and len(mkp2)>self.minKps:

                        # get only key-points
                        p1 = np.float32([kp["pt"] for kp in mkp1])
                        p2 = np.float32([kp["pt"] for kp in mkp2])
                        if self.FLAG_DEBUG > 1:
                            print 'Calculating Homography for {}...'.format(path)

                        # Calculate homography of fore over back
                        H, status = cv2.findHomography(p1, p2,
                                    cv2.RANSAC, self.ransacReprojThreshold)
                    else: # not sufficient key-points
                        if self.FLAG_DEBUG > 1:
                            print 'Not enough key-points for {}...'.format(path)
                        H = None

                    # test that there is homography
                    if H is not None: # first test
                        # load fore image
                        fore = self.loadImage(path,self.loadshape)
                        h,w = fore.shape[:2] # image shape

                        # get corners of fore projection over back
                        projection = getTransformedCorners((h,w),H)
                        c = imcoors(projection) # class to calculate statistical data
                        lines, inlines = len(status), np.sum(status)

                        # ratio to determine how good fore is in back
                        inlineratio = inlineRatio(inlines,lines)

                        Test = inlineratio>self.inlineThresh \
                                and c.rotatedRectangularity>self.rectangularityThresh

                        text = "inlines/lines: {}/{}={}, " \
                               "rectangularity: {}, test: {}".format(
                            inlines, lines, inlineratio, c.rotatedRectangularity,
                            ("failed","succeeded")[Test])

                        if self.FLAG_DEBUG>1: print text

                        if self.FLAG_DEBUG > 2: # show matches
                            matchExplorer("Match " + text, fore,
                                          self.restored, classified[path], status, H)

                        ####################### probability test ########################
                        if Test: # second test

                            if self.FLAG_DEBUG>1: print "Test succeeded..."
                            while path in self.failed: # clean path in fail registry
                                try: # race-conditions safe
                                    self.failed.remove(path)
                                except ValueError:
                                    pass

                            ################### merging and stitching ###################
                            self.merge(path, H)
                            if not self.centric:
                                break
                        else:
                            self.failed.append(path)
                    else:
                        self.failed.append(path)

                # if all classified have failed then end
                if set(classified.keys()) == set(self.failed):
                    if self.FLAG_DEBUG:
                        print "Restoration finished, these images do not fit: "
                        for index in classified.keys():
                            print index
                    break

        with TimeCode("Post-processing ...\n",
            endmsg= "Post-processing overall time was {time}\n",
            enableMsg= self.FLAG_DEBUG):
            processed = self.post_process_restoration(self.restored)
            if processed is not None:
                self.restored = processed

        #################################### Save image #################################
        if self.save:
            self.saveImage()

        return self.restored # return merged image

    def saveImage(self, path = None, overwrite = None):
        """
        save restored image in path.

        :param path: filename, string to format or path to save image.
                if path is not a string it would be replaced with the string
                "{path}restored_{name}{ext}" to format with the formatting
                "{path}", "{name}" and "{ext}" from the baseImage variable.
        :return: status, saved path
        """
        if path is None:
            path = self.save
        if overwrite is None:
            overwrite = self.overwrite

        bbase, bpath, bname, bext = getData(self.used[0])
        if isinstance(path,basestring):
            # format path if user has specified so
            data = getData(self.save.format(path="".join((bbase, bpath)),
                                            name=bname, ext=bext))
            # complete any data lacking in path
            for i,(n,b) in enumerate(zip(data,(bbase, bpath, bname, bext))):
                if not n: data[i] = b
        else:
            data = bbase,bpath,"restored_",bname,bext
        # joint parts to get string
        fn = "".join(data)
        mkPath(getPath(fn))

        if not overwrite:
            fn = increment_if_exits(fn)

        r = cv2.imwrite(fn,self.restored)
        if self.FLAG_DEBUG and r:
            print "Saved: {}".format(fn)
            return True, fn
        else:
            print "{} could not be saved".format(fn)
            return False, fn

    def merge(self, path, H, shape = None):
        """
        merge image to main restored image.

        :param path: file name to load image
        :param H: Transformation matrix of image in path over restored image.
        :param shape: custom shape to load image in path
        :return: self.restored
        """
        if shape is None:
            shape = self.loadshape

        fore = self.loadImage(path,shape) # load fore image

        if self.histMatch: # apply histogram matching
            fore = hist_match(fore, self.restored)

        if self.FLAG_DEBUG > 1: print "Merging..."

        # pre process alpha mask
        alpha = self.pre_process_fore_Mask(self.restored,fore,H)

        # process expert alpha mask if alpha was not provided by the user
        if alpha is None and self.expert is not None:

            if not hasattr(self,"_restored_mask"):

                bname = "".join(getData(self.used[-1])[-2:])
                try:
                    bdata = self.expert[bname]
                except KeyError:
                    raise KeyError("{} is not in self.expert".format(bname))

                bsh = bdata["shape"]
                bm_retina = contours2mask(bdata["coors_retina"],bsh)
                bm_otic_disc = contours2mask(bdata["coors_optic_disc"],bsh)
                bm_defects = contours2mask(bdata["coors_defects"],bsh)

                self._restored_mask = np.logical_and(np.logical_or(np.logical_not(bm_retina),
                                            bm_defects), np.logical_not(bm_otic_disc))

            fname = "".join(getData(path)[-2:])
            try:
                fdata = self.expert[fname]
            except KeyError:
                raise KeyError("{} is not in self.expert".format(fname))

            fsh = fdata["shape"]
            fm_retina = contours2mask(fdata["coors_retina"],fsh)
            #fm_otic_disc = contours2mask(fdata["coors_optic_disc"],fsh)
            fm_defects = contours2mask(fdata["coors_defects"],fsh)

            fmask = np.logical_and(fm_retina,np.logical_not(fm_defects))

            self._restored_mask = maskm = np.logical_and(self._restored_mask,fmask)

            h, w = self.restored.shape[:2]
            alpha = cv2.warpPerspective(maskm.copy().astype(np.uint8), H, (w, h))

        ################################### SUPERPOSE ###################################

        # fore on top of back
        alpha_shape = fore.shape[:2]
        if self.grow_scene: # this makes the images bigger if possible
            # fore(x,y)*H = fore(u,v) -> fore(u,v) + back(u,v)
            ((left,top),(right,bottom)) = pad_to_fit_H(fore.shape, self.restored.shape, H)
             # moved transformation matrix with pad
            H_back = FLOAT([[1,0,left],[0,1,top],[0,0,1]]) # in back
            H_fore = H_back.dot(H) # in fore
            # need: top_left, bottom_left, top_right,bottom_right
            h2,w2 = self.restored.shape[:2]
            w,h = int(left + right + w2),int(top + bottom + h2)
            # this memory inefficient, image is copied to prevent cross-references
            self.restored = cv2.warpPerspective(self.restored.copy(), H_back, (w, h))
            fore = cv2.warpPerspective(fore.copy(), H_fore, (w, h))
        else: # this keeps back shape
            H_fore = H
            H_back = np.eye(3)
            h, w = self.restored.shape[:2]
            fore = cv2.warpPerspective(fore.copy(), H_fore, (w, h))

        if alpha is None: # if no pre-processing function for alpha implemented
            alpha = self.post_process_fore_Mask(self.restored,fore)

        if alpha is None: # create valid mask for stitching
            alpha = cv2.warpPerspective(np.ones(alpha_shape), H_fore, (w, h))

        self.restored = overlay(self.restored, fore, alpha) # overlay fore on top of back

        if H_fore is None: # H is not modified use itself
            H_fore = H

        if self.FLAG_DEBUG > 1: # show merging result
            plotim("Last added from {}".format(path), self.restored).show()

        ####################### update base features #######################
        # make projection to test key-points inside it
        if self.FLAG_DEBUG > 1: print "Updating key-points..."
        # fore projection in restored image
        projection = getTransformedCorners(fore.shape[:2],H_fore)
        # update key-points
        newkps, newdesc = [], []
        for _,_,p,kps,desc in self.feature_list:
            # append all points in the base image and update their position
            if p in self.used: # transform points in back
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

        if self.FLAG_DEBUG > 2: # show keypints in merging
            plotim("merged Key-points", # draw key-points in image
                   cv2.drawKeypoints(
                       im2shapeFormat(self.restored,self.restored.shape[:2]+(3,)),
                              [dict2keyPoint(index) for index in kps_base],
                              flags=4, color=(0,0,255))).show()
        if self.FLAG_DEBUG: print "This image has been merged: {}...".format(path)
        self.used.append(path) # update used

        return self.restored

    def post_process_restoration(self, image):
        pass

    def post_process_fore_Mask(self, back, fore):
        pass

    def pre_process_fore_Mask(self, back, fore, H):
        pass

class RetinalRestore(ImRestore):
    """
    Restore retinal images by merging and stitching techniques. These parameters are
    added to :class:`ImRestore`:

    :param heavynoise: True to process noisy images, False to
        process normal images.
    :param lens: flag to determine if lens are applied. True
        to simulate lens, False to not apply lens.
    :param enclose: flag to enclose and return only retinal area.
        True to return ROI, false to leave image "as is".
    """
    def __init__(self, filenames, **opts):
        super(RetinalRestore,self).__init__(filenames, **opts)
        self.maskforeground = opts.get("maskforeground",lambda img: retinal_mask(img,True))
        self.heavynoise=opts.get("heavynoise",False)
        self.lens = opts.get("lens",True)
        self.enclose = opts.get("enclose",False)

    #__init__.__doc__ = ImRestore.__init__.__doc__+__init__.__doc__

    def post_process_fore_Mask(self, back, fore):
        """
        Function evaluated just after superposition and before overlay.

        :param back: background image
        :param fore: foreground image
        :return: alpha mask
        """

        def _mask(back,fore):
            """
            get bright alpha mask (using histogram method)

            :param back: BGR background image
            :param fore: BGR foreground image
            :return: alpha mask
            """
            # TODO, not working for every retianl scenario
            foregray = brightness(fore)
            # get window with Otsu to prevent expansion
            thresh,w = cv2.threshold(foregray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return getBrightAlpha(brightness(back).astype(float),
                               foregray.astype(float), window=w)

        def get_beta_params(P):
            """
            Automatically find parameters for bright alpha masks.

            :param P: gray image
            :return: beta1,beta2
            """
            # process histogram for uint8 gray image
            hist, bins = np.histogram(P.flatten(), 256)

            # get Otsu thresh as beta2
            beta2 = bins[getOtsuThresh(hist)]
            return np.min(P), beta2 # beta1, beta2

        def mask(back, fore):
            """
            Get bright alpha mask (using Otsu method)

            :param back: BGR background image
            :param fore: BGR foreground image
            :return: alpha mask
            """
            # find retinal area and its alpha
            mask_back, alpha_back = retinal_mask(back,biggest=True,addalpha=True)
            mask_fore, alpha_fore = retinal_mask(fore,biggest=True,addalpha=True)

            # convert uint8 to float
            backgray = brightness(back).astype(float)
            foregray = brightness(fore).astype(float)

            # scale from 0-1 to 0-255
            backm = alpha_back*255
            forem = alpha_fore*255

            # get alpha masks fro background and foreground
            backmask = bandstop(3, *get_beta_params(backm[mask_back.astype(bool)]))(backgray)
            foremask = bandpass(3, *get_beta_params(forem[mask_fore.astype(bool)]))(foregray)

            # merge masks
            alphamask = normalize(foremask * backmask * (backm/255.))
            return alphamask


        pshape = (400,400) # process shape
        # rescaling of the image to process mask
        if pshape is not None:
            oshape = back.shape[:2]
            back = cv2.resize(back,pshape)
            fore = cv2.resize(fore,pshape)

        # get alpha mask
        alphamask = mask(back,fore)

        # rescaling mask to original shape
        if pshape is not None:
            alphamask = cv2.resize(alphamask,oshape[::-1])

        return alphamask

    def post_process_restoration(self, image):
        """
        Post-process a merged retinal image.

        :param image: retinal image
        :return: filtered and with simulated lens
        """
        # detect how much noise to process and convert it to beta parameters
        if self.heavynoise: # slower but interactive for heavy noise
            params = getBilateralParameters(image.shape)
        else:
            params = 15,82,57 # 21,75,75 # faster and for low noise

        # filter using parameters and bilateral filter
        image = cv2.bilateralFilter(image, *params)

        # simulation of lens
        if self.lens:
            # call function to overlay lens on image
            try:
                image = simulateLens(image)
            except Exception as e:
                warnings.warn("simulate lens failed with error {}".format(e))

        # crop retinal area only
        if self.enclose:
            # convert to gray
            gray = brightness(image)
            # get object mask
            mask = foreground(gray)
            # get contour of biggest area
            cnt = thresh_biggestCnt(mask)
            # get enclosure box
            x,y,w,h = cv2.boundingRect(cnt)
            # crop image
            if len(image.shape)>2:
                image = image[x:x+w,y:y+h,:]
            else:
                image = image[x:x+w,y:y+h]

        return image

def True_or_String(string):
    """
    Process string to get string.

    :return: "" then it is true, else returns the string
    """
    if string == "":
        return True
    return string

def feature_creator(string):
    """
    Converts a string to a feature object.

    :param string: any supported feature detector in openCV. the format is
            "[a-]<sift|surf|orb>[-flann]" (str) Ex: "a-sift-flann" where
            "a-" or "-flann" are optional.
    :return: feature object
    """
    if string == "":
        return None
    return Feature().config(string)

def tuple_creator(string):
    """
    Process string to get tuple.

    :param string: string parameters with "," (colon) as separator
            Ex: param1,param2,param3
    :return: "" then it is true, else returns the tuple
    """
    if string == "":
        return None
    return tuple([int(i) for i in string.split(",")])

def loader_creator(string):
    # TODO interpret string to get loader with any feature
    return

class NameSpace(object):
    pass

def shell(args=None, namespace=None):
    """
    Shell to run in terminal the imrestore program

    :param args: (None) list of arguments. If None it captures the
            arguments in sys.
    :param namespace: (None) namespace to place variables. If None
            it creates a namespace.
    :return: namespace
    """
    if namespace is None:
        namespace = NameSpace()
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Restore images by merging and stitching "
                                                 "techniques.",
                                     epilog=__doc__)
    parser.add_argument('filenames', nargs='*',
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
                            'original images. If None (e.g "") it loads the original '
                            'images to process the features but it can incur to performance'
                            ' penalties if images are too big and RAM memory is scarce')
    parser.add_argument('-b','--baseImage', default=True, type=True_or_String,
                       help='First image to merge to. If None (e.g "") '
                            'selects image with most features'
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
                       help='Apply histogram matching to foreground '
                            'image with merge image as template')
    parser.add_argument('-s','--save', default=True,
                       help='Customize image name used to save the restored image.'
                            'By default it saves in path with name restored_{base_image}')
    parser.add_argument('--restorer',choices = ['RetinalRestore','ImRestore'],
                        default='RetinalRestore',
                       help='imrestore is for images in general but it can be parametrized. '
                            'By default it has the profile "retinalRestore" for retinal '
                            'images but its general behaviour can be restorerd by '
                            'changing it to "imrestore"')
    parser.add_argument('--expert', default=None,
                       help='path to expert variables')

    # parse sys and get argument variables
    args = vars(parser.parse_args(args=args, namespace=namespace))

    # this is needed because the shell process wildcards before it gets to argparse
    # creating a list in the path thus it must be filtered. Use quotes in shell
    # to prevent this behaviour
    args['filenames'] = [p for p in args['filenames'] if os.path.isfile(p) or "*" in p]

    # print parsed arguments
    if args['debug']>2:
        print "Parsed Arguments\n",args

    # use configuration
    r = args.pop("restorer")
    if r == 'RetinalRestore':
        if args['debug']: print "Configured for retinal restoration..."
        obj = RetinalRestore(**args)
    elif r == 'ImRestore':
        if args['debug']: print "Configured for general restoration..."
        obj = ImRestore(**args)
    else:
        raise Exception("no restoration class called {}".format(r))

    if namespace is not None:
        # update namespace from early stages so it can have access to the restorer
        namespace.restorer = obj

    # start restoration
    obj.restore()

    return namespace # return given namespace

if __name__ == "__main__":
    shell() # run the shell
    # TODO visualizator for alpha mask
    # TODO implement standard deviation in bright areas to detect optic disk