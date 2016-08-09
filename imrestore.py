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

    (7) Stitching and Merging
        (7.1) Histogram matching* (color)
        (7.2) Segmentation*
        (7.3) Alpha mask calculation*
        (7.4) Overlay

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
        if not well processed and precise information is given or calculated
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
# program imports
import os
import cv2
import warnings
import numpy as np
from RRtoolbox.tools.lens import simulateLens
from RRtoolbox.lib.config import MANAGER, FLOAT
from RRtoolbox.lib.image import hist_match
from RRtoolbox.lib.directory import getData, getPath, mkPath, increment_if_exits
from RRtoolbox.lib.cache import MemoizedDict
from RRtoolbox.lib.image import loadFunc, Imcoors
from RRtoolbox.lib.arrayops.mask import brightness, foreground, thresh_biggestCnt
from multiprocessing.pool import ThreadPool as Pool
from RRtoolbox.tools.selectors import hist_map, hist_comp, entropy
from RRtoolbox.tools.segmentation import retinal_mask, get_layered_alpha
from RRtoolbox.lib.root import TimeCode, glob, lookinglob, Profiler, VariableNotSettable
from RRtoolbox.lib.descriptors import Feature, inlineRatio
from RRtoolbox.tools.segmentation import get_bright_alpha, Bandpass, Bandstop
from RRtoolbox.lib.plotter import MatchExplorer, Plotim, fastplt
from RRtoolbox.lib.arrayops.filters import getBilateralParameters
from RRtoolbox.lib.arrayops.convert import getSOpointRelation, dict2keyPoint
from RRtoolbox.lib.arrayops.basic import getTransformedCorners, transformPoint, \
    im2shapeFormat,normalize, getOtsuThresh, contours2mask, pad_to_fit_H, overlay


def check_valid(fn):
    """
    checks that a file is valid for loading.
    :param fn: filename
    :return: True for valid, False for invalid.
    """
    test = os.path.isfile(fn)
    if test and getData(fn)[-2].startswith("_"):
        return False
    return test

class ImRestore(object):
    """
    Restore images by merging and stitching techniques.

    :param filenames: list of images or string to path which uses glob filter in path.
            Loads image array from path, url, server, string
            or directly from numpy array (supports databases)
    :param debug: (0) flag to print messages and debug data.
            0 -> do not print messages.
            1 -> print normal messages.
            2 -> print normal and debug messages.
            3 -> print all messages and show main results.
                (consumes significantly more memory).
            4 -> print all messages and show all stage results.
                (consumes significantly more memory).
            5 -> print all messages, show all results and additional data.
                (consumes significantly more memory).
    :param feature: (None) feature instance. It contains the configured
            detector and matcher.
    :param pool: (None) use pool Ex: 4 to use 4 CPUs.
    :param cachePath: (None) saves memoization to specified path. This is
            useful to save some computations and use them in next executions.
            If True it creates the cache in current path.

            .. warning:: Cached data is not guaranteed to work between different
                        configurations and this can lead to unexpected program
                        behaviour. If a different configuration will be used it
                        is recommended to clear the cache to recompute values.
    :param clearCache: (0) clear cache flag.
            * 0 do not clear.
            * 1 re-compute data but other cache data is left intact.
            * 2 All CachePath is cleared before use.
            Notes: using cache can result in unexpected behaviour
                if some configurations does not match to the cached data.
    :param loader: (None) custom loader function used to load images.
            If None it loads the original images in color.
    :param process_shape: (400,400) process shape, used to load pseudo images
            to process features and then results are converted to the
            original images. The smaller the image more memory and speed gain
            If None it loads the original images to process the features but it
            can incur to performance penalties if images are too big and RAM
            memory is scarce.
    :param load_shape: (None) custom shape used load images which are being merged.
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
    :param maskforeground:(False)
            * True, limit features area using foreground mask of input images.
                This mask is calculated to threshold a well defined object.
            * Callable, Custom function to produce the foreground image which
                receives the input gray image and must return the mask image
                where the keypoints will be processed.
    :param noisefunc: True to process noisy images or provide function.
    :param save: (False)
            * True, saves in path with name _restored_{base_image}
            * False, does not save
            * Image name used to save the restored image.
    :param overwrite: If True and the destine filename for saving already
            exists then it is replaced, else a new filename is generated
            with an index "{filename}_{index}.{extension}"
    """

    def __init__(self, filenames, **opts):
        self.profiler = opts.pop("profiler",None)
        if self.profiler is None:
            self.profiler = Profiler("ImRestore init")

        self.log_saved = [] # keeps track of last saved file.

        # for debug
        self.verbosity = opts.pop("verbosity", 1)

        ################################## GET IMAGES ####################################
        if filenames is None or len(filenames)==0: # if images is empty use demonstration
            #test = MANAGER["TESTPATH"]
            #if self.verbosity: print "Looking in DEMO path {}".format(test)
            #fns = glob(test + "*",check=check_valid)
            raise Exception("List of filenames is Empty")
        elif isinstance(filenames, basestring):
            # if string assume it is a path
            if self.verbosity: print "Looking as {}".format(filenames)
            fns = glob(filenames,check=check_valid)
        elif not isinstance(filenames, basestring) and \
                        len(filenames) == 1 and "*" in filenames[0]:
            filenames = filenames[0] # get string
            if self.verbosity: print "Looking as {}".format(filenames)
            fns = glob(filenames,check=check_valid)
        else: # iterator containing data
            fns = filenames # list file names

        # check images
        if not len(fns)>1:
            raise Exception("list of images must be "
                            "greater than 1, got {}".format(len(fns)))

        self.filenames = fns

        # for multiprocessing
        self.pool = opts.pop("pool",None)
        if self.pool is not None: # convert pool count to pool class
            NO_CPU = cv2.popNumberOfCPUs()
            if self.pool <= NO_CPU:
                self.pool = Pool(processes = self.pool)
            else:
                raise Exception("pool of {} exceeds the "
                                "number of processors {}".format(self.pool,NO_CPU))
        # for features
        self.feature = opts.pop("feature",None)
        # init detector and matcher to compute descriptors
        if self.feature is None:
            self.feature = Feature(pool=self.pool, debug=self.verbosity)
            self.feature.config(name='a-sift-flann')
        else:
            self.feature.pool = self.pool
            self.feature.debug = self.verbosity

        # select method to order images to feed in superposition
        self.selectMethod = opts.pop("selectMethod",None)
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
        self.distanceThresh = opts.pop("distanceThresh",0.75) # filter ratio

        # threshold for inlineRatio
        self.inlineThresh = opts.pop("inlineThresh",0.2) # filter ratio
        # ensures adequate value [0,1]
        assert self.inlineThresh<=1 and self.inlineThresh>=0

        # threshold for rectangularity
        self.rectangularityThresh = opts.pop("rectangularityThresh",0.5) # filter ratio
        # ensures adequate value [0,1]
        assert self.rectangularityThresh<=1 and self.rectangularityThresh>=0

        # threshold to for RANSAC reprojection
        self.ransacReprojThreshold = opts.pop("ransacReprojThreshold",5.0)

        self.centric = opts.pop("centric",False) # tries to attach as many images as possible
        # it is not memory efficient to compute descriptors from big images
        self.process_shape = opts.pop("process_shape", (400, 400)) # use processing shape
        self.load_shape = opts.pop("load_shape", None) # shape to load images for merging
        self.minKps = 3 # minimum len of key-points to find Homography
        self.histMatch = opts.pop("hist_match",False)
        self.denoise=opts.pop("denoise", None)

        ############################## OPTIMIZATION MEMOIZEDIC ###########################
        self.cachePath = opts.pop("cachePath",None)
        self.clearCache = opts.pop("clearCache",0)
        if self.cachePath is not None:
            if self.cachePath is True:
                self.cachePath = os.path.abspath(".") # MANAGER["TEMPPATH"]
            if self.cachePath == "{temp}":
                self.cachePath = self.cachePath.format(temp=MANAGER["TEMPPATH"])
            self.feature_dic = MemoizedDict(os.path.join(self.cachePath, "descriptors"))
            if self.verbosity: print "Cache path is in {}".format(self.feature_dic._path)
            if self.clearCache==2:
                self.feature_dic.clear()
                if self.verbosity: print "Cache path cleared"
        else:
            self.feature_dic = {}

        self.expert = opts.pop("expert",None)
        if self.expert is not None:
            self.expert = MemoizedDict(self.expert) # convert path

        # to select base image ahead of any process
        baseImage = opts.pop("baseImage",None)
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

        if self.verbosity: print "No. images {}...".format(len(fns))

        # make loader
        self.loader = opts.pop("loader",None) # BGR loader
        if self.loader is None: self.loader = loadFunc(1)
        self._loader_cache = None # keeps last image reference
        self._loader_params = None # keeps last track of last loading options to reload

        self.save = opts.pop("save",False)
        self.grow_scene = opts.pop("grow_scene",True)
        self.maskforeground = opts.pop("maskforeground",False)
        self.overwrite = opts.pop("overwrite",False)

        # do a check of the options
        if opts:
            raise Exception("Unknown keyword(s) {}".format(opts.keys()))

        # processing variables
        self._feature_list = None
        self.used = None
        self.failed = None
        self.restored = None
        self.kps_base,self.desc_base = None,None

    @property
    def denoise(self):
        return self._noisefunc
    @denoise.setter
    def denoise(self, value):
        if value is False:
            value = None
        if value is True:
            value = "mild"
        if value in ("mild","heavy","normal",None) or callable(value):
            self._noisefunc = value
        else:
            raise Exception("denoise '{}' not recognised".format(value))
    @denoise.deleter
    def denoise(self):
        del self._noisefunc

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

    def load_image(self, path=None, shape=None):
        """
        load image from source

        :param path: filename, url, .npy, server, image in string
        :param shape: shape to convert image
        :return: BGR image
        """
        params = (path, shape)
        if self._loader_cache is None or params != self._loader_params:
            # load new image and cache it
            img = self.loader(path) # load image
            if shape is not None:
                img = cv2.resize(img,shape)
            self._loader_cache = img # this keeps a reference
            self._loader_params = params
            return img
        else: # return cached image
            return self._loader_cache

    def compute_keypoints(self):
        """
        Computes key-points from file names.

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
            mask = None
            if callable(self.maskforeground):
                mask = self.maskforeground(img)
            if self.maskforeground is True:
                mask = foreground(img)

            if mask is not None and self.verbosity > 4:
                fastplt(overlay(img.copy(),mask*255,alpha=mask*0.5),block=True,
                        title="{} mask to detect features".format(getData(path)[-2]))
            return mask

        with TimeCode("Computing features...\n", profiler=self.profiler,
                      profile_point=("Computing features",),
                      endmsg="Computed feature time was {time}\n",
                      enableMsg=self.verbosity) as timerK:

            self._feature_list = [] # list of key points and descriptors
            for index,path in enumerate(fns):
                img = self.load_image(path, self.load_shape)
                lshape = img.shape[:2]
                try:
                    point = Profiler(msg=path, tag="cached")
                    if self.cachePath is None or self.clearCache==1 \
                            and path in self.feature_dic:
                        raise KeyError # clears entry from cache
                    kps, desc, pshape = self.feature_dic[path] # thread safe
                    if pshape is None:
                        raise ValueError
                except (KeyError, ValueError) as e: # not memorized
                    point = Profiler(msg=path, tag="processed")
                    if self.verbosity: print "Processing features for {}...".format(path)
                    if lshape != self.process_shape:
                        img = cv2.resize(img, self.process_shape)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    # get features
                    kps, desc = self.feature.detectAndCompute(img, getMask(img))
                    pshape = img.shape[:2] # get process shape
                    # to memoize
                    self.feature_dic[path] = kps, desc, pshape

                # re-scale keypoints to original image
                if lshape != pshape:
                    # this necessarily does not produce the same result
                    """
                    # METHOD 1: using Transformation Matrix
                    H = getSOpointRelation(process_shape, lshape, True)
                    for kp in kps:
                        kp["pt"]=tuple(cv2.perspectiveTransform(
                            np.array([[kp["pt"]]]), H).reshape(-1, 2)[0])
                    """
                    # METHOD 2:
                    rx,ry = getSOpointRelation(pshape, lshape)
                    for kp in kps:
                        x,y = kp["pt"]
                        kp["pt"] = x*rx,y*ry
                        kp["path"] = path
                else:
                    for kp in kps: # be very carful, this should not appear in self.feature_dic
                        kp["path"] = path # add paths to key-points

                # number of key-points, index, path, key-points, descriptors
                self._feature_list.append((len(kps),index,path,kps,desc))
                if self.verbosity: print "\rFeatures {}/{}...".format(index + 1, len(fns)),
                # for profiling individual processing times
                if self.profiler is not None: self.profiler._close_point(point)

        return self._feature_list
    
    def pre_selection(self):
        """
        This method selects the first restored image so that self.restored is initialized
        with a numpy array and self.used should specify the used image preferably in
        self.feature_list.

        :return: None
        """
        ########################### Pre-selection from a set ############################
        baseImage = self.baseImage # baseImage option should not be update
        # initialization and base image selection
        if baseImage is None: # select first image as baseImage
           _,_,baseImage,self.kps_base,self.desc_base = self.feature_list[0]
        elif isinstance(baseImage,basestring):
            self.kps_base,self.desc_base = self.feature_dic[baseImage]
        elif baseImage is True: # sort images
            self.feature_list.sort(reverse=True) # descendant: from bigger to least
            # select first for most probable
            _,_,baseImage,self.kps_base,self.desc_base = self.feature_list[0]
        else:
            raise Exception("baseImage must be None, True or String")

        if self.verbosity: print "baseImage is", baseImage
        self.used = [baseImage] # select first image path
        # load first image for merged image
        self.restored = self.load_image(baseImage, self.load_shape)

    def restore(self):
        """
        Restore using file names (self.file_names) with base image (self.baseImage
        calculated from self.pre_selection()) and other configurations.

        :return: self.restored
        """
        self.pre_selection()
        self.failed = [] # registry for failed images
        fns = self.filenames # in this process fns should not be changed
        ########################## Order set initialization #############################
        if self._orderValue: # obtain comparison with structure (value, path)
            if self._orderValue == 1: # entropy
                comparison = zip(*entropy(fns, loadfunc=loadFunc(1, self.process_shape),
                                          invert=False)[:2])
                if self.verbosity: print "Configured to sort by entropy..."
            elif self._orderValue == 2: # histogram comparison
                comparison = hist_comp(fns, loadfunc=loadFunc(1, self.process_shape),
                                       method=self.selectMethod)
                if self.verbosity:
                    print "Configured to sort by {}...".format(self.selectMethod)
            elif self._orderValue == 3:
                comparison = self.selectMethod(fns)
                if self.verbosity: print "Configured to sort by Custom Function..."
            else:
                raise Exception("DEBUG: orderValue {} does "
                    "not correspond to {}".format(self._orderValue,self.selectMethod))
        elif self.verbosity: print "Configured to sort by best matches"

        with TimeCode("Restoring ...\n",profiler=self.profiler,
                      profile_point=("Restoring",),
                      endmsg= "Restoring overall time was {time}\n",
                      enableMsg= self.verbosity) as timerR:

            while True:
                with TimeCode("Matching ...\n",profiler=self.profiler,
                              profile_point=("Matching",),
                              endmsg= "Matching overall time was {time}\n",
                              enableMsg= self.verbosity) as timerM:
                    ################### remaining keypoints to match ####################
                    # initialize key-point and descriptor base list
                    kps_remain,desc_remain = [],[]
                    for _,_,path,kps,desc in self.feature_list:
                        # append only those which are not in the base image
                        if path not in self.used:
                            kps_remain.extend(kps)
                            desc_remain.extend(desc)

                    if not kps_remain: # if there is not image remaining to stitch break
                        if self.verbosity: print "All images used"
                        break

                    desc_remain = np.array(desc_remain) # convert descriptors to array

                    ############################ Matching ###############################
                    # select only those with good distance (hamming, L1, L2)
                    raw_matches = self.feature.matcher.knnMatch(desc_remain,
                                    trainDescriptors = self.desc_base, k = 2) #2
                    # If path=2, it will draw two match-lines for each key-point.
                    classified = {}
                    for m in raw_matches:
                        # filter by Hamming, L1 or L2 distance
                        if m[0].distance < m[1].distance * self.distanceThresh:
                            m = m[0]
                            kp1 = kps_remain[m.queryIdx]  # keypoint in query image
                            kp2 = self.kps_base[m.trainIdx]  # keypoint in train image

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


                with TimeCode("Merging ...\n",profiler=self.profiler,
                              profile_point=("Merging",),
                              endmsg= "Merging overall time was {time}\n",
                              enableMsg= self.verbosity) as timerH:

                    # feed key-points in order according to order set
                    for rank, path in ordered:
                        point = Profiler(msg=path) # profiling point
                        ######################### Calculate Homography ###################
                        mkp1,mkp2 = zip(*classified[path]) # probably good matches
                        if len(mkp1)>self.minKps and len(mkp2)>self.minKps:

                            # get only key-points
                            p1 = np.float32([kp["pt"] for kp in mkp1])
                            p2 = np.float32([kp["pt"] for kp in mkp2])
                            if self.verbosity > 4:
                                print 'Calculating Homography for {}...'.format(path)

                            # Calculate homography of fore over back
                            H, status = cv2.findHomography(p1, p2,
                                        cv2.RANSAC, self.ransacReprojThreshold)
                        else: # not sufficient key-points
                            if self.verbosity > 1:
                                print 'Not enough key-points for {}...'.format(path)
                            H = None

                        # test that there is homography
                        if H is not None: # first test
                            # load fore image
                            fore = self.load_image(path, self.load_shape)
                            h,w = fore.shape[:2] # image shape

                            # get corners of fore projection over back
                            projection = getTransformedCorners((h,w),H)
                            c = Imcoors(projection) # class to calculate statistical data
                            lines, inlines = len(status), np.sum(status)

                            # ratio to determine how good fore is in back
                            inlineratio = inlineRatio(inlines,lines)

                            Test = inlineratio>self.inlineThresh \
                                    and c.rotatedRectangularity>self.rectangularityThresh

                            text = "inlines/lines: {}/{}={}, " \
                                   "rectangularity: {}, test: {}".format(
                                inlines, lines, inlineratio, c.rotatedRectangularity,
                                ("failed","succeeded")[Test])

                            if self.verbosity>1: print text

                            if self.verbosity > 3: # show matches
                                MatchExplorer("Match " + text, fore,
                                              self.restored, classified[path], status, H)

                            ####################### probability test #####################
                            if Test: # second test

                                if self.verbosity>1: print "Test succeeded..."
                                while path in self.failed: # clean path in fail registry
                                    try: # race-conditions safe
                                        self.failed.remove(path)
                                    except ValueError:
                                        pass

                                ################### merging and stitching ################
                                self.merge(path, H)

                                # used for profiling
                                if self.profiler is not None:
                                    self.profiler._close_point(point)

                                if not self.centric:
                                    break
                            else:
                                self.failed.append(path)
                        else:
                            self.failed.append(path)

                # if all classified have failed then end
                if set(classified.keys()) == set(self.failed):
                    if self.verbosity:
                        print "Restoration finished, these images do not fit: "
                        for index in classified.keys():
                            print index
                    break

        with TimeCode("Post-processing ...\n",profiler=self.profiler,
                      profile_point=("Post-processing",),
                      endmsg= "Post-processing overall time was {time}\n",
                      enableMsg= self.verbosity) as timerP:
            processed = self.post_process_restoration(self.restored)
            if processed is not None:
                self.restored = processed

        # profiling post-processing
        self.time_postprocessing = timerP.time_end

        #################################### Save image ##################################
        if self.save:
            self.save_image()

        return self.restored # return merged image

    def save_image(self, path = None, overwrite = None):
        """
        save restored image in path.

        :param path: filename, string to format or path to save image.
                if path is not a string it would be replaced with the string
                "{path}restored_{name}{ext}" to format with the formatting
                "{path}", "{name}" and "{ext}" from the baseImage variable.
        :param overwrite: If True and the destine filename for saving already
            exists then it is replaced, else a new filename is generated
            with an index "{filename}_{index}.{extension}"
        :return: saved path, status (True for success and False for fail)
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
            data = bbase,bpath,"_restored_",bname,bext
        # joint parts to get string
        fn = "".join(data)
        mkPath(getPath(fn))

        if not overwrite:
            fn = increment_if_exits(fn)

        if cv2.imwrite(fn,self.restored):
            if self.verbosity: print "Saved: {}".format(fn)
            self.log_saved.append(fn)
            return fn, True
        else:
            if self.verbosity: print "{} could not be saved".format(fn)
            return fn, False

    def merge(self, path, H, shape = None):
        """
        Merge image to main restored image.

        :param path: file name to load image
        :param H: Transformation matrix of image in path over restored image.
        :param shape: custom shape to load image in path
        :return: self.restored
        """
        alpha = None

        if shape is None:
            shape = self.load_shape

        fore = self.load_image(path, shape) # load fore image

        if self.histMatch: # apply histogram matching
            fore = hist_match(fore, self.restored)

        if self.verbosity > 1: print "Merging..."

        # process expert alpha mask if alpha was not provided by the user
        if self.expert is not None:

            # process _restored_mask if None
            if not hasattr(self,"_restored_mask"):
                # from path/name.ext get only name.ext
                bname = "".join(getData(self.used[-1])[-2:])
                try:
                    bdata = self.expert[bname]
                    bsh = bdata["shape"]
                    bm_retina = contours2mask(bdata["coors_retina"],bsh)
                    bm_otic_disc = contours2mask(bdata["coors_optic_disc"],bsh)
                    bm_defects = contours2mask(bdata["coors_defects"],bsh)

                    self._restored_mask = np.logical_and(np.logical_or(np.logical_not(bm_retina),
                                                bm_defects), np.logical_not(bm_otic_disc))
                except Exception as e:
                    #exc_type, exc_value, exc_traceback = sys.exc_info()
                    #lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    warnings.warn("Error using expert {} to create self._restored_mask:"
                                  " {}{}".format(bname,type(e),e.args))

            # only if there is a _restored_mask
            if hasattr(self,"_restored_mask"):
                fname = "".join(getData(path)[-2:])
                try:
                    fdata = self.expert[fname]
                    fsh = fdata["shape"]
                    fm_retina = contours2mask(fdata["coors_retina"],fsh)
                    #fm_otic_disc = contours2mask(fdata["coors_optic_disc"],fsh)
                    fm_defects = contours2mask(fdata["coors_defects"],fsh)

                    fmask = np.logical_and(fm_retina,np.logical_not(fm_defects))

                    self._restored_mask = maskm = np.logical_and(self._restored_mask,fmask)

                    h, w = self.restored.shape[:2]
                    alpha = cv2.warpPerspective(maskm.copy().astype(np.uint8), H, (w, h))
                except Exception as e:
                    warnings.warn("Error using expert {} to create alpha mask:"
                                  " {}{}".format(fname,type(e),e.args))

        if alpha is None:
            # pre process alpha mask
            alpha = self.pre_process_fore_Mask(self.restored,fore,H)

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

        if self.verbosity > 3: # show merging result
            fastplt(alpha, title="alpha mask from {}".format(path),block=True)

        self.restored = overlay(self.restored, fore, alpha) # overlay fore on top of back

        if H_fore is None: # H is not modified use itself
            H_fore = H

        if self.verbosity > 4: # show merging result
            Plotim("Last added from {}".format(path), self.restored).show()

        ####################### update base features #######################
        # make projection to test key-points inside it
        if self.verbosity > 1: print "Updating key-points..."
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
        # update self.kps_base and self.desc_base
        self.kps_base = newkps
        self.desc_base = np.array(newdesc)

        if self.verbosity > 4: # show keypints in merging
            Plotim("merged Key-points",  # draw key-points in image
                   cv2.drawKeypoints(
                       im2shapeFormat(self.restored,self.restored.shape[:2]+(3,)),
                              [dict2keyPoint(index) for index in self.kps_base],
                              flags=4, color=(0,0,255))).show()
        if self.verbosity: print "This image has been merged: {}...".format(path)
        self.used.append(path) # update used

        return self.restored

    def post_process_restoration(self, image):
        """
        Post-process a merged retinal image.

        :param image: retinal image
        :return: filtered and with simulated lens
        """
        if callable(self.denoise):
            return self.denoise(image)
        # detect how much noise to process and convert it to beta parameters
        if self.denoise is not None:
            # slower but interactive for heavy noise
            # filter using parameters and bilateral filter
            params = getBilateralParameters(image.shape, self.denoise)
            return cv2.bilateralFilter(image, *params)

    def post_process_fore_Mask(self, back, fore):
        """
        Method to post-process fore mask used after fore and back are transformed
        to a new domain for merging. This method is called by the merge method in the
        event that an alpha mask has not been created by self.pre_process_fore_Mask
        method or obtained with the self.expert variables.

        :param back: background image. This is called by method self.merge
                with self.restored
        :param fore: fore ground image.
        :return: alpha mask with shape (None,None)
        """
        pass

    def pre_process_fore_Mask(self, back, fore, H):
        """
        Method to pre-process fore mask used before fore and back are transformed
        to a new domain for merging.

        :param back:
        :param fore:
        :param H:
        :return: alpha mask with shape (None,None)
        """
        pass

class RetinalRestore(ImRestore):
    """
    Restore retinal images by merging and stitching techniques. These parameters are
    added to :class:`ImRestore`:

    :param lens: flag to determine if lens are applied. True
        to simulate lens, False to not apply lens.
    :param enclose: flag to enclose and return only retinal area.
        True to return ROI, false to leave image "as is".
    """
    def __init__(self, filenames, **opts):
        # overwrite variables
        opts["denoise"]=opts.pop("denoise", True)
        opts["maskforeground"] = opts.pop("maskforeground",lambda img: retinal_mask(img,True))
        # create new variables
        self.lens = opts.pop("lens",False)
        self.enclose = opts.pop("enclose",False)
        # call super class
        super(RetinalRestore,self).__init__(filenames, **opts)

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
            # TODO, not working for every retinal scenario
            foregray = brightness(fore)
            # get window with Otsu to prevent expansion
            thresh,w = cv2.threshold(foregray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return get_bright_alpha(brightness(back).astype(float),
                                    foregray.astype(float), window=w)


        pshape = (400,400) # process shape
        # rescaling of the image to process mask
        if pshape is not None:
            oshape = back.shape[:2]
            back = cv2.resize(back,pshape)
            fore = cv2.resize(fore,pshape)

        # get alpha mask
        alphamask = get_layered_alpha(back,fore)

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
        image_ = super(RetinalRestore,self).post_process_restoration(image)
        if image_ is not None:
            image = image_

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

def feature_creator(string):
    """
    Converts a string to a feature object.

    :param string: any supported feature detector in openCV. the format is
            "[a-]<sift|surf|orb>[-flann]" (str) Ex: "a-sift-flann" where
            "a-" or "-flann" are optional.
    :return: feature object
    """
    return Feature().config(string)

def tuple_creator(string):
    """
    Process string to get tuple.

    :param string: string parameters with "," (colon) as separator
            Ex: param1,param2,...,paramN
    :return: tuple
    """
    tp = []
    func = string_interpreter()
    for i in string.split(","):
        try:
            tp.append(func(i))
        except:
            tp.append(i)
    return tuple(tp)

def loader_creator(string):
    """
    creates an image loader.

    :param string: flag, x size, y size. Ex 1: "0,100,100" loads gray images of shape
            (100,100) in gray scale. Ex 2: "1" loads images in BGR color and with
            original shapes. Ex 3: "0,200,None" loads gray images of shape (200,None)
            where None is calculated to keep image ratio.
    :return: loader
    """
    params = tuple_creator(string)
    try:
        flag = params[0]
    except:
        flag=1
    try:
        x = params[1]
    except:
        x=None
    try:
        y = params[2]
    except:
        y=None
    return loadFunc(flag,(x,y))

def denoise_creator(string):
    """
    creates an function to de-noise images using bilateral filter.

    :param string: d, sigmaColor, sigmaSpace. Ex: 27,75,75 creates the
            filter to de-noise images.
    :return: denoiser
    """
    d, sigmaColor, sigmaSpace = tuple_creator(string)
    def denoiser(image):
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    return denoiser

def string_interpreter(empty=None, commahandler=None, handle=None):
    """
    create a string interpreter
    :param empty: (None) variable to handle empty strings
    :param commahandler: (tuple_creator) function to handle comma separated strings
    :return: interpreter function
    """
    def interprete_string(string):
        if string == "":
            return empty
        if "," in string:
            if commahandler is None:
                return tuple_creator(string)
            else:
                return commahandler(string)
        if string.lower() == "none":
            return None
        if string.lower() == "true":
            return True
        if string.lower() == "false":
            return False
        if handle is None:
            try:
                return int(string)
            except:
                return string
        else:
            return handle(string)
    interprete_string.__doc__="""
        Interpret strings.

        :param string: string to interpret.
        :return: interpreted string. If empty string (i.e. '') it returns {}.
                If 'None' returns None. If 'True' returns True. If 'False' returns False.
                If comma separated it applies {} else applies {}.
        """.format(empty, commahandler, handle)
    return interprete_string

class NameSpace(object):
    """
    used to store variables
    """

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
                                     epilog=__doc__ +
                                            "\nContributions and bug reports are appreciated."
                                            "\nauthor: David Toro"
                                            "\ne-mail: davsamirtor@gmail.com"
                                            "\nproject: https://github.com/davtoh/RRtools")
    parser.add_argument('filenames', nargs='*',
                        help='List of images or path to images. Glob operations can be '
                             'achieved using the wildcard sign "*". '
                             'It can load image from files, urls, servers, strings'
                             'or directly from numpy arrays (supports databases)'
                             'Because the shell process wildcards before it gets '
                             'to the parser it creates a list of filtered files in '
                             'the path. Use quotes in shell to prevent this behaviour '
                             'an let the restorer do it instead e.g. "/path/to/images/*.jpg". '
                             'if "*" is used then folders and filenames that start with an '
                             'underscore "_" are ignored by the restorer')
    parser.add_argument('-v','--verbosity',type=int,default=1,
                        help="""(0) flag to print messages and debug data.
                                0 -> do not print messages.
                                1 -> print normal messages.
                                2 -> print normal and debug messages.
                                3 -> print all messages and show main results.
                                    (consumes significantly more memory).
                                4 -> print all messages and show all results.
                                    (consumes significantly more memory).
                                5 -> print all messages, show all results and additional data.
                                    (consumes significantly more memory).""")
    parser.add_argument('-f','--feature', type=string_interpreter(commahandler=feature_creator),
                        help='Configure detector and matcher')
    parser.add_argument('-u','--pool', action='store', type=int,
                        help='Use pool Ex: 4 to use 4 CPUs')
    parser.add_argument('-c','--cachePath',default=None,
                        help="""
                           saves memoization to specified path. This is useful to save
                           some computations and use them in next executions.
                           Cached data is not guaranteed to work between different
                           configurations and this can lead to unexpected program
                           behaviour. If a different configuration will be used it
                           is recommended to clear the cache to recompute values.
                           If True it creates the cache in current path.
                           """)
    parser.add_argument('-e','--clearCache', type=int, default=0,
                        help='clear cache flag.'
                            '* 0 do not clear.'
                            '* 1 re-compute data but other cache data is left intact.'
                            '* 2 All CachePath is cleared before use.'
                            'Notes: using cache can result in unexpected behaviour '
                            'if some configurations does not match to the cached data.')
    parser.add_argument('--loader', type=string_interpreter(commahandler=loader_creator),
                        nargs='?', help='Custom loader function used to load images. '
                            'By default or if --loader flag is empty it loads the '
                            'original images in color. The format is "--loader colorflag, '
                            'x, y" where colorflag is -1,0,1 for BGRA, gray and BGR images '
                            'and the load shape are represented by x and y. '
                            'Ex 1: "0,100,100" loads gray images of shape (100,100) in '
                            'gray scale. Ex 2: "1" loads images in BGR color and with '
                            'original shapes. Ex 3: "0,200,None" loads gray images of shape '
                            '(200,None) where None is calculated to keep image ratio.')
    parser.add_argument('-p','--process_shape', default=(400,400), type=string_interpreter(),
                        nargs='?', help='Process shape used to convert to pseudo images '
                            'to process features and then convert to the '
                            'original images. The smaller the image more memory and speed '
                            'gain. By default process_shape is 400,400'
                            'If the -p flag is empty it loads the original '
                            'images to process the features but it can incur to performance'
                            ' penalties if images are too big and RAM memory is scarce')
    parser.add_argument('-l','--load_shape', default=None, type=string_interpreter(),
                        nargs='?', help='shape used to load images which are beeing merged.')
    parser.add_argument('-b','--baseImage', default=True, type=string_interpreter(), nargs='?',
                        help='Specify image''s name to use from path as first image to merge '
                            'in the empty restored image. By default it selects the image '
                            'with most features. If the -b flag is empty it selects the '
                            'first image in filenames as base image')
    parser.add_argument('-m','--selectMethod',
                        help='Method to sort images when matching. This '
                            'way the merging order can be controlled.'
                            '* (None) Best matches'
                            '* Histogram Comparison: Correlation, Chi-squared,'
                            'Intersection, Hellinger or any method found in hist_map'
                            '* Entropy'
                            '* custom function of the form: rating,fn <-- selectMethod(fns)')
    parser.add_argument('-d','--distanceThresh', type = float, default=0.75,
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
    parser.add_argument('-s','--save', default=True, type = string_interpreter(False), nargs='?',
                        help='Customize image name used to save the restored image.'
                            'By default it saves in path with name "_restored_{base_image}".'
                            'if the -s flag is specified empty it does not save. Formatting '
                            'is supported so for example the default name can be achived as '
                            '"-s {path}_restored_{name}{ext}"')
    parser.add_argument('-o','--overwrite', action='store_true',
                        help = 'If True and the destine filename for saving already '
                            'exists then it is replaced, else a new filename is generated '
                            'with an index "{filename}_{index}.{extension}"')
    parser.add_argument('-g','--grow_scene', action='store_true',
                        help='Flag to allow image to grow the scene so that that the final '
                            'image can be larger than the base image')
    parser.add_argument('-y','--denoise', default=None,
                        type=string_interpreter(False,commahandler=denoise_creator),
                        help="Flag to process noisy images. Use mild, normal, heavy or "
                            "provide parameters for a bilateral filter as "
                            "'--denoise d,sigmaColor,sigmaSpace' as for example "
                            "'--denoise 27,75,75'. By default it is None which can be "
                            "activated according to the restorer, if an empty flag is "
                            "provided as '--denoise' it deactivates de-noising images.")
    parser.add_argument('-a','--lens', action='store_true',
                        help='Flag to apply lens to retinal area. Else do not apply lens')
    parser.add_argument('-k','--enclose', action='store_true',
                        help='Flag to enclose and return only retinal area. '
                            'Else leaves image "as is"')
    parser.add_argument('-z','--restorer',choices = ['RetinalRestore','ImRestore'],
                        default='RetinalRestore',
                        help='imrestore is for images in general but it can be parametrized. '
                            'By default it has the profile "retinalRestore" for retinal '
                            'images but its general behaviour can be restorerd by '
                            'changing it to "imrestore"')
    parser.add_argument('-x','--expert', default=None,help='path to the expert variables')
    parser.add_argument('-q','--console', action='store_true',
                        help='Enter interactive mode to let user execute commands in console')
    parser.add_argument('-w','--debug', action='store_true', # https://pymotw.com/2/pdb/
                        help='Enter debug mode to let programmers find bugs. In the debugger '
                             'type "h" for help and know the supported commands.')
    parser.add_argument('--onlykeys', action='store_true',
                        help='Only compute keypoints. This is useful when --cachePath is '
                            'used and the user wants to have the keypoints cached beforehand')

    # parse sys and get argument variables
    args = vars(parser.parse_args(args=args, namespace=namespace))

    # shell variables
    debug = args.pop('debug')
    console = args.pop('console')
    onlykeys =  args.pop('onlykeys')

    # debugger
    if debug:
        print "debug ON."
        import pdb; pdb.set_trace()

    # this is needed because the shell process wildcards before it gets to argparse
    # creating a list in the path thus it must be filtered. Use quotes in shell
    # to prevent this behaviour
    if len(args['filenames'])>1:
        args['filenames'] = [p for p in args['filenames'] if check_valid(p) or "*" in p]
    else:
        args['filenames'] = args['filenames'][0]

    # print parsed arguments
    if args['verbosity']>1:
        print "Parsed Arguments\n",args

    # use configuration
    use_restorer = args.pop("restorer")
    if use_restorer == 'RetinalRestore':
        if args['verbosity']: print "Configured for retinal restoration..."
        self = RetinalRestore(**args)
    elif use_restorer == 'ImRestore':
        if args['verbosity']: print "Configured for general restoration..."
        for key in ['enclose', 'lens']:
            args.pop(key) # clean up unused key
        self = ImRestore(**args)
    else:
        raise Exception("no restoration class called {}".format(use_restorer))

    if namespace is not None:
        # update namespace from early stages so it can have access to the restorer
        namespace.restorer = self

    if console:
        print "interactive ON."
        print "restoring instance is 'namespace.restorer' or 'self'"
        print "Ex: type 'self.restore()' to proceed with restoration."
        import code; code.interact(local=locals())
    elif onlykeys:
        self.compute_keypoints()
    else:
        # start restoration
        self.restore()

    return namespace # return given namespace

if __name__ == "__main__":
    shell() # run the shell
    # TODO visualizator for alpha mask
    # TODO implement standard deviation in bright areas to detect optic disk