# -*- coding: utf-8 -*-
# ----------------------------    IMPORTS    ---------------------------- #
# three-party
import cv2
import numpy as np
# custom
from RRtoolbox.lib.config import MANAGER, FLAG_DEBUG
from RRtoolbox.lib.cache import MemoizedDict, memoize
from RRtoolbox.lib.directory import getData
from RRtoolbox.lib.plotter import MatchExplorer,fastplt, plotPointsContour, plt, Plotim
from RRtoolbox.lib.arrayops.convert import invertH,sh2oh,spairs2opairs, translateQuadrants, dict2keyPoint
from RRtoolbox.lib.arrayops.basic import overlay,superpose,vertexesAngles, relativeQuadrants, getTransformedCorners,transformPoint
from RRtoolbox.lib.arrayops.filters import bilateralFilter, normsigmoid
from RRtoolbox.lib.descriptors import ASIFT,MATCH, inlineRatio
from RRtoolbox.lib.image import PathLoader,loadFunc,ImCoors, hist_match, try_loads
from RRtoolbox.tools.segmentation import get_bright_alpha
from RRtoolbox.lib.root import TimeCode

from glob import glob

def watchData(request, loader, data, onlyTest = False):
    """
    Show data of format obtained from testRates

    :param request: (fore_name,back_name)
    :param loader: image loader from path
    :param data: dictionary containing requests (each request is a key).
    :param onlyTest: if True show only test results, else also show stitched images and some other data.
    :return: request data.
    """
    if FLAG_DEBUG: print "request is: ",request
    temp =  data[request]
    scaled_fore, scaled_back = list(PathLoader(request, loader))
    if temp:
        kp_pairs, status, H = temp["kp_pairs"],temp["status"],temp["H"]
        if H is not None:
            #shapes = original_fore.shape,scaled_fore.shape,original_back.shape,scaled_back.shape
            #H2 = sh2oh(H,*shapes) #### sTM to oTM
            #kp_pairs2 = spairs2opairs(kp_pairs,*shapes)
            f = plt.figure()
            f.add_subplot()
            ax = plotPointsContour(temp["projection"]) # FIXME, causes crashes, i left it purportedly here to solve the problem
            plt.hold(True)
            ax = plotPointsContour(temp["rotatedBox"], lcor="r", deg=True)
            asum = np.sum(temp["angles"])
            txt = """rectangularity: {rectangularity}:{rotatedRectangularity}, quadrant sum: {quadrant_sum},
                  ragularity: {regularity}, inline rate: {inlineRatio}""".format(**temp)
            #ax.figure.canvas.set_window_title(txt)
            plt.title(txt)
            f.add_subplot()
            if not onlyTest:
                win,mwin = "stitch","mask for stitch"
                try:
                    mask1 = get_bright_alpha(scaled_back.astype(float), scaled_fore.astype(float))
                    m1 = fastplt(mask1,"gray",mwin)
                except:
                    if FLAG_DEBUG: print mwin, "crashed"

                try:
                    merged1 = superpose(scaled_back, scaled_fore, H, mask1)[0]
                    p1 = fastplt(merged1,"gray",win)
                except:
                    if FLAG_DEBUG: print win, "crashed"

                win,mwin = "inverted stitch", "mask for inverted stitch"
                try:
                    mask2 = get_bright_alpha(scaled_fore.astype(float), scaled_back.astype(float))
                    m2 = fastplt(mask2,"gray",mwin)
                except:
                    if FLAG_DEBUG: print mwin, "crashed"

                try:
                    merged2 = superpose(scaled_fore,scaled_back, invertH(H), mask2)[0]
                    p2 = fastplt(merged2,"gray",win)
                except:
                    if FLAG_DEBUG: print win, "crashed"

            notes = "inlines/lines:{inlines}/{lines},\noverall test: {overall_test}".format(**temp)
            if FLAG_DEBUG:
                print "quadrants:", temp["quadrants_translated"]
                print notes
            win = 'matching result: '+notes
            try:
                vis = MatchExplorer(win, scaled_fore, scaled_back, kp_pairs, status, H, show=False)
                fastplt(vis.img,title=win,block=True)
            except:
                if FLAG_DEBUG: print win, " crashed"
        else:
            if FLAG_DEBUG: print "Homography is None"
    else:
        win = 'matching result: None'
        try:
            vis = MatchExplorer(win, scaled_fore, scaled_back, show=False)
            fastplt(vis.img,title=win,block=True)
        except:
            if FLAG_DEBUG: print win, " crashed"
    return temp

def qualifyData(data, loader= None, saveTo = None, autoqualify = False, showOnlyPassedTest= False, clear = False):
    """
    Qualify data of format obtained from testRates.

    :param data: dictionary containing all data
    :param loader: use loader for images
    :param saveTo: save data to path
    :param autoqualify: qualify all data without asking human intervention and using algorithm
    :param showOnlyPassedTest: if True shows only the goods tests to qualify, else do not show.
    :param clear: re-do all saved tests.
    :return:
    """
    if loader is None:
        rzyf,rzxf = 400,400 # dimensions to scale foregrounds
        loader = loadFunc(0,dsize=(rzxf, rzyf))
    requests = data.keys()
    #for request in requests:
    #    watchData(request,loader,data,True)
    if saveTo: # persists
        qualification = MemoizedDict(saveTo + "qualification")
        algorithmFails = MemoizedDict(saveTo + "algorithmFails")
        if clear:
            qualification.clear()
            algorithmFails.clear()
    else: # just save to normal dictionary
        qualification,algorithmFails = {},{}
    for i,request in enumerate(requests):
        print "test No{} of {}".format(i,len(requests))
        if request not in qualification: # qualify
            overall_test = None
            if autoqualify or showOnlyPassedTest:
                temp =  data[request]
                if temp:
                    overall_test = temp["overall_test"]
                else:
                    overall_test = False

                if autoqualify:
                    print "qualification {} was given to request: {}".format(overall_test,request)

                if not showOnlyPassedTest or showOnlyPassedTest and not overall_test:
                    qualification[request] = (overall_test,"autoqualified")
                    continue # only let pass those that were good to be qualified

            watchData(request,loader,data,True)
            if raw_input("did these match? (y/n):").lower() in ("yes","y","yeah","true"):
                correct = True
            else:
                correct = False
            if overall_test is not None and overall_test != correct:
                print "test seem to have failed with {}/{}(auto/user) qualification in request {}".format(overall_test,correct,request)
                algorithmFails[request] = (overall_test,correct)
            notes = raw_input("notes?: ")
            if overall_test is not None:
                qualification[request] = (correct,"autoqualified {} with notes: ".format(overall_test)+notes)
            else:
                qualification[request] = (correct,notes)
            print request, "was marked as", correct
    return qualification,algorithmFails

def asif_demo(fn_back =None,fn_fore = None, **opts):
    """
    Demo to test ASIFT, Merge, Plotters.

    :param fn_back: file name of background image
    :param fn_fore: file name of foreground image
    :param **opts: demo options
        flag options:
        flag_filter_scaled (False): filter scaled images
        flag_filter_original (False): filter original images
        flag_filter_out (False): filter pos-processed images
        flag_invertH (False): test inversion of Trasformation Matrix obtained from Homography
        flag_show_match (True): show match keypoint of scaled images
        flag_show_result (True): show restored image
        flag_save_perspective (False): save calculated perspective of foreground image
        flag_save_result (False): save restored image

        value obtions:
        fore_scale (400,400): tuple of W,H to convert original foreground to scaled image
        back_scale (400,400): tuple of W,H to convert original background to scaled image
        feature: ('sift-flann') use base descriptor
    :return: dictionary of results
            if there is not match the keys are:
                ['status', 'kp1', 'kp2', 'kp_pairs', 'desc1', 'desc2','H']
            if there is match then additional keys are:
                ['img_perspective', 'img_restored', 'H_original', 'kp_pairs_original']
            if flag_invertH additional kes: ['H_inverted']
    """

    flag_filter_scaled = opts.get("flag_filter_scaled",False)
    flag_filter_original = opts.get("flag_filter_original",False)
    flag_filter_out = opts.get("flag_filter_out",False)

    flag_invertH = opts.get("flag_invertH",False)

    flag_show_match = opts.get("flag_show_match",True)
    flag_show_result = opts.get("flag_show_result",True)

    flag_save_perspective = opts.get("flag_save_perspective",False)
    flag_save_result = opts.get("flag_save_result",False)
    feature_name = opts.get("feature",'sift-flann')

    #### LOADING
    ls = []
    if fn_fore: ls.append(fn_fore)
    ls.append('im1_2.jpg')
    original_fore = try_loads(ls) # foreground

    ls = []
    if fn_back: ls.append(fn_back)
    ls.append('im1_1.jpg')
    original_back = try_loads(ls) # background

    #### SCALING
    rzyf,rzxf = opts.get("fore_scale",(400,400)) # dimensions to scale foreground
    scaled_fore = cv2.resize(cv2.cvtColor(original_fore,cv2.COLOR_RGB2GRAY), (rzxf, rzyf))

    rzyb,rzxb = opts.get("back_scale",(400,400)) # dimensions to scale background
    scaled_back = cv2.resize(cv2.cvtColor(original_back,cv2.COLOR_RGB2GRAY), (rzxb, rzyb))

    #### PRE-PROCESSING
    if flag_filter_scaled:  # persistent by @root.memoize
        d,sigmaColor,sigmaSpace = 50,100,100
        scaled_fore = bilateralFilter(scaled_fore,d,sigmaColor,sigmaSpace)
        scaled_back = bilateralFilter(scaled_back,d,sigmaColor,sigmaSpace)
        print "merged image filtered with bilateral filter d={},sigmaColor={},sigmaSpace={}".format(d,sigmaColor,sigmaSpace)
    if flag_filter_original:  # persistent by @root.memoize
        d,sigmaColor,sigmaSpace = 50,100,100
        original_fore = bilateralFilter(original_fore,d,sigmaColor,sigmaSpace)
        original_back = bilateralFilter(original_back,d,sigmaColor,sigmaSpace)
        print "merged image filtered with bilateral filter d={},sigmaColor={},sigmaSpace={}".format(d,sigmaColor,sigmaSpace)


    results = {} # dictionary to contain results
    #### FEATURE DETECTOR  # persistent by @root.memoize
    print "finding keypoints with its descriptos..."
    #result = ASIFT_multiple([scaled_fore, scaled_back]) # OR use ASIFT for each image
    kp1,desc1 = ASIFT(feature_name, scaled_fore, mask=None)
    results["kp1"],results["desc1"] = kp1,desc1 # collect descriptors foreground
    kp2,desc2 = ASIFT(feature_name, scaled_back, mask=None)
    results["kp2"],results["desc2"] = kp2,desc2 # collect descriptors background

    #### MATCHING  # persistent by @root.memoize
    print "matching..."
    #H, status, kp_pairs = MATCH_multiple(result)[0] # OR use MATCH
    H, status, kp_pairs = MATCH(feature_name,kp1,desc1,kp2,desc2)
    results["H"],results["status"],results["kp_pairs"] = H.copy(), status, kp_pairs # collect match results

    if H is not None:

        original_fore = hist_match(original_fore, original_back)
        if flag_invertH:
            kp_pairs = [(j,i) for i,j in kp_pairs]
            H = invertH(H)
            results["H_inverted"] = H # collect inversion of H
            tmp1,tmp2,tmp3,tmp4 = original_fore,scaled_fore,original_back,scaled_back
            original_fore,scaled_fore,original_back,scaled_back = tmp3,tmp4,tmp1,tmp2

        shapes = original_fore.shape,scaled_fore.shape,original_back.shape,scaled_back.shape
        H2 = sh2oh(H, *shapes) #### sTM to oTM
        results["H_original"] = H2.copy()
        kp_pairs2 = spairs2opairs(kp_pairs,*shapes)
        results["kp_pairs_original"] = kp_pairs2

        if flag_show_match: # show matching
            win = 'matching result'
            print "waiting to close match explorer..."
            vis = MatchExplorer(win, original_fore, original_back, kp_pairs2, status, H2)
            #vis = MatchExplorer(win, scaled_fore, scaled_back, kp_pairs, status, H)

        # get perspective from the scaled to original Transformation matrix
        bgra_fore = cv2.cvtColor(original_fore,cv2.COLOR_BGR2BGRA) # convert BGR to BGRA
        fore_in_back = cv2.warpPerspective(bgra_fore,H2,(original_back.shape[1],original_back.shape[0])) # get perspective
        results["img_perspective"] = fore_in_back.copy() # collect perspective
        foregray = cv2.cvtColor(fore_in_back,cv2.COLOR_BGRA2GRAY).astype(float) # convert formats to float
        fore_in_back = fore_in_back.astype(float) # convert to float to make operations
        saveas = "perspective.png"
        if flag_save_perspective:
            cv2.imwrite(saveas,fore_in_back) # save perspective
            print "perspective saved as: "+saveas
        # find alpha and do overlay
        alpha = fore_in_back[:,:,3].copy()
        for i in xrange(1): # testing damage by iteration
            backgray = cv2.cvtColor(original_back.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(float)
            fore_in_back[:,:,3]= n = get_bright_alpha(backgray, foregray, alpha) #### GET ALFA MASK
            fastplt(n)
            original_back = overlay(original_back, fore_in_back) #### MERGING
        original_back = original_back.astype(np.uint8) # convert back to uint8
        #### POS-PROCESSING
        if flag_filter_out: # filter  # persistent by @root.memoize
            # http://docs.opencv.org/modules/imgproc/doc/filtering.html
            d,sigmaColor,sigmaSpace =50,100,100 # best guess: (50,100,10), opencv: (9,75,75), d=-1 is filter distance until sigma
            original_back = bilateralFilter(original_back,d,sigmaColor,sigmaSpace)
            saveas = "merged_bilateralfilter_d_{}_sigmaColor_{}_sigmaSapace_{}.png".format(d,sigmaColor,sigmaSpace)
            title = "bilateral filtered d={},sigmaColor={},sigmaSpace={}".format(d,sigmaColor,sigmaSpace)
        else:
            saveas = "merged_nofilter.png"
            title = "merged image"
        print "image merged..."
        if flag_show_result: # plot result
            fastplt(cv2.cvtColor(original_back,cv2.COLOR_BGR2RGB), title = title)
        if flag_save_result:
            cv2.imwrite(saveas,original_back) # save result
            print "result saved as: "+saveas
        results["img_restored"] = original_back # collect image result
        print "process finished... "
        #raw_input("")
    return results

def asif_demo2(fn_back =None,fn_fore = None, **opts):

    """
    Demo to test ASIFT, Merge, Plotters,etc while extending images.

    :param fn_back: file name of background image
    :param fn_fore: file name of foreground image
    :param **opts: demo options
        flag options:
        flag_filter_scaled (False): filter scaled images
        flag_filter_original (False): filter original images
        flag_filter_out (False): filter pos-processed images
        flag_invertH (False): test inversion of Trasformation Matrix obtained from Homography
        flag_show_match (True): show match keypoint of scaled images
        flag_show_result (True): show restored image
        flag_save_perspective (False): save calculated perspective of foreground image
        flag_save_result (False): save restored image

        value obtions:
        fore_scale (400,400): tuple of W,H to convert original foreground to scaled image
        back_scale (400,400): tuple of W,H to convert original background to scaled image
        feature: ('sift-flann') use base descriptor
    :return: dictionary of results
            if there is not match the keys are:
                ['status', 'kp1', 'kp2', 'kp_pairs', 'desc1', 'desc2','H']
            if there is match then additional keys are:
                ['img_perspective', 'img_restored', 'H_original', 'kp_pairs_original']
            if flag_invertH additional kes: ['H_inverted']
    """

    feature_name = opts.get("feature",'sift-flann')
    #### LOADING
    fn_fore = fn_fore or MANAGER["TESTPATH"] + 'im1_2.jpg' # foreground is placed to background
    #original_fore = cv2.imread(fn_fore) # foreground
    print fn_fore, " Loaded..."

    fn_back = fn_back or MANAGER["TESTPATH"] + 'im1_1.jpg' # background
    #original_back = cv2.imread(fn_back) # background
    print fn_back, " Loaded..."

    #### SCALING
    rzyf,rzxf = opts.get("fore_scale",(400,400)) # dimensions to scale foreground
    scaled_fore = cv2.resize(cv2.imread(fn_fore, 0), (rzxf, rzyf))

    rzyb,rzxb = opts.get("back_scale",(400,400)) # dimensions to scale background
    scaled_back = cv2.resize(cv2.imread(fn_back, 0), (rzxb, rzyb))

    #### FEATURE DETECTOR  # persistent by @root.memoize
    print "finding keypoints with its descriptos..."
    #result = ASIFT_multiple([scaled_fore, scaled_back]) # OR use ASIFT for each image
    kp1,desc1 = ASIFT(feature_name, scaled_fore, mask=None)
    kp2,desc2 = ASIFT(feature_name, scaled_back, mask=None)
    #### MATCHING  # persistent by @root.memoize
    print "matching..."
    #H, status, kp_pairs = MATCH_multiple(result)[0] # OR use MATCH
    H, status, kp_pairs = MATCH(feature_name,kp1,desc1,kp2,desc2)

    if H is not None:
        #shapes = original_fore.shape,scaled_fore.shape,original_back.shape,scaled_back.shape
        #H2 = sh2oh(H,*shapes) #### sTM to oTM
        #kp_pairs2 = spairs2opairs(kp_pairs,*shapes)
        print "waiting to close match explorer..."
        win,mwin = "stitch","mask for stitch"
        mask1 = get_bright_alpha(scaled_back.astype(float), scaled_fore.astype(float))
        m1 = fastplt(mask1,"gray",mwin)
        merged1 = superpose(scaled_back, scaled_fore, H, mask1)[0]
        p1 = fastplt(merged1,"gray",win)
        win,mwin = "inverted stitch", "mask for inverted stitch"
        mask2 = get_bright_alpha(scaled_fore.astype(float), scaled_back.astype(float))
        m2 = fastplt(mask2,"gray",mwin)
        merged2 = superpose(scaled_fore,scaled_back, invertH(H), mask2)[0]
        p2 = fastplt(merged2,"gray",win)
        win = 'matching result'
        vis = MatchExplorer(win, scaled_fore, scaled_back, kp_pairs, status, H)

def extractCSV(data, saveTo = None):
    import csv
    saveTo = saveTo or testRates.func_name+".csv"
    with open(saveTo,"a+") as csvfile:
        wr = csv.writer(csvfile, delimiter=";", dialect='excel')
        wr.writerows(data) # save columns # FIXME not implemented yet

def getDicDescriptor(path, loader, feature_name, dic = None):
    """

    :param path:
    :param loader:
    :param feature_name:
    :param dic:
    :return:
    """
    if dic is None: dic = {}
    if path in dic:
        return dic[path]
    else:
        if FLAG_DEBUG: print "finding ASIFT of {}".format(path)
        im = loader(path)
        kps,desc = ASIFT(feature_name, im)
        for kp in kps:
            kp["shape"] = im.shape
            kp["path"] = path
        dic[path] = kps,desc
        return kps,desc

def testRates(images = None, **opts):
    """

    :param opts:
    feature = 'sift-flann'
    loader = "400,400"
    saveTo = None,
    autoqualify = False,
    showOnlyPassedTest= False,
    clearAll = False,
    clearData = clearAll,
    clearQualification = clearAll
    :return:
    """

    def checkrelativity(angles):
        if len([i for i in angles if i<=1.571])>0: # at least one is less than 90.01 degress
            return True
        return False

    def checkquadrands(quadrants):
        if np.array(quadrants).dtype.type is not np.string_:
            quadrants = translateQuadrants(quadrants) # translate to string
        unique = np.unique(quadrants)
        if len(unique) == len(quadrants) or len(unique) == len(quadrants)/2:
            return True
        return False

    feature_name = opts.get("feature",'sift-flann')
    saveTo = opts.get("saveTo",None)#"/mnt/4E443F99443F82AF/restoration_data/"
    clearAll = opts.get("clearAll",False)
    if saveTo:
        descriptors = MemoizedDict(saveTo + "descriptors")
        data = MemoizedDict(saveTo + "data")
        if opts.get("clearData",clearAll):
            descriptors.clear()
            data.clear()
    else:
        descriptors,shapes,data = {},{},{}
    #### LOADING
    if images is None:
        print "looking in path {}".format(MANAGER["TESTPATH"])
        fns = glob(MANAGER["TESTPATH"] + "*.jpg")
    elif isinstance(images,basestring):
        print "looking as {}".format(images)
        fns = glob(images)
    else: # iterator containing data
        fns = images
    #fns = fns[:3]
    print "testing {} files...".format(len(fns))
    #### SCALING
    loader = opts.get("loader",None)
    if isinstance(loader, basestring):
        loader = loadFunc(0,dsize=eval(loader))
    if loader is None:
        rzyf,rzxf = 400,400 # dimensions to scale foregrounds
        loader = loadFunc(0,dsize=(rzxf, rzyf))
    #ims = pathLoader(fns,loader) # load just when needed
    addNewData = False # add any posterior custom data that was forgotten in the calculation
    counter,countNone = 0,0
    #### MATCHING
    with TimeCode("matching...",endmsg="Overall time is "):
        for i,fore_name in enumerate(fns):
            for j,back_name in enumerate(fns):
                if j>i: # do not test itself and inverted tests
                    counter +=1
                    if FLAG_DEBUG: print "comparision No.{}".format(counter)
                    if (fore_name,back_name) not in data:
                        with TimeCode("finding keypoints with its descriptors..."):
                            # FIXME inefficient code ... just 44 descriptors generate 946 Homographies
                            (kps1,desc1) = getDicDescriptor(fore_name,loader,feature_name,descriptors)
                            (kps2,desc2) = getDicDescriptor(back_name,loader,feature_name,descriptors)
                            H, status, kp_pairs = MATCH(feature_name,kps1,desc1,kps2,desc2)
                            if H is None:
                                countNone +=1
                                if FLAG_DEBUG: print "comparison {},{} is None".format(fore_name,back_name)
                                data[(fore_name,back_name)] = None
                            else:
                                projection = getTransformedCorners(kps1[0]["shape"][:2],H)
                                angles = vertexesAngles(projection) # angles in radians
                                relativity_test = checkrelativity(angles)
                                quadrants = relativeQuadrants(projection)
                                quadrants_translated = translateQuadrants(quadrants)
                                quadrants_test = checkquadrands(quadrants_translated)
                                inlines=np.sum(status)
                                lines = len(status)
                                c = ImCoors(projection)
                                inlineratio = inlineRatio(inlines,lines)
                                temp = dict(H = H,
                                            status=status,
                                            kp_pairs=kp_pairs,
                                            kps1=kps1,
                                            kps2=kps2,
                                            inlines=inlines,
                                            lines = lines,
                                            projection = projection,
                                            quadrants = quadrants,
                                            angles = angles,
                                            relativity_test = relativity_test,
                                            quadrants_test = quadrants_test,
                                            quadrants_translated=quadrants_translated,
                                            area_rec = c.rectangularArea,
                                            area = c.area,
                                            rotatedBox = c.rotatedBox,
                                            rotatedRectangularity = c.rotatedRectangularity,
                                            regularity = c.regularity,
                                            rectangularity = c.rectangularity,
                                            inlineRatio = inlineratio,
                                            overall_test = inlineratio>0.8 and c.rectangularity>0.5,
                                            quadrant_sum = np.sum(quadrants,0))
                                data[(fore_name,back_name)] = temp # return temporal data to memoized dict
                    elif addNewData: # add data after calculations.
                        temp = data[(fore_name,back_name)] # get from memoized dict temporal data
                        if temp:
                            pass #data[(fore_name,back_name)] = temp[0]
                        else:
                            countNone +=1
                            if FLAG_DEBUG: print "comparison {},{} is None".format(fore_name,back_name)
    print "{} of {} were None while calculating...".format(countNone,counter)
    if opts.get("qualify",False):
        with TimeCode("Qualifying data..."):
            qualification,algorithmFails = qualifyData(data=data,loader=loader, saveTo=saveTo,
                                                       autoqualify=opts.get("autoqualify",False),
                                                       showOnlyPassedTest=opts.get("showOnlyPassedTest",False),
                                                       clear = opts.get("clearQualification",clearAll))
        return data,descriptors,loader,qualification,algorithmFails
    else:
        return data,descriptors,loader

def stitch_multiple(images = None, **opts):
    """

    :param opts:
    feature = 'sift-flann'
    loader = "400,400"
    saveTo = None,
    autoqualify = False,
    showOnlyPassedTest= False,
    clearAll = False,
    clearData = clearAll,
    clearQualification = clearAll
    :return:
    """
    """
    Notes:
    * inlineratio is really useful to determine if a match is adequate for a merging, but
    it is not good to use when stitching more than 2 images because each time the stitching
    grown the ratios decrease. inlineratio
    """
    from RRtoolbox.lib.descriptors import init_feature, ASIFT_iter,filter_matches

    centric = True # tries to attach as many images as possible to each match process
    # centric is quicker since it does not have to process too many match computations

    feature_name = opts.get("feature",'sift-flann')
    saveTo = opts.get("saveTo",None)#"/mnt/4E443F99443F82AF/restoration_data/"
    clearAll = opts.get("clearAll",False)
    if saveTo:
        descriptors_dic = MemoizedDict(saveTo + "descriptors")
        data = MemoizedDict(saveTo + "data") # this is the result of testRates if it exists
        if opts.get("clearData",clearAll):
            descriptors_dic.clear()
    else:
        descriptors_dic,shapes,data = {},{},{}
    #### LOADING
    if images is None:
        print "looking in path {}".format(MANAGER["TESTPATH"])
        fns = glob(MANAGER["TESTPATH"] + "*.jpg")
    elif isinstance(images,basestring):
        print "looking as {}".format(images)
        fns = glob(images)
    else: # iterator containing data
        fns = images
    #fns = fns[:3]
    print "testing {} files...".format(len(fns))
    #### SCALING
    loader = opts.get("loader",None)
    if isinstance(loader, basestring):
        loader = loadFunc(0,dsize=eval(loader))
    if loader is None:
        rzyf,rzxf = 800,800 # dimensions to scale foregrounds
        loader = loadFunc(0,dsize=(rzxf, rzyf))
    #ims = pathLoader(fns,loader) # load just when needed

    from copy import deepcopy
    with TimeCode("finding descriptors...",endmsg="Overall time is "):
        descriptors_list = []
        """ # this is used if no memoizeDict is used
        for i,(kps,desc) in enumerate(ASIFT_iter(ims,feature_name)):
            descriptors_list.append((len(kps),i,kps,desc))
            if FLAG_DEBUG: print "computing descriptor {}/{}...".format(i,len(ims))"""
        for i, path in enumerate(fns):
            kps,desc = getDicDescriptor(path,loader,feature_name,descriptors_dic)
            for kp in kps:
                kp["modified"] = [] # statistical data
                kp["pt_original"] = kp["pt"]
            descriptors_list.append((len(kps),i,path,kps,desc))
            if FLAG_DEBUG: print "descriptor {}/{}...".format(i+1,len(fns))

    #### MATCHING
    matcher = init_feature(feature_name)[1] # it must get matcher object of cv2 here to prevent conflict with memoizers
    # BFMatcher.knnMatch() returns k best matches where k is specified by the user
    descriptors_list.sort(reverse=True) # descendant: from bigger to least
    _,_,path,kps_base,desc_base = descriptors_list[0] # select first with most descriptors
    used = [path] # select first image path
    failed = [] # registry for failed images
    merged = loader(path) # load first image

    with TimeCode("matching...",endmsg="Overall time is "):
        # TODO make a merger to take a first image and then stitch more to it
        while True:
            kps_remain,desc_remain = [],[] # initialize keypoint and descriptor list of candidates
            for _,_,path,kps,desc in descriptors_list:
                if path not in used: # append only those which are not in the base image
                    kps_remain.extend(kps)
                    desc_remain.extend(desc)
                    for kp in kps:
                        assert kp["modified"] == []

            if not kps_remain: # if there is not image remaining to stitch break
                print "all images used"
                break

            desc_remain = np.array(desc_remain) # convert descriptors to array
            # select only those with good hamming distance
            raw_matches = matcher.knnMatch(desc_remain, trainDescriptors = desc_base, k = 2) #2
            # If k=2, it will draw two match-lines for each keypoint.
            # So we have to pass a status if we want to selectively draw it.
            #p1, p2, kp_pairs = filter_matches(kps_remain, kps_base, raw_matches) #ratio test of 0.75
            # descriptors_dic[kp_pairs[0][0]["name"]][0]
            ratio = 0.75 # filter ratio
            classified = {}
            for m in raw_matches:
                if len(m) == 2 and m[0].distance < m[1].distance * ratio: # by Hamming distance
                    m = m[0]
                    kp1 = kps_remain[m.queryIdx]  # keypoint with Index of the descriptor in query descriptors
                    kp2 = kps_base[m.trainIdx]  # keypoint with Index of the descriptor in train descriptors

                    key = kp1["path"]
                    assert key not in used
                    assert kp2["path"] in used

                    if key in classified:
                        classified[key].append((kp1,kp2))
                    else:
                        classified[key] = [(kp1,kp2)]

            ordered = sorted([(len(v),k) for k,v in classified.items()],reverse=True) # order with best matches

            for v,k in ordered:
                mkp1,mkp2 = zip(*classified[k]) # probably good matches
                p1 = np.float32([kp["pt"] for kp in mkp1])
                p2 = np.float32([kp["pt"] for kp in mkp2])
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                if False and H is not None: # test recursivity
                    H, status = cv2.findHomography(np.float32([p for p,s in zip(p1,status) if s]),
                                                   np.float32([p for p,s in zip(p2,status) if s]), cv2.RANSAC, 10.0)
                # FIXME it seems that when the keypoints correspond to a slanted image the homography cannot
                # minimize the error resulting in erratic transformation functions resulting in it been discarded
                # H should work for the merged image, status specifies the inlier and outlier points
                if H is not None: #first test
                    scaled_fore = loader(k) # load fore image
                    h,w = scaled_fore.shape[:2] #mkp1[0]["shape"][:2]
                    projection = getTransformedCorners((h,w),H) # get corners of dore projection over back
                    c = ImCoors(projection) # class to calculate statistical data
                    lines, inlines = len(status), np.sum(status)
                    inlineratio = inlineRatio(inlines,lines) # ratio to determine how good fore is in back
                    text = "inlines/lines: {}/{}={} and rect {}".format(
                        inlines, lines, inlineratio, c.rotatedRectangularity)
                    merged2 = cv2.drawKeypoints(cv2.cvtColor(merged,cv2.COLOR_GRAY2BGR),
                                          [dict2keyPoint(i) for i in kps_base],
                                          flags=4, color=(255,0,0))
                    MatchExplorer("match " + text, scaled_fore, merged2, classified[k], status, H)
                    if inlineratio>0.2 and c.rotatedRectangularity>0.5: # second test
                        while True: # clean fail registry
                            try:
                                failed.remove(k)
                            except:
                                break
                        merged, H_back, H_fore = superpose(merged, scaled_fore, H) # create new base
                        #fastplt(merged,"gray")
                        #Plotim("last added with "+text,merged).show()
                        projection = getTransformedCorners((h,w),H_fore)
                        newkps, newdesc = [], []
                        '''
                        # newkps, newdesc = [], []
                        # THIS WORKS BUT DOES NOT UPDATE OTHER DATA THAT IS USED FOR VISUALIZATION
                        # MAKING THE NOTION THAT THE ALGORITHM IS NOT WORKING
                        for kp,dsc in zip(kps_base,desc_base):
                            pt = tuple(transformPoint(kp["pt"],H_back))
                            kp["pt"] = pt
                            kp["modified"].append(("H_back",H_back))
                            if cv2.pointPolygonTest(projection, pt, False) == -1: #include only those outside fore
                                newkps.append(kp)
                                newdesc.append(dsc)
                        _,_,_,kps,desc = filter(lambda x: x[2] == k, descriptors_list)[0] # get all the keypoints in that photo
                        # not all fore points are in back
                        for kp in kps: # update keypoint positions
                            kp["pt"] = tuple(transformPoint(kp["pt"],H_fore))
                            kp["modified"].append(("H_fore",H_fore))
                        newkps.extend(kps)
                        newdesc.extend(desc)
                        #kps_base = newkps
                        #desc_base = np.array(newdesc)
                        '''
                        for _,_,path,kps,desc in descriptors_list:
                            if path in used: # append only those which are not in the base image
                                for kp,dsc in zip(kps,desc): # kps,desc
                                    pt = tuple(transformPoint(kp["pt"],H_back))
                                    kp["pt"] = pt
                                    kp["modified"].append(("H_back",H_back))
                                    if cv2.pointPolygonTest(projection, pt, False) == -1: #include only those outside fore
                                        newkps.append(kp)
                                        newdesc.append(dsc)
                            elif path == k:
                                for kp,dsc in zip(kps,desc): # kps,desc
                                    kp["pt"] = tuple(transformPoint(kp["pt"],H_fore))
                                    kp["modified"].append(("H_fore",H_fore))
                                    newkps.append(kp)
                                    newdesc.append(dsc)

                        kps_base = newkps
                        desc_base = np.array(newdesc)
                        used.append(k)
                        assert len(kps_base),len(desc_base)
                        #mkp1 = deepcopy(mkp1) # copy keypoints of fore
                        #for kp in mkp1:
                        #    kp["pt"] = kp["pt_original"] # restore its original keypoints for visualization
                        # visualize the match if data is in merged
                        #vis = MatchExplorer("match in merged", scaled_fore, merged, zip(mkp1,mkp2), status, H_fore)
                        """
                        # draw keypoints in image
                        Plotim("keypoints",
                               cv2.drawKeypoints(cv2.cvtColor(merged,cv2.COLOR_GRAY2BGR),
                                          [dict2keyPoint(i) for i in kps_base],
                                          flags=4, color=(0,0,255))).show()"""
                        if not centric:
                            break
                    else:
                        failed.append(k)
                else:
                    failed.append(k)

            if set(classified.keys()) == set(failed):
                print "Ended, these images do not fit: "
                for i in classified.keys():
                    print i
                break

    base,path,name,ext = getData(used[0])
    name = "merged_"+name
    cv2.imwrite("".join((base,"/home/davtoh/Desktop/",name,ext)),merged)

if __name__ == "__main__":
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    memoize.ignoreAll = True
    asif_demo("im1_2.jpg","im1_1.jpg")
    #asif_demo2()
    images="../TESIS/DATA_RAW/IMAGES/RETINA/*.jpg"
    #stitch_multiple(images=images,
    #                saveTo = "/mnt/4E443F99443F82AF/restoration_data/", clearAll=False)
    #testRates(images=images,
                #saveTo = "/mnt/4E443F99443F82AF/restoration_data/",autoqualify=True,qualify=True,showOnlyPassedTest=True)