__author__ = 'Davtoh'

import os

import cv2
import numpy as np

from RRtoolbox.lib.arrayops import contours2mask
from RRtoolbox.lib.cache import MemoizedDict
from RRtoolbox.lib.directory import getData, getPath, mkPath, increment_if_exits
from RRtoolbox.lib.image import getcoors, loadFunc, drawcoorarea, Image
from RRtoolbox.lib.root import glob, NameSpace
from RRtoolbox.shell import string_interpreter
from imrestore import check_valid


class ImageExpert(Image):
    def __init__(self, data, shape = None):
        if isinstance(data,basestring):
            fn = data
            data = {"fn":fn,"shape":shape}
        else:
            fn = data["fn"]
            shape = data["shape"] # loads image with shape of coordinates

        self.data = data
        base,path,name,ext = getData(fn)
        super(ImageExpert,self).__init__(path=path, ext=ext, name=name, shape=shape)

    def coordinates(self, key, msg=None, times=None, review= False):
        self.data["shape"] = self.BGRA.shape[:2] # update loaded shape

        if msg is None:
            msg = "{i}/{limit} Select region for {key}:"
        if times is None:
            times = 1
        assert isinstance(times,int) or times>0 or times == True

        # init default dictionary to format
        default = dict(key=key,
                       name=self.name,
                       path=self.path,
                       fn=self.data["fn"],
                       shape=self.data["shape"])
        # get old coordinates
        old = self.data.get(key)
        # init new coordinates
        coor_list = []
        if review and old is not None:
            default["limit"] = len(old)
            for coors in old:
                formatted = msg.format(i=len(coor_list)+1,**default)
                coors = getcoors(self.BGRA,formatted,drawcoorarea,coors=coors)
                coor_list.append(coors)
                if not isinstance(times,bool):
                    times -= 1
        elif old is not None:
            coor_list = old

        if review or old is None:
            default["limit"] = times
            while times:
                default["remain"] = times
                formatted = msg.format(i=len(coor_list)+1,**default)
                coors = getcoors(self.BGRA,formatted,drawcoorarea)
                if not coors:
                    break
                coor_list.append(coors)
                if not isinstance(times,bool):
                    times -= 1
        #if not coor_list: # ensures there is coors inisite coor_list
        #    coor_list = [[]]
        self.data[key] = coor_list
        return coor_list

class Expert(object):
    """
    Class to generate images expert data (experimental)
    """
    def __init__(self, path, data = None, shape = None, modify=False, reset=False,
                 review = False, inpath=None, ask=False, contents="*.*",
                 filter=check_valid):
        """

        :param path:
        :param data:
        :param modify:
        :param reset:
        :param review:
        :param inpath:
        :param ask:
        :param contents:
        :param filter: function to filter out files
        """
        if isinstance(path, basestring):
            path = os.path.abspath(path) # ensures absolute path
            self.fns = fns = glob(path, contents=contents, check=filter)
        else:
            fns = [os.path.abspath(set) for set in path]
        if not fns:
            raise Exception("not images to get expert data")

        # mode 0 lest the user run all un-cached tests,
        # mode 1 lets the user run all tests and correct cached tests.
        # mode 2 lets is used to change the fields in the cached data
        if data is None:
            data = {}
        elif isinstance(data,basestring):
            data = MemoizedDict(data)
        else: # it should be dictionary
            pass
        self.data = data
        self.ask = ask
        self.fns = fns
        self.reset = reset
        self.review = review
        self.inpath = inpath
        self.modify = modify
        self.shape = shape

    def start(self):
        data = self.data
        for i,fn in enumerate(self.fns):
            inpath = self.inpath
            if self.ask and not raw_input("check {}?(yes/no)".format(fn)).lower() in ("not","no","n"):
                continue
            print "{}/{} checking {}".format(i+1,len(self.fns),fn)
            key = Expert.get_key(fn)
            exp = None
            memo_inpath = None
            if inpath is not None:
                p = os.path.join(os.path.split(fn)[0],inpath)
                memo_inpath = MemoizedDict(p)
                if key in memo_inpath:
                    exp = ImageExpert(memo_inpath[key],shape=self.shape)
            if exp is None and key in data and not self.reset:
                exp = ImageExpert(data[key],shape=self.shape)
            elif exp is None:
                exp = ImageExpert(fn,shape=self.shape)

            # cache new data or replace previous test
            exp.coordinates(msg="Select retinal area for {}".format(key),
                            key="coors_retina",review=self.review)
            exp.coordinates(msg="Select optic disc for {}".format(key),
                            key="coors_optic_disc",review=self.review)
            exp.coordinates(msg="Select defects (inside retina)-{i}/{limit}:",
                            key="coors_defects",review=self.review,times=True)
            exp.coordinates(msg="Select noisy areas -{i}/{limit}:",
                            key="coors_noisy",review=self.review,times=True)

            if self.modify:
                self.new_modify(exp.data)

            # register expert data
            exp_data = dict(exp.data)
            data[key] = exp_data # ensures that memoizedDicts are converted to dict
            if inpath is not None and memo_inpath is not None:
                memo_inpath[key] = exp_data

    @classmethod
    def get_key(self,fn):
        return "".join(getData(fn)[-2:])

    def new_modify(self, dict):
        pass


def shell(args=None, namespace=None):
    """
    Shell to run in terminal the expert data generator program

    :param args: (None) list of arguments. If None it captures the
            arguments in sys.
    :param namespace: (None) namespace to place variables. If None
            it creates a namespace.
    :return: namespace
    """
    if namespace is None:
        namespace = NameSpace()
    import argparse
    parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                     description = "Create expert data for images",
                                     epilog="\nContributions and bug reports are appreciated."
                                            "\nauthor: David Toro"
                                            "\ne-mail: davsamirtor@gmail.com"
                                            "\nproject: https://github.com/davtoh/RRtools")
    parser.add_argument('path', nargs=1,
                        help='')
    parser.add_argument('-o', '--output', default="{path}/_expert", nargs='?', action="store",
                        const="{path}/_expert", type = string_interpreter(),
                        help='Customize output folder for expert data')
    parser.add_argument('-s', '--subfolders', action='store_true',
                        help='Look for images in sub folders')
    parser.add_argument('-f', '--from', type = str,
                        help='Start from a given pattern')
    parser.add_argument('-i', '--inpath', type = str,
                        help='Save folder of expert data also in the path of the image')
    parser.add_argument('-m', '--modify', action='store_true',
                        help='')
    parser.add_argument('-r', '--reset', action='store_true',
                        help='')
    parser.add_argument('-v', '--review', action='store_true',
                        help='')
    parser.add_argument('-a', '--ask', action='store_true',
                        help='')
    parser.add_argument('-c', '--contents', type = str, default="*.*",
                        help='pattern to look in folder')

    # parse sys and get argument variables
    args = vars(parser.parse_args(args=args, namespace=namespace))
    setspath = args.pop("path")[0]
    bfrom = args.pop("from")
    expertname = args.pop("output").format(path = setspath)
    subfolders = args.pop("subfolders")
    if subfolders:
        def check_dir(path):
            return os.path.isdir(path) and not path.endswith(expertname)
        imsets = glob(os.path.join(setspath,"") + "*", check=check_dir) # only folders
    else:
        imsets = [setspath]

    start = False
    for imset in imsets:
        if bfrom is None or bfrom in imset:
            start = True
        if start:
            exp = Expert(imset,data=expertname,**args)
            exp.start()


def crop_expert(fn, outpath = None, expertpath=None, loader=None, preview=None,
                form = None, startfrom = None, name = None, modify=False, reset=False,
                review = False, ask=False, help = False):
    """
    Crop input image and save ROIs

    :param fn: file name
    :param outpath: (None)
    :param expertpath:
    :param loader: (loadFunc(1))
    :param preview: (rect)
    :param form: crop shape type supported by :func:`getROI`
    :param startfrom: start from an specific pattern in path
    :param name: default name to give to cropped images
            which includes (ImageItem,save_path) items
    :param modify:
    :param reset:
    :param review:
    :param ask:
    :param help: help the user by providinf some coordinates (experimental)
    :return: ROI object, list of transformations
    """
    from RRtoolFC.GUI.forms import getROI
    #from RRtoolbox.lib.arrayops import foreground
    from RRtoolbox.tools.segmentation import retinal_mask

    imsets = glob(fn) # only folders
    if preview is None:
        preview = True
    if form is None:
        form = "rect"
    if loader is None:
        loader = loadFunc(1)

    start = False
    for impath in imsets:
        if startfrom is None or startfrom in impath:
            start = True
        if start:
            image = loader(impath).astype(np.float32)
            print "loaded",impath

            a,b,c,d = getData(impath)
            if name is not None:
                c = name

            # get path to save ROIs
            if outpath is None: # create default path
                outpath2 = a+b+c # add new folder with name of the image
            else: # custom sting path
                outpath2 = outpath

            if expertpath is None: # create default path
                expertpath2 = os.path.join(outpath2,"_expert")
            else: # custom sting path
                expertpath2 = expertpath

            # make path if it does not exists
            mkPath(getPath(outpath2))

            # save ROI
            fn = os.path.join(outpath2,"_original_"+c+d)
            if cv2.imwrite(fn, image):
                print "Original image saved as {}".format(fn)
            else:
                "Original image {} could not be saved".format(fn)
                fn = impath

            # get expert data
            exp = Expert(fn,data=expertpath2, shape=image.shape, modify=modify, reset=reset,
                         review = review, ask=ask, filter=os.path.isfile)
            contours, _ = cv2.findContours(retinal_mask(image.copy()),
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            main_key = getData(fn)[-2]+d
            if help and main_key not in exp.data: # help the user by guessing initial threshold
                exp.data[main_key] = {"fn":fn,"shape":image.shape[:2],"coors_retina":contours}
            exp.start()
            expertfield = exp.data.values()[0]

            # get ROIs
            while not raw_input("get ROI?(y,n)").lower() in ("n","not","no"):
                # get ROI
                roi, crop = getROI(image, preview=preview, form= form, crop=False)
                fn = increment_if_exits(os.path.join(outpath2,"{}{}".format(c,d)),force=True)

                imroi = roi.getArrayRegion(image, crop)
                # save ROI
                if cv2.imwrite(fn, imroi):
                    print "Saved: {}".format(fn)
                else:
                    "{} could not be saved".format(fn)

                info = {}
                # automatically calculate expert data from parent image
                for field,val in expertfield.iteritems():
                    if field.startswith("coors_"):
                        mask = contours2mask(val,shape=image.shape)
                        mask = roi.getArrayRegion(mask, crop)
                        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        info[field] = contours
                    elif "fn" == field:
                        info[field] = fn
                    elif "shape" == field:
                        info[field] = imroi.shape[:2]
                    else:
                        raise Exception("Expert structure not supported")

                exp.data["".join(getData(fn)[-2:])] = info # register expert data of crop


if False and __name__ == "__main__": # for a folder with many sets
    """
    Example using Expert class: generate expert data contained in results/ folder
    """
    def check_dir(path):
        return os.path.isdir(path) and not path.endswith(expertname)

    setspath = "./results/"
    expertname = "_expert"

    imsets = glob(os.path.join(setspath,"") + "*", check=check_dir) # only folders

    bfrom = None
    start = False
    for imset in imsets:
        if bfrom is None or bfrom in imset:
            start = True
        if start:
            exp = Expert(imset,data=os.path.join(setspath,expertname),inpath=expertname,review=False)
            exp.start()



if __name__ == "__main__": # for a folder with many sets
    """
    Example using crop_expert function: generate expert data from X image and save
    perspectives in another path with expert data along.
    """
    setpath = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/DATABASE/Database_DRIVE/test/images/"
    outpath = "./results/"
    base_name = "test{}"

    startfrom = "01_test"
    loader=None
    preview=True
    form = "rect"
    debug = True

    for i,fn in enumerate(glob(setpath)):
        name = base_name.format(i+1)
        base_out = os.path.join(outpath,name)
        crop_expert(fn=fn, outpath=base_out, startfrom=startfrom, help=True,
                    loader=loader, preview=preview, form= form, name=name)

        # debug automatic cropping with expert data
        if debug:
            print "################ DEBUG ################"
            exp = Expert(base_out,data=os.path.join(base_out,"_expert"),review=True)
            exp.start()
            print "############## END DEBUG ##############"


if __name__ == "__main__":
    """
    Call expert program from terminal
    """
    # shell("./results/ --subfolders".split()) # call expert shell
    shell() # call expert shell
