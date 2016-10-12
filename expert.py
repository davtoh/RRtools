__author__ = 'Davtoh'

import os
import cv2
from RRtoolbox.lib.root import glob, NameSpace
from RRtoolbox.lib.cache import MemoizedDict
from RRtoolbox.lib.image import getcoors, loadFunc, drawcoorarea, try_loads
from RRtoolbox.lib.directory import getData, getPath, mkPath, increment_if_exits
from imrestore import check_valid
from RRtoolbox.shell import tuple_creator, string_interpreter

class Image(object):
    """
    Structure to load and save images
    """
    def __init__(self, name=None, ext=None, path=None, shape=None, verbosity=False):
        self._loader = loadFunc(-1,dsize=None,throw=False)
        self._shape = None
        self.shape=shape # it is the inverted of dsize
        self.ext=ext
        self.name=name
        self.path=path
        self._RGB=None
        self._RGBA = None
        self._gray=None
        self._BGRA=None
        self._BGR=None
        self.overwrite = False
        self.verbosity = verbosity
        self.log_saved = None
        self.log_loaded = None
        self.last_loaded = None

    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self,value):
        if value != self._shape:
            if value is not None:
                value = value[1],value[0] # invert, for value is shape and we need dsize
            self._loader = loadFunc(-1,dsize=value,throw=False)
        self._shape = value
    @shape.deleter
    def shape(self):
        del self._shape

    @property
    def ext(self):
        if self._ext is None:
            return ""
        return self._ext
    @ext.setter
    def ext(self,value):
        try:
            if not value.startswith("."): # ensures path
                value = "."+value
        except:
            pass
        self._ext = value
    @ext.deleter
    def ext(self):
        del self._ext

    @property
    def path(self):
        if self._path is None:
            return ""
        return self._path
    @path.setter
    def path(self,value):
        try:
            if value[-1] not in ("/","\\"): # ensures path
                value += "/"
        except:
            pass
        self._path = value
    @path.deleter
    def path(self):
        del self._path

    @property
    def BGRA(self):
        if self._BGRA is None:
            self.load()
        return self._BGRA
    @BGRA.setter
    def BGRA(self,value):
        self._BGRA = value
    @BGRA.deleter
    def BGRA(self):
        self._BGRA = None

    @property
    def BGR(self):
        if self._BGR is None:
            self.load()
        return self._BGR
    @BGR.setter
    def BGR(self,value):
        self._BGR = value
    @BGR.deleter
    def BGR(self):
        self._BGR = None

    @property
    def RGB(self):
        if self._RGB is None:
            self._RGB = cv2.cvtColor(self.BGR, cv2.COLOR_BGR2RGB)
        return self._RGB
    @RGB.setter
    def RGB(self,value):
        self._RGB = value
    @RGB.deleter
    def RGB(self):
        self._RGB = None

    @property
    def RGBA(self):
        if self._RGBA is None:
            self._RGBA = cv2.cvtColor(self.BGRA, cv2.COLOR_BGRA2RGBA)
        return self._RGBA
    @RGBA.setter
    def RGBA(self,value):
        self._RGBA = value
    @RGBA.deleter
    def RGBA(self):
        self._RGBA = None

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.BGR, cv2.COLOR_BGR2GRAY)
        return self._gray
    @gray.setter
    def gray(self,value):
        self._gray = value
    @gray.deleter
    def gray(self):
        self._gray = None

    def save(self, name=None, image=None, overwrite = None):
        """
        save restored image in path.

        :param name: filename, string to format or path to save image.
                if path is not a string it would be replaced with the string
                "{path}restored_{name}{ext}" to format with the formatting
                "{path}", "{name}" and "{ext}" from the baseImage variable.
        :param image: (self.BGRA)
        :param overwrite: If True and the destine filename for saving already
            exists then it is replaced, else a new filename is generated
            with an index "{filename}_{index}.{extension}"
        :return: saved path, status (True for success and False for fail)
        """
        if name is None:
            name = self.name
        if name is None:
            raise Exception("name parameter needed")

        if image is None:
            image = self.BGRA

        if overwrite is None:
            overwrite = self.overwrite

        bbase, bpath, bname = getData(self.path)
        bext = self.ext
        # format path if user has specified so
        data = getData(name.format(path="".join((bbase, bpath)),
                                   name=bname, ext=bext))
        # complete any data lacking in path
        for i,(n,b) in enumerate(zip(data,(bbase, bpath, bname, bext))):
            if not n: data[i] = b
        # joint parts to get string
        fn = "".join(data)
        mkPath(getPath(fn))

        if not overwrite:
            fn = increment_if_exits(fn)

        if cv2.imwrite(fn,image):
            if self.verbosity: print "Saved: {}".format(fn)
            if self.log_saved is not None: self.log_saved.append(fn)
            return fn, True
        else:
            if self.verbosity: print "{} could not be saved".format(fn)
            return fn, False

    def load(self, name = None, path = None, shape = None):
        if name is None:
            name = self.name
        if path is None: path = self.path
        if path is None: path = ""
        if shape is not None:
            self.shape = shape

        data = try_loads([name,name+self.ext], paths=path, func= self._loader, addpath=True)
        if data is None:
            raise Exception("Image not Loaded")

        img, last_loaded = data
        if self.log_loaded is not None: self.log_loaded.append(last_loaded)
        if self.verbosity:
            print "loaded: {}".format(last_loaded)
        self.last_loaded = last_loaded

        self._RGB=None
        self._RGBA = None
        self._gray=None

        if img.shape[2] == 3:
            self.BGR = img
            self.BGRA = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        else:
            self.BGRA = img
            self.BGR = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        return self


class ImageExpert(Image):
    def __init__(self,data):
        if isinstance(data,basestring):
            fn = data
            shape = None
            data = {"fn":fn,"shape":shape}
        else:
            fn = data["fn"]
            shape = data["shape"]

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
    def __init__(self, setpath, data = None, modify=False, reset=False,
                 review = False, inpath=None, ask=False, contents="*.*",
                 filter=check_valid):
        if isinstance(setpath,basestring):
            if os.path.isdir(setpath):
                setpath = os.path.abspath(setpath) # ensures absolute path
                self.fns = fns = glob(setpath,contents=contents,check=filter)
            else:
                raise Exception("setpath must be a directory and got {}".format(setpath))
        else:
            fns = [os.path.abspath(set) for set in setpath]
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
                    exp = ImageExpert(memo_inpath[key])
            if exp is None and key in data and not self.reset:
                exp = ImageExpert(data[key])
            elif exp is None:
                exp = ImageExpert(fn)

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

if __name__ == "__main__":
    # shell("./results/ --subfolders".split()) # call expert shell
    shell() # call expert shell

if False and __name__ == "__main__": # for a folder with many sets

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


