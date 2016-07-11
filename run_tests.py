#import pdb; pdb.set_trace()
#import objgraph
from imrestore import shell, mkPath, getPath, os
from RRtoolbox.lib.root import stdoutLOG
from glob import glob
from RRtoolbox.lib.config import MANAGER

linux = False
win = True

if linux:
    MANAGER["TEMPPATH"] = "/mnt/4E443F99443F82AF/temp"
    MANAGER.save()

    root = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/classified/"
    stdoutLOG(root + "log")
    fns = [i for i in glob(root + "*") if os.path.isdir(i)] # only folders

    #fns = ["/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/classified/set15"]
    fns = ["/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/classified/set26"]
    #fns = ["/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/classified/set_room"]
    for fn in fns:
        log = stdoutLOG(fn + "/log")
        txt = "processing for set {}".format(fn)
        print "*"*len(txt)
        print txt
        print "*"*len(txt)
        #shell("{0}/*.* -s {0}/results/restored.png -v 3".format(fn).split())
        # with cache
        shell("{0}/*.* -s {0}/results/restored.png -c /mnt/4E443F99443F82AF/restoration_data/".format(fn).split())
        log.close()
if win:

    MANAGER["TEMPPATH"] = "E:/temp"
    MANAGER.save()

    root = "E:/MEGAsync/TESIS/DATA_RAW/classified/"
    stdoutLOG(root + "log")
    fns = [i for i in glob(root + "*") if os.path.isdir(i)] # only folders

    fns = ["E:/MEGAsync/TESIS/DATA_RAW/classified/set26"]
    for fn in fns:
        log = stdoutLOG(fn + "/log")
        txt = "processing for set {}".format(fn)
        print "*"*len(txt)
        print txt
        print "*"*len(txt)
        #shell("{0}/*.* -s {0}/results/restored.png -v 3".format(fn).split())
        # with cache
        shell("{0}/*.* -s {0}/results/restored.png -c E:/restoration_data/".format(fn).split())
        log.close()

