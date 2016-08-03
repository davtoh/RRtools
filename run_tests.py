from imrestore import shell, os
from RRtoolbox.lib.root import StdoutLOG
from glob import glob
from RRtoolbox.lib.config import MANAGER

win = False

if win:
    base = "E:"
else: # linux
    base = "/mnt/4E443F99443F82AF"

#MANAGER["TEMPPATH"] = "{0}/temp".format(base)
#MANAGER.save()

root = "{0}/MEGAsync/TESIS/DATA_RAW/classified/".format(base)
StdoutLOG(root + "_log")

fns = [i for i in glob(root + "*") if os.path.isdir(i)] # only folders
fns = ["{0}/MEGAsync/TESIS/DATA_RAW/classified/set17".format(base)]
#fns = ["{0}/MEGAsync/TESIS/DATA_RAW/classified/set26".format(base)]
#fns = ["{0}/MEGAsync/TESIS/DATA_RAW/classified/set_room".format(base)]
for fn in fns:
    log = StdoutLOG(fn + "/_log")
    txt = "processing for set {}".format(fn)
    print "*"*len(txt)
    print txt
    print "*"*len(txt)
    #shell("{0}/*.* -s {0}/results/restored.png -v 3".format(fn).split())
    # with cache
    #shell("{0}/*.* -s {0}/results/restored.png -c {1}/restoration_data/".format(fn,base).split())
    shell("{0}/*.* -oa -c {1}/restoration_data/".format(fn,base).split())
    log.close()
