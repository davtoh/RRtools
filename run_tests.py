"""
script to run imrestore shell and leave log in the restoration place
"""

from imrestore import shell, os, glob
from RRtoolbox.lib.root import StdoutLOG
from RRtoolbox.lib.config import MANAGER
from time import strftime

use_cache = True # cache keypoints
use_subfolder = False # save restored images in subfolder
use_lens = True # use lens simualtion
use_RetinalRestore = True
overwrite = True # overwrite restored image
set_name = "set[1-2]"  # "*"
win = os.name == "nt"
home = os.path.expanduser("~")

if win: # windows
    base = "E:"
else: # linux or others
    base = "/mnt/4E443F99443F82AF"

#MANAGER["TEMPPATH"] = "{0}/temp".format(base)
#MANAGER.save()

root = os.path.join(base,"MEGAsync/TESIS/DATA_PROCESSED/classified")

with StdoutLOG(os.path.join(root,"_log")) as log_master: # Mater log

    # glob using special function from RRtoolbox
    fns = [i for i in glob(os.path.join(root,set_name),check=os.path.isdir)] # only folders

    print("These sets will be processed {}".format(fns))

    # recursive running over master folder
    for fn in fns:

        with StdoutLOG(os.path.join(fn,"_log"), add_stdout=True) as log_child: # child log

            # options
            short_opts = []
            long_opts = []
            if overwrite:
                short_opts.append("o")
            if use_lens:
                short_opts.append("a")
            if use_subfolder:
                long_opts.append("-s {}/results/restored.png".format(fn))
            if use_cache:
                long_opts.append("-c {}/_restoration_data/".format(root))
            if use_RetinalRestore:
                long_opts.append("-z RetinalRestore")
            else:
                long_opts.append("-z ImRestore")

            command = "{0}/*.*".format(fn)
            if short_opts:
                command += " -{}".format("".join(short_opts))
            if long_opts:
                command += " " + " ".join(long_opts)
            # header
            txt1 = "{}".format(strftime("%Y/%m/%d %I:%M:%S"))
            txt2 = "Running command '{}'".format(command)
            l = len(max(txt1,txt2))
            print('{{:*^{}}}'.format(l).format(txt1))
            print(txt2)
            print("*"*l)

            shell(command.split())
            #log_child.close()

    #log_master.close()