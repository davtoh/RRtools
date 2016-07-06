__author__ = 'Davtoh'

import os
import csv
from RRtoolbox.lib.root import glob
from RRtoolbox.lib.plotter import edger
from RRtoolbox.lib.cache import memoizedDict

root = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/"
fns = glob(root+"*")
mode = 0

data = memoizedDict(os.path.abspath(__file__).split(".")[0] + "_cached")
for fn in fns:
    if fn not in data:
        obj = edger(fn)
        obj.show(clean=False)
        data[fn] = {"th1":obj.th1, "th2":obj.th2}
        obj.clean()
    elif mode == 1: # replace previous test
        cached = data[fn]
        obj = edger(fn)
        obj.th1 = cached["th1"]
        obj.th2 = cached["th2"]
        obj.show(clean=False)
        data[fn] = {"th1":obj.th1, "th2":obj.th2}
        obj.clean()
    elif mode == 2: # unsaved independent session
        obj = edger(fn)
        obj.th1 = 0
        obj.th2 = 0
        obj.show(clean=False)
        obj.clean()

if False:
    data = zip(*data)
    data.insert(0,headers)
    with open('experimental_data.csv', 'wb') as csvfile:
        wr = csv.writer(csvfile, delimiter=";", dialect='excel')
        wr.writerows(data)

