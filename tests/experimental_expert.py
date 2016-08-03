__author__ = 'Davtoh'

import os
import csv
from RRtoolbox.lib.root import glob
from RRtoolbox.lib.cache import MemoizedDict
from RRtoolbox.lib.image import getcoors, loadFunc, drawcoorarea
from RRtoolbox.lib.directory import getData

root = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/"
root = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/classified/set15/"
fns = glob(root+"*.*")
mode = 1
# mode 0 lest the user run all un-cached tests,
# mode 1 lets the user run all tests and correct cached tests.
# mode 2 lets is used to change the fields in the cached data

data = MemoizedDict(os.path.abspath(__file__).split(".")[0] + "_cached")
loader = loadFunc(1)

for fn in fns:
    print "checking",fn
    key = "".join(getData(fn)[-2:])
    if key in data and mode > 0:
        data_fn = data[key]
    else:
        data_fn = {}
    if key not in data or mode == 1: # cache new data or replace previous test
        print "test for",fn
        img = loader(fn)
        coors_retina = [getcoors(img,"select retinal area for {}".format(key),drawcoorarea,coors=data_fn.get("coors_retina"))]
        coors_optic_disc = [getcoors(img,"select optic disc for {}".format(key),drawcoorarea,coors=data_fn.get("coors_optic_disc"))]
        defects_c = data_fn.get("coors_defects")
        defects = []
        if defects_c is not None:
            for i in defects_c:
                if i:
                    coors = getcoors(img,"select defects (inside retina):",drawcoorarea,coors=i)
                    defects.append(coors)
        while True:
            coors = getcoors(img,"select defects (inside retina):",drawcoorarea)
            defects.append(coors)
            if not coors:
                break

        data[key] = {"fn":fn,
                    "coors_retina":coors_retina,
                    "coors_optic_disc":coors_optic_disc,
                    "coors_defects":defects,
                    "shape":img.shape}

    if mode == 2 and data_fn:
        data_fn["shape"] = loader(fn).shape
        data[key] = data_fn

if False:
    data = zip(*data)
    data.insert(0,headers)
    with open('experimental_data.csv', 'wb') as csvfile:
        wr = csv.writer(csvfile, delimiter=";", dialect='excel')
        wr.writerows(data)

