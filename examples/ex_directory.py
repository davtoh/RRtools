from RRtoolbox.lib.directory import getPath, getData, changedir, strdifference, increment_if_exits

import os
print "working dir: ",os.getcwd()
a = "/basura/testfile2"
a = r'/Users/Davtoh/Dropbox/PYTHON/projects/escriptors/im5_1.jpg'
b = os.path.abspath(a)
c = os.path.dirname(b)
d = os.path.abspath("../relative_path")
print "\n",b,"\n",c,"\n",d,"\n"

def test_getPath(a):
    print a
    print os.path.isfile(a) # "examples/testfile":False
    print os.path.isdir(a) # "examples/testfile": False
    k = getPath(a)
    print k
    print os.path.isfile(k)
    print os.path.isdir(k)

def test_getData(a):
    data = getData(a)
    print data

if __name__ == "__main__":
    #test_getData(a)
    print increment_if_exits("/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/classified/set15/results/restored.png")