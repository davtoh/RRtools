from __future__ import print_function
from tests.experimental_restoration import *

if True:
    back = "/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/tests/im1_1.jpg"
    fore = "/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/tests/im1_2.jpg"
    results = asif_demo(back,fore)
    print(list(results.keys()))

#asif_demo2()
#testRates()
#stich()