__author__ = 'Davtoh'

from RRtoolbox.tools.selectors import EntropyPlot
from RRtoolbox.lib import directory as dr
from tesisfunctions import IMAGEPATH
from RRtoolbox.lib.root import glob

#fns= glob(IMAGEPATH + "IMG*.jpg")[0:3]
root = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/" # lighting/
fns = glob(root+"*")[:3]
#mainpath = "C:\Users\Davtoh\Dropbox\Proyecto Reconstruccion de imagenes\RESULTADOS\\"
#fns= glob.glob(mainpath+"im1*.*")
print "Selecting: ",fns

obj = EntropyPlot(fns)
obj.show()