__author__ = 'Davtoh'
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
from RRtoolbox.lib.plotter import Edger

rootpath = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/lighting/"
#basename = "im1_1.jpg"
#basename = "_good.jpg"
basename = "IMG_20150730_125411.jpg"
#img = cv2.imread(rootpath+basename)
obj = Edger(rootpath + basename)
obj.show(clean=False)
edges = obj.edge

