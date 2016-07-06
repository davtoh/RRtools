__author__ = 'Davtoh'
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
import cv2
from tesisfunctions import IMAGEPATH, bilateralFilter
import glob

rootpath = IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/"
#fns= glob.glob(rootpath+"IMG_20150730_131444.jpg")
imlist= glob.glob(rootpath+"IMG*.jpg")
#rootpath = r"C:\Users\Davtoh\Dropbox\PYTHON\projects\tesis"+"\\"
#fns= glob.glob(rootpath+"im*.jpg")
size = (400,400)
# aperturesize for sobel() must be 1, 3, 5, or 7
th1,th2,apertureSize,L2gradient=3000,6000,7,True
isSIZE = True
isEQUA = False
isCLAHE = False
clipLimit,tileGridSize=2.0,(8,8)
isBFilter = False
d,sigmaColor,sigmaSpace =10,20,20 # best guess: (50,100,10), opencv: (9,75,75), d=-1 is filter distance until sigma
for fn in imlist:
    img = cv2.imread(fn,0)
    basename = fn.split('\\')[-1]
    name = [""]
    if isSIZE:
        img = cv2.resize(img,size)
        name.append("SIZE"+str(size[0])+"_"+str(size[1])+"_")
    if isEQUA:
        img = cv2.equalizeHist(img)
        name.append("EQUA_")
    if isCLAHE:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        img = clahe.apply(img)
        name.append("CLAHE"+str(clipLimit)+"_"+str(tileGridSize[0])+"_"+str(tileGridSize[1])+"_")
    if isBFilter:
        img = bilateralFilter(img,d,sigmaColor,sigmaSpace)
        name.append("BFILTER"+str(d)+"_"+str(sigmaColor)+"_"+str(sigmaSpace)+"_")

    #image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None
    name.append("E"+str(th1)+"_"+str(th2)+"_"+str(apertureSize)+"_"+str(L2gradient)+"_")
    edges = cv2.Canny(img,th1,th2,apertureSize=apertureSize,L2gradient=L2gradient) # ,True,3,True
    initname,header = basename.split(".")
    strname = "edge_"+initname+"_"+"".join(name)+"."+header
    bgr = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    #kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],np.uint8)
    #edges = cv2.dilate(edges,kernel,iterations = 1)
    #edges = cv2.erode(edges,kernel,iterations = 1)
    #kernel = np.ones((10,10),np.uint8)
    #edges= cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    #kernel = np.ones((2,2),np.uint8)
    #edges= cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    #edges = cv2.filter2D(edges,-1,kernel)
    #plotim("img",edges).show()
    #thresh,th = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #plotim("img",th).show()
    #edges[img<thresh] = 0 # make edges in black black color disappear

    bgr[edges != 0] = (0, 255, 0)
    cv2.imwrite(rootpath+strname,bgr)
    print fn, " saved as ",strname
