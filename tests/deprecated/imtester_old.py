__author__ = 'Davtoh'
import cv2
from tesisfunctions import plotim,overlay,padVH
import numpy as np
from RRtoolbox.lib.image import fig2bgra
from matplotlib import pyplot as plt
from tesisfunctions import sigmoid, histogram

# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
# http://people.csail.mit.edu/sparis/bf_course/

def detectType(self,type,i="",info=""):
    if type==self.binary:
        info += ", type"+i+"= binary"
    elif type==self.binary_inv:
        info += ", type"+i+"= binary inv"
    elif type==self.tozero:
        info += ", type"+i+"= tozero"
    elif type==self.tozero_inv:
        info += ", type"+i+"= tozero inv"
    elif type==self.trunc:
        info += ", type"+i+"= truncate"
    return info

def applythresh(img,type,adaptativetoggle,threshtoggle,th,blocksz,c,i="",ti="",info="",title=""):
    # here any threshold is made to img
    if adaptativetoggle:
        if threshtoggle: # Mean img,maximum,cv2.ADAPTIVE_THRESH_MEAN_C,type,blocksz,C
            thresh = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,type,blocksz,c)
            title += ". Mean"+ti+": "
        else: # Gauss img,maximum,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,type,blocksz,C
            thresh = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,type,blocksz,c)
            title += ". Gauss"+ti+": "
        info += "blocksz"+i+"= "+str(blocksz)+", c"+i+"= "+str(c)
    else:
        if threshtoggle: # normal img,thresh,maximum,type
            thresh = cv2.threshold(img,th,1,type)[1]
            title += ". Normal"+ti+": "
        else: # otsu img,thresh,maximum,type+cv2.THRESH_OTSU
            thresh = cv2.threshold(img,th,1,type+cv2.THRESH_OTSU)[1]
            title += ". Otsu"+ti+": "
        info += "th"+i+"= "+str(th)
    return thresh,info,title

def fomatinfo(info,words=9):
    infolist = info.split()
    newlist = [[""]]
    j=1
    for i in xrange(len(infolist)):
        if i>words*j:
            j+=1
            newlist.append([""])
        newlist[-1][0]+=infolist[i]+" "
    return newlist

def visualize(self):
    # here self.data is passed to the render
    ## Image data ##
    self.img = self.data.copy() # image
    self.sample = self.img
    temp = self.img.shape
    self.maxY = temp[0] # maximum
    self.maxX = temp[1]
    self.minY = 0
    self.minX = 0
    # for rendered visualization
    self.y = 0
    self.x = 0
    self.rx2=self.maxX
    self.ry2=self.maxY
    self.rx1=0
    self.ry1=0
    # for coordinates
    self.mousemoved = False
    self.event = 0
    self.flags = 0
    self.updaterenderer()  # update render

def updatevisualization(self,image,channel,th = None,items=None,thresh1=None,thresh2 = None):
    # here self.data is updated
    if th is not None : th = th*self.maximum
    # UPDATE PLOT #
    if th is not None and self.overlay:
        image = overlay(image.copy(), th, alpha=th)
    sz = image.shape
    # PREPARE ITEMS TO PLOT
    if self.showhist and not self.portablehist:
        data = [fig2bgra(histogram(image,False))]
    else:
        data = []
        if self.showimg or th is None:
            data.append(image)
        if self.showchannel:
            data.append(channel)
        if th is not None: data.append(th)
    if self.showhist and self.portablehist:
        hst = fig2bgra(histogram(image,False))
        sz = hst.shape[1]/2,hst.shape[0]/2#(self.rW/2,self.rH)
        if items is None: items = [[cv2.resize(hst,sz)]]
        else: items.insert(0,[cv2.resize(hst,sz)])
    if items is None:
        items = [["zoom(x{self.rxZoom}({self.rx1}-{self.rx2}), y{self.ryZoom}({self.ry1}-{self.ry2})) "]]
    self.controlText= items
    if thresh1 is not None and thresh2 is not None:
        self.data = padVH([data,[thresh1*self.maximum,thresh2*self.maximum]],100,100)[0] # MAKE VISUALIZATION
    else:
        self.data = padVH([data],100,100)[0] # MAKE VISUALIZATION
    visualize(self) # SHOW VISUALIZATION

def builtcmd(self):
    # here commands for cmd are built
    self.cmdlist = ["limitrender","limitaxes","showcoors","showcontrol",
                    "showpixel","staticcoors","showpixelvalue","pixels",
                    "showimg","showhist","adaptoggle1","adaptoggle1","thtoggle1",
                    "thtoggle2","portablehist","cmdformatter","doubleth",
                    "operation","overlay","sigmoide","showchannel"] # filter commands
    #self.doubleth # "self._timeitems"
    binary=["if self.doubleth: self.type2=self.binary","if not self.doubleth: self.type1=self.binary"]
    binary_inv=["if self.doubleth: self.type2=self.binary_inv","if not self.doubleth: self.type1=self.binary_inv"]
    tozero=["if self.doubleth: self.type2=self.tozero","if not self.doubleth: self.type1=self.tozero"]
    tozero_inv=["if self.doubleth: self.type2=self.tozero_inv","if not self.doubleth: self.type1=self.tozero_inv"]
    truncate=["if self.doubleth: self.type2=self.trunc","if not self.doubleth: self.type1=self.trunc"]
    normal = ["if self.doubleth: self.thtoggle2=1","if self.doubleth: self.adaptoggle2=0",
              "if not self.doubleth: self.thtoggle1=1","if not self.doubleth: self.adaptoggle1=0"]
    otsu = ["if self.doubleth: self.thtoggle2=0","if self.doubleth: self.adaptoggle2=0",
            "if not self.doubleth: self.thtoggle1=0","if not self.doubleth: self.adaptoggle1=0"]
    mean = ["if self.doubleth: self.thtoggle2=1","if self.doubleth: self.adaptoggle2=1",
            "if not self.doubleth: self.thtoggle1=1","if not self.doubleth: self.adaptoggle1=1"]
    gaussian = ["if self.doubleth: self.thtoggle2=0","if self.doubleth: self.adaptoggle2=1",
                "if not self.doubleth: self.thtoggle1=0","if not self.doubleth: self.adaptoggle1=1"]
    self.cmdeval = {"pixels":["showpixel","showpixelvalue"],
        "+1th1":"self.th1=(1+self.th1)%256",
        "1th/2":"self.th1=(self.th1/2)%256",
        "-1th":"self.th1=(self.th1-1)%256",
        "+2th":"self.th2=(1+self.th2)%256",
        "2th/2":"self.th2=(self.th2/2)%256",
        "-2th":"self.th2=(self.th2-1)%256",
        "and":"self.operation=1",
        "or":"self.operation=0",
        # FILTERS
        "convolution":"self.filter=0", "gaussian filter":"self.filter=1",
        "blur":"self.filter=2", "median filter":"self.filter=3",
        "bilateral filter":"self.filter=4", "no filter":"self.filter=5",
        # TYPES
        "binary":binary, "binary inv":binary_inv,
        "tozero":tozero, "tozero inv":tozero_inv,
        "truncate":truncate,
        # TYPES 2
        "normal threshold":normal, "otsu threshold":otsu,
        "adaptive mean threshold":mean, "adaptive gaussian threshold":gaussian,
        "end":["self.computefunc(self)"]}

    sz = self.data0.shape
    if len(sz)==2:
        self.channel = self.data0.copy()
        self.cmdeval[self.channels[0]] = "self.channel=self.data0.copy()"
    elif len(sz)==3:
        self.channel = self.data0[:,:,0].copy()
        for i in range(sz[2]):
            self.cmdeval[self.channels[i]] = "self.channel=self.data0[:,:,"+str(i)+"].copy()"

    self.cmdlist.extend(self.cmdeval.keys())

def compute(self,image=None):
    # this function is designed to work with countless initializations of image after builtcmd(self)
    if image is None:
        image = self.data0

    # DETECT CHANNEL
    sz = image.shape
    info = "unknown channel"
    if len(sz)==2:
        if np.array_equal(self.channel,image):
            info = self.channels[0]
    elif len(sz)==3:
        for i in range(sz[2]):
            if np.array_equal(self.channel,self.data0[:,:,i]):
                info = self.channels[i]
                break

    chimg = self.channel
    if self.sigmoide:
        info += ". Sigmoide: alfa= "+str(self.alfa)+", beta= "+str(self.beta)
        chimg = sigmoid(chimg.astype("float"),self.alfa,self.beta).astype("uint8")

    # FILTERS
    flags = self.filter
    ksize = self.ksize
    if flags==0 or type(ksize) is np.ndarray: # convolution
        if type(ksize) is not np.ndarray: # d is divisor
            ksize = np.ones(ksize,np.float32)/self.d
        chimg = cv2.filter2D(chimg,-1,ksize)
        info += ". Convolution: ksize= "+str(tuple(ksize.shape))
    elif flags==1: # gaussian filter: highly effective in removing Gaussian noise
        chimg = cv2.GaussianBlur(chimg,ksize,0) # src, ksize, sigmaX
        info += ". Gaussian F: ksize= "+str(ksize)
    elif flags==2: # blur: averaging
        chimg = cv2.blur(chimg,ksize)
        info += ". Blur F: ksize= "+str(ksize)
    elif flags==3: # median filter: does not change colors: highly effective in removing salt-and-pepper noise
        chimg = cv2.medianBlur(chimg,self.d) # d is ksize
        info += ". Median F: d= "+str(self.d)
    elif flags==4: # bilateral filtering: highly effective at noise removal while preserving edges
        chimg = cv2.bilateralFilter(chimg,self.d,self.sigmaColor,self.sigmaSpace) # d, sigmaColor, sigmaSpace
        info += ". Bilateral F: d= "+str(self.d)+", sigmaColor= "+str(self.sigmaColor)+", sigmaSpace= "+str(self.sigmaSpace)
    else:
        info += ". No Filter"

    # APPLY THRESHOLD
    info += ". Maximum: "+str(self.maximum)
    if self.doubleth:
        thresh1,info1,title1 = applythresh(chimg,self.type1,self.adaptoggle1,self.thtoggle1,self.th1,self.blocksz1,self.c1,i="1",ti="1")
        thresh2,info2,title2 = applythresh(chimg,self.type2,self.adaptoggle2,self.thtoggle2,self.th2,self.blocksz2,self.c2,i="2",ti="2")
        if self.operation:
            self.th = thresh1 & thresh2
            op = " AND"
        else:
            self.th = thresh1 | thresh2
            op = " OR"
        # DETECT TYPE
        info1 = detectType(self,self.type1,i="1",info=info1)
        info2 = detectType(self,self.type2,i="2",info=info2)
        info += title1+info1+op+ title2[1:]+info2
    else:
        self.th,info1,title1 = applythresh(chimg,self.type1,self.adaptoggle1,self.thtoggle1,self.th1,self.blocksz1,self.c1,i="1",ti="1")
        # DETECT TYPE
        info += title1+detectType(self,self.type1,i="1",info=info1)
        thresh1 = None
        thresh2 = None

    self.info = info
    items = [["zoom(x{self.rxZoom}({self.rx1}-{self.rx2}), y{self.ryZoom}({self.ry1}-{self.ry2})) "]]
    items.extend(fomatinfo(info,words=5))
    # UPDATE PLOT #
    updatevisualization(self,image,chimg,self.th,items,thresh1,thresh2)

def imtester(img,win="imtester plot",plotter=plotim):

    def windowfunc(self):
        cv2.namedWindow(self.win,self.wintype)  # create window
        cv2.resizeWindow(self.win,self.rW,self.rH)
        builtcmd(self) # preparing cmd: use when replacing image
        self.computefunc(self) # computing operations

    if type(img) is plotim:
        self = img
        #if not self.delayplot: self.delayplot=300
    else:
        self = plotter(win,img,(191,191,191))
        self.rW = 1080
        self.rH = 700
        self.showcoors = 0
        #self.delayplot=300

    self.info = ""
    self.textbackground = (191,191,191,200)
    self.errorbackground = (0,0,255,150)
    self.data0 = img.copy()
    self.computefunc = compute
    self.windowfunc = windowfunc
    # for filters
    self.d = 9
    self.sigmaColor = 75
    self.sigmaSpace = 75
    self.ksize = (5,5)
    self.filter=5
    # for thresholds
    self.blocksz1=11
    self.c1=30
    self.blocksz2=11
    self.c2=30
    self.maximum=255
    self.th1= 255/2
    self.th2 = 0
    # types of threshold
    self.type1=cv2.THRESH_BINARY
    self.type2=cv2.THRESH_BINARY
    self.binary_inv = cv2.THRESH_BINARY_INV
    self.binary = cv2.THRESH_BINARY
    self.tozero =  cv2.THRESH_TOZERO
    self.tozero_inv = cv2.THRESH_TOZERO_INV
    self.trunc = cv2.THRESH_TRUNC
    # control
    self.showimg=0
    self.showhist=0
    self.portablehist=0
    self.thtoggle1=1
    self.thtoggle2=1
    self.adaptoggle1=0
    self.adaptoggle2=0
    self.showchannel = 0
    self.alfa = 10
    self.beta = 10
    self.sigmoide = 0
    self.doubleth=0
    self.operation=1
    self.overlay = 1
    sz = self.data0.shape
    if len(sz)==2:
        self.channels = ["channel gray"]
    else:
        self.channels = ["channel b","channel g","channel r","channel c"]
    self.show()
    return self

if __name__=="__main__":
    #img= cv2.resize(cv2.imread(r"asift2fore.png"),(400,400))
    img = cv2.resize(cv2.imread(r'im1_2.jpg'),(400,400))
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    test = imtester(img)
    print test.info