__author__ = 'Davtoh'
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html
from tesisfunctions import Plotim,overlay
import cv2
import numpy as np
import tesisfunctions as tf

fn1 = r'im1_2.jpg'
#fn1 = tf.IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_115534_1.jpg"
name = fn1.split('\\')[-1].split(".")[0]

fore = cv2.imread(fn1)
fore = cv2.resize(fore,(800,800))
padval = 100
percent = 20
if percent>1: percent/=100.0

# draw SEED
P = tf.brightness(fore)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
P2 = clahe.apply(P)
thresh,P2 = cv2.threshold(P2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
P2 = cv2.distanceTransform(P2,cv2.DIST_LABEL_PIXEL,5)
P2 = tf.normalize(P2).astype(np.uint8)*255
P2 = cv2.cvtColor(P2,cv2.COLOR_GRAY2BGR)
fore = tf.pad(fore,(np.min(fore[:,:,0]),np.min(fore[:,:,1]),np.min(fore[:,:,2])),padval,True)
P2 = tf.pad(P2,(np.min(P2[:,:,0]),np.min(P2[:,:,1]),np.min(P2[:,:,2])),padval,True)

frame = fore.copy()

# set up the ROI for tracking
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

P = tf.brightness(fore)
th,mask = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#mask = cv2.inRange(hsv_roi, np.array((0., 0.,176.)), np.array((179.,159.,255.)))
#Plotim("mask", mask).show()
fore = overlay(fore, mask, alpha=mask * 0.2)
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# setup initial location of window
h,w = 10,10
#x,y = ov.convertXY(0,0,frame.shape,(h,w),4) # simply hardcoded the values
#x,y = np.mean([tf.getThreshCenter1(mask),tf.getThreshCenter2(mask),tf.getThreshCenter3(mask),tf.getThreshCenter4(mask)],0).astype(int)
x,y = tf.getThreshCenter2(mask) # x,y
track_window = (x,y,w,h)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT , 10, 10 )
color = (200,0,200)
frame = fore.copy()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
ret, track_window = cv2.CamShift(dst, track_window, term_crit)
ROI = np.zeros(mask.shape,dtype=np.uint8)
cv2.ellipse(ROI,ret, 1, 0, -1)
cv2.ellipse(frame,ret, color, True, -1)
cv2.imshow('img2',frame)
noclose = True
maxcount = 50
count = 0
while(np.sum(mask[ROI==1]==0)/np.sum(ROI).astype(float)<=percent): # np.all(mask[ROI==1])
    # np.sum(mask[ROI==1]==0)/np.sum(ROI).astype(float)<=percent
    frame = fore.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw it on image
    cv2.ellipse(frame,ret, color, True, -1)
    cv2.imshow('img2',frame)
    ROI = np.zeros(mask.shape,dtype=np.uint8)
    cv2.ellipse(ROI,ret, 1, 0, -1)

    count+=1
    if count>maxcount:
        break
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        noclose = False
        break

cv2.ellipse(fore,ret, 0, False, -1)
#fore = tf.croppad(fore,padval)
while(noclose):
    cv2.imshow('img2',fore)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()


"""
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html
from Plotim import Plotim,np,cv2
import tesisfunctions as tf
import overlay as ov
import hypothesis_functions as tf

fn1 = r'C:\Users\Davtoh\Dropbox\PYTHON\projects\tesis\im1_1.jpg'
fn1 = r"C:\Users\Davtoh\Documents\2015_01\Tesis tests\retinal photos\ALCATEL ONE TOUCH IDOL X\left_DAVID\IMG_20150730_115534_1.jpg"
name = fn1.split('\\')[-1].split(".")[0]

fore = cv2.imread(fn1)
fore = cv2.resize(fore,(800,800))
padval = 100
percent = 40
if percent>1: percent/=100.0

P = tf.brightness(fore)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
P2 = clahe.apply(P)
thresh,P2 = cv2.threshold(P2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
P2 = cv2.distanceTransform(P2,cv2.DIST_LABEL_PIXEL,5)
P2 = tf.normalizeToRange(P2).astype(np.uint8)
P2 = cv2.cvtColor(P2,cv2.COLOR_GRAY2BGR)
fore = tf.pad(fore,(np.min(fore[:,:,0]),np.min(fore[:,:,1]),np.min(fore[:,:,2])),padval,True)
P2 = tf.pad(P2,(np.min(P2[:,:,0]),np.min(P2[:,:,1]),np.min(P2[:,:,2])),padval,True)

frame = fore.copy()

# set up the ROI for tracking
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

P = tf.brightness(fore)
th,mask = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#mask = cv2.inRange(hsv_roi, np.array((0., 0.,176.)), np.array((179.,159.,255.)))
#Plotim("mask", mask).show()
fore = ov.overlay(fore,mask,alfa=mask*0.2)
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# setup initial location of window
h,w = 10,10
#x,y = ov.convertXY(0,0,frame.shape,(h,w),4) # simply hardcoded the values
#x,y = np.mean([tf.getThreshCenter1(mask),tf.getThreshCenter2(mask),tf.getThreshCenter3(mask),tf.getThreshCenter4(mask)],0).astype(int)
x,y = tf.getThreshCenter2(mask) # x,y
track_window = (x,y,w,h)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT , 10, 10 )
color = (200,0,200)
frame = fore.copy()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
ret, track_window = cv2.CamShift(dst, track_window, term_crit)
ROI = np.zeros(mask.shape,dtype=np.uint8)
cv2.ellipse(ROI,ret, 1, 0, -1)
cv2.ellipse(frame,ret, color, True, -1)
cv2.imshow('img2',frame)
noclose = True
maxcount = 50
count = 0
while(np.sum(mask[ROI==1]==0)/np.sum(ROI).astype(float)<=percent): # np.all(mask[ROI==1])
    # np.sum(mask[ROI==1]==0)/np.sum(ROI).astype(float)<=percent
    frame = fore.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw it on image
    cv2.ellipse(frame,ret, color, True, -1)
    cv2.imshow('img2',frame)
    ROI = np.zeros(mask.shape,dtype=np.uint8)
    cv2.ellipse(ROI,ret, 1, 0, -1)

    count+=1
    if count>maxcount:
        break
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        noclose = False
        break

cv2.ellipse(fore,ret, 0, False, -1)
#fore = tf.croppad(fore,padval)
while(noclose):
    cv2.imshow('img2',fore)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()

"""