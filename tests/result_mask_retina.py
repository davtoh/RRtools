from __future__ import print_function
from __future__ import absolute_import
from .tesisfunctions import brightness,retinalmask,IMAGEPATH
import glob
import cv2
imlist= glob.glob("im*.jpg")
imlist.extend(glob.glob("good_*.jpg"))
for fn in imlist:
    try:
        img = cv2.imread(fn)
        name = "masked_"+fn
        P = brightness(img)
        mask = retinalmask(P)
        img[mask==0]=0
        cv2.imwrite(name,img)
        print(fn, " saved as ",name)
    except:
        if img is None:
            print(fn, " does not exists")
        else:
            print(fn, " could not be processed")