from __future__ import absolute_import
from builtins import zip
__author__ = 'Davtoh'

from .tesisfunctions import hist_cdf

import cv2
import csv
import glob
imlist= glob.glob("im*.jpg")
names = ["_S_","_H_","_V_","_fgray_"]
names = ["_fb_","_fg_","_fr_","_fgray_"]
headers = []
data = []
for fn in imlist:
    fore = cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2HSV)
    fb,fg,fr=cv2.split(fore)
    fgray = cv2.cvtColor(fore,cv2.COLOR_BGR2GRAY)
    for i,img in enumerate([fb,fg,fr,fgray]):
        hist,cdf = hist_cdf(img)
        headers.append(fn+names[i]+"hist")
        data.append(list(hist))
        headers.append(fn+names[i]+"cdf")
        data.append(list(cdf))
data = list(zip(*data))
data.insert(0,headers)
with open('experimental_data.csv', 'wb') as csvfile:
    wr = csv.writer(csvfile, delimiter=";", dialect='excel')
    wr.writerows(data)

