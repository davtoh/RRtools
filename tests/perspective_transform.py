import cv2
import numpy as np
from matplotlib import pyplot as plt
from RRtoolbox.lib.plotter import Plotim
from RRtoolbox.lib.image import getcoors, drawcoorperspective
fn1 = r"im1_1.jpg"
img = cv2.imread(fn1)  # (width, height)
rows,cols,ch = img.shape # (height, width, channel)

# Top_left,Top_right,Bottom_left,Bottom_right -> point[col,row] -> point[x,y]

points =getcoors(img,"get pixel coordinates", updatefunc=drawcoorperspective)
pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]]) # top_left,top_right,bottom_left,bottom_right
if points:
    pts1 = np.float32(points)
else:
    pts1 = pts2

M = cv2.getPerspectiveTransform(pts1,pts2)
print M.shape
dst = cv2.warpPerspective(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
