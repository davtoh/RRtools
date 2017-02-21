from __future__ import print_function


import glob
from tools import entropy

mainpath = ""
imlist= glob.glob(mainpath+"im1*.*g")
print("Selecting: ",imlist)
sortedD,sortedImages,D,imlist = entropy(imlist)
print("Information amount: ", D)
print("Reference image: ", sortedImages[0])
print("Target image(s): ", sortedImages[1:])