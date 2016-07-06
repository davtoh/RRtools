from tesisfunctions import simulateLens_approx,IMAGEPATH,plotim
import glob
import cv2
imlist= glob.glob(IMAGEPATH+"IMG*.jpg")
#rootpath = r"C:\Users\Davtoh\Dropbox\PYTHON\projects\RRtoolbox\RRtools"+"\\"
#fns= glob.glob(rootpath+"descriptor_Result.png")
#fns.extend(glob.glob(rootpath+"good_*.jpg"))
img = not None
for fn in imlist:
    try:
        img = cv2.imread(fn)
        name = "approxlens_"+fn.split('\\')[-1]
        plotim(name,simulateLens_approx(img)[0]).show()
        #cv2.imwrite(rootpath+name,simulateLens_approx(img)[0])
        #print fn, " saved as ",name
    except:
        if img is None:
            print fn, " does not exists"
        else:
            print fn, " could not be processed"