from tesisfunctions import simulateLens, IMAGEPATH
import glob
import cv2
imlist= glob.glob(IMAGEPATH+"IMG*.jpg")
#rootpath = r"C:\Users\Davtoh\Dropbox\PYTHON\projects\tesis"+"\\"
#fns= glob.glob(rootpath+"descriptor_Result.png")
#fns.extend(glob.glob(rootpath+"good_*.jpg"))
for fn in imlist:
    try:
        img = cv2.imread(fn)
        name = "lens_"+fn.split('\\')[-1]
        cv2.imwrite(rootpath+name,simulateLens(img,parameters=None)[0])
        print fn, " saved as ",name
    except:
        if img is None:
            print fn, " does not exists"
        else:
            print fn, " could not be processed"