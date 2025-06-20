import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
# file_path = 'P:\CS39440 Major Project\pcot-exomars-pancam-major-project\HenryFiles\Camera Calibration\Tutorial\gradient.png'
file_path = 'HenryFiles/Camera Calibration/right images/aupe-cal_c25_RWAC01_T00_P00_BS.png'
folder_path = "HenryFiles/Camera Calibration/left images/"
# folder_path = "HenryFiles/Camera Calibration/right images/"

# img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"

# ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
# ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
# ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

def attempt_thresh(image_files):
    for file in image_files:

        img = cv.imread(file)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY)

        cv.imshow('Thresholded Image', thresh)
        cv.waitKey(0)
 
image_files = glob.glob(folder_path + "*.png")

attempt_thresh(image_files)
