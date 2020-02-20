#%%
from imutils import contours
from skimage import measure
import PIL
import numpy as np
import argparse
import imutils
import cv2
def countLargeBlobs(img):
    image = img
    gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold the image to reveal light regions in the
    # blurred image
    thresh = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY)[1]
    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    # This threshold represent the amount of pixels needed to consider a blob as "LARGE"
    PIXEL_THRESHOLD = 15
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
    
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
    
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > PIXEL_THRESHOLD:
            mask = cv2.add(mask, labelMask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return len(cnts)

#%%
import numpy as np
import pickle
from matplotlib import pyplot as plt

def read_t(t=0.25,root="../IsingData/"):
    if t > 0.:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    else:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=All.pkl','rb'))
    return np.unpackbits(data).astype(int).reshape(-1,1600)
# exemple sur 1 image à temp = 2

for i,t in enumerate(np.arange(0.25,4.01,0.25)):
	nb_of_1 = 0
	nb_of_2 = 0
	nb_of_3 = 0
	nb_of_4 = 0
	nb_of_above = 0
	
	print("=================== Sample Temperature = ", t, " ===================")
	for i in range(10000) :
		img = read_t(t)[i,:].reshape((40,40)).astype(dtype=np.uint8)
		img[img == 1] = 255

		whiteBlobs = countLargeBlobs(img)
		blackBlobs = countLargeBlobs(-1*img+255)

		if ( whiteBlobs + blackBlobs == 1 ) :
			nb_of_1 = nb_of_1 + 1

		if ( whiteBlobs + blackBlobs == 2 ) :
			nb_of_2 = nb_of_2 + 1

		if ( whiteBlobs + blackBlobs == 3 ) :
			nb_of_3 = nb_of_3 + 1

		if ( whiteBlobs + blackBlobs == 4 ) :
			nb_of_4 = nb_of_4 + 1

		if ( whiteBlobs + blackBlobs > 4 ) :
			nb_of_above = nb_of_above + 1

		#print("t = {0}, i = {1}".format(t, i))

	print("Number of 1 = ", nb_of_1)
	print("Number of 2 = ", nb_of_2)
	print("Number of 3 = ", nb_of_3)
	print("Number of 4 = ", nb_of_4)
	print("Number of above = ", nb_of_above)