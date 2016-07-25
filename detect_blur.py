
import cv2


def variance_of_laplacian(image):
	#compute the Laplacian of the image and return the focus measure (variance)
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return  lap.var(), lap


def  tenengrad(img):
    import numpy as np
    ksize = 5
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0, ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1 , ksize=5)

    FM = np.multiply(sobelx,sobelx) + np.multiply(sobelx,sobelx)
    #print FM.shape
    focusMeasure = cv2.mean(FM)
    #print "tenengrad==> " , focusMeasure
    return focusMeasure

def normalizedGraylevelVariance(image):
    mu, sigma = cv2.meanStdDev(image)
    focusMeasure = (sigma * sigma ) / mu
    #print focusMeasure[0]
    return focusMeasure[0]


