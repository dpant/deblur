import cv2
import sys
import numpy as np
from detect_blur import *
from  deblur_weiner import *
from utils import *

def detectface(imgPath,cPath,sf=1.3):

    faceCascade = cv2.CascadeClassifier(cPath)
    # Read the image
    image = cv2.imread(imgPath)
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayimg,scaleFactor=sf,minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

    print "Found {0} face".format(len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #imgx = image[y:y+h,x:x+w]
        #cv2.imshow("Facesx", imgx)
        #cv2.waitKey(0)
        #sys.exit()

    cv2.imshow("Faces", image)
    cv2.waitKey(0)
    return faces

def blurFaces(imgPath,faces):
    import os
    image = cv2.imread(imgPath)
    result_image = image.copy()
    index = 0
    for (x, y, w, h) in faces:
        index +=1 
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img_one_face = image[y:y+h,x:x+w]
        one_face_blur = cv2.GaussianBlur(img_one_face,(3, 3), 30)

        result_image[y:y+img_one_face.shape[0], x:x+img_one_face.shape[1]] = one_face_blur # blur single
        
        #face_file_name = "./face_" + str(y) + ".jpg"
        
        path = os.path.split(os.path.abspath(imgPath))
        (file_name,ext) = os.path.splitext(path[1])
        resized_image_name = file_name + "_blurred_" +  str(index) + "_" +  ext
        #print resized_image_name
        face_file_name = path[0] + "/" + resized_image_name

        cv2.imwrite(face_file_name, result_image)
        cv2.imshow("Facesx", result_image)
        result_image[y:y+img_one_face.shape[0], x:x+img_one_face.shape[1]] = img_one_face # restore

        cv2.waitKey(0)


def detectFaceblurNormal(imgPath, faces):
    image = cv2.imread(imgPath)
    result_image = image.copy()
    fullimg_VOL, _ = variance_of_laplacian(image)
    cv2.imshow("Laplacian_FULL_IMGx",_)
    cv2.waitKey(0)
    print "Variance Of Laplacian (Full Image):", imgPath, "==>", fullimg_VOL
    # print "Tenegrad blurness Index for Full Image (R,G,B):" , imgPath ,  "==>" , tenengrad(image) , "\n"
    index = 0
    img_lap_var = []
    for (x, y, w, h) in faces:
        img_one_face = image[y:y + h, x:x + w]
        img_lap_var.append(variance_of_laplacian(img_one_face)[0])

    if len(faces) == 1:  # just one face compare with the whole image
        print "Analyzing Face #1"
        img_one_face = image[y:y + h, x:x + w]
        cv2.imshow("Facesx", img_one_face)
        face_file_name = "./face_" + str(1) + ".jpg"
        cv2.imwrite(face_file_name, img_one_face)
        print "Variance Of Laplacian for Single Face:", "==>", img_lap_var[0]

        if (img_lap_var[0] < fullimg_VOL * 0.7):
            print "Face is Blur"
        else:
            print "Face is not Blur"
    else:
        for (x, y, w, h) in faces:
            print "Analyzing Face #", index + 1
            img_one_face = image[y:y + h, x:x + w]
            cv2.imshow("Facesx", img_one_face)
            lap = variance_of_laplacian(img_one_face)[1]
            cv2.imshow("Laplacian", lap)
            face_file_name = "./face_" + str(index) + ".jpg"
            cv2.imwrite(face_file_name, img_one_face)
            if (img_lap_var[index] < np.max(img_lap_var) * 0.5):
                print "Face is Blur"
            else:
                print "Face is not Blur"
            print "Variance Of Laplacian for Single Face:", "==>", img_lap_var[index]
            print "\n"
            index += 1
            cv2.waitKey(0)

def detect_blob(im):
    # Standard imports
    #import cv2
    #import numpy as np;

    # Read image
    #print "img_path ==> " , img
    #im = cv2.imread(img)
    #print im , np.max(im)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector(params)

    # Detect blobs.
    keypoints = detector.detect(im)
    print "keypoints==>", keypoints

    if len(keypoints) == 0:
        return
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

def detect_canny_image_char():
    for i in range(8,9):
        image = "./edge_" + str(i) + ".jpg"
        #image = "./deblurred_" + str(i) + ".jpg"
        img = cv2.imread(image)
        print img

        print i
        print image
        detect_blob(image)
#detect_blob("edge_17.jpg")
#detect_canny_image_char()
