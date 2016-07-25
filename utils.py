import cv2
import sys
import os

def resizeImg(imgPath, pxl_w=800):
    import os
    image = cv2.imread(imgPath)
    if (image.shape[1] < pxl_w):
        print  "Warning Image:", imgPath, "is smaller than", pxl_w, "in height. No Rescaling done"
        return imgPath
    r = (pxl_w * 1.0) / image.shape[1]
    dim = (pxl_w, int(image.shape[0] * r))
    # print dim
    # print imgPath
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    path = os.path.split(os.path.abspath(imgPath))
    (file_name, ext) = os.path.splitext(path[1])
    resized_image_name = file_name + "_" + str(pxl_w) + "_" + ext
    resized_image_name = file_name + "_" + str(pxl_w) + "_" + ".png"
    # print resized_image_name
    new_fname = path[0] + "/" + resized_image_name
    print new_fname
    cv2.imwrite(new_fname, resized)
    cv2.imwrite(new_fname, resized)
    return new_fname



def convertjpgtopng(image,name):
    img = cv2.imread(image)
    newx,newy = img.shape[1]/4,img.shape[0]/4 #new size (w,h)
    newimage = cv2.resize(img,(newx,newy))
    cv2.imwrite(name, newimage)