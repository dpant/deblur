#!/usr/bin/env python


import glob
import os
import sys
import random

import cv2
from PIL import Image
import numpy as np


def dilate(ary, N, iterations): 

    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[(N-1)/2,:] = 1
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[:,(N-1)/2] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))



def find_components(edges, max_components=16):
    count = 20
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    #print dilation
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * dilated_image).show()
    return contours





def downscale_image(im, max_dim=2048):
    """Shrink im until its longest dimension is <= max_dim.

    Returns new_image, scale (where scale <= 1).
    """
    a, b = im.size
    if max(a, b) <= max_dim:
        return 1.0, im

    scale = 1.0 * max_dim / max(a, b)
    new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
    return scale, new_im

def findCountours_num(path):

    orig_im = Image.open(path)
    scale, im = downscale_image(orig_im)

    edges = cv2.Canny(np.asarray(im), 100, 200)
    cv2.imwrite('egde.jpg', edges)
        # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(np.asarray(edges), contours, -1, (255,255,155), 3)
    print len(contours)
    return len(contours)

def findText(fpath,ind):
    orig_im = Image.open(fpath)
    scale, im = downscale_image(orig_im)

    edges = cv2.Canny(np.asarray(im), 100, 200)
    #print edges
    #print "MAX=====>" ,np.max(edges)

    path = os.path.split(os.path.abspath(fpath))
    (file_name, ext) = os.path.splitext(path[1])

    if not os.path.exists("output"):
        os.makedirs("output")
    canny_dir = path[0] + '/' + 'edge'

    if not os.path.exists(canny_dir):
        os.makedirs(canny_dir)

    tmp_file = canny_dir + '/' + file_name + "_canny_" + str(ind) + ".jpg"

    fn = 'egde_' + str(ind) + ".jpg"
    cv2.imwrite(tmp_file ,edges)
    #tes = cv2.imread(fn)
    import face_detect
    #face_detect.detect_blob(edges)

    #print "Mean======>" , np.mean(edges)
    #print "# of W===> " , edges[edges == 255]
    #print "# of W===> ", len(edges[edges == 255])
    canny_edges = len(edges[edges == 255])
    #print "MAX_1=====>", np.max(tes)
    # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(np.asarray(edges), contours, -1, (255,255,155), 3)
    #print len(contours)
    #print "xyz==>" , im
    #cv2.imshow("window title", np.asarray(edges))
    #cv2.waitKey()
    #sys.exit()

    border_contour = None

    edges = 255 * (edges > 0).astype(np.uint8)

    # Remove ~1px borders using a rank filter.
    #cv2.imwrite('egde_f.jpg',edges)
    contours = find_components(edges)

    #cv2.drawContours(np.asarray(edges), contours, -1, (255,255,155), 3)
    #print len(contours)
    #print "xyz==>" , im
    #cv2.imshow("window title", np.asarray(edges))
    #cv2.waitKey()
    #sys.exit()

    if len(contours) == 0:
        print '%s -> (no text!)' % fpath
        return 0,canny_edges
    else:
        #print len(contours)
        return len(contours),canny_edges

