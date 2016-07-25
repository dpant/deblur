__author__ = 'dpant'
import numpy as np
import cv2
import deblur_weiner


def do_focus_deblur(fname):
    import matlab.engine
    eng = matlab.engine.connect_matlab()

    parameters_blur_radius = range(5,18)

    deblur_weiner.do_deblurring(eng,parameters_blur_radius,fname)

def do_motion_deblur(fname):
    import matlab.engine
    eng = matlab.engine.connect_matlab()

    parameters_blur_radius = range(5,18)

    deblur_weiner.do_deblurring(eng,parameters_blur_radius,fname,False)