from face_detect import *
from deblur import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='deblur_util')
    debug = True
    parser.add_argument('imgPath', nargs='*', help='specify the image path')
    parser.add_argument('--cPath', default= 'haarcascade_frontalface_default.xml', nargs='?', help='cascade file path for face detection')
    parser.add_argument('--do_blur', default= 0, nargs='?', help='Generates the blurred face image in the current image directory')
    parser.add_argument('--detect_face_blur', default=0, nargs='?', help='detect the blur based on faces present in the image')
    parser.add_argument('--defocus_deblur', default=0, nargs='?', help='correct the blur due to defocus of camera')
    parser.add_argument('--motion_deblur', default=0, nargs='?', help='correct the blur due to linear motion of camera')
    args = parser.parse_args()


    imgPathList = args.imgPath
    cPath = args.cPath
    """
    imgPathList = ["C:\VM2_SHARED\FinalProject\inputs\\text\set_3\IMG_5085.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_3\IMG_5086.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_3\IMG_5087.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_4\IMG_5091.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_4\IMG_5092.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_4\IMG_5093.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_4\IMG_5094.jpg",
                   "C:\VM2_SHARED\FinalProject\inputs\gen\Capture.png" , \
                   "C:\VM2_SHARED\FinalProject\inputs\gen\\scene_camera_blur.png"
                   ]

    imgPathList = ["C:\VM2_SHARED\FinalProject\inputs\\text\set_3\IMG_5087.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_3\IMG_5086.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_3\IMG_5085.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_4\IMG_5091.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_4\IMG_5092.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_4\IMG_5093.jpg", \
                   "C:\VM2_SHARED\FinalProject\inputs\\text\set_4\IMG_5094.jpg",
                   "C:\VM2_SHARED\FinalProject\inputs\gen\Capture.png", \
                   "C:\VM2_SHARED\FinalProject\inputs\gen\scene_camera_blur.png"
                   ]
    """
    for imgPath in imgPathList: # multiple images can be specified at once.
        print "Processing image:" , imgPath
        new_img_path = resizeImg(imgPath,800);
        imgPath = new_img_path
        imm = cv2.imread(imgPath)
        print "Variance of Laplacian:", variance_of_laplacian(imm)[0]
        print "Tenengrad value (Sobel):", tenengrad(imm)[0]

        if args.detect_face_blur:
            faces = detectface(imgPath,cPath)
            detectFaceblurNormal(imgPath,faces)

        if args.do_blur:
            blurFaces(imgPath,faces)
            sys.exit()

        if args.defocus_deblur:
            do_focus_deblur(imgPath)

        if args.motion_deblur:
            do_motion_deblur(imgPath)



