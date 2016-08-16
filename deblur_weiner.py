import cv2
import matlab
import numpy as np
from detect_blur import *
import edged
import os

def normalizeImage(img, dtype=np.uint8):
    """ This function normalizes an image from any range to 0->255.

    Note: This sounds simple, but be very careful about getting this right. I
    heavily suggest you follow the steps listed below.

    1. Set 'out' equal to 'img' subtracted by the minimum of the image. For
    example, if your image range is from 10->20, subtracting the minimum will
    make the range from 0->10. The benefit of this is that if the range is from
    -10 to 10, it will set the range to be from 0->20. Since we will have to
    deal with negative values, it is important you normalize the function this
    way. We do the rounding for you so do not do any casting or rounding for
    the input values (a value like 163.92 will be cast to 163 by the return
    statement that casts everything to a uint8).

    2. Now, multiply 'out' by (255 / max) where max is the max value of 'out'.
    This max is computed after you subtract the minimum (not before).

    3. return 'out'.

    Args:
        img (numpy.ndarray): A grayscale or color image represented in a numpy
                             array. (dtype = np.float)

        dtype (numpy.dtype): (Optional) The type parameter for casting the
                             output array.

    Returns:
        out (numpy.ndarray): A grasycale or color image of dtype uint8, with
                             the shape of img, but values ranging from 0->255.
    """
    out = np.array(img.shape)

    # WRITE YOUR CODE HERE.
    out = np.ndarray(shape=img.shape,dtype = np.float64) #redifine to correct type
    out = np.copy(img)
    min_intensity =  np.amin(img)
    out = out - min_intensity
    max_intensity = np.amax(out)

    out = out * (255/max_intensity)


    # END OF FUNCTION.
    return out.astype(dtype)



def do_deblurring(eng, parameters_blur_radius,inp_fname,defocus=True):
    discArray = []
    volArray = []
    tenegradArray = []
    cannyEdgeArray = []
    CANNY_VAL = 10500
    if defocus:
        for diskSize in parameters_blur_radius:
            PSF = eng.fspecial('disk', float(diskSize));
            vol , tenegrad , canny_edge = do_deblurring_core(eng, diskSize, inp_fname,PSF)
            discArray.append(diskSize)
            volArray.append(vol)
            tenegradArray.append(tenegrad)
            cannyEdgeArray.append(canny_edge)

    else:
        for diskSize in parameters_blur_radius:
            #[-90.0,-45.0,0.0,45.0,90.0]
            for orient in [0]:
                PSF = eng.fspecial('motion', float(diskSize),orient);
                vol, tenegrad, canny_edge = do_deblurring_core(eng, str(diskSize) + "_" + str(orient) , inp_fname, PSF)
                discArray.append(diskSize)
                volArray.append(vol)
                tenegradArray.append(tenegrad)
                cannyEdgeArray.append(canny_edge)

    index = 0
    for elem in cannyEdgeArray:
        if elem < CANNY_VAL:
            print "Optimal PSF value: ", discArray[index]
        index +=1

    print "discArray--> " , discArray
    print "volArray--> " , volArray
    print "tenegradArray-->", tenegradArray
    print "cannyEdgeArray-->" , cannyEdgeArray

def do_deblurring_core(eng, diskSize,inp_fname,PSF):
    #print fname
    I = eng.im2double(eng.imread(inp_fname))
    #print "IMG" , np.min(I) , np.max(I)


    #eng.figure(1);eng.imshow(I); eng.title('Source image');
    #cv2.imshow("blurred_input", I)
    #figure(1); imshow(I); title('Source image');
    # Blur image


    #print type(PSF) , "\n"
    #print PSF
    sum = 0.0
    for i in range(len(PSF)):
        for j in range(len(PSF[0])):
            if(PSF[i][j] == 0):
                PSF[i][j] = .0000000000001
            sum += PSF[i][j]
    #print "PSF SUM " , sum
    #print diskSize
    #PSF[PSF == 0] = .000001;
    #PSF = fspecial('gaussian', 17, 5)
    #noise_var = 0.0001;
    #noise_var = 0.00001;
    noise_var = 0.00001;

    #adjust noise
    noise_img_open = cv2.imread(inp_fname)
    if(variance_of_laplacian(noise_img_open)[0] < 10 ):
        noise_var /=10

    I = eng.edgetaper(I,PSF);
    #estimated_nsr = noise_var / eng.var(I[:])
    I_numpy  = np.array(I._data).reshape(I.size[::-1]).T
    val = float(np.var(I_numpy))
    estimated_nsr = noise_var/val

    #print "(estimated_nsr)", type(estimated_nsr) , estimated_nsr
    #print "diskSize==> " ,diskSize
    #print "IMG111", np.min(I), np.max(I)
    deb = eng.deconvwnr(I, PSF,estimated_nsr )
    O_numpy = np.array(deb._data).reshape(deb.size[::-1]).T
    #print "deb sum==> ", np.sum(deb)
    #print "deb min==> ", np.min(deb)
    #print "deb max==> ", np.max(deb)
    #print "O_numpy_shape:" , O_numpy.shape

    # eng.imwrite(deb, 'x.jpg') does not work!!!
    # rescale the value to 0-255 as opencv works on that range
    #minv = np.min(O_numpy)
    #O_numpy -= minv
    #max = np.max(O_numpy)

    #O_numpy /= max
    #print "np.max(O_numpy)-->" , np.max(O_numpy)
    #print "np.min(O_numpy)-->" , np.min(O_numpy)
    #O_numpy *= 254.0
    #matlab index are RGB while opencv sis BGR
    for i in range(3):
        #O_numpy[..., i] = normalizeImage(O_numpy[..., i])
        O_numpy[..., 3-i-1] = normalizeImage(O_numpy[..., i])

    #print "np.max(O_numpy)-->" , np.max(O_numpy)
    #print "np.min(O_numpy)-->" , np.min(O_numpy)
    #O_numpy *= 255.0
    #print "VOL:==>" ,face_detect.variance_of_laplacian(O_numpy)[0]
    #print "tenegrad" , face_detect.tenengrad(O_numpy)


    #eng.figure(diskSize); eng.imshow(deb);
    #eng.imwrite(deb,"x.jpg")
    path = os.path.split(os.path.abspath(inp_fname))
    (file_name, ext) = os.path.splitext(path[1])

    if not os.path.exists("output"):
        os.makedirs("output")

    directory = "output/" + file_name
    #print directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    tmp_file = directory + '/' + file_name + "_deblurred_" + str(diskSize) + ".jpg"

    fname = tmp_file

    cv2.imwrite(fname ,O_numpy)
    #print "# of countors"
    #crop.findCountours_num(fname)
    #print "Text Found?"
    _,cannyedgesmean = findText(fname,diskSize)

    #cv2.waitKey(0)
    #figure(n); imshow(deconvlucy(I, PSF, 100));
    #figure(n); imshow(deconvblind(I, PSF, 100)) ,title(n);
    return variance_of_laplacian(O_numpy)[0] , tenengrad(O_numpy)[0] , cannyedgesmean
