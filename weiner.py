
"""
=====================
Image Deconvolution
=====================

In this example, we deconvolve a noisy version of an image using Wiener
and unsupervised Wiener algorithms. This algorithms are based on
linear models that can't restore sharp edge as much as non-linear
methods (like TV restoration) but are much faster.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import color, data, restoration
import cv2
from scipy.signal import convolve2d as conv2




import numpy as np
from numpy.fft import fft, ifft, ifftshift

"""
     implementination of Wiener deconvolution including the signal to noise ratio
     numpy implementation does not have snr parameter.

"""

def wiener_deconv(signal, kernel, snr):

	kernel = np.hstack((kernel, np.zeros(len(signal)  - len(kernel))))
	H = fft(kernel)
	deconv = np.real(ifft (fft (signal)* np.conj(H) / (H * np.conj(H) + snr**2)))
	return deconv



def weiner_filter(img,psf):
    astro = color.rgb2gray(data.astronaut())

    print "running Weiner for a channel"

    out = np.ndarray(shape=img.shape, dtype=np.float64)  # redifine to correct type
    out = np.copy(img)
    min_intensity = np.amin(img)
    out = out - min_intensity
    max_intensity = np.amax(out)

    out = out * (1.0 / max_intensity)

    img = out
    #psf = np.ones((5, 5)) / 25.0
    astro = conv2(astro, psf, 'same')
    astro += 0.22 * astro.std() * np.random.standard_normal(astro.shape)

    #deconvolved, _ = restoration.unsupervised_wiener(img, psf)
    #deconvolved  = restoration.wiener(img, psf, 500)
    deconvolved  = restoration.richardson_lucy(img, psf, 200)

    return deconvolved


def generate_PSF(kernlen=21, nsig=3):
    import numpy as np
    import scipy.ndimage.filters as fi


    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    value = fi.gaussian_filter(inp, nsig)
    #print value
    return value


def run_weiner(outimg,psf):
    image = '/home/dpant/CP/FinalProject/images_self/ab.png'
    image =  "/home/dpant/CP/FinalProject/text.png"
    img_org = cv2.imread(image)
    img = img_org
    num_channels = img.shape[2]
    #img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    #hdr_image = np.zeros(images[0].shape, dtype=np.float64)
    final_image = np.zeros(img.shape, dtype=np.float64)
    #img = restoration.denoise_bilateral(img)
    #(img, sigma_color=0.05, sigma_spatial=15)
    xyz = psf * 255.0
    cv2.imwrite("denoise.png",xyz)
    #psf = fspecial(8,img.shape[0])
    #print np.sum(psf)
    #import sys
    #sys.exit()
    for channel in range(num_channels):

        # collect the current layer of each input image
        layer_stack = img[:, :, channel]
        #layer_stack = [img[:, :, channel] for img in images]
        channel_res = weiner_filter(layer_stack,psf)
        min = np.min(channel_res)
        channel_res -= min
        max = np.max(channel_res)
        channel_res = channel_res/max
        channel_res *= 255.0
        # Sample image intensities
        #intensity_samples = sampleIntensities(layer_stack, num_points)
        final_image[..., channel] = channel_res

    blur_out = cv2.GaussianBlur(final_image, (5,5), 2)
    final_image = cv2.addWeighted(final_image, 1.5, blur_out, -0.5, 0);
    cv2.imwrite(outimg,final_image)

np.set_printoptions(precision=3,threshold=1000,linewidth=1000)
#threshold=None, edgeitems=None,
def fspecial(rad,mat_size=0):
    if not mat_size:
        mat_size = (int) (2* rad + 1)

    def getnc(x):
        return x - mat_size/2

    mat = np.zeros( (mat_size,mat_size) )
    for i in range(mat_size):
        for j in range(mat_size):
            x = getnc(i)
            y = getnc(j)
            r = np.sqrt(x*x + y*y)
            #print r ,  rad
            #if(r<=rad):
            #    print x, y
            mat[i,j] = (r <= rad)
    print mat
    mat = mat/np.sum(mat)
    print mat
    print mat.shape

    return mat



if __name__ == '__main__':

  psf = np.loadtxt('psf8.txt', dtype=np.float, comments='#', delimiter=',')
  #psf = np.ones((17, 17)) / (17.0 * 17.0)

  v = generate_PSF(5,0.5)
  #print  np.sum(v)
  for cnt in np.arange(1,19,1.0):
    #psf = generate_PSF(cnt,0.5)
    #psf = np.ones((cnt, cnt)) / (cnt * cnt * 1.0)
    outimg = "final_" + str(cnt) + ".png"
    psf = fspecial(cnt)
    run_weiner(outimg,psf)


