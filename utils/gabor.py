import cv2
import numpy as np


def gabor_hash(specle_image, wavelength=10., k_scale=1 / 10, orientation=45.,
               SpatialFrequencyBandwidth=1., SpatialAspectRatio=0.5,
               cutoff_sigma=4., do_threshold=True):
    orientation = -orientation / 180 * np.pi
    sigma = 0.5 * wavelength * SpatialFrequencyBandwidth
    shape = 1 + 2 * int(np.ceil(cutoff_sigma * sigma))
    shape = (shape, shape)

    gabor_filter_imag = cv2.getGaborKernel(
        ksize=shape, sigma=sigma, theta=orientation, lambd=wavelength,
        gamma=SpatialAspectRatio, psi=np.pi / 2)
    im_gabor = cv2.filter2D(specle_image.astype(np.float32), -1,
                            gabor_filter_imag)
    gabor_im_small = cv2.resize(im_gabor, None, fx=k_scale, fy=k_scale,
                                interpolation=cv2.INTER_CUBIC)

    if do_threshold:
        gabor_hash = gabor_im_small > 0
        return gabor_hash
    else:
        return gabor_im_small


def hash_using_conv_kernel(specle_image, filter, k_scale=1 / 10,
                           do_threshold=False):
    im_gabor = cv2.filter2D(specle_image.astype(np.float32), -1,
                            filter[::-1, ::-1],
                            borderType=cv2.BORDER_REPLICATE)

    gabor_im_small = cv2.resize(
        im_gabor, None, fx=k_scale, fy=k_scale, interpolation=cv2.INTER_CUBIC
    )

    if do_threshold:
        gabor_hash = gabor_im_small > 0
        return gabor_hash
    else:
        return gabor_im_small
