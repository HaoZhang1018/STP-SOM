import numpy as np
from PIL import Image
import tensorflow as tf
import scipy.stats as st
from skimage import io,data,color
from functools import reduce
import cv2

from scipy import ndimage
from scipy import signal
#import scipy.misc as 


def load_images_no_norm(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0    
    return img
    
    
def gauss_kernel(kernlen, nsig, channels):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def blur(x,kernlen, nsig,channel):
    kernel_var = gauss_kernel(kernlen, nsig, channel)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

def fspecial_gauss(size, sigma):
    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def fir_filter_wind(hd, w):

    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = np.clip(h, a_min=0, a_max=np.max(h))
    h = h / np.sum(h)

    return h

def nyquist_filter_generator(nyquist_freq, ratio, kernel_size):

    assert isinstance(nyquist_freq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'

    if isinstance(nyquist_freq, list):
        nyquist_freq = np.asarray(nyquist_freq)

    nbands = nyquist_freq.shape[0]

    kernel = np.zeros((kernel_size, kernel_size, nbands))  # generic kerenel (for normalization purpose)

    fcut = 1 / np.double(ratio)

    for j in range(nbands):
        alpha = np.sqrt(((kernel_size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(nyquist_freq[j])))
        H = fspecial_gauss((kernel_size, kernel_size), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(kernel_size, 0.5)

        kernel[:, :, j] = np.real(fir_filter_wind(Hd, h))
    MTF_kernal = tf.expand_dims(kernel,-1)
    return MTF_kernal

def downgrade_images(I_MS, I_PAN, ratio, sensor=None,flag_PAN_MTF=0):
    """
    downgrade MS and PAN by a ratio factor with given sensor's gains
    """
    
    if sensor=='QB':
        flag_resize_new = 2
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = np.asarray([0.15])
    elif sensor=='IKONOS':
        flag_resize_new = 2             #MTF usage
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = np.asarray([0.17])
    elif sensor=='GeoEye1':
        flag_resize_new = 2             # MTF usage
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = np.asarray([0.16])     
    elif sensor=='WV2':
        flag_resize_new = 2             # MTF usage
        GNyq = np.asarray([0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.27])
        GNyqPan = np.asarray([0.11])
    elif sensor=='WV3':
        flag_resize_new = 2             #MTF usage
        GNyq = 0.29 * np.ones(8)
        GNyqPan = np.asarray([0.15])
    else:
        flag_resize_new = 1
    
    '''the default downgrading method is gaussian'''
    if flag_resize_new == 1:     
        I_PAN_LR = I_PAN[:,0::int(ratio), 0::int(ratio),:]
        I_MS_LR =  I_MS[:,0::int(ratio), 0::int(ratio),:]
                                                    
    elif flag_resize_new==2:        
        N=41
        MS_MTF_kernal = tf.cast(nyquist_filter_generator(GNyq, ratio, N),dtype=tf.float32)
        MS_filter = tf.nn.depthwise_conv2d(I_MS, MS_MTF_kernal, [1, 1, 1, 1], padding='SAME')
        I_MS_LR =  MS_filter[:,0::int(ratio), 0::int(ratio),:]
           
        if flag_PAN_MTF==1:
            #fir filter with window method
            PAN_MTF_kernal = tf.cast(nyquist_filter_generator(GNyqPan, ratio, N),dtype=tf.float32)
            PAN_filter = tf.nn.depthwise_conv2d(I_PAN, PAN_MTF_kernal, [1, 1, 1, 1], padding='SAME')
            I_PAN_LR =  PAN_filter[:,0::int(ratio), 0::int(ratio),:]
                        
        else:
            #bicubic resize
            I_PAN_LR =  I_PAN[:,0::int(ratio), 0::int(ratio),:]
                          
    return I_MS_LR,I_PAN_LR



def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)



def load_raw_high_images(file):
    raw = rawpy.imread(file)
    im_raw = raw.postprocess(use_camera_wb = True, half_size = False, no_auto_bright=True, output_bps=16)
    #im_raw = np.maximum(im_raw - 512,0)/ (65535 - 512)
    im_raw = np.float32(im_raw/65535.0)
    im_raw_min = np.min(im_raw)
    im_raw_max = np.max(im_raw)
    a_weight = np.float32(im_raw_max - im_raw_min)
    im_norm = np.float32((im_raw - im_raw_min) / a_weight)
    return im_norm, a_weight

def load_raw_images(file):
    raw = rawpy.imread(file)
    im_raw = raw.postprocess(use_camera_wb = True, half_size = False, no_auto_bright=True, output_bps=16)
    #im_raw = np.maximum(im_raw - 512,0)/ (65535 - 512)
    im_raw = np.float32(im_raw/65535.0)
    im_raw_min = np.min(im_raw)
    im_raw_max = np.max(im_raw)
    a_weight = np.float32(im_raw_max - im_raw_min)
    im_norm = np.float32((im_raw - im_raw_min) / a_weight)
    return im_norm, a_weight

def load_raw_low_images(file):
    raw = rawpy.imread(file)
    im_raw = raw.postprocess(use_camera_wb = True, half_size = False, no_auto_bright=True, output_bps=16)
    im_raw = np.maximum(im_raw - 512.0,0)/ (65535.0 - 512.0)
    im_raw = np.float32(im_raw)
    im_raw_min = np.min(im_raw)
    print(im_raw_min)
    im_raw_max = np.max(im_raw)
    print(im_raw_max)
    a_weight = np.float32(im_raw_max - im_raw_min)
    im_norm = np.float32((im_raw - im_raw_min) / a_weight)
    print(a_weight)
    return im_norm, a_weight

def load_images_and_norm(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    norm_coeff = np.float32(img_max - img_min)
    return img_norm, norm_coeff

def load_images_and_a_and_norm(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    a_weight = np.float32(img_max - img_min)
    return img, img_norm, a_weight

def load_images_and_a_003(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    img_norm = (np.maximum(img_norm, 0.03)-0.03) / 0.97
    a_weight = np.float32(img_max - img_min)
    return img_norm, a_weight


def load_images_no_norm(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0


def load_images_uint16(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 65535.0

def load_images_hsv(file):
    im = io.imread(file)
    hsv = color.rgb2hsv(im)

    return hsv

def save_images(filepath, result_1, result_2 = None, result_3 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    result_3 = np.squeeze(result_3)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)
    if not result_3.any():
        cat_image = cat_image
    else:
        cat_image = np.concatenate([cat_image, result_3], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype(np.uint8))
    im.save(filepath, 'png')


def save_images_noise(filepath, result_1, result_2 = None, result_3 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    result_3 = np.squeeze(result_3)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)
    if not result_3.any():
        cat_image = cat_image
    else:
        cat_image = np.concatenate([cat_image, result_3], axis = 1)

    im = Image.fromarray(np.clip(abs(cat_image) * 255.0,0, 255.0).astype('uint8'))
    im.save(filepath, 'png')
    
    

