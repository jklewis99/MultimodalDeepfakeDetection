import cv2
import numpy as np
from Utils.image import min_dim_image_resize

def antidiag_avg(x):
    '''
    Average antidiagonal elements of a 2d array

    x : 2D np.array

    Return:
    x1d : 1D np.array of antidiagonal averages
    '''

    x1d = [np.mean(x[::-1, :].diagonal(i)) for i in range(-x.shape[0] + 1, x.shape[1])]
    return np.array(x1d)

def squared_image(img):
    '''
    method to remove the first column and/or first row to make the dimensions of the array even

    img: 2D nparray
    '''
    
    height, width = img.shape
    return img[height % 2 :, width % 2 :]

def dct_antidiagonal_on_image(img, information_size=128):
    '''
    method to take the dct of a 2D array and calculate its antidiagonal average and save as a numpy array
    
    img: np array
    '''
    
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = min_dim_image_resize(img, information_size) 
    img = img / 255
    img = squared_image(img)
    dct = cv2.dct(img)
    dct = np.log(dct - dct.min() + 1)
    norm_dct = (dct - dct.min()) / (dct.max() - dct.min()) # normalize
    return antidiag_avg(norm_dct[:information_size, :information_size])

def dct_antidiagonal_on_sequence(imgs):
    '''
    imgs: array of sequential landmark imgs (nparrays)
    '''
    antidiag_avgs = []
    for img in imgs:
        antidiag_avg = dct_antidiagonal_on_image(img)
        antidiag_avgs.append(antidiag_avg)
    antidiag_avgs = np.stack(antidiag_avgs)
    return antidiag_avgs