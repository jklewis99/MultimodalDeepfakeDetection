
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import time

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
    return img[height % 2 :, width % 2:]

def dct_frames(path, video):
    imgs = os.listdir(os.path.join(path, video))
    
    for img_name in imgs:
        img = cv2.imread(os.path.join(path, video, img_name), 0) / 255
        img = squared_image(img)
        t = dct = cv2.dct(img)
        dct = np.log(dct - dct.min() + 1)
        norm = (dct - dct.min()) / (dct.max() - dct.min())
        dct_antidiag_avg = antidiag_avg(dct)
        np.save('{}/{}'.format(os.path.join(path, video), img_name[:-4]), dct_antidiag_avg)

DATA_FOLDER = 'preprocessed_data'

folders = os.listdir(DATA_FOLDER) # will return a list of folders with names of the videos
start = time.time()
for vid in folders:
    dct_frames(DATA_FOLDER, vid)
    print('Finished processing dct of frames in video {}'.format(vid))
process_time = time.time() - start
print('DONE!\nProcessed the frames of {} videos in {:.2f} s'.format(len(folders), process_time))