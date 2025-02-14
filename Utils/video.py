import numpy as np
import torch
import cv2
import math
from .image import *


def read_video(filename):
    """
    returns list of numpy arrays
    """

    cap = cv2.VideoCapture(filename)
    array = []

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        array.append(frame)

    cap.release()
    return array


def resize_and_crop_video(frames, dim):
    smframes = []
    xshift, yshift = 0, 0
    for i in range(len(frames)):
        if frames[i].shape[0] > frames[i].shape[1]:
            img = image_resize(frames[i], width=dim)
            yshift, xshift = (frames[i].shape[0] - frames[i].shape[1]) // 2, 0
            y_start = (img.shape[0] - img.shape[1]) // 2
            y_end = y_start + dim
            smframes.append(img[y_start:y_end, :, :])
        else:
            img = image_resize(frames[i], height=dim)
            yshift, xshift = 0, (frames[i].shape[1] - frames[i].shape[0]) // 2
            x_start = (img.shape[1] - img.shape[0]) // 2
            x_end = x_start + dim
            smframes.append(img[:, x_start:x_end, :])

    return smframes, (xshift, yshift)


# if __name__ == '__main__':
#     a = [np.ones((1080, 1920, 3)) for i in range(100)]
#     b = resize_and_crop_video(a, 128)
#     print(b[0].shape)
