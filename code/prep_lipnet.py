import numpy as np
import cv2
import torch

def save_mouth_sequence(video, video_id, save_path=None):
    video = np.stack(video, axis=0).astype(np.float32)
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0
    if save_path:
        torch.save(video, '{}/{}-lipnet-mouths.pt'.format(save_path, video_id))
    return video