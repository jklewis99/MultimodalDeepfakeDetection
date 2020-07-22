import cv2
from Utils.dct_processing import dct_antidiagonal_on_sequence
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='1D DCT of Landmarks Transcription')
parser.add_argument('--folder-input', default=None, help=r"Folder containing all of the landmarks (ex. path\to\mouth)")
parser.add_argument('--save-output', default=None, help="Location to save 1D dct landmark batches")

mouth_video = []
nose_video = []
eye1_video =[]
eye2_video = []
eyes_video = []

mouth_video.append(mouth)
nose_video.append(nose)
eye1_video.append(eye1)
eye2_video.append(eye2)
eyes_video.append(eyes)

dct_mouth_video = dct_antidiagonal_on_sequence(mouth_video)
dct_nose_video = dct_antidiagonal_on_sequence(nose_video)
dct_eye1_video = dct_antidiagonal_on_sequence(eye1_video)
dct_eye2_video = dct_antidiagonal_on_sequence(eye2_video)
dct_botheyes_video = dct_antidiagonal_on_sequence(eyes_video)

np.save('{}/{}-mouth-dct'.format(save_path, video_id), dct_mouth_video)
np.save('{}/{}-nose-dct'.format(save_path, video_id), dct_nose_video)
np.save('{}/{}-eye1-dct'.format(save_path, video_id), dct_eye1_video)
np.save('{}/{}-eye2-dct'.format(save_path, video_id), dct_eye2_video)
np.save('{}/{}-botheyes-dct'.format(save_path, video_id), dct_botheyes_video)

def dct_stuff(input_path, landmark, save_path, batch_size=24):
    frame_ids = []
    imgs = []
    for frame_id in os.listdir(os.path.join(input_path, landmark)):
        frame_ids.append(int(frame_id[:-4]))
        imgs.append(cv2.imread(os.path.join(input_path, landmark, frame_id)))

    dct_batch = []
    count = 0
    for frame_id, landmark_img in zip(frame_ids, imgs):
        if count != frame_id:
            np.save(f'{save_path}/{landmark}/', np.stack(dct_batch))
        antidiag_avg = dct_antidiagonal_on_image(img)

        dct_batch.append(antidiag_avg)

    np.save('{}/{}-mouth-dct'.format(save_path, video_id), dct_mouth_video)
def main():
    pass

if __name__ =='__main__':
    main()