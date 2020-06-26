import cv2 as cv
import os
import pandas as pd
from matplotlib import pyplot as plt
from moviepy.editor import *
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np

DATA_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'

def extract_frames(video_list, path):

    for i in range(len(video_list)):
        video_file = video_list[i]
        capture = cv.VideoCapture(os.path.join(path, video_file))
        count = 0
        incr = 4
        if not os.path.exists('../output/video_{}'.format(i)):
            os.mkdir('../output/video_{}'.format(i))
        
        # write every {incr} frames to a new folder for each video
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                # we have reached the end of the video
                break

            cv.imwrite('../output/video_{}/frame_{}.png'.format(i, count), frame)
            capture.set(1, count)
            count += incr
        if i < 1:
            break

def extract_spectrogram(video_list, path):
    for i in range(len(video_list)):
        video_file = VideoFileClip(os.path.join(path, video_list[i]))
        audio = video_file.audio
        sample_rate = audio.fps
        audio_data = audio.to_soundarray()
        print(audio_data.shape)
        # spectrogram visualization
        NFFT = sample_rate / 25

        fig, ax = plt.subplots(figsize=(14, 4))
        spectrum, freqs, time, im = ax.specgram(audio_data.mean(axis=1), 
                                                            NFFT=NFFT,
                                                            pad_to=4096, 
                                                            Fs=sample_rate, 
                                                            noverlap=512, 
                                                            mode='magnitude', 
                                                            cmap='inferno')
        fig.colorbar(im)

        print('spectrum: {}\nfreqs: {}\ntime: {}'.format(spectrum, freqs, time))

        if not os.path.exists('../output/video_{}'.format(i)):
            os.mkdir('../output/video_{}'.format(i))
        np.save('../output/video_{}/spectrogram_{}'.format(i, i), audio_data)
        break

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    
def get_meta_from_json(path, json_file):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df

def main():
    train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))

    json_file = [file for file in train_list if file.endswith('json')][0]

    meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER, json_file)
    meta_train_df.head()

    fake_train_sample_video = list(
        meta_train_df.loc[meta_train_df.label == 'FAKE'].sample(3).index)

    extract_spectrogram(fake_train_sample_video, os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER))
    extract_frames(fake_train_sample_video, os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER))

if __name__ == '__main__':
    main()