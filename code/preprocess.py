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
            frame_ft = fourier_tranform(frame, '')
            # cv.imwrite('../output/video_{}/frame_{}.png'.format(i, count), frame)
            cv.imwrite('../output/video_{}/fourier_frame_{}.png'.format(i, count), frame_ft)
            plt.savefig('../output/video_{}/1D_power_spectrum_frame_{}.png'.format(i, count))
            capture.set(1, count)
            count += incr
        break

def extract_spectrogram(video_list, path):
    for i in range(len(video_list)):
        video_file = VideoFileClip(os.path.join(path, video_list[i]))
        audio = video_file.audio
        sample_rate = audio.fps
        audio_data = audio.to_soundarray()

        # NOT SURE IF THIS IS FAST AND EFFICIENT FOR SPECTROGRAM DATA
        # SO I AM JUST SAVING THE RAW AUDIO INTO A NUMPY ARRAY
        # spectrogram visualization

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

def fourier_tranform(img, dest):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # need to ensure log0 is not used
    epsilon = 1e-8

    ft = np.fft.fft2(img)
    fshift = np.fft.fftshift(ft) + epsilon

    # scale the magnitudes to better distribution
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    one_dim_power_spectrum = azimuthal_average(magnitude_spectrum)
    return magnitude_spectrum
    # eventually, we want to return this as a tensor
    # return one_dim_power_spectrum

def azimuthal_average(img):
    # get the indices of the image
    height, width = np.indices(img.shape)

    center = np.array([(width.max() - width.min()) / 2, (height.max() - height.min()) / 2])

    # returns an array of the length of each radius from center to corner
    radius = np.hypot(width - center[0], height - center[1])
    np.hypot
    # sorts the ____
    indices = np.argsort(radius.flat)
    radius_sorted = radius.flat[indices]
    image_sorted = img.flat[indices]

    # convert radius to integer (will get check only one pixel in radial bin per radius length)
    radius_int = radius_sorted.astype(int)

    # find all the pixels in the radial bin
    delta_r = radius_int[1:] - radius_int[:-1]  # assumes all radii are represented
    radius_idx = np.where(delta_r)[0]           # location of changed radius
    num_radius_bin = radius_idx[1:] - radius_idx[:-1]       # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    cumulative_sum = np.cumsum(image_sorted, dtype=float)
    total_bin = cumulative_sum[radius_idx[1:]] - cumulative_sum[radius_idx[:-1]]

    radial_profile = total_bin / num_radius_bin

    visualize_radial_spectrum(radial_profile)
    return radial_profile

def visualize_radial_spectrum(radial_profile):
    t = np.arange(0, len(radial_profile))
    return plt.plot(t, radial_profile)


def main():
    train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))

    json_file = [file for file in train_list if file.endswith('json')][0]

    meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER, json_file)
    meta_train_df.head()

    fake_train_sample_video = list(
        meta_train_df.loc[meta_train_df.label == 'FAKE'].index)

    extract_spectrogram(fake_train_sample_video, os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER))
    extract_frames(fake_train_sample_video, os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER))

if __name__ == '__main__':
    main()