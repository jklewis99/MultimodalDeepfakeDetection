import cv2 as cv
import os
import pandas as pd
from matplotlib import pyplot as plt
from moviepy.editor import *
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
from scipy import signal

DATA_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'

def extract_frames(file_name, path, dest):
    '''
    a preprocessing method that converts the frames of the input video into image files

    video_file: video file to be processed
    path: directory from which the files are located
    dest: directory to which the spectrogram numpy arrays will be saved
    '''

    capture = cv.VideoCapture(os.path.join(path, file_name))
    count = 0
    incr = 25
    
    # write every {incr} frames to a new folder for each video
    while capture.isOpened():
        print('YOUOIJO')
        success, frame = capture.read()
        if not success:
            # we have reached the end of the video
            break
        power_spectrum_1D = fourier_tranform(frame, '')
        np.save('{}/{}/azimuthal_{}'.format(dest, file_name, count), power_spectrum_1D)
        capture.set(1, count)
        count += incr

def extract_spectrogram(file_name, path, dest):
    '''
    a preprocessing method that takes the audio of the input video and converts the sample
    into a numpy array defining the spectrogram of the sample

    video_list: array of videos for preprocessing
    path: directory from which the files are located
    dest: directory to which the spectrogram numpy arrays will be saved
    '''
    
    video_file = VideoFileClip(os.path.join(path, file_name))
    audio = video_file.audio
    sample_rate = audio.fps
    audio_sample = audio.to_soundarray()
    audio_sample = audio_sample.mean(axis=1) # convert from stereo to mono
    frequencies, times, spectrogram = signal.spectrogram(audio_sample, fs=sample_rate, nfft= sample_rate/25)

    np.save('{}/{}/{}'.format(dest, file_name, 'spectrogram'), spectrogram)

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
    return one_dim_power_spectrum
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

    # visualize_radial_spectrum(radial_profile)
    return radial_profile

def visualize_radial_spectrum(radial_profile):
    t = np.arange(0, len(radial_profile))
    return plt.plot(t, radial_profile)

def visualize_spectrogram(times, frequencies, spectrogram):
    fig, ax = plt.subplots(figsize=(14, 4))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    ax.set_title('SPECTROGRAM')
    ax.pcolormesh(times, frequencies, 20*np.log(spectrogram), cmap='Greys')
    plt.show()

def load_spectrogram(path, file_name):
    spectrogram_np = np.load(os.path.join(path, file_name))

def stride_search(img, target_size=(128, 128)):
    '''
    A method to loop through an image and search the image for faces by tiling with overlaps

    img - the original image on which we want to find the face
    target_size - desired width and height of the output image (the input size for blazeface)

    split landscape video into three overlapping frames of size height, unless video is
    in portrait mode, then only take top of image
    '''

    height, width, _ = img.shape

    split_size = min(height, width) # size of each tile
    x_step = (width - split_size) // 2  # amount to move horizontally
    y_step = (height - split_size) // 2  # amount to move vertically
    num_v = 1                       # number of tiles in vertical direction
    num_h = 3 if width > height else 1       # number of tiles in horizontal direction

    # array that contains the resized tiles
    tiles = np.zeros((num_v * num_h, target_size[1], target_size[0], 3), dtype=np.uint8)
    i = 0
    for tile_row in range(num_v):
        tile_start_y = tile_row * y_step
        for tile_col in range(num_h):
            tile_start_x = tile_col * x_step
            tile = img[tile_start_y : tile_start_y + split_size, tile_start_x : tile_start_x + split_size]
            tiles[i] = cv.resize(tile, target_size, interpolation=cv.INTER_AREA)
            i+=1

    resize_info = [split_size / target_size[0], split_size / target_size[1], 0, 0]
    return tiles, resize_info

def main():
    train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))

    json_file = [file for file in train_list if file.endswith('json')][0]

    meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER, json_file)
    meta_train_df.head()

    fake_train_sample_video = list(
        meta_train_df.loc[meta_train_df.label == 'FAKE'].sample(3).index)

    dest = '../output'
    for video_file in fake_train_sample_video:
        if not os.path.exists('{}/{}'.format(dest, video_file)):
            os.mkdir('{}/{}'.format(dest, video_file))
        extract_spectrogram(video_file, os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER), dest)
        extract_frames(video_file, os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER), dest)

if __name__ == '__main__':
    main()