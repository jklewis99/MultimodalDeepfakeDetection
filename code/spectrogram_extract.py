from scipy import signal
import cv2
from moviepy.editor import *
import numpy as np
from Utils.misc import create_directory
import torch

CV2_FRAMECOUNT_ID = int(cv2.CAP_PROP_FRAME_COUNT)
CV2_FPS_ID = int(cv2.CAP_PROP_FPS)


def main():

    input_dir = sys.argv[1]
    faces_dir = sys.argv[2]
    output_dir = sys.argv[3]

    fakelist = [os.path.join(input_dir, 'fake', p)
                for p in os.listdir(os.path.join(input_dir, 'fake'))]

    reallist = [os.path.join(input_dir, 'real', p)
                for p in os.listdir(os.path.join(input_dir, 'real'))]

    face_fakelist = [os.path.join(faces_dir, 'fake', p)
                     for p in os.listdir(os.path.join(faces_dir, 'fake'))]

    face_reallist = [os.path.join(faces_dir, 'real', p)
                     for p in os.listdir(os.path.join(faces_dir, 'real'))]

    for flist, facelist, split in zip([fakelist, reallist], [face_fakelist, face_reallist], ['fake', 'real']):
        for vid_path, face_path in zip(flist, facelist):
            fileid = get_file_id(vid_path)
            outpath = os.path.join(output_dir, split, fileid)
            create_directory(outpath)
            feats_from_vid(vid_path, face_path, outpath)


def get_file_id(path):
    file_id = path.split('/')[-1]
    file_id = file_id.split('.')[0]
    return file_id


def spectrogram(video_file, normalize=True):
    """Generates spectrogram of audio content in a video file 

    Args:
        video_file (String): path to video file

    Returns:
        [type]: [description]
    """
    videoclip = VideoFileClip(video_file)
    cap = cv2.VideoCapture(video_file)

    framecount = int(cv2.VideoCapture.get(cap, CV2_FRAMECOUNT_ID))
    fps = cv2.VideoCapture.get(cap, CV2_FPS_ID)
    sample_rate = videoclip.audio.fps

    cap.release()
    audio = videoclip.audio.to_soundarray()
    audio = audio.astype('float32') / 32767  # normalize audio
    if len(audio.shape) > 1:
        if audio.shape[1] == 1:
            audio = audio.squeeze()
        else:
            audio = audio.mean(axis=1)  # multiple channels, average

    frequencies, times, Sxx = signal.spectrogram(
        audio, fs=sample_rate, nperseg=int(sample_rate/fps), noverlap=0)
    Sxx = 10 * np.log10(Sxx + np.finfo(float).eps)
    frequencies.shape, times.shape, Sxx.shape
    if normalize:
        mean = Sxx.mean()
        std = Sxx.std()
        Sxx = Sxx - mean
        Sxx = Sxx / std
    return frequencies, times, Sxx


def feats_from_vid(vid_path, face_path, outpath):
    frequencies, times, Sxx = spectrogram(vid_path)
    Sxx = Sxx.transpose()

    vname = get_file_id(vid_path)

    frameidx = [int(p[-8:-4]) for p in os.listdir(face_path)]
    if len(frameidx) == 0:
        return

    currentseq = []
    counter = 0
    for i, vec in enumerate(Sxx):
        fname = f'{vname}-{i:04d}.jpg'

        fpath = os.path.join(face_path, fname)
        if not os.path.exists(fpath):
            if len(currentseq) != 0:
                currentseq = np.stack(currentseq)
                currentseq = torch.FloatTensor(currentseq)
                outfile = os.path.join(outpath, vname)
                torch.save(
                    currentseq, f'{outfile}-{counter:03d}-{currentseq.shape[0]}.pt')
                counter += 1
                currentseq = []
            continue

        if len(currentseq) == 24:
            currentseq = np.stack(currentseq)
            currentseq = torch.FloatTensor(currentseq)
            outfile = os.path.join(outpath, vname)
            torch.save(currentseq,
                       f'{outfile}-{counter:03d}-{currentseq.shape[0]}.pt')
            counter += 1
            currentseq = []

        currentseq.append(vec)


if __name__ == '__main__':
    main()
