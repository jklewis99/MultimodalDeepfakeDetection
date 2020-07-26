import librosa
import soundfile as sf
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

    fakelist = sorted([os.path.join(input_dir, 'fake', p)
                       for p in os.listdir(os.path.join(input_dir, 'fake'))])

    reallist = sorted([os.path.join(input_dir, 'real', p)
                       for p in os.listdir(os.path.join(input_dir, 'real'))])

    face_fakelist = sorted([os.path.join(faces_dir, 'fake', p)
                            for p in os.listdir(os.path.join(faces_dir, 'fake'))])

    face_reallist = sorted([os.path.join(faces_dir, 'real', p)
                            for p in os.listdir(os.path.join(faces_dir, 'real'))])

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


def load_audio(path):
    videoclip = VideoFileClip(video_file)
    audio = videoclip.audio
    # if audio is None:

    audio = audio.set_fps(16000).to_soundarray()
    return sound


def parse_audio(audio_path, normalize=True):
    sample_rate = 16000  # The sample rate for the data/model features
    window_size = .02  # Window size for spectrogram generation (seconds)
    window_stride = .01  # Window stride for spectrogram generation (seconds)

    y = load_audio(audio_path)

    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=self.window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)

    return spect


def spectrogram(video_file, normalize=True, sample_rate=44_100):
    """Generates spectrogram of audio content in a video file 

    Args:
        video_file (String): path to video file

    Returns:
        [type]: [description]
    """
    cap = cv2.VideoCapture(video_file)
    framecount = int(cv2.VideoCapture.get(cap, CV2_FRAMECOUNT_ID))
    fps = cv2.VideoCapture.get(cap, CV2_FPS_ID)
    cap.release()

    videoclip = VideoFileClip(video_file)
    audio = videoclip.audio
    if audio is None:
        Sxx = np.zeros((framecount, 533))
    else:
        audio = audio.set_fps(sample_rate).to_soundarray()
        if len(audio.shape) > 1:
            if audio.shape[1] == 1:
                audio = audio.squeeze()
            else:
                audio = audio.mean(axis=1)  # multiple channels, average

        frequencies, times, Sxx = signal.spectrogram(
            audio, fs=sample_rate, nperseg=int(sample_rate/fps), noverlap=0)
        Sxx = 10 * np.log10(Sxx + np.finfo(float).eps)

    if normalize:
        mean = Sxx.mean()
        std = Sxx.std()
        Sxx = Sxx - mean
        Sxx = Sxx / std
    return Sxx


def feats_from_vid(vid_path, face_path, outpath):
    # Sxx = parse_audio(vid_path)
    Sxx = spectrogram(vid_path)
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
