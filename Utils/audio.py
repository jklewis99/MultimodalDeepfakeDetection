import librosa
import numpy as np
from moviepy.editor import VideoFileClip

def load_audio(video_path, fps=16000):
    '''
    takes a path to a video file and returns the monophonic audio sample
    Args:
        video_path: realtive or absolute path to a video file
    '''
    videoclip = VideoFileClip(video_path)
    # TODO: if audio does not exist, add an empty AudioFileClip object and set
    # video to that empty audio
    # if videoclip.audio is None:
    #     videoclip.set_audio
    audio = videoclip.audio.set_fps(fps).to_soundarray()
    if len(audio.shape) > 1:
        if audio.shape[1] == 1:
            audio = audio.squeeze()
        else:
            audio = audio.mean(axis=1)  # multiple channels, average
    return audio

def parse_audio_spect(video_path, win_stride, win_size, sample_rate, window):
    '''
    takes a path to a video file and returns the spectrogram of that video's \
    audio sample and returns the log of that spectrogram
    
    Args:
        win_stride = size of the stride between audio samples
        win_size = size of the window between audio samples
        sample_rate = intended sample rate of audio
        window = window string value to define librosa's window type
    '''

    sound_array = load_audio(video_path, fps=sample_rate)

    n_fft = int(sample_rate * win_size)
    win_length = n_fft
    hop_length = int(sample_rate * win_stride)
    # STFT
    D = librosa.stft(sound_array, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    return spect