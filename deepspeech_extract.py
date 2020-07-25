import argparse
import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import torch
from DeepSpeech2.deepspeech_pytorch.config import SpectConfig
from DeepSpeech2.deepspeech_pytorch.model import DeepSpeech
from DeepSpeech2.deepspeech_pytorch.utils import load_decoder
from moviepy.editor import VideoFileClip
from scipy import signal
from Utils.audio import parse_audio_spect
from Utils.misc import create_directory

parser = argparse.ArgumentParser(description='DeepSpeech2 Feature Extraction')
parser.add_argument('input', help="Folder containing fake and real folders, inside those folders \
                                   all of the video files are stored")
parser.add_argument('output', help="Saves DeepSpeech2 tensors to this file_path. It will create a fake \
                                    and real folder, in which the video label folder will be \
                                    created containing the tensors")
parser.add_argument('-device', default='cuda', help="device for model (default cuda)")
parser.add_argument('-pretrained', 
                    default='DeepSpeech2/pretrained_weights/librispeech_pretrained_v2.pth',
                    help='path to pretrained weights for DeepSpeech2 model')

def get_deepspeech2_features(model, input_path, save_path, video_id, 
                             window_stride = .01,
                             window_size = .02,
                             sample_rate = 16000,
                             window = 'hamming'):
    spect = parse_audio_spect(input_path, window_stride, window_size, sample_rate, window)
    tspect = spect.transpose()
    spect = torch.cuda.FloatTensor(spect).unsqueeze(0).unsqueeze(0)
    print(spect.shape)
    # torch.save(spect, f'{save_path}/{video_id}-{i:03d}-{spect.shape[1]}.pt')
    length = torch.IntTensor([spect.size(3)]).int().to("cuda")
    out, out_size, features = model(spect, length)
    print(tspect)
    return out, out_size, features

def main():
    args = parser.parse_args()
    model = DeepSpeech.load_model(args.pretrained)
    model.to(args.device)

    window_stride = .01
    window_size = .02
    sample_rate = 16000
    window = 'hamming' #SpectConfig.window.value

    decoder = 'greedy'
    lm_path = None          # Path to an (optional) kenlm language model for use with beam search
    alpha = 0.8             # Language model weight
    beta = 1                # Language model word bonus (all words)
    cutoff_top_n = 40       # Cutoff number in pruning, only top cutoff_top_n characters with highest probs in 
                            # vocabulary will be used in beam search
    cutoff_prob = 1.0       # Cutoff probability in pruning,default 1.0, no pruning.
    beam_width = 10         # Beam width to use
    lm_workers = 1          # Number of LM processes to use

    decoder = load_decoder(decoder_type=decoder,
                            labels=model.labels,
                            lm_path=lm_path,
                            alpha=alpha,
                            beta=beta,
                            cutoff_top_n=cutoff_top_n,
                            cutoff_prob=cutoff_prob,
                            beam_width=beam_width,
                            lm_workers=lm_workers)

    subfolders = ['real', 'fake']
    for subfolder in subfolders:
        file_list = [video_label for video_label in os.listdir(os.path.join(args.input, subfolder))]
        for video_id in file_list:
            input_path = os.path.join(args.input, subfolder, video_id)
            output_path = create_directory(os.path.join(args.output, subfolder, video_id))
            out, out_size, features = get_deepspeech2_features(model, input_path, output_path, video_id)

            print(features.shape)
            decoded_output = decoder.decode(out, out_size)
            print(decoded_output)

if __name__ == '__main__':
    main()
