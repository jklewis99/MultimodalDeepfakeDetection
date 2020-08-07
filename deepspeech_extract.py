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
import time

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

def save_deepspeech2_features(model, input_path, save_path, video_id, 
                             window_stride = .01,
                             window_size = .02,
                             sample_rate = 16000,
                             window = 'hamming'):
    spect = parse_audio_spect(input_path, window_stride, window_size, sample_rate, window)
    spect = spect.transpose()
    input_size = torch.IntTensor([100]).int().to('cuda') # 1 sec input (100 data points)

    # list_outputs = []      
    # every 100 data points on the spectrogram is 1 sec of audio
    for sec in range(len(spect) // 100):
        one_sec_spect = spect[sec * 100 : min((sec+1) * 100, len(spect))]
        one_sec_spect = torch.FloatTensor(one_sec_spect.transpose()).unsqueeze(0).unsqueeze(0).to('cuda')
        out, out_size, features = model(one_sec_spect, input_size)
        # list_outputs.append((out, out_size))
        torch.save(features, f'{save_path}/{video_id[:-4]}-{sec:03d}-{features.shape[0]}.pt')
    
    # verify_output(list_outputs, model)

    return out, out_size, features

def verify_output(list_outputs, model):
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
    
    for i, (out, out_size) in enumerate(list_outputs):
        decoded_output = decoder.decode(out, out_size)
        print(decoded_output[0])
    
def main():
    start = time.time()
    args = parser.parse_args()
    model = DeepSpeech.load_model(args.pretrained)
    model.to(args.device)

    window_stride = .01
    window_size = .02
    sample_rate = 16000
    window = 'hamming' #SpectConfig.window.value

    subfolders = ['real', 'fake']
    for subfolder in subfolders:
        file_list = [video_label for video_label in os.listdir(os.path.join(args.input, subfolder))]
        for video_id in file_list:
            input_path = os.path.join(args.input, subfolder, video_id)
            output_path = create_directory(os.path.join(args.output, subfolder, video_id))
            save_deepspeech2_features(model, input_path, output_path, video_id)

    print(time.time()-start)

if __name__ == '__main__':
    main()
