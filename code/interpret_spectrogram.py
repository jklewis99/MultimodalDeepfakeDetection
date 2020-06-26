import os
from matplotlib import pyplot as plt
import numpy as np

path = '../output/video_0'

spectrograms_list = [file for file in os.listdir(path) if file.endswith('.npy')]


for spectrogram in spectrograms_list:
    spectrogram_np = np.load(os.path.join(path, spectrogram))

    # may not be constant, so we may need an object for sample rate and audio
    sample_rate = 44100
    
    # one method of getting the necessary data, but there are better ones
    NFFT = sample_rate / 25

    fig, ax = plt.subplots(figsize=(14, 4))
    spectrum, freqs, time, im = ax.specgram(spectrogram_np.mean(axis=1), 
                                                        NFFT=NFFT,
                                                        pad_to=4096, 
                                                        Fs=sample_rate, 
                                                        noverlap=512, 
                                                        mode='magnitude', 
                                                        cmap='inferno')
    print('spectrum: {}\nfreqs: {}\ntime: {}'.format(spectrum, freqs, time))
    fig.colorbar(im)
    plt.show()

