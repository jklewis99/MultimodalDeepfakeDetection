import argparse
import os
import torch
from Data.frimagenet_dataset import FrimagenetDataset

parser = argparse.ArgumentParser(description='FrimageNet Demo File')
parser.add_argument('spectrogram', help="place with spectrogram features")
parser.add_argument('xception', help="place with xception features")
# r'J:\reu\MultimodalDeepfakeDetection\output\spectrogram_features'
# r'J:\reu\MultimodalDeepfakeDetection\output\xception_features'
# J:\reu\MultimodalDeepfakeDetection\output\spectrogram_features J:\reu\MultimodalDeepfakeDetection\output\xception_features

def main():
    args = parser.parse_args()
    data = FrimagenetDataset(args.spectrogram, args.xception)
    count_real = 0
    for i, sample in enumerate(data):
        if sample[1].tolist() == 1:
            count_real += 1
        print(sample)
    print('Real', count_real)
    print('Total', len(data))
    
if __name__ == '__main__':
    main()