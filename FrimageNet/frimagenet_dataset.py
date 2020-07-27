import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# from Utils.errors import *
from torch.utils.data import Dataset

class FrimagenetDataset(Dataset):
    '''
    FrimageNet data set for concatenating XceptionNet Features and Spectrogram features
    '''
    def __init__(self, spectrogram_folder, xception_features_folder):
        """
        Args:
            spectrogram_folder (string): Path to the csv file with annotations.
            xception_features_folder (string): Directory with all the images.
        """
        self.classification = []
        self.encode_map = {
            'real': 1,
            'fake': 0
        }
        self.features = self.__get_feats(spectrogram_folder, xception_features_folder)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.classification[idx]

    def __get_feats(self, spect_directory, xception_directory, seq_size=24, nfirst=25, max_spect_feats=700, max_xcept_feats=2048):
        samples = []
        labels = ['real', 'fake']
        for label in labels:
            xception_vidpaths = sorted([os.path.join(xception_directory, label, vid) for vid in os.listdir(os.path.join(xception_directory, label))])
            spect_vidpaths = sorted([os.path.join(spect_directory, label, vid) for vid in os.listdir(os.path.join(spect_directory, label))])

            for xcept_path, spect_path in zip(xception_vidpaths, spect_vidpaths):
                # loops through the paths to the video labels of xception features and spectrogram features folders
                sorted_vid_xcept = sorted(os.listdir(xcept_path))
                sorted_vid_spect = sorted(os.listdir(spect_path))

                for xcept_feat, spect_feat in zip(sorted_vid_xcept, sorted_vid_spect):
                    # loops throught the individual files in each respective video_id folder for the xception features and spectrogram features
                    if (xcept_feat != spect_feat):
                        # the labels are not identical, so alignment is off. Return error
                        print(f'{xcept_feat} != {spect_feat} ')
                        # raise NonAligned

                    if xcept_feat[-5:] == f'{seq_size}.pt':
                        xcept = torch.load(os.path.join(xcept_path, xcept_feat))[:, :max_xcept_feats]
                        spect = torch.load(os.path.join(spect_path, spect_feat))[:, :max_spect_feats]
                        samples.append(torch.cat((xcept, spect), dim=-1))
                        self.classification.append(torch.tensor(self.encode_map[label]))
        self.classification = torch.stack(self.classification)
        return torch.stack(samples)