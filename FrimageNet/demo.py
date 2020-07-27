import argparse
import os
import torch.optim as optim
from frimagenet_dataset import FrimagenetDataset
from frimagenet import FrimageNet
from torch import nn
from training import train

parser = argparse.ArgumentParser(description='FrimageNet Demo File')
parser.add_argument('spectrogram', help="place with spectrogram features")
parser.add_argument('xception', help="place with xception features")
# J:\reu\code\output\spectrogram_features J:\reu\code\output\xception_features

def main():
    args = parser.parse_args()
    device = 'cuda'
    model = FrimageNet(2748).to(device)
    # data = FrimagenetDataset(args.spectrogram, args.xception)
    # count_real = 0
    # for i, sample in enumerate(data):
    #     if sample[1].tolist() == 1:
    #         count_real += 1
    #     print(sample)
    # print('Real', count_real)
    # print('Total', len(data))
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, args.spectrogram, args.xception, loss_function, optimizer)
    
if __name__ == '__main__':
    main()