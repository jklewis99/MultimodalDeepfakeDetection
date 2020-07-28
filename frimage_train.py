# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# from Utils.errors import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/frimage')

# %%
labelmap = {'real': 0, 'fake': 1}


# %%
spec_path = '/home/itdfh/data/dfdc-subset/train_spectrograms_part-5'
xcep_path = '/home/itdfh/data/dfdc-subset/train_xception_part-5'


# %%
def tensor_file_lists(spec_path, xcep_path, max_files=None, perc=.9):
    spec_files_train, xcep_files_train = [], []
    spec_files_val, xcep_files_val = [], []

    for label in ['real', 'fake']:
        train_files = []
        val_files = []

        all_files = os.listdir(os.path.join(spec_path, label))

        for i, p in enumerate(all_files):
            base_dir = os.path.join(label, p)
            full_base_dir = os.path.join(spec_path, base_dir)
            if i < len(all_files) * .9:
                train_files.extend([os.path.join(base_dir, p)
                                    for p in os.listdir(full_base_dir)])
            else:
                val_files.extend([os.path.join(base_dir, p)
                                  for p in os.listdir(full_base_dir)])

        spec_files_train.extend([(os.path.join(spec_path, p), labelmap[label])
                                 for p in train_files if p[-5:] == '24.pt'])
        xcep_files_train.extend([(os.path.join(xcep_path, p), labelmap[label])
                                 for p in train_files if p[-5:] == '24.pt'])

        spec_files_val.extend([(os.path.join(spec_path, p), labelmap[label])
                               for p in val_files if p[-5:] == '24.pt'])
        xcep_files_val.extend([(os.path.join(xcep_path, p), labelmap[label])
                               for p in val_files if p[-5:] == '24.pt'])

    return spec_files_train, xcep_files_train, spec_files_val, xcep_files_val


# %%
spec_files_train, xcep_files_train, spec_files_val, xcep_files_val = tensor_file_lists(
    spec_path, xcep_path)


# %%
class FrimagenetDataset(Dataset):
    '''
    FrimageNet data set for concatenating XceptionNet Features and Spectrogram features
    '''

    def __init__(self, spec_files, xcep_files, seq_size=24, max_spec_size=700):
        """
        Args:
            spectrogram_folder (string): Path to the csv file with annotations.
            xception_features_folder (string): Directory with all the images.
        """
        self.max_spec_size = max_spec_size
        self.seq_size = seq_size

        self.spec_files, self.xcep_files = spec_files, xcep_files

    def __len__(self):
        return len(self.spec_files)

    def __getitem__(self, idx):
        sf, label = self.spec_files[idx]
        xf, label = self.xcep_files[idx]

        # loading spec_feats with 0 padding
        spec_feats = torch.zeros((self.seq_size, self.max_spec_size))
        specs = torch.load(sf, map_location=torch.device('cpu'))[
            :, :self.max_spec_size]
        spec_feats[:, :specs.shape[-1]] = specs

        xcep_feats = torch.load(xf, map_location=torch.device('cpu'))
        x = torch.cat((xcep_feats, spec_feats), dim=-1)
        label = torch.tensor(label).long()
        return x, label


# %%
trainset = FrimagenetDataset(spec_files_train, xcep_files_train)
valset = FrimagenetDataset(spec_files_val, xcep_files_val)


class FrimageNet(nn.Module):
    def __init__(self, feature_size, num_layers=2, num_hidden_nodes=1024, device='cuda'):
        super(FrimageNet, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.num_hidden_nodes = num_hidden_nodes

        # input dim is 167, output 200
        self.lstm = nn.LSTM(feature_size, num_hidden_nodes,
                            batch_first=True, num_layers=num_layers)
        # fully connected
        self.fc1 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden_nodes, 2)
        self.softmax = nn.Softmax()

    def forward(self, x, hidden):
        y, hidden = self.lstm(x, hidden)
        y = self.fc1(y)
        y = y[:, -1, :]
        y = self.act(y)
        y = self.fc2(y)
        y = F.log_softmax(y, dim=1)
        return y, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.num_hidden_nodes).zero_().to(self.device),
                  weight.new(self.num_layers, batch_size, self.num_hidden_nodes).zero_().to(self.device))
        return hidden


model = FrimageNet(2748)


def train(model, trainset, loss_function, optimizer, valset=None, epochs=1000, batch_size=50, device='cuda'):
    global writer
    epsilon = 1e-6

    model = model.to(device)
    trainloader = DataLoader(trainset, shuffle=True,
                             batch_size=batch_size, drop_last=True)

    if valset is not None:
        valloader = DataLoader(valset, shuffle=True,
                               batch_size=batch_size, drop_last=True)

    hidden = model.init_hidden(batch_size)
    for h in hidden:
        h = h.to(device)

    print_every = 5
    i = 0
    losses = []
    accs = []

    vaccs = []
    vlosses = []

    running_loss = 0.0
    running_acc = 0.0
    # again, normally you would NOT do 100 epochs, it is toy data
    for epoch in range(epochs):
        for inp, labels in trainloader:  # renamed sequence to inp because inp is a batch of sequences
            optimizer.zero_grad()
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            inp = inp.float().to(device)
            labels = labels.to(device)

            # Step 2. Run our forward pass.
            tag_scores, h = model(inp, hidden)
            tag_scores = tag_scores.add(epsilon)
            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(tag_scores, labels)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            loss.backward()
            optimizer.step()

            running_acc += torch.mean((tag_scores.argmax(dim=1)
                                       == labels).float()).item()

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every-1:
                print('[%d, %5d] loss: %.3f - acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_every, running_acc * 100 / print_every))

                writer.add_scalar('train/loss', running_loss / print_every, i)
                writer.add_scalar('train/acc', running_acc *
                                  100 / print_every, i)

                losses.append(running_loss / print_every)
                accs.append(running_acc * 100 / print_every)

                running_loss = 0.0
                running_acc = 0.0
            i += 1

        if valset is not None:
            with torch.no_grad():
                val_accs, val_losses = [], []
                for inp, labels in valloader:
                    inp = inp.float().to(device)
                    labels = labels.to(device)

                    tag_scores, h = model(inp, hidden)
                    loss = loss_function(tag_scores, labels)

                    val_accs.append(torch.mean((tag_scores.argmax(dim=1)
                                                == labels).float()).item())
                    val_losses.append(loss)

                val_accs = torch.mean(torch.tensor(val_accs))
                val_losses = torch.mean(torch.tensor(val_losses))

                writer.add_scalar('val/loss', val_accs * 100, epoch)
                writer.add_scalar('val/acc', val_losses, epoch)

                vaccs.append(val_accs)
                vlosses.append(val_losses)

    return losses, accs, vlosses, vaccs


loss_function = nn.NLLLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
losses, accs, vlosses, vaccs = train(model, trainset, loss_function,
                                     optimizer, epochs=100, batch_size=200)
