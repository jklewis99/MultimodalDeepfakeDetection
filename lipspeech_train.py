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

writer = SummaryWriter('runs/lipspeech')

# %%
labelmap = {'real': 0, 'fake': 1}


# %%
ds_path = '/home/itdfh/data/dfdc-subset/train-6-deepspeech'
ln_path = '/home/itdfh/data/dfdc-subset/train-6-lipnet'


# %%
def tensor_file_lists(ds_path, ln_path, max_files=None, perc=.9):

    ds_files_train, ln_files_train = [], []
    ds_files_val,   ln_files_val = [], []

    for label in ['real', 'fake']:
        train_files = []

        val_files = []

        all_files = os.listdir(os.path.join(ds_path, label))

        for i, p in enumerate(all_files):
            base_dir = os.path.join(label, p)
            full_base_dir = os.path.join(ds_path, base_dir)
            if i < len(all_files) * .9:
                train_files.extend([os.path.join(base_dir, p)
                                    for p in os.listdir(full_base_dir)])
            else:
                val_files.extend([os.path.join(base_dir, p)
                                  for p in os.listdir(full_base_dir)])

        ds_files_train.extend([(os.path.join(ds_path, p[:-5]+'50.pt'), labelmap[label])
                               for p in train_files if p[-5:] == '50.pt'])
        ln_files_train.extend([(os.path.join(ln_path, p[:-5]+'30.pt'), labelmap[label])
                               for p in train_files if p[-5:] == '50.pt'])

        ds_files_val.extend([(os.path.join(ds_path, p[:-5]+'50.pt'), labelmap[label])
                             for p in val_files if p[-5:] == '50.pt'])
        ln_files_val.extend([(os.path.join(ln_path, p[:-5]+'30.pt'), labelmap[label])
                             for p in val_files if p[-5:] == '50.pt'])

    return ds_files_train, ln_files_train, ds_files_val, ln_files_val


# %%
ds_files_train, ln_files_train, ds_files_val, ln_files_val = tensor_file_lists(
    ds_path, ln_path)

clean_ln_files_train = [ln_files_train[i] for i, (f, label) in enumerate(
    ln_files_train) if os.path.exists(f)]
clean_ds_files_train = [ds_files_train[i] for i, (f, label) in enumerate(
    ln_files_train) if os.path.exists(f)]
clean_ln_files_val = [ln_files_val[i]
                      for i, (f, label) in enumerate(ln_files_val) if os.path.exists(f)]
clean_ds_files_val = [ds_files_val[i]
                      for i, (f, label) in enumerate(ln_files_val) if os.path.exists(f)]
ln_files_train = clean_ln_files_train
ds_files_train = clean_ds_files_train
ln_files_val = clean_ln_files_val
ds_files_val = clean_ds_files_val

# %%


class LipSpeechDataset(Dataset):
    '''
    LipSpeech data set for concatenating lntionNet Features and dstrogram features
    '''

    def __init__(self, ds_files, ln_files, seq_size=24, max_ds_size=700):
        """
        Args:
            DeepSpeech (string): Path to the csv file with annotations.
            LipNet (string): Directory with all the images.
        """
        self.max_ds_size = max_ds_size
        self.seq_size = seq_size

        self.ds_files, self.ln_files = ds_files, ln_files

    def __len__(self):
        return len(self.ds_files)

    def __getitem__(self, idx):
        dsf, label = self.ds_files[idx]
        lnf, label = self.ln_files[idx]

        ds_feats = torch.load(dsf).transpose(0, 1)
        ln_feats = torch.load(lnf).transpose(0, 1)

        label = torch.tensor(label).long()

        return ds_feats, ln_feats, label


# %%
trainset = LipSpeechDataset(ds_files_train, ln_files_train, max_ds_size=0)
valset = LipSpeechDataset(ds_files_val, ln_files_val, max_ds_size=0)


class LSTMFC(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=1024, num_layers=2, device='cuda'):
        super(LSTMFC, self).__init__()

        self.device = device

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, num_layers=num_layers)
        # fully connected
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x, hidden):
        y, hidden = self.lstm(x, hidden)
        y = y[:, -1, :]
        y = self.fc(y)
        y = self.act(y)

        return y, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden


class LSTMFC(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=1024, num_layers=2, device='cuda'):
        super(LSTMFC, self).__init__()

        self.device = device

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, num_layers=num_layers)
        # fully connected
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x, hidden):
        y, hidden = self.lstm(x, hidden)
        print(y.shape)
        y = y[:, -1, :]
        y = self.fc(y)
        y = self.act(y)

        return y, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden


class LipSpeechNet(nn.Module):
    def __init__(self, out_dim=2, hidden_dim=1024, device='cuda'):
        super(LipSpeechNet, self).__init__()

        self.device = device

        self.lstmfc_ds = LSTMFC(
            1024, hidden_dim, hidden_dim, device=self.device)
        self.lstmfc_ln = LSTMFC(
            512, hidden_dim, hidden_dim, device=self.device)

        # fully connected
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self, batch_size):
        return self.lstmfc_ds.init_hidden(batch_size), self.lstmfc_ln.init_hidden(batch_size)

    def forward(self, x_ds, x_ln, hidden_ds, hidden_ln):
        y_ds, hidden_ds = self.lstmfc_ds(x_ds, hidden_ds)
        y_ln, hidden_ln = self.lstmfc_ln(x_ln, hidden_ln)
        y = torch.cat((y_ds, y_ln), 1)

        y = self.fc1(y)
        y = self.act(y)

        y = self.fc2(y)
        y = self.softmax(y)

        return y, hidden_ds, hidden_ln


model = LipSpeechNet().cuda()


def train(model, trainset, loss_function, optimizer, valset=None, epochs=1000, batch_size=50, device='cuda'):
    global writer
    epsilon = 1e-6

    model = model.to(device)
    trainloader = DataLoader(trainset, shuffle=True,
                             batch_size=batch_size, drop_last=True)

    if valset is not None:
        valloader = DataLoader(valset, shuffle=True,
                               batch_size=batch_size, drop_last=True)

    print_every = 5
    i = 0
    losses = []
    accs = []

    vaccs = []
    vlosses = []

    running_loss = 0.0
    running_acc = 0.0

    for epoch in range(epochs):
        for x_ds, x_ln, labels in trainloader:
            optimizer.zero_grad()

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            x_ds = x_ds[:, 0].float().to(device)
            x_ln = x_ln[:, 0].float().to(device)

            labels = labels.to(device)

            hidden_ds, hidden_ln = model.init_hidden(batch_size=batch_size)

            # Step 2. Run our forward pass.
            out, hidden_ds, hidden_ln = model(x_ds, x_ln, hidden_ds, hidden_ln)
            out = out.add(epsilon)

            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(out, labels)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            loss.backward()
            optimizer.step()

            running_acc += torch.mean((out.argmax(dim=1)
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
                for x_ds, x_ln, labels in valloader:
                    x_ds = x_ds[:, 0].float().to(device)
                    x_ln = x_ln[:, 0].float().to(device)
                    labels = labels.to(device)

                    out, hidden_ds, hidden_ln = model(
                        x_ds, x_ln, hidden_ds, hidden_ln)
                    loss = loss_function(out, labels)

                    val_accs.append(torch.mean((out.argmax(dim=1)
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
optimizer = optim.Adam(model.parameters(), lr=1e-5)
train(model, trainset, loss_function, optimizer,
      epochs=1000, batch_size=10, valset=valset)
