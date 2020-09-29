#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
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
dct_path = '/home/itdfh/data/dfdc-subset/train-6-dct-all'
spc_path = '/home/itdfh/data/dfdc-subset/train-6-spectrograms'


# ## Load data

# ### Listing files

# In[2]:


def tensor_file_lists(dct_path, spc_path, max_files=None, perc=.9):

    dct_files_train, spc_files_train = [], []
    dct_files_val,   spc_files_val = [], []

    for label in ['real', 'fake']:
        train_files = []

        val_files = []

        all_files = os.listdir(os.path.join(dct_path, label))

        for i, p in enumerate(all_files):
            base_dir = os.path.join(label, p)
            full_base_dir = os.path.join(dct_path, base_dir)
            if i < len(all_files) * .9:
                train_files.extend([os.path.join(base_dir, p)
                                    for p in os.listdir(full_base_dir)])
            else:
                val_files.extend([os.path.join(base_dir, p)
                                  for p in os.listdir(full_base_dir)])

        dct_files_train.extend([(os.path.join(dct_path, p[:-6]+'30.npy'), labelmap[label])
                                for p in train_files if p[-6:] == '30.npy'])
        spc_files_train.extend([(os.path.join(spc_path, p[:-6]+'30.pt'), labelmap[label])
                                for p in train_files if p[-6:] == '30.npy'])

        dct_files_val.extend([(os.path.join(dct_path, p[:-6]+'30.npy'), labelmap[label])
                              for p in val_files if p[-6:] == '30.npy'])
        spc_files_val.extend([(os.path.join(spc_path, p[:-6]+'30.pt'), labelmap[label])
                              for p in val_files if p[-6:] == '30.npy'])

    return dct_files_train, spc_files_train, dct_files_val, spc_files_val


# %%
dct_files_train, spc_files_train, dct_files_val, spc_files_val = tensor_file_lists(
    dct_path, spc_path)


# In[3]:


spc_files_train[0], dct_files_train[0]


# In[4]:


ls / home/itdfh/data/dfdc-subset/train-6-spectrograms/real/uoptulivph/


# ### Checking match

# In[5]:


a = {
    'dct_exist': 0,
    'dct_not_exist': 0,
    'spc_exist': 0,
    'spc_not_exist': 0
}
for (dct_f, _), (spc_f, _) in tqdm(zip(dct_files_train, spc_files_train), total=len(spc_files_train)):
    if not os.path.exists(dct_f):
        a['dct_not_exist'] += 1
    else:
        a['dct_exist'] += 1
    if not os.path.exists(spc_f):
        a['spc_not_exist'] += 1
    else:
        a['spc_exist'] += 1

a


# ### Keeping matches

# In[6]:


clean_spc_files_train = [spc_files_train[i] for i, (f, label) in enumerate(
    spc_files_train) if os.path.exists(f)]
clean_dct_files_train = [dct_files_train[i] for i, (f, label) in enumerate(
    spc_files_train) if os.path.exists(f)]
spc_files_train = clean_spc_files_train
dct_files_train = clean_dct_files_train


# In[7]:


a = {
    'dct_exist': 0,
    'dct_not_exist': 0,
    'spc_exist': 0,
    'spc_not_exist': 0
}
for (dct_f, _), (spc_f, _) in tqdm(zip(dct_files_train, spc_files_train), total=len(spc_files_train)):
    if not os.path.exists(dct_f):
        a['dct_not_exist'] += 1
    else:
        a['dct_exist'] += 1
    if not os.path.exists(spc_f):
        a['spc_not_exist'] += 1
    else:
        a['spc_exist'] += 1

a


# ### `FourierDataset`

# In[8]:


class FourierDataset(Dataset):
    '''
    LipSpeech data set for concatenating lntionNet Features and dstrogram features
    '''

    def __init__(self, dct_files, spc_files, max_spc_size=700):
        """
        Args:
            DeepSpeech (string): Path to the csv file with annotations.
            LipNet (string): Directory with all the images.
        """
        self.max_spc_size = max_spc_size
        self.dct_files, self.spc_files = dct_files, spc_files

    def __len__(self):
        return len(self.dct_files)

    def __getitem__(self, idx):
        dctf, label = self.dct_files[idx]
        spcf, label = self.spc_files[idx]

        dct_feats = np.load(dctf)
        dct_feats = torch.tensor(dct_feats)

        specs = torch.load(spcf, map_location=torch.device('cpu'))[
            :, :self.max_spc_size]
        spc_feats = torch.zeros((specs.shape[0], self.max_spc_size))
        spc_feats[:, :specs.shape[-1]] = specs

        fourier_feats = torch.cat([dct_feats.float(), spc_feats], dim=1)

        label = torch.tensor(label).long()

        return fourier_feats, label


# %%
trainset = FourierDataset(dct_files_train, spc_files_train)
valset = FourierDataset(dct_files_val, spc_files_val)


# In[12]:


class FourierNet(nn.Module):
    def __init__(self, feature_size, num_layers=2, num_hidden_nodes=512, device='cuda'):
        super(FourierNet, self).__init__()
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
        print(x.device, hidden[0].device)
        y, hidden = self.lstm(x, hidden)    # returns the two outputs
        y = y[:, -1, :]  # get only the last output
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = F.softmax(y, dim=1)

        return y, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.num_hidden_nodes).zero_().to(self.device),
                  weight.new(self.num_layers, batch_size, self.num_hidden_nodes).zero_().to(self.device))
        return hidden


# In[15]:


model = FourierNet(1465)


# In[16]:


model


# In[17]:


def train(model, trainset, loss_function, optimizer, epochs=1000, batch_size=50, device='cuda'):
    model = model.to(device)
    trainloader = DataLoader(trainset, shuffle=True,
                             batch_size=batch_size, drop_last=True)

    hidden = model.init_hidden(batch_size)
    for h in hidden:
        h = h.to(device)

    print_every = 12
    i = 0
    losses = []
    accs = []
    running_loss = 0.0
    running_acc = 0.0
    # again, normally you would NOT do 100 epochs, it is toy data
    for epoch in range(epochs):
        for inp, labels in trainloader:  # renamed sequence to inp because inp is a batch of sequences
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            print(inp.shape, labels.shape)
            model.zero_grad()

            inp = inp.float().to(device)
            labels = labels.to(device)

            print(inp.device, labels.device,
                  hidden[0].device, hidden[0].device)
            # Step 2. Run our forward pass.
            tag_scores, h = model(inp, hidden)

            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(tag_scores, labels)
            loss.backward()
            optimizer.step()

            running_acc += torch.mean((tag_scores.argmax(dim=1)
                                       == labels).float()).item()

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every-1:
                print('[%d, %5d] loss: %.3f - acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_every, running_acc * 100 / print_every))

                losses.append(running_loss / print_every)
                accs.append(running_acc * 100 / print_every)

                running_loss = 0.0
                running_acc = 0.0
            i += 1
    return losses, accs


# In[18]:

model = model.cuda()
loss_function = nn.NLLLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1)
losses, accs = train(model, trainset, loss_function,
                     optimizer, epochs=1000, batch_size=100)


# In[ ]:
