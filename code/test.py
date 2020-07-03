# %% 
import pickle
import torch 
from torch.utils.data import Dataset, DataLoader

with open("train_data.txt", "rb") as fp:   # Unpickling
    training_data = pickle.load(fp)
label, data, name = training_data[0]
print(data.shape)
print(len(training_data))
print(label)

class FourierDataset(Dataset):
    def __init__(self, data):
        """
        data: a list of (label: string, fourier_data: numpy array, name: string)
    
        """
        self.data = []
        for elt in data:
            label, spects, name = elt
            label = torch.tensor([0, 1] if label=='FAKE' else [1, 0])

            # Moving window sequence generation without overalap 
            # other ideas: 1. Random sampling, 2. Moving qindow with overlap
            # this data will be shuffled
            for i in range(0, 24 * (spects.shape[0] // 24), 24):
                spect = torch.tensor(spects[i:i+24, :])
                self.data.append((spect, label))

    
    def __getitem__(self, idx):
        return self.data[idx] # spect (24, 167), label (2)

    def __len__(self):
        return len(self.data)

dataset = FourierDataset(training_data)

print(len(dataset))

dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

for data in dataloader:
    inp, label = data
    print(inp.shape, label.shape)
    exit()
