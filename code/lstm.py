import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import *
from torch.utils.data import Dataset, DataLoader
from blazeface import BlazeFace
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import pickle

DATA_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NET = BlazeFace().to(gpu)
NET.load_weights("../input/blazeface.pth")
NET.load_anchors("../input/anchors.npy")

class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(167, 200, batch_first=True) # input dim is 167, output 200
        self.fc1 = nn.Linear(200, 200)                  # fully connected
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(200, 2)   
        self.softmax = nn.Softmax() 

    def forward(self, x, hidden):
        y, hidden = self.lstm(x, hidden)    # returns the two outputs
        y = y[:, -1, :] # get only the last output
        y = self.fc1(y)
        y = self.fc2(y)
        y = F.softmax(y, dim=1)

        return y, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(1, batch_size, 200).zero_(),
                  weight.new(1, batch_size, 200).zero_())
        return hidden



class FourierDataset(Dataset):
    def __init__(self, data):
        """
        data: a list of (label: string, fourier_data: numpy array, name: string)
    
        """
        self.data = []
        for elt in data:
            label, spects, name = elt
            label = torch.tensor(0 if label=='FAKE' else 1)

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


sequence = 24 # 1 sec of video
feature_size = 167 # length of spatial frequency


def read_video(filename):
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    count = 0
    images = []
    
    while success:
        tiles, resize_info = stride_search(image)
        detections = NET.predict_on_image(tiles[1])
        blazeface_endpoints = get_face_endpoints(tiles[1], detections)[0] # take the first face only
        # we need to resize them on the original image and get the amount shifted to prevent negative values
        split_size = 128 * resize_info[1]               # in this case it will be 1080
        x_shift = (image.shape[1] - split_size) // 2    # determine how much we shifted for this tile
        face_endpoints = (int(blazeface_endpoints[0] * resize_info[0]), 
                          int(blazeface_endpoints[1] * resize_info[0] + x_shift), 
                          int(blazeface_endpoints[2] * resize_info[0]), 
                          int(blazeface_endpoints[3] * resize_info[0] + x_shift))
        # next we need to expand the rectangle to be 240, 240 pixels (for this training example)
        #   we can do this equally in each direction, kind of
        face_width = face_endpoints[3] - face_endpoints[1]
        face_height = face_endpoints[2] - face_endpoints[0]
        buffer = 20
        face_box = image[max(0, face_endpoints[0] - buffer) : min(face_endpoints[2] + buffer, image.shape[0]),
                         max(0, face_endpoints[1] - buffer) : min(face_endpoints[3] + buffer, image.shape[1])]
        # print(face_box.shape) # almost a square or very close to it 
        face = cv2.resize(face_box, (240, 240))
        images.append(face)
        # cv2.imshow("face", face)
        success, image = vidcap.read()
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if images:
        return np.stack(images)

def get_spects(vid):

    spects = []
    for i in range(vid.shape[0]):
        img = vid[i]
        spects.append(fourier_tranform(img, ''))

    return np.stack(spects)

def get_face_endpoints(img, detections, with_keypoints=False):
    if isinstance(detections, torch.Tensor):
        detections=detections.cpu().numpy()

    if detections.ndim == 1:
        detections=np.expand_dims(detections, axis=0)

    # print("Found %d face(s)" % detections.shape[0])
    # print('Face endpoints on 128x128 image:')
    detected_faces_endpoints = []

    for i in range(detections.shape[0]): # dependent on number of faces found
        ymin=detections[i, 0] * img.shape[0]
        xmin=detections[i, 1] * img.shape[1]
        ymax=detections[i, 2] * img.shape[0]
        xmax=detections[i, 3] * img.shape[1]

        detected_faces_endpoints.append((ymin, xmin, ymax, xmax))
        
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

        if with_keypoints:
            for k in range(6):
                kp_x=detections[i, 4 + k*2] * img.shape[1]
                kp_y=detections[i, 4 + k*2 + 1] * img.shape[0]
                circle=patches.Circle((kp_x, kp_y), radius = 0.5, linewidth = 1,
                                        edgecolor = "lightskyblue", facecolor = "none",
                                        alpha = detections[i, 16])
                # ax.add_patch(circle)

    return detected_faces_endpoints

def prepare_data():
    # Here we check the train data files extensions.
    train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
    ext_dict = []
    for file in train_list:
        file_ext = file.split('.')[1]
        if (file_ext not in ext_dict):
            ext_dict.append(file_ext)

    print(f"Extensions: {ext_dict}")

    # Let's count how many files with each extensions there are.
    for file_ext in ext_dict:
        print(
            f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")

    test_list = list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))
    ext_dict = []
    for file in test_list:
        file_ext = file.split('.')[1]
        if (file_ext not in ext_dict):
            ext_dict.append(file_ext)
    print(f"Extensions: {ext_dict}")
    for file_ext in ext_dict:
        print(
            f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")

    json_file = [file for file in train_list if file.endswith('json')][0]
    print(f"JSON file: {json_file}")

    meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER, json_file)
    meta_train_df.head()

    fake_train_sample_video = list(
        meta_train_df.loc[meta_train_df.label == 'FAKE'].sample(90).index)
    real_train_sample_video = list(
        meta_train_df.loc[meta_train_df.label == 'REAL'].index)

    training_data = []
    for video_file in fake_train_sample_video:
        try:
            data = process_video_data(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
            training_data.append(('FAKE', data, video_file))# (X, 24, 167)
        except:
            continue

    for video_file in real_train_sample_video:
        try:
            data = process_video_data(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
            training_data.append(('REAL', data, video_file))
        except:
            continue

    random.shuffle(training_data)

    with open('train_data.txt', 'wb') as fp:   # pickling
        pickle.dump(training_data, fp)

    return training_data

def read_data():
    with open("train_data.txt", "rb") as fp:   # Unpickling
        training_data = pickle.load(fp)
    return training_data

def process_video_data(video_file):
    stack = read_video(video_file)
    stack = stack.mean(axis=-1) / 255
    return get_spects(stack)

def prepare_spect(spect):
    return torch.tensor(spect)

def convert_scores(label):
    return torch.tensor([1, 0]) if label == 'FAKE' else torch.tensor([0, 1])

def train(training_data):
    batch_size = 69
    model = MyLSTM()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    training_data = FourierDataset(training_data)
    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    
    for data in trainloader:
        inp, label = data
        print(inp.shape, label.shape)

    hidden = model.init_hidden(batch_size)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        sequence, labels = next(iter(trainloader))
        tag_scores = model(sequence.float(), hidden)
        print(tag_scores)
    
    print_every = 10

    for epoch in range(100):  # again, normally you would NOT do 100 epochs, it is toy data
        
        running_loss = 0.0
        running_acc = 0.0
        i = 0
        for sequence, labels in trainloader:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            sequence = sequence.float()

            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn the azimuthal average
            # graph into Tensors

            # Step 3. Run our forward pass.
            tag_scores, h = model(sequence, hidden)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            # print(tag_scores, '\n\n\n', target, '\n\n')
            # print('tag', tag_scores, 'labels', labels)
            loss = loss_function(tag_scores, labels)
            loss.backward()
            optimizer.step()

            running_acc += (torch.sum((tag_scores.argmax(dim=1) == labels).float()).item() / 100)

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every-1:
                print('[%d, %5d] loss: %.3f - acc: %.3f' %
                    (epoch + 1, i + 1, running_loss / print_every, running_acc * 100 / print_every))
                running_loss = 0.0
                running_acc = 0.0
            i+=1

    # See what the scores are after training
    with torch.no_grad():
        sequence, labels = next(iter(trainloader))
        tag_scores = model(sequence, hidden)
        print(tag_scores)

    # print('finished training')
    # print('Testing on one of the things:')
    # print(len(training_data))
    # example = training_data[0]
    # print('Video: ', example[2], 'Label: ', example[0])
    # print('Our guess:', model(FourierDataset(example[1])[:], hidden))

def main():
    # prepare_data()
    '''
    stack = read_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, 'aagfhgtpmv.mp4'))
    print(stack.shape)
    stack = stack.mean(axis=-1) / 255
    spects = get_spects(stack)
    # print(spects.shape)


    print(spects[0])
    plt.plot(spects[0])
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Power Spectrum')
    plt.show()
    '''
    training_data = read_data()
    train(training_data)

if __name__ == '__main__':
    main()