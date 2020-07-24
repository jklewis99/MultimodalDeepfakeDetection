import argparse
import cv2
import numpy as np
import os
import torch
from LipNet.model import LipNet
from torch import nn
from Utils.misc import create_directory, get_frame_ids

parser = argparse.ArgumentParser(description='LipNet Processing Transcription')
parser.add_argument('input', help="Folder containing fake and real folders, inside those folders \
                                   all of the video id folders that each contain a mouth folder \
                                   of aligned mouth images")
parser.add_argument('output', help="Saves LipNet tensor to this file_path. It will create a fake \
                                    and real folder, in which the video label folder will be \
                                    created containing the tensors")
parser.add_argument('-seq-size', default=24, help="Maximum size of sequence (default 24)")
parser.add_argument('-device', default='cuda', help="device for model (default cuda)")
parser.add_argument('-pretrained', 
                    default='LipNet/pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt',
                    help='path to pretrained weights for LipNet model')

def normalize_mouth_sequence(video):
    video = np.stack(video, axis=0).astype(np.float32)
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0
    return video

def get_lipnet_features(model, input_path, save_path, video_id, seq_size=24, img_type='jpg', device='cuda'):
    '''
    export the features from the LipNet model. Prepares sequences of seq_size \
        number of images, converts them to tensors, and then feeds them \
        through the LipNet model

    Args
    -------------------------------------------------------------------------
    model: trained LipNet model 
    input_path: path to folder containing the mouths with 4 digit ids
    save_path: path in which LipNet features will be saved
    video_id: video name
    -------------------------------------------------------------------------
    
    KeywordArgs
    -------------------------------------------------------------------------
    seq_size: number of images per sequence (default 24)
    img_type: file type of saved images (default 'jpg')
    device: 'cuda' or 'cpu'
    ------------------------------------------------------------------------- 
    '''
    frame_ids = get_frame_ids(input_path, img_type=img_type)

    mouth_batch = []
    batch_sequences = []

    for i in range(max(frame_ids) + 1):
        file_name = f'{i:04d}.{img_type}'
        file_path = os.path.join(input_path, file_name)

        im = cv2.imread(file_path)
        if im is None:
            if len(mouth_batch) != 0:
                batch_sequences.append(np.stack(mouth_batch))    
            mouth_batch = []
            continue

        if len(mouth_batch) == seq_size:
            batch_sequences.append(np.stack(mouth_batch))
            mouth_batch = []

        mouth = cv2.resize(im, (128, 64))
        mouth_batch.append(mouth)

    if len(mouth_batch) != 0:
        batch_sequences.append(np.stack(mouth_batch))

    feature_sequences = []
    translationf = []
    for i, seq in enumerate(batch_sequences):
        seq = normalize_mouth_sequence(seq)
        if seq.shape[1] == seq_size:
            translation, features = model(seq[None, ...].to(device))
            torch.save(features, f'{save_path}/{video_id}-{i:03d}-{seq.shape[1]}.pt')
            feature_sequences.append(features)
            translationf.append(translation)
    
    return feature_sequences, translationf

def main():
    args = parser.parse_args()
    landmark = 'mouth'
    subfolders = ['real', 'fake']
    
    model = LipNet()
    model = model.to(args.device)
    net = nn.DataParallel(model).to(args.device)

    pretrained_dict = torch.load(args.pretrained, map_location=torch.device(args.device))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict.keys() and 
                        v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    # print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    # print('miss matched params:{}'.format(missed_params))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    for subfolder in subfolders:
        file_list = [video_label for video_label in os.listdir(os.path.join(args.input, subfolder))]
        for video_id in file_list:
            input_path = os.path.join(args.input, subfolder, video_id, landmark)
            output_path = create_directory(os.path.join(args.output, subfolder, video_id))
            # the assumption is that this input path contains a mouth folder
            get_lipnet_features(model, input_path, output_path, video_id, seq_size=args.seq_size, device=args.device)

if __name__ == '__main__':
    main()
