import argparse
import cv2
import numpy as np
import os
import torch
from Utils.misc import create_directory, get_frame_ids

parser = argparse.ArgumentParser(description='LipNet Processing Transcription')
parser.add_argument('--folder-input', default=None, help="Folder containing all of the video id folders\
                                                          that each contain a mouth folder of aligned mouth images")
parser.add_argument('--save-output', default=None, 
                    help="Saves LipNet tensor to this file_path. It will create a fake and real folder, in which the\
                          video label folder will be created containing the tensors")
parser.add_argument('--batch-size', default=24, help="Maximum size of batch")

def normalize_mouth_sequence(video):
    video = np.stack(video, axis=0).astype(np.float32)
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0
    return video

def create_mouth_batch(input_path, save_path, batch_size=24, img_type='jpg'):
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

        if len(mouth_batch) == batch_size:
            batch_sequences.append(np.stack(mouth_batch))
            mouth_batch = []

        mouth = cv2.resize(im, (128, 64))
        mouth_batch.append(mouth)
    if len(mouth_batch) != 0:
        batch_sequences.append(np.stack(mouth_batch))

    for i, seq in enumerate(batch_sequences):
        seq = normalize_mouth_sequence(seq)
        torch.save(seq, f'{save_path}/mouth-seq-{i:03d}-{seq.shape[1]}.pt')

def main():
    args = parser.parse_args()
    landmark = 'mouth'
    subfolders = ['real', 'fake']
    
    for subfolder in subfolders:
        file_list = [video_label for video_label in os.listdir(os.path.join(args.folder_input, subfolder))]
        for video_id in file_list:
            input_path = os.path.join(args.folder_input, subfolder, video_id, landmark)
            output_path = create_directory(os.path.join(args.save_output, subfolder, video_id))
            # the assumption is that this input path contains a mouth folder
            create_mouth_batch(input_path, output_path, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
