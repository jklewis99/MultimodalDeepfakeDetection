import cv2
from Utils.dct_processing import dct_antidiagonal_on_image
from Utils.misc import create_directory
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='1D DCT of Landmarks Transcription')
parser.add_argument('--folder-input', default=None, help=r"Folder containing all of the landmark folders (ex. path\to\video_id)")
parser.add_argument('--save-output', default=None, help="Location to save 1D dct landmark batches")
parser.add_argument('--batch_size', default=24, help="Maximum size of batch")

def dct_batch(input_path, landmark, save_path, batch_size):
    '''
    method to create a antidiagonal average of DCT for each frame in an image and\
        save a numpy array of batch size number of antidiagonal averages
    
    input_path: path where the jpg images of the landmarks are stored (must be labled as the frame number)
    landmark: string of the landmark that will be processed
    save_path: directory in which a folder named {landmark} will be create, in which numpy batches will be saved
    batch_size: size of batches
    '''
    frame_ids = []
    path_to_landmark = os.path.join(input_path, landmark)
    if not os.path.exists(path_to_landmark):
        print(f'Error: No folder named "{landmark}" in {input_path}')
        return
    landmarks = os.listdir(path_to_landmark)
    for frame_id in landmarks:
        if frame_id.endswith('.jpg'):
            frame_ids.append(int(frame_id[:-4]))

    dct_batch = []
    batch_sequences = []

    for i in range(max(frame_ids) + 1):
        file_name = f'{i:04d}.jpg'
        file_path = os.path.join(path_to_landmark, file_name)

        im = cv2.imread(file_path)
        if im is None:
            if len(dct_batch) != 0:
                batch_sequences.append(np.stack(dct_batch))    
            dct_batch = []
            continue

        if len(dct_batch) == batch_size:
            batch_sequences.append(np.stack(dct_batch))
            dct_batch = []

        antidiag_avg = dct_antidiagonal_on_image(im)
        dct_batch.append(antidiag_avg)
    if len(dct_batch) != 0:
        batch_sequences.append(np.stack(dct_batch))

    for i, seq in enumerate(batch_sequences):
        np.save(f'{save_path}/dct-{i:03d}-{seq.shape[0]}', seq)

def main():
    args = parser.parse_args()
    landmarks = ['mouth', 'nose', 'left-eye', 'right-eye', 'both-eyes']
    subfolders = ['real', 'fake']
    
    for subfolder in subfolders:
        file_list = [video_label for video_label in os.listdir(os.path.join(args.folder_input, subfolder))]
        output_path = create_directory(os.path.join(args.save_output, subfolder))
        for video_id in file_list:
            input_path = os.path.join(args.folder_input, subfolder, video_id)
            output_path = create_directory(os.path.join(output_path, video_id))
            # the assumption is that this input path contains 5 folders: mouth, nose, left-eye, right-eye, and both-eyes
            for landmark in landmarks:
                landmark_output_path = create_directory(os.path.join(output_path, landmark))
                dct_batch(input_path, landmark, landmark_output_path, args.batch_size)

if __name__ =='__main__':
    main()