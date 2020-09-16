import argparse
import cv2
import numpy as np
import os
from Utils.dct_processing import dct_antidiagonal_on_image
from Utils.misc import create_directory, get_frame_ids

parser = argparse.ArgumentParser(
    description='1D DCT of Landmarks Transcription')
parser.add_argument('input', type=str, default=None,
                    help=r"Folder containing all of the landmark folders (ex. path\to\video_id)")
parser.add_argument('output', default=None,
                    help="Location to save 1D dct landmark sequences")
parser.add_argument('-seq-size', type=int, default=30,
                    help="Maximum size of sequence (default 30)")


def landmark_dct_seq(input_path, video_id, landmark, save_path, seq_size):
    '''
    method to create an antidiagonal average of DCT for each frame in an image and\
        save numpy arrays of sequences size number of antidiagonal averages

    input_path: path where the jpg images of the landmarks are stored (must be labled as the frame number)
    landmark: string of the landmark that will be processed
    save_path: directory in which a folder named {landmark} will be create, in \
        which numpy sequences will be saved
    seq_size: size of sequences
    '''
    frame_ids = []
    path_to_landmark = os.path.join(input_path, landmark)
    if not os.path.exists(path_to_landmark):
        print(f'Error: No folder named "{landmark}" in {input_path}')
        return
    frame_ids = get_frame_ids(path_to_landmark)

    if len(frame_ids) == 0:
        return

    dct_seq = []
    batch_sequences = []

    for i in range(max(frame_ids) + 1):
        file_name = f'{i:04d}.jpg'
        file_path = os.path.join(path_to_landmark, file_name)

        im = cv2.imread(file_path)
        if im is None:
            if len(dct_seq) != 0:
                batch_sequences.append(np.stack(dct_seq))
            dct_seq = []
            continue

        if len(dct_seq) == seq_size:
            batch_sequences.append(np.stack(dct_seq))
            dct_seq = []

        antidiag_avg = dct_antidiagonal_on_image(im)
        dct_seq.append(antidiag_avg)
    if len(dct_seq) != 0:
        batch_sequences.append(np.stack(dct_seq))

    for i, seq in enumerate(batch_sequences):
        if seq.shape[0] == seq_size:
            np.save(f'{save_path}/{video_id}-{i:03d}-{seq.shape[0]}', seq)


def main():
    args = parser.parse_args()
    landmarks = ['mouth', 'nose', 'both-eyes']
    subfolders = ['real', 'fake']
    for subfolder in subfolders:
        file_list = [video_label for video_label in os.listdir(
            os.path.join(args.input, subfolder))]
        for video_id in file_list:
            input_path = os.path.join(args.input, subfolder, video_id)
            # the assumption is that this input path contains 3 folders: mouth, nose, and both-eyes
            for landmark in landmarks:
                landmark_output_path = create_directory(
                    os.path.join(args.output, subfolder, video_id, landmark))
                landmark_dct_seq(input_path, video_id, landmark,
                                 landmark_output_path, args.seq_size)


if __name__ == '__main__':
    main()
