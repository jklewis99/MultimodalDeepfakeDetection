import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_frame_ids(input_path, img_type='jpg'):
    mouth_files = os.listdir(input_path)
    frame_ids = []
    for frame_id in mouth_files:
        if frame_id.endswith('.' + img_type):
            frame_ids.append(int(frame_id[:-4]))
    return frame_ids