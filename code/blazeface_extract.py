import os
import math
import cv2
import sys
import np
from BlazeFace.blazeface import init_model
from Utils.video import *


def main():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    fakefilelist = [os.path.join(input_dir, 'fake', p) for p in os.listdir(os.path.join(
        input_dir, 'fake')) if p[-4:] == '.mp4']

    realfilelist = [os.path.join(input_dir, 'real', p) for p in os.listdir(os.path.join(
        input_dir, 'real')) if p[-4:] == '.mp4']

    # filelist = [*realfilelist, *fakefilelist]

    net = init_model()

    for p in realfilelist:
        fileid = get_file_id(p)
        out_dir = os.path.join(output_dir, 'real', fileid)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        faces = extract_faces(p, net, padding=30, save_path=out_dir)

    for p in realfilelist:
        fileid = get_file_id(p)
        out_dir = os.path.join(output_dir, 'fake', fileid)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        faces = extract_faces(p, net, padding=30, save_path=out_dir)


def get_file_id(path):
    filename = path.split('/')[-1]
    fileid = '.'.join(filename.split('.')[:-1])
    return fileid


def extract_faces(filepath, net, padding=10, save_path=None, mark_landmarks=False):
    """
    returns an array of face images
    """
    frames = read_video(filepath)
    smframes, (xshift, yshift) = resize_and_crop_video(frames, 128)

    # Detection
    smframesnp = np.stack(smframes)
    detections = net.predict_on_batch(smframesnp)

    output = []
    for i, detection in enumerate(detections):
        frame = frames[i]

        size = min(frame.shape[0], frame.shape[1])
        detection = detection.cpu().numpy()

        ymin = math.floor(detection[0, 0] * size + yshift) - padding
        ymin = max(0, ymin)

        xmin = math.floor(detection[0, 1] * size + xshift) - padding
        xmin = max(0, xmin)

        ymax = math.floor(detection[0, 2] * size + yshift) + padding
        ymax = min(frame.shape[0], ymax)

        xmax = math.floor(detection[0, 3] * size + xshift) + padding
        xmax = min(frame.shape[1], xmax)

        # mark landmarks:
        if mark_landmarks:
            for k in range(6):
                kp_x = math.floor(detection[0, 4 + k*2] * size + xshift)
                kp_y = math.floor(detection[0, 4 + k*2 + 1] * size + yshift)
                frame = cv2.circle(frame, (kp_x, kp_y), 2, (255, 0, 0), 2)

        face = frame[ymin:ymax, xmin:xmax]
        face = cv2.resize(face, (128, 128))

        if save_path is not None:
            fileid = get_file_id(filepath)
            cv2.imwrite(f'{save_path}/{fileid}-{i:04d}.png',
                        cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        output.append(face)

    return output


if __name__ == '__main__':
    main()
