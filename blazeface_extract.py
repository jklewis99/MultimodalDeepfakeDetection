import os
import math
import cv2
import sys
import numpy as np
from BlazeFace.blazeface import init_model
from Utils.video import *


def main():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    fakefilelist = [os.path.join(input_dir, 'fake', p) for p in os.listdir(os.path.join(
        input_dir, 'fake')) if p[-4:] == '.mp4']

    realfilelist = [os.path.join(input_dir, 'real', p) for p in os.listdir(os.path.join(
        input_dir, 'real')) if p[-4:] == '.mp4']

    net = init_model()

    for p in realfilelist:
        fileid = get_file_id(p)
        out_dir = os.path.join(output_dir, 'real', fileid)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        faces = extract_faces(p, net, padding=30, save_path=out_dir)

    for p in fakefilelist:
        fileid = get_file_id(p)
        out_dir = os.path.join(output_dir, 'fake', fileid)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        faces = extract_faces(p, net, padding=30, save_path=out_dir)


def get_file_id(path):
    filename = path.split('/')[-1]
    fileid = '.'.join(filename.split('.')[:-1])
    return fileid


def get_iou(last_face, current_face):
    '''
    a method to calculate the intersection over union of the current face of interest and the
    face of interest from the previous frame

    last_face: the bounding box (x1, y1, x2, y2) of the face found in the previous frame 
    current_face: the bounding box (x1, y1, x2, y2) of one of the faces found in the current frame 
    '''
    if last_face is None:
        return 1
    # rectangle of intersection:
    x1 = max(last_face[0], current_face[0])
    x2 = min(last_face[2], current_face[2])
    y1 = max(last_face[1], current_face[1])
    y2 = min(last_face[3], current_face[3])

    # intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # area of the last face
    last_face_area = max(
        0, last_face[2] - last_face[0]) * max(0, last_face[3] - last_face[1])

    # area of the current face
    current_face_area = max(
        0, current_face[2] - current_face[0]) * max(0, current_face[3] - current_face[1])

    # intersection over union
    iou = inter_area / (last_face_area + current_face_area - inter_area)

    return iou


def extract_faces(filepath, net, padding=10, save_path=None, mark_landmarks=False):
    """
    returns an array of face images
    """
    frames = read_video(filepath)
    smframes, (xshift, yshift) = resize_and_crop_video(frames, 128)

    # Detection
    smframesnp = np.stack(smframes)
    detections = net.predict_on_batch(smframesnp)

    # TODO: Fill the detection gap

    output = []
    last_face = None

    for i, detection in enumerate(detections):
        frame = frames[i]

        size = min(frame.shape[0], frame.shape[1])
        detection = detection.cpu().numpy()

        iou = -1
        face_of_interest = None

        for face_num in range(len(detection)):
            ymin = math.floor(detection[face_num, 0] * size + yshift) - padding
            ymin = max(0, ymin)

            xmin = math.floor(detection[face_num, 1] * size + xshift) - padding
            xmin = max(0, xmin)

            ymax = math.floor(detection[face_num, 2] * size + yshift) + padding
            ymax = min(frame.shape[0], ymax)

            xmax = math.floor(detection[face_num, 3] * size + xshift) + padding
            xmax = min(frame.shape[1], xmax)

            # check if this detected face was closest to the last face detected
            temp_iou = get_iou(last_face, [xmin, ymin, xmax, ymax])
            if temp_iou > iou:
                face_of_interest = [xmin, ymin, xmax, ymax]
                iou = temp_iou

            # mark landmarks:
            if mark_landmarks:
                for k in range(6):
                    kp_x = math.floor(
                        detection[face_num, 4 + k*2] * size + xshift)
                    kp_y = math.floor(
                        detection[face_num, 4 + k*2 + 1] * size + yshift)
                    frame = cv2.circle(frame, (kp_x, kp_y), 2, (255, 0, 0), 2)

        # update the last_face rectangle
        if iou > 0:  # if the iou is zero, then we have found a different face
            # EDGE CASE: blazeface does not find the same face after enough frames
            # and the subject moves enough that no face will ever be saved
            last_face = face_of_interest
            face = frame[face_of_interest[1]: face_of_interest[3],
                         face_of_interest[0]: face_of_interest[2]]
            face = cv2.resize(face, (299, 299))

            if save_path is not None:
                fileid = get_file_id(filepath)
                cv2.imwrite(f'{save_path}/{fileid}-{i:04d}.jpg',
                            cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            output.append(face)
        else:
            print('Found a different or no faces. Skipping...')

    return output


if __name__ == '__main__':
    main()
