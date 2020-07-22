import face_alignment
import cv2
import os
import time
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='Landmark Processing Transcription')
parser.add_argument('--folder-input', default=None, help="Folder containing all of the video id folders that contain face images")
parser.add_argument('--save-output', default=None, 
                    help="Saves 1D dct, LipNet tensor, and (optionally) landmark images to this file_path")
parser.add_argument('--landmark-save', default=False, help="Boolean that determines if landmark images will be saved (default: False)")

# create the directory for each video and set it as the current directory to save each frame inside
def create_directory(save_path, label, video_file, file_type='.mp4'):
    name = video_file.split(file_type)[0]
    if not os.path.isdir('{}/{}/{}'.format(save_path, label, name)):
        os.mkdir('{}/{}/{}'.format(save_path, label, name))
    return '{}/{}/{}'.format(save_path, label, name)

def create_landmark_dir(save_path, landmark):
    if not os.path.isdir(f'{save_path}/{landmark}'):
        os.mkdir(f'{save_path}/{landmark}')
'''
Landmarks from face_alignment are located as follows:
    Landmark | indices
    face:      0,  16
    eyebrow1:  17, 21
    eyebrow2:  22, 26 
    nose:      27, 30
    nostril:   31, 35
    eye1:      36, 41
    eye2:      42, 47
    lips:      48, 59
    teeth:     60, 67
'''

def image_resize(image, min_dim=128, inter = cv2.INTER_AREA):
    '''
    method to resize an image based on a given width or height
    '''
    # initialize the dimensions of the image to be resized and
    # grab the image size
    global time_to_resize
    start = time.time()
    dim = None
    width = None
    height = None
    
    if image.shape[0] < image.shape[1]:
        height = min_dim
    else: 
        width = min_dim
    (h, w) = image.shape[:2]

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    time_to_resize += time.time() - start
    return resized, (w/dim[0], h/dim[1])

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
 
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
 
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def get_position(size, padding=0.25):
    
    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                    0.553364, 0.490127, 0.42689]
    
    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                    0.784792, 0.824182, 0.831803, 0.824182]
    
    x, y = np.array(x), np.array(y)
    
    x = (x + padding) / (2 * padding + 1)
    y = (y + padding) / (2 * padding + 1)
    x = x * size
    y = y * size
    return np.array(list(zip(x, y)))

def get_frame(img_id):
    return img_id[-8:-4]

def get_landmarks_from_directory(path, fa):
    '''
    process the faces in a directory and find their landmark points
    '''
    faces = os.listdir(path)
    labels = []
    read_images = []
    for face in faces:
        read_images.append(cv2.imread(os.path.join(path, face)))
        labels.append(get_frame(face))
    read_images = list(filter(lambda im: not im is None, read_images))
    list_landmark_points = [fa.get_landmarks(img, detected_faces=[np.array([0, 0, 298, 298])]) for img in read_images]
    
    return list_landmark_points, read_images, labels

def landmark_boundaries(front256, img):
    '''
    get the center of our landmarks and return the bounding box of the landmark
    *** function assumes the img is affine transformed ***
    '''
    x, y = front256[31:].mean(0).astype(np.int32) # mouth
    mouth = get_landmark_box(img, x, y, 80, square=False)
    x, y = front256[10:19].mean(0).astype(np.int32) # nose?
    nose = get_landmark_box(img, x, y, 40)
    x, y =  np.concatenate([front256[0:5], front256[19: 25]]).mean(0).astype(np.int32) # eye1?
    eye1 = get_landmark_box(img, x, y, 40)
    x, y = np.concatenate([front256[5:10], front256[25: 31]]).mean(0).astype(np.int32) # eye2?
    eye2 = get_landmark_box(img, x, y, 40)
    x, y = np.concatenate([front256[0:10], front256[19: 31]]).mean(0).astype(np.int32) # both eyes
    eyes = get_landmark_box(img, x, y, 100, square=False)
    return mouth, nose, eye1, eye2, eyes

def get_landmark_box(img, x, y, w, square=True):
    if square:
        img = img[y - w : y + w, x - w : x + w, ...]
    else:
        img = img[y - w // 2: y + w // 2, x - w : x + w, ...]
    return img

def process_faces(fa, input_path, video_id, save_path=None, save_landmarks=False):
    list_dir_landmarks, faces_array, labels = get_landmarks_from_directory(os.path.join(input_path, video_id), fa)
    front256 = get_position(256)
    count = 0

    create_landmark_dir(save_path, 'mouth')
    create_landmark_dir(save_path, 'both-eyes')
    create_landmark_dir(save_path, 'nose')
    create_landmark_dir(save_path, 'left-eye')
    create_landmark_dir(save_path, 'right-eye')

    for frame, preds, face in zip(labels, list_dir_landmarks, faces_array):
        if preds is not None:
            # get the list of landmarks
            # shape = preds[0] # this command works on my computer, but not lewis
            shape = preds[0][0] # this command works on Lewis, but not my computer
            shape = shape[17:] # diregard the face endpoints
            M = transformation_from_points(np.matrix(shape), np.matrix(front256)) # transform the face
        
            img = cv2.warpAffine(face, M[:2], (256, 256))
            mouth, nose, eye1, eye2, eyes = landmark_boundaries(front256, img)
            
            mouth = cv2.resize(mouth, (256, 128))
            nose = cv2.resize(nose, (128, 128))
            eye1 = cv2.resize(eye1, (128, 128))
            eye2 = cv2.resize(eye2, (128, 128))
            eyes = cv2.resize(eyes, (256, 128))
            
            if save_landmarks and save_path:
                cv2.imwrite(f'{save_path}/mouth/{frame}.jpg', mouth)
                cv2.imwrite(f'{save_path}/nose/{frame}.jpg', nose)
                cv2.imwrite(f'{save_path}/left-eye/{frame}.jpg', eye1)
                cv2.imwrite(f'{save_path}/right-eye/{frame}.jpg', eye2)
                cv2.imwrite(f'{save_path}/both-eyes/{frame}.jpg', eyes)

        else:
            count += 1
            print('No Preds:', count)

def main():
    args = parser.parse_args()

    # will fail if there is no folder labeled 'fake'
    fakefilelist = [video_label for video_label in os.listdir(os.path.join(
        args.folder_input, 'fake'))]

    realfilelist = [video_label for video_label in os.listdir(os.path.join(
        args.folder_input, 'real'))]

    save_landmarks = args.landmark_save.lower() == 'true'
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')

    start = time.time()
    count_processed = 0
    
    if not os.path.isdir('{}/{}'.format(args.save_output, 'fake')):
        os.mkdir('{}/{}'.format(args.save_output, 'fake'))
    for vid in fakefilelist:
        path = create_directory(args.save_output, 'fake', vid, file_type=' ')
        process_faces(fa, os.path.join(args.folder_input, 'fake'), vid, save_path=path, save_landmarks=save_landmarks)
        print('Finished processing video {}'.format(vid))
        count_processed += 1

    if not os.path.isdir('{}/{}'.format(args.save_output, 'real')):
        os.mkdir('{}/{}'.format(args.save_output, 'real'))
    for vid in realfilelist:
        path = create_directory(args.save_output, 'real', vid, file_type=' ')
        process_faces(fa, os.path.join(args.folder_input, 'real'), vid, save_path=path, save_landmarks=save_landmarks)
        print('Finished processing video {}'.format(vid))
        count_processed += 1

    process_time = time.time() - start
    print('PROCESS TIME: {:.3f} s for {} videos (out of {})'.format(process_time, count_processed, len(fakefilelist)+len(realfilelist)))
    
if __name__ == '__main__':
    main()
