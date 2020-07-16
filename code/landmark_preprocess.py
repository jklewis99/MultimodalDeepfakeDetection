import face_alignment
import cv2
import os
from matplotlib import pyplot as plt
import time
import random
import math

DATA_FOLDER = 'train_sample_videos'

time_to_detect_landmarks = 0
time_to_detect_tracking = 0
time_to_render_bounds = 0
time_to_read = 0
time_to_write = 0
time_to_resize = 0

# create the directory for each video and set it as the current directory to save each frame inside
def create_directory(video_file, file_type='.mp4'):
    name = video_file.split(file_type)[0]
    os.mkdir('preprocessed_data/{}'.format(name))
    return 'preprocessed_data/{}'.format(name)

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
def landmark_bounding_box(landmarks, offset, resize_info, img_shape):
    '''
    a top-level method for getting a bounding box of the landmarks detected
    by FaceAlign

    landmarks: array of size 68 with tuples of the x, y coordinates of a landmark
    offset: x, y tuple defining the shift in the x and y direction when using psuedo-facetracking
    resize_info: x, y for rescaling the dimensions because of initial rescale
    '''
    # create list ranges for landmark bounding boxes
    eye1_range = list(range(17, 22)) + list(range(36, 42))
    eye2_range = list(range(22, 27)) + list(range(42, 48))
    nose_range = list(range(27, 36))
    mouth_range = list(range(48, 60))
    
    eye1_rectangle = get_bounding_box(eye1_range, landmarks, offset, resize_info)
    eye1_rectangle = verify_bounds(eye1_rectangle, img_shape)
    
    eye2_rectangle = get_bounding_box(eye2_range, landmarks, offset, resize_info)
    eye2_rectangle = verify_bounds(normalize(eye2_rectangle), img_shape)
    
    eyes_rectangle = [min(eye1_rectangle[0], eye2_rectangle[0]),
                      min(eye1_rectangle[1], eye2_rectangle[1]), 
                      max(eye1_rectangle[2], eye2_rectangle[2]), 
                      max(eye1_rectangle[3], eye2_rectangle[3])]

    nose_rectangle = get_bounding_box(nose_range, landmarks, offset, resize_info)
    nose_rectangle = verify_bounds(normalize(nose_rectangle), img_shape)
    
    mouth_rectangle = get_bounding_box(mouth_range, landmarks, offset, resize_info)
    mouth_rectangle = verify_bounds(normalize(mouth_rectangle, size=(64, 128)), img_shape)
    
    return eye1_rectangle, eye2_rectangle, eyes_rectangle, nose_rectangle, mouth_rectangle

def get_bounding_box(index_range, landmarks, offset, resize_info):
    '''
    a sub-level helper method for getting a bounding box based on the index_range

    index_range: the indices that need to be searched for a given facial feature (nose, eye, etc.)
    landmarks: array of size 68 with tuples of the x, y coordinates of a landmark
    offset: x, y tuple defining the shift in the x and y direction when using psuedo-facetracking
    resize_info: x, y for rescaling the dimensions because of initial rescale
    '''
    x1 = y1 = math.inf
    x2 = y2 = 0
    
    for i in index_range:
        x1 = min(landmarks[i][0] + offset[0], x1)
        y1 = min(landmarks[i][1] + offset[1], y1)
        x2 = max(landmarks[i][0] + offset[0], x2)
        y2 = max(landmarks[i][1] + offset[1], y2)
    
    # additional space provide space in case of error in detection
    x_cushion = int(max((x2-x1)*resize_info[0]*0.08, 6))
    y_cushion = int(max((y2-y1)*resize_info[0]*0.08, 6))
    
    return [int(x1*resize_info[0])-x_cushion, 
            int(y1*resize_info[1])-y_cushion, 
            int(x2*resize_info[0])+x_cushion, 
            int(y2*resize_info[1])+y_cushion]

def normalize(bounding_box, size=(28, 28)):
    '''
    a sub-level helper method to rescale each facial feature into the 
    correct scale while maintaining aspect ration

    bounding_box: the x1, y1, x2, y2 mutable array that has already given 
                  the rectangle of a facial feature
    size: y, x tuple defining the desired height and width, respectively
    '''
    w = bounding_box[2]-bounding_box[0]
    h = bounding_box[3]-bounding_box[1]
    
    if w < size[1] and h < size[0]:
        # if both of the current width and height are too small, increase
        # as needed
        bounding_box[0] -= int(math.floor((size[1] - w )/ 2))
        bounding_box[2] += int(math.ceil((size[1] - w) / 2))
        bounding_box[1] -= int(math.floor((size[0] - h) / 2))
        bounding_box[3] += int(math.ceil((size[0] - h) / 2))
    elif w < size[1]:
        # only increase width
        to_increase = ((size[1] / size[0]) * h) - w
        bounding_box[0] -= int(math.floor(to_increase / 2))
        bounding_box[2] += int(math.ceil(to_increase / 2))
    elif h < size[0]:
        # only increase height
        to_increase = (w / (size[1] / size[0])) - h
        bounding_box[1] -= int(math.floor(to_increase / 2))
        bounding_box[3] += int(math.ceil(to_increase / 2))
    else:
        # both height and width are bigger than the desired size
        wid_rat = size[1] / w
        hei_rat = size[0] / h
        if wid_rat > hei_rat:
            # increase width
            to_increase = ((size[1] / size[0]) * h) - w
            bounding_box[0] -= int(math.floor(to_increase / 2))
            bounding_box[2] += int(math.ceil(to_increase / 2))
        else:
            # increase height
            to_increase = (w / (size[1] / size[0])) - h
            bounding_box[1] -= int(math.floor(to_increase / 2))
            bounding_box[3] += int(math.ceil(to_increase / 2))
            
    return bounding_box

def verify_bounds(rectangle, img_dim):
    # make sure we won't get any index out of bounds errors
    rectangle[0] = int(max(0, rectangle[0]))
    rectangle[1] = int(max(0, rectangle[1]))
    rectangle[2] = int(min(img_dim[1]-1, rectangle[2]))
    rectangle[3] = int(min(img_dim[0]-1, rectangle[3]))
    
    return rectangle
    
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    '''
    method to resize an image based on a given width or height
    '''
    # initialize the dimensions of the image to be resized and
    # grab the image size
    global time_to_resize
    start = time.time()
    dim = None
    (h, w) = image.shape[:2]
    
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

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

def face_resize_square(boundaries, img, size=299, inter=cv2.INTER_AREA):
    # method to resize each face to 299x299
    w = boundaries[2]-boundaries[0]
    h = boundaries[3]-boundaries[1]
    if w > h:
        to_increase = w - h
        boundaries[1] -= math.floor(to_increase/2)
        boundaries[3] += math.ceil(to_increase/2)
    else:
        to_increase = h - w
        boundaries[0] -= math.floor(to_increase/2)
        boundaries[2] += math.ceil(to_increase/2)
        
    boundaries = verify_bounds(boundaries, img.shape)
    return image_resize(img[boundaries[1] : boundaries[3], boundaries[0] : boundaries[2], :], width=size)[0]

def face_track(frame, fa, last_bounding_box=None):
    '''
    we want to limit our search space based on the last bounding box to speed up landmark detection
    
    frame: frame of video (already resized)
    last_bounding_box: rectangluar endpoints of last detected face (x1, y1, x2, y2)
    fa: pretrained detection model
    '''
    global time_to_detect_landmarks
    
    if last_bounding_box is None:
        start = time.time()
        preds = fa.get_landmarks(frame)
        time_to_detect_landmarks += time.time() - start
        return preds, (0, 0)
    
    # expand the last bounding box in all directions
    offset_x = max(0, last_bounding_box[0]-20) # maintain offsets as floats
    offset_y = max(0, last_bounding_box[1]-20)
    x1 = int(offset_x)                                       # new x1
    y1 = int(offset_y)                                       # new y1
    x2 = int(min(frame.shape[1]-1, last_bounding_box[2]+20)) # new x2
    y2 = int(min(frame.shape[0]-1, last_bounding_box[3]+20)) # new y2

    start = time.time()
    preds = fa.get_landmarks(frame[y1:y2, x1:x2, :])
    time_to_detect_landmarks += time.time() - start
    
    return preds, (offset_x, offset_y)

def detect_landmarks(path, file_name, fa, folder_to_write):
    '''
    primary method to take each frame of a video and save the face and landmarks from each frame
    
    path: directory of video
    file_name: name of the video
    fa: model for face and landmark detections
    folder_to_write: local directory in top-level video folders and their landmarks and frames will be save
    '''
    capture = cv2.VideoCapture(os.path.join(path, file_name))
    width  = int(capture.get(3))
    height = int(capture.get(4))
    
    global time_to_detect_tracking, time_to_render_bounds, time_to_read, time_to_write, time_to_resize
    
    face_box = None
    frame_num = 1
    
    while capture.isOpened():
        start = time.time()
        success, orig_frame = capture.read()
        time_to_read += time.time() - start
        if not success:
            # we have reached the end of the video
            break
        start = time.time()
        frame, resize_info = image_resize(orig_frame, width=400)
        time_to_resize += time.time() - start
        
        start = time.time()        
        preds, offset = face_track(frame, fa, face_box)
        time_to_detect_tracking += time.time() - start
        
        if preds:
            for i in range(len(preds[1])):
                face_box = preds[1][i]
                
                face_box[0] += offset[0]
                face_box[2] += offset[0]
                face_box[1] += offset[1]
                face_box[3] += offset[1]
                start = time.time()
                eye1, eye2, both_eyes, nose, mouth = landmark_bounding_box(preds[0][i], offset, resize_info, orig_frame.shape)
                time_to_render_bounds += time.time()-start
                
                face = face_resize_square([int(face_box[0]*resize_info[0]), int(face_box[1]*resize_info[0]), 
                                           int(face_box[2]*resize_info[1]), int(face_box[3]*resize_info[1])], 
                                           orig_frame)
                eye1, _ = image_resize(orig_frame[eye1[1]:eye1[3], eye1[0]:eye1[2], :], width=128)
                eye2, _ = image_resize(orig_frame[eye2[1]:eye2[3], eye2[0]:eye2[2], :], width=128)
                both_eyes, _ = image_resize(orig_frame[both_eyes[1]:both_eyes[3], both_eyes[0]:both_eyes[2], :], width=128)
                nose, _ = image_resize(orig_frame[nose[1]:nose[3], nose[0]:nose[2], :], width=128)
                mouth, _ = image_resize(orig_frame[mouth[1]:mouth[3], mouth[0]:mouth[2], :], width=128)

                # save each image in the correct folder
                start = time.time()
                cv2.imwrite(folder_to_write + r'/face_face{}_frame{}.jpg'.format(i, frame_num), face)                                                                                             int(face_box[0]*resize_info[0]):int(face_box[2]*resize_info[0]), :])
                cv2.imwrite(folder_to_write + r'/eye1_face{}_frame{}.jpg'.format(i, frame_num), eye1)
                cv2.imwrite(folder_to_write + r'/eye2_face{}_frame{}.jpg'.format(i, frame_num), eye2)
                cv2.imwrite(folder_to_write + r'/botheyes_face{}_frame{}.jpg'.format(i, frame_num), both_eyes)
                cv2.imwrite(folder_to_write + r'/nose_face{}_frame{}.jpg'.format(i, frame_num), nose)
                cv2.imwrite(folder_to_write + r'/mouth_face{}_frame{}.jpg'.format(i, frame_num), mouth)
                time_to_write += time.time() - start
        else:
            print('Video Error: No face landmarks detected so no face was saved')
        frame_num += 1
        
    capture.release()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)

vids = os.listdir(DATA_FOLDER)
start = time.time()
count_successful = 0
videos_failed = []
for vid in vids:
    if os.path.isdir('preprocessed_data/{}'.format(vid.split('.mp4')[0])):
        continue
    folder2write = create_directory(vid)
    try:
        detect_landmarks(DATA_FOLDER, vid, fa, folder2write)
        
    except:
        print('ERROR: Unsuccessful processing of {}'.format(vid))
        videos_failed.append(vid)
        continue
    count_successful += 1 
    print('Finished processing video {}'.format(vid))
    print('Time: {} s'.format(time.time()-start))
    
    
    
process_time = time.time() - start
print('PROCESS TIME: {:.3f} s for {} videos (out of {})'.format(process_time, count_successful, len(vids)))
print('\tLandmark Detection: {:.3f} s'.format(time_to_detect_landmarks))
print('\tTracking time: {:.3f} s'.format(time_to_detect_tracking))
print('\tMake rectangles: {:.3f} s'.format(time_to_render_bounds))
print('\tRead Time: {:.3f} s'.format(time_to_read))
print('\tWrite Time: {:.3f} s'.format(time_to_write))
print('\tResize: {:.3f} s'.format(time_to_resize))
print('Videos that failed:\n', videos_failed)