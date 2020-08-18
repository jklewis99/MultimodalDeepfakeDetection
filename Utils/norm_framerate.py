import os
import argparse
from misc import create_directory
import cv2

parser = argparse.ArgumentParser(description='Convert videos to 30 fps')
parser.add_argument('input', help="Folder containing fake and real folders, inside those folders \
                                   all of the video files are stored")
parser.add_argument('output', help="Saves videos with new framerate. It will create a fake \
                                    and real folder")
parser.add_argument('-fps', default=30, help="intended frame rate")

def main():
    args = parser.parse_args()
    subfolders = ['real', 'fake']
    print('response')
    for subfolder in subfolders:
        file_list = [video_label for video_label in os.listdir(os.path.join(args.input, subfolder))]
        for video_id in file_list:
            input_path = os.path.join(args.input, subfolder, video_id)
            output_path = create_directory(os.path.join(args.output, subfolder))
            os.system(f'ffmpeg -i {input_path} -filter:v fps=fps={args.fps} {os.path.join(output_path, video_id)}')

def test():
    CV2_FRAMECOUNT_ID = int(cv2.CAP_PROP_FRAME_COUNT)
    CV2_FPS_ID = int(cv2.CAP_PROP_FPS)
    video_file = r'J:\reu\code\input\deepfake-detection-challenge\ii\fake\aagfhgtpmv.mp4'
    cap = cv2.VideoCapture(video_file)
    framecount = int(cv2.VideoCapture.get(cap, CV2_FRAMECOUNT_ID))
    fps = cv2.VideoCapture.get(cap, CV2_FPS_ID)
    cap.release()
    print(fps)

if __name__ == '__main__':
    test()