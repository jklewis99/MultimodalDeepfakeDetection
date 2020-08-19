import os
import argparse
import concurrent.futures
import subprocess
from misc import create_directory, split_list

parser = argparse.ArgumentParser(description='Convert videos to 30 fps')
parser.add_argument('input', help="Folder containing fake and real folders, inside those folders \
                                   all of the video files are stored")
parser.add_argument('output', help="Saves videos with new framerate. It will create a fake \
                                    and real folder")
parser.add_argument('-fps', default=30, help="intended frame rate")

def main():
    args = parser.parse_args()
    subfolders = ['real', 'fake']
    args_list = []
    for subfolder in subfolders:
        file_list = [video_label for video_label in os.listdir(os.path.join(args.input, subfolder))]
        for video_id in file_list:
            input_path = os.path.join(args.input, subfolder, video_id)
            output_path = create_directory(os.path.join(args.output, subfolder))
            args_list.append((input_path, os.path.join(output_path, video_id)))

    args_list = split_list(args_list, 20)
    for arg_tuples in args_list:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(set_fps, input_path, output_path, args.fps)
                       for input_path, output_path in arg_tuples]

            for f in concurrent.futures.as_completed(results):
                print(f.result())

def set_fps(input_path, output_path, fps):
    # os.system(f'ffmpeg -i {input_path} -filter:v fps=fps={args.fps} {os.path.join(output_path, video_id)}')
    subprocess.check_output(f'ffmpeg -i {input_path} -filter:v fps=fps={fps} {output_path}', shell=True)

if __name__ == '__main__':
    main()