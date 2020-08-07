# MultimodalDeepfakeDetection

An open-source repository that provides the code for the research conducted at the NSF University of Missouri REU Summer 2020

## Command to process landmarks

```shell
python landmark_preprocess.py path/to/processed/videos path/to/save/output
```

Input folder should be a folder containing top-level folders `real` and `fake`. These folders contain video-label-named folders which will contain faces in `JPEG` file format.

## Command to process 1D DCT of Landmarks

```shell
python dct.py path/to/processed/video/ path/to/save/output
```

Input folder should be a folder containing top-level folders `real` and `fake`. These folders contain video-label-named folders in which landmark-named folders (`mouth`, `nose`, `both-eyes`) are hosted. This `dct.py` script will take these images and save them in numpy sequences of `seq-size`. See `dct.py --help` for more information.

## Command to process LipNet Sequences of Mouth

```shell
python lipnet_sequence.py path/to/processed/video-landmarks/ path/to/save/output
```

Input folder should be a folder containing top-level folders `real` and `fake`.

```
input_folder
    - real
        - *.mp4
    - fake
        - *.mp4
```

These folders contain video-label-named folders in which landmark-named folders (`mouth`, `nose`, `both-eyes`) are hosted. This `lipnet_sequence.py` script will take the mouths in this directory and save them into torches and then send this sequence through the [LipNet](https://github.com/Fengdalu/LipNet-PyTorch) model and extract the final features before the fully connected layer. See `lipnet_sequence.py --help` for more information.
