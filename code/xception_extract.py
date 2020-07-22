from Xception.xception import xception
import torch
import os
import cv2
import numpy as np
import sys
from tqdm import tqdm

BATCH_SIZE = 24


def feats_from_dir(path, outpath, model):

    vname = path.split('/')[-1]

    frameidx = [int(p[-8:-4]) for p in os.listdir(path)]
    if len(frameidx) == 0:
        return
    imgseqs = []
    currentseq = []

    for i in range(max(frameidx) + 1):
        fname = f'{vname}-{i:04d}.jpg'
        fpath = os.path.join(path, fname)

        im = cv2.imread(fpath)

        if im is None:
            if len(currentseq) != 0:
                currentseq = np.stack(currentseq)
                imgseqs.append(currentseq)
            currentseq = []
            continue

        if len(currentseq) == 24:
            currentseq = np.stack(currentseq)
            imgseqs.append(currentseq)
            currentseq = []

        im = (im / 255. - .5) / .5
        currentseq.append(im)

    if len(currentseq) != 0:
        currentseq = np.stack(currentseq)
        imgseqs.append(currentseq)

    for i, seq in enumerate(imgseqs):
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        seq = torch.Tensor(seq.transpose(0, 3, 1, 2))
        seq = seq.cuda()
        feats = model.last_feature_layer(seq)
        outfile = os.path.join(outpath, vname)
        torch.save(feats, f'{outfile}-{i:03d}-{feats.shape[0]}.pt')


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    fakelist = [os.path.join(input_dir, 'fake', p)
                for p in os.listdir(os.path.join(input_dir, 'fake'))]

    reallist = [os.path.join(input_dir, 'real', p)
                for p in os.listdir(os.path.join(input_dir, 'real'))]

    net = xception().to('cuda')

    for split in ('fake', 'real'):
        for path in tqdm(fakelist, desc=f'Processing {split} files'):
            fileid = get_file_id(path)
            outpath = os.path.join(output_dir, split, fileid)
            feats_from_dir(path, outpath, net)


def get_file_id(path):
    return path.split('/')[-1]


    # print(files)
if __name__ == '__main__':
    main()
