from Xception.xception import xception
import torch
import os
import cv2
import numpy as np
import sys
from tqdm import tqdm


def feats_from_dir(path, outpath, model):

    vname = path.split('/')[-1]

    frameidx = [int(p[-8:-4]) for p in os.listdir(path)]
    imgseqs = []
    currentseq = []
    for i in range(max(frameidx) + 1):
        fname = f'{vname}-{i:04d}.png'
        fpath = os.path.join(path, fname)
        im = cv2.imread(fpath)
        if im is None:
            currentseq = np.stack(currentseq)
            imgseqs.append(currentseq)
            currentseq = []
        else:
            currentseq.append(im)

    currentseq = np.stack(currentseq)
    imgseqs.append(currentseq)

    for i, seq in enumerate(imgseqs):
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        seq = torch.Tensor(seq.transpose(0, 3, 1, 2)).cuda()
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

    for path in tqdm(reallist, desc='Processing fake files'):
        fileid = get_file_id(path)
        outpath = os.path.join(output_dir, 'real', fileid)
        feats_from_dir(path, outpath, net)

    for path in tqdm(fakelist, desc='Processing fake files'):
        fileid = get_file_id(path)
        outpath = os.path.join(output_dir, 'fake', fileid)
        feats_from_dir(path, outpath, net)


def get_file_id(path):
    return path.split('/')[-1]

    # print(files)
if __name__ == '__main__':
    main()
