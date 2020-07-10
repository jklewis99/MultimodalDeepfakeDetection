import torch
import sys
from torch import nn

from model import LipNet
from utils import *


if(__name__ == '__main__'):
    opt = __import__('options')

    device = f'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    model = LipNet()
    model = model.to(device)
    net = nn.DataParallel(model).to(device)

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(
            opt.weights, map_location=torch.device(device))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items(
        ) if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items(
        ) if not k in pretrained_dict.keys()]
        print(
            'loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    video, img_p = load_video(sys.argv[1], device=device)
    y, yf = model(video[None, ...].to(device))

    print(y.shape, yf.shape)
