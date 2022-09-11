import os
import time
import warnings

import torch
import json
import numpy as np
import pandas as pd

from helpers import read_video, display_video, ind_to_label, normalize_video_by_frames

from utils import get_loaders
from model import CompressByTuckerVideoRecognizer
import tensorly as tl
from tensorly.decomposition import tucker

tl.set_backend('pytorch')

warnings.filterwarnings("ignore")


def test(params):
    train_loader, val_loader, test_loader = get_loaders(params)

    input_shape = (164, 240, 320, 3)
    tucker_ranks = params['tucker_ranks']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model = CompressByTuckerVideoRecognizer(input_shape=input_shape, tucker_ranks=tucker_ranks, n_classes=24).to(device)
    checkpoint = torch.load('training/test/checkpoint_epoch14.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    count_top_1 = 0
    count_top_5 = 0
    count = 0

    # x = torch.Tensor([[1, 2, 3], [3, 4, 5]])
    # a, b = torch.topk(x, dim=1, k=2)
    # print(a, b)
    # raise ValueError

    print('Start testing...')
    for i, samples in enumerate(val_loader):
        # Get data
        cores, factor_matrices_1, factor_matrices_2, factor_matrices_3, factor_matrices_4, labels = samples

        # Forward pass
        outputs = model((cores.to(device),
                         factor_matrices_1.to(device),
                         factor_matrices_2.to(device),
                         factor_matrices_3.to(device),
                         factor_matrices_4.to(device)))
        outputs = torch.nn.functional.softmax(outputs.to('cpu'), dim=1)
        labels = labels.detach().numpy()
        predicts = torch.argmax(outputs, dim=1).detach().numpy()
        _, predicts_top_5 = torch.topk(outputs, dim=1, k=5)
        predicts_top_5 = predicts_top_5.detach().numpy()
        # print(predicts_top_5)

        # print('-----------------------')
        # print('gts', labels)
        # print('pre', predicts)
        # print(torch.topk(outputs, dim=1, k=5))
        count_top_1 += len(np.where(labels == predicts)[0])
        for j in range(len(predicts_top_5)):
            if labels[j] in predicts_top_5[j]:
                count_top_5 += 1

    # print(count)
    print(count_top_1, count_top_5)
    print('Done')


if __name__ == '__main__':
    params = json.load(open('config.json', 'r'))

    test(params)

