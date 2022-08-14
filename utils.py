import os

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from operator import itemgetter
import tensorly as tl
from tensorly.decomposition import tucker
from functools import partial

from helpers import read_video, normalize_video_by_frames, read_ucf101_dataset_annotation

tl.set_backend('pytorch')


class UCF101Dataset(Dataset):
    def __init__(self, length, dataset, base_data_path):
        self.length = length
        self.dataset = dataset
        self.base_data_path = base_data_path
        self.rmv_list = []

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vid_path, label = self.dataset['Path'][idx], self.dataset['LabelInd'][idx]
        vid_path = os.path.join(self.base_data_path, vid_path)
        # vid_path = self.base_data_path + '/' + vid_path
        # vid, _, _ = torchvision.io.read_video(vid_path)
        # vid = read_video(vid_path, resize=(240, 240), n_frames=100)
        vid = read_video(vid_path)
        vid = torch.from_numpy(vid)

        sample = {
            'video': vid.type(torch.FloatTensor) / 255,
            'label': int(label) - 1,
            'path': vid_path
        }

        return sample


def collate_fn1(data):
    inds = []
    labels = []
    for i in range(len(data)):
        vid, label = data[i]['video'], data[i]['label']
        inds.append(i)
        labels.append(label - 1)
    videos = list(itemgetter(*inds)([normalize_video_by_frames(d['video'], 164) for d in data]))
    videos = torch.stack(videos, 0)
    if len(labels) == 1:
        videos = videos.unsqueeze(0)

    return videos, torch.from_numpy(np.array(labels))


def collate_fn2(data, tucker_ranks):
    labels = []
    list_core = []
    list_factor_matrix_1 = []
    list_factor_matrix_2 = []
    list_factor_matrix_3 = []
    list_factor_matrix_4 = []

    for i in range(len(data)):
        vid, label = data[i]['video'], data[i]['label']
        labels.append(label)
        vid = normalize_video_by_frames(vid, 164)
        core, factors = tucker(vid, rank=tucker_ranks, init='random', tol=10e-5, random_state=12345)
        list_core.append(tl.tensor_to_vec(core))
        list_factor_matrix_1.append(tl.tensor_to_vec(factors[0]))
        list_factor_matrix_2.append(tl.tensor_to_vec(factors[1]))
        list_factor_matrix_3.append(tl.tensor_to_vec(factors[2]))
        list_factor_matrix_4.append(tl.tensor_to_vec(factors[3]))

    list_cores = torch.stack(list_core, 0)
    list_factor_matrix_1 = torch.stack(list_factor_matrix_1, 0)
    list_factor_matrix_2 = torch.stack(list_factor_matrix_2, 0)
    list_factor_matrix_3 = torch.stack(list_factor_matrix_3, 0)
    list_factor_matrix_4 = torch.stack(list_factor_matrix_4, 0)
    if len(labels) == 1:
        list_cores = list_cores.unsqueeze(0)
        list_factor_matrix_1 = list_factor_matrix_1.unsqueeze(0)
        list_factor_matrix_2 = list_factor_matrix_2.unsqueeze(0)
        list_factor_matrix_3 = list_factor_matrix_3.unsqueeze(0)
        list_factor_matrix_4 = list_factor_matrix_4.unsqueeze(0)

    return list_cores, list_factor_matrix_1, list_factor_matrix_2, list_factor_matrix_3, list_factor_matrix_4, \
           torch.from_numpy(np.array(labels)).long()


def get_loaders(params):
    train_df = pd.read_csv(params['train_set'])[:32]
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(params['val_set'])[:16]
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    test_df = pd.read_csv(params['test_set'])[:16]

    train_ucf_dataset = UCF101Dataset(length=len(train_df), dataset=train_df, base_data_path=params['data_dir'])
    val_ucf_dataset = UCF101Dataset(length=len(val_df), dataset=val_df, base_data_path=params['data_dir'])
    test_ucf_dataset = UCF101Dataset(length=len(test_df), dataset=test_df, base_data_path=params['data_dir'])

    train_loader = DataLoader(dataset=train_ucf_dataset,
                              batch_size=params['batch_size'],
                              num_workers=2,
                              collate_fn=partial(collate_fn2, tucker_ranks=params['tucker_ranks']),
                              # collate_fn=collate_fn1,
                              shuffle=True)

    val_loader = DataLoader(dataset=val_ucf_dataset,
                            batch_size=params['batch_size'],
                            num_workers=2,
                            collate_fn=partial(collate_fn2, tucker_ranks=params['tucker_ranks']),
                            # collate_fn=collate_fn1,
                            shuffle=True)

    test_loader = DataLoader(dataset=test_ucf_dataset,
                             batch_size=params['batch_size'],
                             num_workers=2,
                             collate_fn=partial(collate_fn2, tucker_ranks=params['tucker_ranks']),
                             # collate_fn=collate_fn1,
                             shuffle=True)

    return train_loader, val_loader, test_loader
