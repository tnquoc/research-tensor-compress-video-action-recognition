import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from operator import itemgetter
import tensorly as tl
from tensorly.decomposition import tucker, non_negative_tucker, non_negative_tucker_hals
from functools import partial
from scipy.linalg import hankel

from torchvideotransforms import video_transforms, volume_transforms
from helpers import read_video, normalize_video_by_frames, read_ucf101_dataset_annotation, display_video, get_hankel_matrix

tl.set_backend('pytorch')


class UCF101Dataset(Dataset):
    def __init__(self, length, dataset, base_data_path, transform=None):
        self.length = length
        self.dataset = dataset
        self.base_data_path = base_data_path
        self.rmv_list = []
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vid_path, label = self.dataset['Path'][idx], self.dataset['LabelInd'][idx]
        vid_path = os.path.join(self.base_data_path, vid_path)
        vid = read_video(vid_path, resize=(160, 120))
        if 'IsAugment' in self.dataset and self.dataset['IsAugment'][idx]:
            vid = self.transform(vid)
            vid = np.array(vid)
        # display_video(vid)
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


def collate_fn2(data, tucker_ranks, device):
    labels = []
    list_core = []
    list_factor_matrix_1 = []
    list_factor_matrix_2 = []
    list_factor_matrix_3 = []
    list_factor_matrix_4 = []

    for i in range(len(data)):
        vid, label = data[i]['video'], data[i]['label']
        labels.append(label)
        if device != 'cpu':
            # print(torch.cuda.get_device_name(device))
            vid = normalize_video_by_frames(vid, 164).to(device)
            core, factors = tucker(vid, rank=tucker_ranks, init='random', tol=10e-5, random_state=12345)
            list_core.append(tl.tensor_to_vec(core.cpu()))
            list_factor_matrix_1.append(tl.tensor_to_vec(factors[0].cpu()))
            list_factor_matrix_2.append(tl.tensor_to_vec(factors[1].cpu()))
            list_factor_matrix_3.append(tl.tensor_to_vec(factors[2].cpu()))
            list_factor_matrix_4.append(tl.tensor_to_vec(factors[3].cpu()))
            del vid
            del core
            del factors
        else:
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


def collate_fn3(data, tucker_ranks, hankel_indices, device):
    labels = []
    list_core = []
    list_factor_matrix_1 = []
    list_factor_matrix_2 = []
    list_factor_matrix_3 = []
    list_factor_matrix_4 = []
    list_hankel_matrix = []

    for i in range(len(data)):
        vid, label = data[i]['video'], data[i]['label']
        labels.append(label)
        if device != 'cpu':
            # print(torch.cuda.get_device_name(device))
            vid = normalize_video_by_frames(vid, 164).to(device)
            core, factors = tucker(vid, rank=tucker_ranks, init='random', tol=10e-5, random_state=12345)
            core_unfold = tl.unfold(core.cpu(), 0).detach().numpy().T
            hankel_matrix = get_hankel_matrix(core_unfold, hankel_indices)
            list_core.append(tl.tensor_to_vec(core.cpu()))
            list_factor_matrix_1.append(tl.tensor_to_vec(factors[0].cpu()))
            list_factor_matrix_2.append(tl.tensor_to_vec(factors[1].cpu()))
            list_factor_matrix_3.append(tl.tensor_to_vec(factors[2].cpu()))
            list_factor_matrix_4.append(tl.tensor_to_vec(factors[3].cpu()))
            list_hankel_matrix.append(torch.from_numpy(hankel_matrix))
            del vid
            del core
            del factors
        else:
            vid = normalize_video_by_frames(vid, 164)
            core, factors = tucker(vid, rank=tucker_ranks, init='random', tol=10e-5, random_state=12345)
            core_unfold = tl.unfold(core, 0).detach().numpy().T
            hankel_matrix = get_hankel_matrix(core_unfold, hankel_indices)
            list_core.append(tl.tensor_to_vec(core))
            list_factor_matrix_1.append(tl.tensor_to_vec(factors[0]))
            list_factor_matrix_2.append(tl.tensor_to_vec(factors[1]))
            list_factor_matrix_3.append(tl.tensor_to_vec(factors[2]))
            list_factor_matrix_4.append(tl.tensor_to_vec(factors[3]))
            list_hankel_matrix.append(torch.from_numpy(hankel_matrix))

    list_cores = torch.stack(list_core, 0)
    list_factor_matrix_1 = torch.stack(list_factor_matrix_1, 0)
    list_factor_matrix_2 = torch.stack(list_factor_matrix_2, 0)
    list_factor_matrix_3 = torch.stack(list_factor_matrix_3, 0)
    list_factor_matrix_4 = torch.stack(list_factor_matrix_4, 0)
    list_hankel_matrix = torch.stack(list_hankel_matrix, 0)
    if len(labels) == 1:
        list_cores = list_cores.unsqueeze(0)
        list_factor_matrix_1 = list_factor_matrix_1.unsqueeze(0)
        list_factor_matrix_2 = list_factor_matrix_2.unsqueeze(0)
        list_factor_matrix_3 = list_factor_matrix_3.unsqueeze(0)
        list_factor_matrix_4 = list_factor_matrix_4.unsqueeze(0)
        list_hankel_matrix = list_hankel_matrix.unsqueeze(0)

    return list_cores, list_factor_matrix_1, list_factor_matrix_2, list_factor_matrix_3, list_factor_matrix_4, \
           list_hankel_matrix, torch.from_numpy(np.array(labels)).long()


def get_loaders(params, device='cpu'):
    video_transform_list = [
        video_transforms.RandomRotation(5),
        video_transforms.RandomHorizontalFlip(),
    ]
    transforms = video_transforms.Compose(video_transform_list)

    if '.csv' in params['train_set']:
        train_df = pd.read_csv(params['train_set'])
    else:
        train_df = read_ucf101_dataset_annotation(params['train_set'])

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(params['val_set'])
    test_df = pd.read_csv(params['test_set'])

    train_ucf_dataset = UCF101Dataset(length=len(train_df), dataset=train_df, base_data_path=params['data_dir'],
                                      transform=transforms)
    val_ucf_dataset = UCF101Dataset(length=len(val_df), dataset=val_df, base_data_path=params['data_dir'])
    test_ucf_dataset = UCF101Dataset(length=len(test_df), dataset=test_df, base_data_path=params['data_dir'])
    hankel_indices = hankel(range(1, 16), range(15, 17)).T - 1

    train_loader = DataLoader(dataset=train_ucf_dataset,
                              batch_size=params['batch_size'],
                              num_workers=0,
                              collate_fn=partial(collate_fn3, tucker_ranks=params['tucker_ranks'],
                                                 hankel_indices=hankel_indices, device=device),
                              shuffle=True)

    val_loader = DataLoader(dataset=val_ucf_dataset,
                            batch_size=params['batch_size'],
                            num_workers=0,
                            collate_fn=partial(collate_fn3, tucker_ranks=params['tucker_ranks'],
                                               hankel_indices=hankel_indices, device=device))

    test_loader = DataLoader(dataset=test_ucf_dataset,
                             batch_size=params['batch_size'],
                             num_workers=0,
                             collate_fn=partial(collate_fn3, tucker_ranks=params['tucker_ranks'],
                                                hankel_indices=hankel_indices, device=device))

    return train_loader, val_loader, test_loader
