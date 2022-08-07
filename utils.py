import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from helpers import read_video


class UCF101Dataset:
    def __init__(self, length, dataset, base_data_path):
        self.length = length
        self.dataset = dataset
        # self.base_data_path = 'data/UCF-101-dataset/UCF-101/'
        self.base_data_path = base_data_path
        self.rmv_list = []

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vid_path, label = self.dataset[idx]
        # vid_path = self.base_data_path + vid_path
        vid_path = os.path.join(self.base_data_path, vid_path)
        # vid, _, _ = torchvision.io.read_video(vid_path)
        # vid = read_video(vid_path, resize=(240, 240), n_frames=100)
        vid = read_video(vid_path)
        vid = torch.from_numpy(vid)

        sample = {
            'video': vid.type(torch.FloatTensor) / 255,
            'label': int(label)
        }

        return sample



