import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, matrix_product_state

import cv2
import torch
# from torchsummary import summary

from helpers import read_video, write_video, display_video, compress_video_by_mps, compress_video_by_tucker, \
    read_ucf101_dataset_annotation, normalize_video_by_frames
from utils import UCF101Dataset
from operator import itemgetter

from multiprocessing import Pool

from model import CompressByTuckerVideoRecognizer


# def collate_fn(data):
#     labels = [int(d['label']) for d in data]
#     ind = range(len(labels))
#     videos = list(itemgetter(*ind)([d['video'] for d in data]))
#     targets = torch.zeros(len(labels), 24).long()
#     for i in range(len(labels)):
#         targets[i][labels[i] - 1] = 1
#
#     return videos, targets

def collate_fn(data):
    select_inds = []
    for i in range(len(data)):
        vid, label = data[i]['video'], data[i]['label']
        if 2 * vid.shape[0] - 1 < 164:
            continue
        select_inds.append(i)
    videos = list(itemgetter(*select_inds)([normalize_video_by_frames(d['video'], 164) for d in data]))
    videos = torch.stack(videos, 0)

    labels = list(itemgetter(*select_inds)([d['label'] for d in data]))
    targets = torch.zeros(len(labels), 24)
    for i in range(len(labels)):
        targets[i][labels[i] - 1] = 1

    return videos, targets


def get_frames(sample):
    vid, label = sample['video'], sample['label']
    return vid.shape[0]


if __name__ == '__main__':
    # train_annotation_file = './data/UCF-101-dataset/UCF101_Action_detection_splits/trainlist01.txt'
    # test_annotation_file = './data/UCF-101-dataset/UCF101_Action_detection_splits/testlist01.txt'
    train_annotation_file = "D:/personal/python_projects/data/UCF-101-dataset/UCF101_Action_detection_splits/trainlist01.txt"
    base_data_path = 'D:/personal/python_projects/data/UCF-101-dataset/UCF-101'

    train_dataset = read_ucf101_dataset_annotation(train_annotation_file)[:64]
    # test_dataset = read_ucf101_dataset_annotation(test_annotation_file)

    train_ucf_dataset = UCF101Dataset(length=len(train_dataset), dataset=train_dataset, base_data_path=base_data_path)
    print(len(train_dataset), len(train_ucf_dataset))

    # n_frames = []
    # for i in range(len(train_ucf_dataset)):
    #     sample = train_ucf_dataset[i]
    #     vid, label = sample['video'], sample['label']
    #     k = vid.shape[0]
    #     print(i, vid.shape[0], 2 * vid.shape[0] - 1)
    #     # if 2 * k - 1 < 164:
    #     #     continue
    #     # display_video(vid)
    #     vid = normalize_video_by_frames(vid, 164)
    #     # n_frames.append(vid.shape[0])
    #     # display_video(vid)
    #     # print(k, vid.shape[0])
    # # n_frames = np.array(n_frames)
    # # print(len(n_frames), np.median(n_frames))
    # raise ValueError

    num_epochs = 10
    batch_size = 8
    learning_rate = 1e-3

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=train_ucf_dataset,
                                              batch_size=batch_size,
                                              num_workers=2,
                                              collate_fn=collate_fn,
                                              shuffle=True)
    input_shape = (164, 240, 320, 3)
    tucker_ranks = (10, 15, 20, 3)
    model = CompressByTuckerVideoRecognizer(input_shape=input_shape, tucker_ranks=tucker_ranks, n_classes=24)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for i, samples in enumerate(data_loader):
            # Forward pass
            vids, labels = samples
            outputs = model(vids)
            loss = criterion(outputs, labels)

            # Backprop and optimize
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, len(data_loader), loss.item()))
            # if (i + 1) % 10 == 0:
            #     print("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}"
            #           .format(epoch + 1, num_epochs, i + 1, len(data_loader), loss.item()))

    torch.save(model.state_dict(), 'model_weights.pth')
