# import torch
# from torchvideotransforms import video_transforms, volume_transforms
# from helpers import read_video, display_video
#
# # from torchvision import datasets, transforms, models
#
# video_transform_list = [
#     video_transforms.RandomRotation(5),
#     video_transforms.RandomHorizontalFlip(),
#     # video_transforms.RandomRotation(15),
#     # video_transforms.RandomHorizontalFlip(),
#     # video_transforms.RandomVerticalFlip()
# ]
# transforms = video_transforms.Compose(video_transform_list)
#
# path = 'D:/personal/python_projects/data/UCF-101-dataset/UCF-101/Basketball/v_Basketball_g01_c01.avi'
# video = read_video(path)
# # display_video(video)
# aug_vids = []
# for i in range(10):
#     aug_video = transforms(video)
#     aug_vids.append(aug_video)
#
# for vid in aug_vids:
#     display_video(vid)

import sys

import csv
import json

import pandas as pd
from sklearn.model_selection import train_test_split


from helpers import read_ucf101_dataset_annotation

# sys.path.append('..')


if __name__ == '__main__':
    # params = json.load(open('config.json', 'r'))

    need_to_augment = 'generated_files/train_set.csv'
    augmented_file = 'generated_files/augmented_train_set_1.csv'
    # df = read_ucf101_dataset_annotation(need_to_augment)
    df = pd.read_csv(need_to_augment)

    with open(augmented_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Path', 'Label', 'LabelInd', 'IsAugment'])
    for i in range(len(df)):
        n_aug = 5

        for j in range(n_aug):
            with open(augmented_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if j == 0:
                    is_augment = 0
                else:
                    is_augment = 1
                writer.writerow([df['Path'][i], df['Label'][i], df['LabelInd'][i], is_augment])

    # df = pd.read_csv('generated_files/test_set.csv')
    # test_set, val_set = train_test_split(df, test_size=0.35)
    # test_set.to_csv('generated_files/test_set.csv')
    # val_set.to_csv('generated_files/val_set.csv')
