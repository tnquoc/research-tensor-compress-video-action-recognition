# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# import tensorly as tl
# from tensorly.decomposition import tucker, matrix_product_state
#
# import torch
# import torchvision
# from torchvision.io import read_video, write_video
#
# video_avi, _, fps = read_video('data/UCF-101-dataset/UCF-101/Archery/v_Archery_g01_c01.avi', pts_unit='sec')
#
# video_avi = video_avi / 255
# for frame in video_avi:
#     cv2.imshow('', np.array(frame))
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#
# raise ValueError
#
#
# random_state = 12345
#
# # read video
# cap = cv2.VideoCapture('data/UCF-101-dataset/UCF-101/Archery/v_Archery_g01_c01.avi')
# # cap = cv2.VideoCapture('v_Archery_g01_c01.avi')
# vid = []
#
# count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         vid.append(frame)
#     else:
#         break
#
# cap.release()
# vid = np.array(vid)
#
# # for frame in vid:
# #     cv2.imshow('', frame)
# #     if cv2.waitKey(25) & 0xFF == ord('q'):
# #         break
# #
# # raise ValueError
#
# x = tl.tensor(vid).astype(np.float)
#
# tucker_rank = [20, 30, 2]
# re_vid = []
# for frame in vid:
#     k = tl.tensor(frame).astype(np.float)
#     core, tucker_factors = tucker(k, rank=tucker_rank, init='random', tol=10e-5, random_state=random_state)
#     re_vid.append((core, tucker_factors))
#
# for core, tucker_factors in re_vid:
#     re_frame = tl.tucker_to_tensor((core, tucker_factors)) / 255
#     cv2.imshow('', re_frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# from model import CompressByTuckerVideoRecognizer

# video_path = './v_Archery_g01_c01.avi'
# vid = read_video(video_path)
# display_video(vid)
# compressed_vid = compress_video_by_mps(vid, ranks=[1, 100, 10, 10, 1], by_frames=False)
# display_video(compressed_vid)
# print(np.mean((255 * (compressed_vid - vid)) ** 2, axis=None))
# compressed_vid = compress_video_by_mps(vid, ranks=[1, 10, 10, 1], by_frames=True)
# display_video(compressed_vid)
# print(np.mean((255 * (compressed_vid - vid)) ** 2, axis=None))

# compressed_vid = compress_video_by_tucker(vid, tucker_ranks=[100, 10, 10, 3], by_frames=False)
# display_video(compressed_vid)
# print(np.mean((255 * (compressed_vid - vid)) ** 2, axis=None))
# compressed_vid = compress_video_by_tucker(vid, tucker_ranks=[10, 10, 3], by_frames=True)
# display_video(compressed_vid)
# print(np.mean((255 * (compressed_vid - vid)) ** 2, axis=None))

# import youtube_dl
# ydl_opts = {}
# with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#     ydl.download(['https://www.youtube.com/watch?v=dfIho5iC370'])
# print(ydl)

import os

print(os.cpu_count())
