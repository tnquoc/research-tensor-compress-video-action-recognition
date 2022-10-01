import cv2
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker, matrix_product_state
from prettytable import PrettyTable


def display_video(vid):
    vid = np.array(vid)
    for frame in vid:
        cv2.imshow('', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.waitKey()


def write_video(vid, file_name, size):
    # vid = vid * 255
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_name, fourcc, 25.0, size)
    for frame in vid:
        frame = (frame * 255).astype(np.uint8)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out.write(frame)

    out.release()


def read_video(video_path, resize=None, n_frames=9999):
    cap = cv2.VideoCapture(video_path)
    vid = []
    count_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # frame = np.array(frame) / 255
        if ret:
            if resize:
                frame = cv2.resize(frame, resize)
            vid.append(frame)

            count_frames += 1
            if count_frames == n_frames:
                break
        else:
            break

    cap.release()

    return np.array(vid)


def compress_video_by_tucker(vid, tucker_ranks, by_frames=True):
    if by_frames:
        if len(tucker_ranks) != 3:
            raise ValueError('Compress by frame needs 3 values of tucker ranks')

        compressed_vid = []
        for frame in vid:
            core, factors = tucker(frame, rank=tucker_ranks, init='random', tol=10e-5, random_state=12345)
            compressed_vid.append(tl.tucker_to_tensor((core, factors)))

    else:
        if len(tucker_ranks) != 4:
            raise ValueError('Compress full video needs 4 values of tucker ranks')

        core, factors = tucker(vid, rank=tucker_ranks, init='random', tol=10e-5, random_state=12345)
        compressed_vid = tl.tucker_to_tensor((core, factors))

    return np.array(compressed_vid)


def compress_video_by_mps(vid, ranks, by_frames=True):
    if by_frames:
        if len(ranks) != 4:
            raise ValueError('Compress by frame needs 4 values of ranks')

        compressed_vid = []
        for frame in vid:
            factors = matrix_product_state(frame, rank=ranks)
            compressed_vid.append(tl.tt_to_tensor(factors))

    else:
        if len(ranks) != 5:
            raise ValueError('Compress full video needs 5 values of ranks')

        factors = matrix_product_state(vid, rank=ranks)
        compressed_vid = tl.tt_to_tensor(factors)

    return np.array(compressed_vid)


def read_ucf101_dataset_annotation(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    f.close()

    dataset = {
        'Path': [],
        'LabelInd': []
    }
    for line in lines:
        vid_path, label = line.split()
        dataset['Path'].append(vid_path)
        dataset['LabelInd'].append(label)

    return pd.DataFrame(dataset)


def normalize_video_by_frames(vid, med):
    n_frames = vid.shape[0]
    if n_frames > med:
        n_frames_to_drop = n_frames - 164
        inds_frames_to_drop = []
        head = 1
        tail = n_frames - 2
        while len(inds_frames_to_drop) < n_frames_to_drop and head <= n_frames // 2 <= tail:
            if head == tail:
                break
            inds_frames_to_drop.append(head)
            inds_frames_to_drop.append(tail)
            head += 2
            tail -= 2
        if len(inds_frames_to_drop) < n_frames_to_drop:
            i = 0
            while len(inds_frames_to_drop) < n_frames_to_drop:
                inds_frames_to_drop.append(i)
                inds_frames_to_drop.append(n_frames - i - 1)
                i += 2
        if len(inds_frames_to_drop) > n_frames_to_drop:
            inds_frames_to_drop = inds_frames_to_drop[:-1]

        inds_frames_to_keep = np.array(list(set(range(n_frames)) - set(inds_frames_to_drop)))
        vid = vid[inds_frames_to_keep]
    else:
        i = 0
        count = 0
        while len(vid) < med and count < n_frames - 1:
            gen_frame = (vid[i] + vid[i + 1]) / 2
            vid = np.insert(vid, i + 1, gen_frame, axis=0)
            count += 1
            i += 2
        if len(vid) > med:
            vid = vid[:-1]
        if len(vid) < med:
            while len(vid) < med:
                i = len(vid) - 1
                gen_frame = (vid[i - 1] + vid[i]) / 2
                vid = np.insert(vid, i + 1, gen_frame, axis=0)

    return vid


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_hankel_matrix(base_matrix, hankel_indices):
    hankel_matrix = []
    for j in range(len(hankel_indices)):
        t = base_matrix[:, hankel_indices[j]]
        hankel_matrix.append(t)
    hankel_matrix = np.vstack(hankel_matrix)
    hankel_matrix = hankel_matrix.dot(hankel_matrix.T)

    return hankel_matrix


label_to_ind = {
    'Basketball': '1',
    'BasketballDunk': '2',
    'Biking': '3',
    'CliffDiving': '4',
    'CricketBowling': '5',
    'Diving': '6',
    'Fencing': '7',
    'FloorGymnastics': '8',
    'GolfSwing': '9',
    'HorseRiding': '10',
    'IceDancing': '11',
    'LongJump': '12',
    'PoleVault': '13',
    'RopeClimbing': '14',
    'SalsaSpin': '15',
    'SkateBoarding': '16',
    'Skiing': '17',
    'Skijet': '18',
    'SoccerJuggling': '19',
    'Surfing': '20',
    'TennisSwing': '21',
    'TrampolineJumping': '22',
    'VolleyballSpiking': '23',
    'WalkingWithDog': '24'
}


ind_to_label = {
    '1': 'Basketball',
    '2': 'BasketballDunk',
    '3': 'Biking',
    '4': 'CliffDiving',
    '5': 'CricketBowling',
    '6': 'Diving',
    '7': 'Fencing',
    '8': 'FloorGymnastics',
    '9': 'GolfSwing',
    '10': 'HorseRiding',
    '11': 'IceDancing',
    '12': 'LongJump',
    '13': 'PoleVault',
    '14': 'RopeClimbing',
    '15': 'SalsaSpin',
    '16': 'SkateBoarding',
    '17': 'Skiing',
    '18': 'Skijet',
    '19': 'SoccerJuggling',
    '20': 'Surfing',
    '21': 'TennisSwing',
    '22': 'TrampolineJumping',
    '23': 'VolleyballSpiking',
    '24': 'WalkingWithDog'
}
