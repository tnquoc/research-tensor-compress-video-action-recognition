import math

import numpy as np
import tensorly as tl
import torch
import torch.nn as nn


class CompressByTuckerVideoRecognizer(nn.Module):
    def __init__(self, input_shape, tucker_ranks, n_classes):
        super(CompressByTuckerVideoRecognizer, self).__init__()
        self.tucker_ranks = tucker_ranks

        self.core_tensor_fc_1 = nn.Linear(int(np.prod(np.array(tucker_ranks))), 100)
        self.core_tensor_fc_2 = nn.Linear(100, 60)
        self.core_tensor_fc_3 = nn.Linear(60, n_classes)
        self.core_tensor_bn_1 = nn.BatchNorm1d(100)
        self.core_tensor_bn_2 = nn.BatchNorm1d(60)

        self.factor_matrix_1_fc_1 = nn.Linear(input_shape[0] * tucker_ranks[0], 100)
        self.factor_matrix_1_fc_2 = nn.Linear(100, 60)
        self.factor_matrix_1_fc_3 = nn.Linear(60, n_classes)
        self.factor_matrix_1_bn_1 = nn.BatchNorm1d(100)
        self.factor_matrix_1_bn_2 = nn.BatchNorm1d(60)

        self.factor_matrix_2_fc_1 = nn.Linear(input_shape[1] * tucker_ranks[1], 100)
        self.factor_matrix_2_fc_2 = nn.Linear(100, 60)
        self.factor_matrix_2_fc_3 = nn.Linear(60, n_classes)
        self.factor_matrix_2_bn_1 = nn.BatchNorm1d(100)
        self.factor_matrix_2_bn_2 = nn.BatchNorm1d(60)

        self.factor_matrix_3_fc_1 = nn.Linear(input_shape[2] * tucker_ranks[2], 100)
        self.factor_matrix_3_fc_2 = nn.Linear(100, 60)
        self.factor_matrix_3_fc_3 = nn.Linear(60, n_classes)
        self.factor_matrix_3_bn_1 = nn.BatchNorm1d(100)
        self.factor_matrix_3_bn_2 = nn.BatchNorm1d(60)

        self.factor_matrix_4_fc_1 = nn.Linear(input_shape[3] * tucker_ranks[3], 100)
        self.factor_matrix_4_fc_2 = nn.Linear(100, 60)
        self.factor_matrix_4_fc_3 = nn.Linear(60, n_classes)
        self.factor_matrix_4_bn_1 = nn.BatchNorm1d(100)
        self.factor_matrix_4_bn_2 = nn.BatchNorm1d(60)

        self.relu = nn.ReLU()

    def forward_core_tensor(self, x):
        x = self.core_tensor_fc_1(x)
        x = self.core_tensor_bn_1(x)
        x = self.relu(x)
        x = self.core_tensor_fc_2(x)
        x = self.core_tensor_bn_2(x)
        x = self.relu(x)
        x = self.core_tensor_fc_3(x)

        return x

    def forward_factor_matrix_1(self, x):
        x = self.factor_matrix_1_fc_1(x)
        x = self.factor_matrix_1_bn_1(x)
        x = self.relu(x)
        x = self.factor_matrix_1_fc_2(x)
        x = self.factor_matrix_1_bn_2(x)
        x = self.relu(x)
        x = self.factor_matrix_1_fc_3(x)

        return x

    def forward_factor_matrix_2(self, x):
        x = self.factor_matrix_2_fc_1(x)
        x = self.factor_matrix_2_bn_1(x)
        x = self.relu(x)
        x = self.factor_matrix_2_fc_2(x)
        x = self.factor_matrix_2_bn_2(x)
        x = self.relu(x)
        x = self.factor_matrix_2_fc_3(x)

        return x

    def forward_factor_matrix_3(self, x):
        x = self.factor_matrix_3_fc_1(x)
        x = self.factor_matrix_3_bn_1(x)
        x = self.relu(x)
        x = self.factor_matrix_3_fc_2(x)
        x = self.factor_matrix_3_bn_2(x)
        x = self.relu(x)
        x = self.factor_matrix_3_fc_3(x)

        return x

    def forward_factor_matrix_4(self, x):
        x = self.factor_matrix_4_fc_1(x)
        x = self.factor_matrix_4_bn_1(x)
        x = self.relu(x)
        x = self.factor_matrix_4_fc_2(x)
        x = self.factor_matrix_4_bn_2(x)
        x = self.relu(x)
        x = self.factor_matrix_4_fc_3(x)

        return x

    def forward(self, data):
        cores, factor_matrices_1, factor_matrices_2, factor_matrices_3, factor_matrices_4 = data
        out_core_tensor = self.forward_core_tensor(cores)
        out_factor_matrix_1 = self.forward_factor_matrix_1(factor_matrices_1)
        out_factor_matrix_2 = self.forward_factor_matrix_2(factor_matrices_2)
        out_factor_matrix_3 = self.forward_factor_matrix_3(factor_matrices_3)
        out_factor_matrix_4 = self.forward_factor_matrix_4(factor_matrices_4)

        sum_out = torch.sum(torch.stack(
            [out_core_tensor, out_factor_matrix_1, out_factor_matrix_2, out_factor_matrix_3, out_factor_matrix_4]),
                            dim=0)

        return sum_out
