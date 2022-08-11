import math

import numpy as np
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tucker, matrix_product_state


# class CompressByTuckerVideoRecognizer(nn.Module):
#     def __init__(self, input_shape, tucker_ranks, n_classes):
#     # def __init__(self, tucker_ranks, n_classes):
#         super(CompressByTuckerVideoRecognizer, self).__init__()
#         self.tucker_ranks = tucker_ranks
#
#         self.core_tensor_fc_1 = nn.Linear(int(np.prod(np.array(tucker_ranks))), 100)
#         # self.core_tensor_fc_1 = nn.LazyLinear(100)
#         self.core_tensor_fc_2 = nn.Linear(100, 60)
#         self.core_tensor_fc_3 = nn.Linear(60, n_classes)
#
#         self.factor_matrix_1_fc_1 = nn.Linear(input_shape[0] * tucker_ranks[0], 100)
#         # self.factor_matrix_1_fc_1 = nn.LazyLinear(100)
#         self.factor_matrix_1_fc_2 = nn.Linear(100, 60)
#         self.factor_matrix_1_fc_3 = nn.Linear(60, n_classes)
#
#         self.factor_matrix_2_fc_1 = nn.Linear(input_shape[1] * tucker_ranks[1], 100)
#         # self.factor_matrix_2_fc_1 = nn.LazyLinear(100)
#         self.factor_matrix_2_fc_2 = nn.Linear(100, 60)
#         self.factor_matrix_2_fc_3 = nn.Linear(60, n_classes)
#
#         self.factor_matrix_3_fc_1 = nn.Linear(input_shape[2] * tucker_ranks[2], 100)
#         # self.factor_matrix_3_fc_1 = nn.LazyLinear(100)
#         self.factor_matrix_3_fc_2 = nn.Linear(100, 60)
#         self.factor_matrix_3_fc_3 = nn.Linear(60, n_classes)
#
#         self.factor_matrix_4_fc_1 = nn.Linear(input_shape[3] * tucker_ranks[3], 100)
#         # self.factor_matrix_4_fc_1 = nn.LazyLinear(100)
#         self.factor_matrix_4_fc_2 = nn.Linear(100, 60)
#         self.factor_matrix_4_fc_3 = nn.Linear(60, n_classes)
#
#     def forward_core_tensor(self, x):
#         # x = tl.tensor_to_vec(x)
#         x = self.core_tensor_fc_1(x)
#         x = self.core_tensor_fc_2(x)
#         x = self.core_tensor_fc_3(x)
#         x = nn.functional.softmax(x, dim=0)
#
#         return x
#
#     def forward_factor_matrix_1(self, x):
#         # x = tl.tensor_to_vec(x)
#         x = self.factor_matrix_1_fc_1(x)
#         x = self.factor_matrix_1_fc_2(x)
#         x = self.factor_matrix_1_fc_3(x)
#         x = nn.functional.softmax(x, dim=0)
#
#         return x
#
#     def forward_factor_matrix_2(self, x):
#         # x = tl.tensor_to_vec(x)
#         x = self.factor_matrix_2_fc_1(x)
#         x = self.factor_matrix_2_fc_2(x)
#         x = self.factor_matrix_2_fc_3(x)
#         x = nn.functional.softmax(x, dim=0)
#
#         return x
#
#     def forward_factor_matrix_3(self, x):
#         # x = tl.tensor_to_vec(x)
#         x = self.factor_matrix_3_fc_1(x)
#         x = self.factor_matrix_3_fc_2(x)
#         x = self.factor_matrix_3_fc_3(x)
#         x = nn.functional.softmax(x, dim=0)
#
#         return x
#
#     def forward_factor_matrix_4(self, x):
#         # x = tl.tensor_to_vec(x)
#         x = self.factor_matrix_4_fc_1(x)
#         x = self.factor_matrix_4_fc_2(x)
#         x = self.factor_matrix_4_fc_3(x)
#         x = nn.functional.softmax(x, dim=0)
#
#         return x
#
#     # def forward(self, x):
#     #     # x = x[0]
#     #     print(x)
#     #     raise ValueError
#     #     core, factors = tucker(np.array(x), rank=self.tucker_ranks, init='random', tol=10e-5, random_state=12345)
#     #     core = torch.from_numpy(tl.tensor_to_vec(core))
#     #     factors = [torch.from_numpy(tl.tensor_to_vec(factor)) for factor in factors]
#     #
#     #     out_core_tensor = self.forward_core_tensor(core)
#     #     out_factor_matrix_1 = self.forward_factor_matrix_1(factors[0])
#     #     out_factor_matrix_2 = self.forward_factor_matrix_2(factors[1])
#     #     out_factor_matrix_3 = self.forward_factor_matrix_3(factors[2])
#     #     out_factor_matrix_4 = self.forward_factor_matrix_4(factors[3])
#     #
#     #     sum_out = torch.sum(torch.stack(
#     #         [out_core_tensor, out_factor_matrix_1, out_factor_matrix_2, out_factor_matrix_3, out_factor_matrix_4]),
#     #                         dim=0)
#     #
#     #     return nn.functional.softmax(sum_out)
#
#     def forward(self, x):
#         output = torch.zeros(len(x), 24)
#         for i in range(len(x)):
#             core, factors = tucker(np.array(x[i]), rank=self.tucker_ranks, init='random', tol=10e-5, random_state=12345)
#             core = torch.from_numpy(tl.tensor_to_vec(core))
#             factors = [torch.from_numpy(tl.tensor_to_vec(factor)) for factor in factors]
#
#             out_core_tensor = self.forward_core_tensor(core)
#             out_factor_matrix_1 = self.forward_factor_matrix_1(factors[0])
#             out_factor_matrix_2 = self.forward_factor_matrix_2(factors[1])
#             out_factor_matrix_3 = self.forward_factor_matrix_3(factors[2])
#             out_factor_matrix_4 = self.forward_factor_matrix_4(factors[3])
#
#             sum_out = torch.sum(torch.stack(
#                 [out_core_tensor, out_factor_matrix_1, out_factor_matrix_2, out_factor_matrix_3, out_factor_matrix_4]),
#                                 dim=0)
#
#             output[i] = nn.functional.softmax(sum_out)
#
#         return output


class CompressByTuckerVideoRecognizer(nn.Module):
    def __init__(self, input_shape, tucker_ranks, n_classes):
        super(CompressByTuckerVideoRecognizer, self).__init__()
        self.tucker_ranks = tucker_ranks

        self.core_tensor_fc_1 = nn.Linear(int(np.prod(np.array(tucker_ranks))), 100)
        self.core_tensor_fc_2 = nn.Linear(100, 60)
        self.core_tensor_fc_3 = nn.Linear(60, n_classes)

        self.factor_matrix_1_fc_1 = nn.Linear(input_shape[0] * tucker_ranks[0], 100)
        self.factor_matrix_1_fc_2 = nn.Linear(100, 60)
        self.factor_matrix_1_fc_3 = nn.Linear(60, n_classes)

        self.factor_matrix_2_fc_1 = nn.Linear(input_shape[1] * tucker_ranks[1], 100)
        self.factor_matrix_2_fc_2 = nn.Linear(100, 60)
        self.factor_matrix_2_fc_3 = nn.Linear(60, n_classes)

        self.factor_matrix_3_fc_1 = nn.Linear(input_shape[2] * tucker_ranks[2], 100)
        self.factor_matrix_3_fc_2 = nn.Linear(100, 60)
        self.factor_matrix_3_fc_3 = nn.Linear(60, n_classes)

        self.factor_matrix_4_fc_1 = nn.Linear(input_shape[3] * tucker_ranks[3], 100)
        self.factor_matrix_4_fc_2 = nn.Linear(100, 60)
        self.factor_matrix_4_fc_3 = nn.Linear(60, n_classes)

    def forward_core_tensor(self, x):
        x = self.core_tensor_fc_1(x)
        x = self.core_tensor_fc_2(x)
        x = self.core_tensor_fc_3(x)
        x = nn.functional.softmax(x, dim=0)

        return x

    def forward_factor_matrix_1(self, x):
        x = self.factor_matrix_1_fc_1(x)
        x = self.factor_matrix_1_fc_2(x)
        x = self.factor_matrix_1_fc_3(x)
        x = nn.functional.softmax(x, dim=0)

        return x

    def forward_factor_matrix_2(self, x):
        x = self.factor_matrix_2_fc_1(x)
        x = self.factor_matrix_2_fc_2(x)
        x = self.factor_matrix_2_fc_3(x)
        x = nn.functional.softmax(x, dim=0)

        return x

    def forward_factor_matrix_3(self, x):
        # x = tl.tensor_to_vec(x)
        x = self.factor_matrix_3_fc_1(x)
        x = self.factor_matrix_3_fc_2(x)
        x = self.factor_matrix_3_fc_3(x)
        x = nn.functional.softmax(x, dim=0)

        return x

    def forward_factor_matrix_4(self, x):
        x = self.factor_matrix_4_fc_1(x)
        x = self.factor_matrix_4_fc_2(x)
        x = self.factor_matrix_4_fc_3(x)
        x = nn.functional.softmax(x, dim=0)

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
