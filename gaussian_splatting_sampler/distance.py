import torch
from torch import linalg
import math

import pickle
import numpy as np

import math
from torch import linalg

torch.set_printoptions(precision=8)


# def Tensorlog_normal(diffs, covs):
#     """
#     diffs: Tensor (N, M, 3)
#     covs: Tensor (M, 3, 3)
#     """
#     m, d, _ = covs.shape
#     device = diffs.device
#     if d == 1:
#         precisions_chol = (torch.sqrt(1 / covs)).reshape(m, d, d)
#     else:
#         precisions_chol = torch.empty((m, d, d), device=device)
#         for k, cov in enumerate(covs):
#             try:
#                 cov_chol = linalg.cholesky(cov, upper=False)
#             except RuntimeError as e:
#                 if "singular U" in str(e):
#                     raise ValueError("协方差矩阵无法进行Cholesky分解")
#                 raise
#             eye = torch.eye(d, device=device)
#             precisions_chol[k] = linalg.solve_triangular(
#                 cov_chol, eye, upper=False, unitriangular=False
#             ).T
#
#     diag_elems = torch.diagonal(precisions_chol, dim1=1, dim2=2)
#     log_det = 2 * torch.sum(torch.log(diag_elems), dim=1) - 3 * torch.log(2 * torch.tensor(math.pi, device=diag_elems.device,
#                                                                                            dtype=diag_elems.dtype))
#     # print('log det:\n',  torch.sum(torch.log(diag_elems), dim=1))
#     log_det = log_det.view(-1, )
#
#     temp1 = torch.einsum('ijk,jkl->ijl', diffs, precisions_chol)
#     log_prob = torch.einsum('ijk,ijk->ij', temp1, temp1)
#
#     # 合成对数概率
#     log_probs = log_prob - log_det.unsqueeze(0)
#
#
#     precisions = torch.einsum('ijk,ilk->ijl', precisions_chol, precisions_chol)
#     return log_probs * 0.5, precisions
#
#
# def TensorKL(means, covs):
#     """
#     PyTorch CUDA版本的KL散度矩阵计算
#     输入均为CUDA Tensor
#     """
#     mus1, mus2 = means
#     Sigmas1, Sigmas2 = covs
#     k1, k2 = mus1.shape[0], mus2.shape[0]
#
#     d = mus1.shape[1]
#
#     diff = mus2.unsqueeze(0) - mus1.unsqueeze(1)  # 维度广播[1](@ref)
#
#     cost_matrix, precisions = Tensorlog_normal(    ## cost matrix N * M; precisions M * 3 * 3
#         diff,
#         Sigmas2
#     )
#     precisions = precisions.unsqueeze(0).repeat(k1, 1, 1, 1)
#     traces = torch.einsum('ijkl,ikl->ij', precisions, Sigmas1)
#
#     # 批量计算log_det (k1,)
#     _, log_det = torch.linalg.slogdet(2 * torch.pi * Sigmas1)  # 稳定行列式计算
#     log_det = 0.5 * log_det
#
#     # 组合最终成本矩阵
#     cost_matrix = 0.5 * (traces - log_det.unsqueeze(1) - d) + cost_matrix
#
#     return cost_matrix


def TensorKL(means, covs):
    """
    PyTorch CUDA版本的KL散度矩阵计算
    输入均为CUDA Tensor
    """
    mus1, mus2 = means # (N1, 3), (N2,3)
    Sigmas1, Sigmas2 = covs # (N1, 3, 3), # (N2, 3, 3)

    mean_diff = mus2.unsqueeze(0) - mus1.unsqueeze(1)  # dim (N1, N2, 3)
    cov_diff = Sigmas2.unsqueeze(0) - Sigmas1.unsqueeze(1)  # dim (N1, N2, 3, 3)

    # 组合最终成本矩阵
    cost_matrix = torch.linalg.norm(mean_diff, dim=-1) ** 2 + torch.linalg.matrix_norm(cov_diff, ord='fro') ** 2

    return cost_matrix
