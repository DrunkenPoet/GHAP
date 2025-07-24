import time
import numpy as np
import math
import torch
import torch.nn as nn
from .utils import balanced_kd_split
from .distance import TensorKL
from tqdm import tqdm

torch.set_printoptions(precision=8)  # 设置显示8位小数


def gaussian_model_reduction(gaussians, ratio, random_seed=42):
    def loading(gaussians, device):
        # 直接在 GPU 上保留张量，避免不必要的 CPU <-> GPU 转移
        means = gaussians.get_xyz.detach().to(device)
        covs = gaussians.get_covariance().detach().to(device)
        weights = gaussians.get_opacity.detach().to(device).reshape(-1)
        return means, weights / weights.sum(), covs

    device = gaussians.get_xyz.detach
    means, weights, covs = loading(gaussians, device)

    downsample_num = int(ratio) if ratio > 1 else max(1, int(means.shape[0] * ratio))
    torch.manual_seed(random_seed)

    kd_reduced = KdGMR_CTD(means, covs, weights, downsample_num,
                            tol=1e-5, max_iter=100, random_state=random_seed)
    kd_reduced.iterative()

    # 收集保留索引
    record_indices = []
    for i, red in enumerate(kd_reduced.reduction_list):
        origin_idx = kd_reduced.index_list[i].to(device)
        inner_idx = torch.argmin(red.cost_matrix, dim=0).to(device)
        record_indices.append(origin_idx[inner_idx])
    keep_idx = torch.cat(record_indices)

    # 构建新的 GaussianModel
    from scene import GaussianModel
    new_model = GaussianModel(sh_degree=3)
    with torch.no_grad():
        for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
            data = getattr(gaussians, attr)[keep_idx]
            setattr(new_model, attr, nn.Parameter(data.clone()))
    return new_model


class GMM_Sampler:
    def __init__(self, means, covs, weights, n, tol=1e-5, max_iter=100, random_state=0, init_method=None,
                 means_init=None, covs_init=None, weights_init=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.means = means.to(self.device)
        self.covs = covs.to(self.device)
        self.weights = weights.to(self.device)
        self.tol = tol
        self.max_iter = max_iter
        self.origin_n = self.weights.shape[0]
        self.new_n = int(n)
        self.random_state = random_state
        self.init_method = init_method
        if init_method == 'user':
            self.means_init = means_init.to(self.device)
            self.covs_init = covs_init.to(self.device)
            self.weights_init = weights_init.to(self.device)
        else:
            self.means_init = self.covs_init = self.weights_init = None

    def _initialize_parameter(self):
        torch.manual_seed(self.random_state)
        if self.init_method is None:
            choice = torch.randperm(self.origin_n, device=self.device)[:self.new_n]
            self.reduced_means = self.means[choice]
            self.reduced_covs = self.covs[choice]
            self.reduced_weights = self.weights[choice]
        else:
            self.reduced_means = self.means_init
            self.reduced_covs = self.covs_init
            self.reduced_weights = self.weights_init
        self.cost_matrix = TensorKL(means=[self.means, self.reduced_means],
                                    covs=[self.covs, self.reduced_covs])

    def _weight_update(self):
        min_vals, _ = torch.min(self.cost_matrix, dim=1)
        cluster = (self.cost_matrix == min_vals.unsqueeze(1))
        plan = cluster * (self.weights / cluster.sum(1)).unsqueeze(1)
        self.ot_plan = plan
        self.reduced_weights = plan.sum(dim=0)
        return torch.sum(self.cost_matrix * plan)

    def _support_update(self):
        wm = self.ot_plan / self.ot_plan.sum(dim=0)
        self.reduced_means = torch.einsum('ij,ik->jk', wm, self.means)
        self.reduced_covs = torch.einsum('ij,ikl->jkl', wm, self.covs)
        self.cost_matrix = TensorKL(means=[self.means, self.reduced_means],
                                   covs=[self.covs, self.reduced_covs])
        return torch.sum(self.cost_matrix * self.ot_plan)

    def iterative(self):
        import gc
        self._initialize_parameter()
        prev_obj = torch.inf
        for n_iter in range(1, self.max_iter + 1):
            with torch.no_grad():
                # Weight update
                obj_w = self._weight_update()
                # Prune zero columns
                nz = torch.where(self.ot_plan.sum(0) != 0)[0]
                self.ot_plan = self.ot_plan[:, nz]
                self.reduced_means = self.reduced_means[nz]
                self.reduced_covs = self.reduced_covs[nz]
                self.reduced_weights = self.reduced_weights[nz]
                # Check convergence
                if abs(prev_obj - obj_w) < self.tol:
                    break
                prev_obj = obj_w

                # Support update
                obj_s = self._support_update()
                if abs(prev_obj - obj_s) < self.tol:
                    break
                prev_obj = obj_s

                # 清理缓存，释放显存
                torch.cuda.empty_cache()
                gc.collect()

        # 最后再清理一次
        torch.cuda.empty_cache()
        gc.collect()


class KdGMR_CTD:
    def __init__(self, means, covs, weights, n, tol=1e-5, max_iter=100, random_state=0):
        self.all_means = means
        self.all_covs = covs
        self.all_weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.new_n = n

    def KDdevide(self):
        max_points = 3000
        max_depth = int(np.log2(self.all_means.shape[0] / max_points))
        indices = torch.arange(len(self.all_means), device=self.all_means.device)
        self.means_list, self.index_list = balanced_kd_split(self.all_means, indices, max_depth=max_depth)
        self.covs_list = [self.all_covs[idx] for idx in self.index_list]
        self.weights_list = [self.all_weights[idx] / self.all_weights[idx].sum() for idx in self.index_list]
        self.batch = len(self.means_list)

    def iterative(self, chosen_num=None):
        self.KDdevide()
        self.reduction_list = []
        targets = [chosen_num] if chosen_num is not None else range(self.batch)
        for i in targets:
            sampler = GMM_Sampler(self.means_list[i], self.covs_list[i], self.weights_list[i],
                                   self.new_n / self.batch, tol=self.tol,
                                   max_iter=self.max_iter, random_state=self.random_state)
            sampler.iterative()
            self.reduction_list.append(sampler)
