import time
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
torch.set_printoptions(precision=8)  # 设置显示8位小数
from typing import Tuple, List
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix


def decompose_covariance_torch(Sigma):
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
    eigenvalues = torch.clamp(eigenvalues, min=1e-6)  # 截断负值

    # 处理反射矩阵（确保右手坐标系）
    dets = torch.det(eigenvectors)
    sign_mask = (dets < 0).unsqueeze(-1).unsqueeze(-1)
    eigenvectors = torch.where(sign_mask, -eigenvectors, eigenvectors)

    # 计算缩放因子
    scaling = torch.sqrt(eigenvalues)

    # 直接转换旋转矩阵到四元数（支持批量）
    quaternion = matrix_to_quaternion(eigenvectors)

    return scaling, quaternion, eigenvectors


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
    del cov_diff, mean_diff, mus1, mus2, Sigmas1, Sigmas2
    return cost_matrix

def balanced_kd_split(
    data: np.ndarray,
    indices: np.ndarray,
    depth: int = 0,
    max_depth: int = None,
    k: int = 2  # 新增参数控制切割次数
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    平衡KD树分割函数
    :param data: 输入数据 (n_samples, n_features)
    :param indices: 对应原始数据的索引
    :param depth: 当前递归深度
    :param max_depth: 最大允许深度（默认基于 k*dim）
    :param k: 每个维度的分割次数，控制总深度为 k * d
    :return: (数据块列表, 对应索引列表)
    """
    n, d = data.shape
    if max_depth is None:
        max_depth = k * d  # 总切割次数为 k*d（由 k 控制）
    if depth >= max_depth or n <= 1:
        return [data], [indices]

    current_dim = depth % d
    sorted_indices = np.argsort(data[:, current_dim])
    sorted_data = data[sorted_indices]
    sorted_original_indices = indices[sorted_indices]

    median_idx = n // 2
    left_data, left_indices = sorted_data[:median_idx], sorted_original_indices[:median_idx]
    right_data, right_indices = sorted_data[median_idx:], sorted_original_indices[median_idx:]

    left_blocks, left_idx = balanced_kd_split(left_data, left_indices, depth + 1, max_depth, k)
    right_blocks, right_idx = balanced_kd_split(right_data, right_indices, depth + 1, max_depth, k)

    return left_blocks + right_blocks, left_idx + right_idx


import numpy as np
from typing import List, Tuple
from itertools import product


def oct_split(
        data: np.ndarray,
        indices: np.ndarray,
        depth: int = 0,
        max_depth: int = None,
        k: int = 1
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    平衡Octree分割函数
    :param data: 输入数据 (n_samples, 3)
    :param indices: 对应原始数据的索引
    :param depth: 当前递归深度
    :param max_depth: 最大允许深度（默认由k控制）
    :param k: 控制切割次数，最大深度默认k
    :return: (数据块列表, 对应索引列表)
    """
    n, d = data.shape
    assert d == 3, "Octree requires 3D data"

    # 设置最大深度（默认k次切割）
    if max_depth is None:
        max_depth = k
    if depth >= max_depth or n <= 1:
        return [data], [indices]

    # 计算三个维度的中位数分割值
    split_vals = []
    for dim in range(3):
        med_val = (np.max(data[:, dim]) + np.min(data[:, dim])) / 2
        split_vals.append(med_val)

    # 生成三个维度的比较条件
    x_le = data[:, 0] <= split_vals[0]
    y_le = data[:, 1] <= split_vals[1]
    z_le = data[:, 2] <= split_vals[2]

    # 生成八叉树子块
    all_blocks, all_indices = [], []
    for mask in product([True, False], repeat=3):
        # 组合三个维度的比较条件
        cond = (x_le if mask[0] else ~x_le) & \
               (y_le if mask[1] else ~y_le) & \
               (z_le if mask[2] else ~z_le)

        sub_data = data[cond]
        sub_idx = indices[cond]

        if len(sub_data) > 0:  # 跳过空块
            # 递归处理子块
            blocks, idx = oct_split(sub_data, sub_idx, depth + 1, max_depth, k)
            all_blocks.extend(blocks)
            all_indices.extend(idx)

    return all_blocks, all_indices

class GMM_Sampler:
    def __init__(self,
                 means,
                 covs,
                 weights,
                 n,
                 tol=1e-5,
                 max_iter=100,
                 random_state=0,
                 # batch_size=None,
                 init_method=None,
                 means_init=None,
                 covs_init=None,
                 weights_init=None
                 ):
        self.cost_index = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f'Data in device {self.device}!')
        self.means = torch.from_numpy(means).to(self.device)
        self.covs = torch.from_numpy(covs).to(self.device)
        self.weights = torch.from_numpy(weights).to(self.device)
        self.tol = tol
        self.max_iter = max_iter
        self.origin_n = self.weights.shape[0]
        self.new_n = int(n)
        self.random_state = random_state
        self.converged_ = False
        self.time_ = []
        # self.batch_size = self.weights.shape[0] if batch_size is None else batch_size
        self.init_method = init_method
        if self.init_method == 'user':
            self.means_init = means_init.to(self.device)
            self.covs_init = covs_init.to(self.device)
            self.weights_init = weights_init.to(self.device)
        else:
            self.means_init = None
            self.covs_init = None
            self.weights_init = None

    def _initialize_parameter(self):
        if self.init_method is None:
            np.random.seed(self.random_state)
            torch.cuda.manual_seed(self.random_state)
            choice = torch.randperm(len(self.weights))[:self.new_n]
            self.reduced_means = self.means[choice]
            self.reduced_covs = self.covs[choice]
            self.reduced_weights = self.weights[choice]

        elif self.init_method == 'user':
            self.reduced_means = self.means_init
            self.reduced_covs = self.covs_init
            self.reduced_weights = self.weights_init

        self.cost_matrix = TensorKL(means=[self.means, self.reduced_means],
                                    covs=[self.covs, self.reduced_covs])

    def _obj(self):
        return torch.sum(self.cost_matrix * self.ot_plan)

    def _weight_update(self):
        min_values, min_indices = torch.min(self.cost_matrix, dim=1)
        self.clustering_matrix = (self.cost_matrix == min_values.unsqueeze(1))
        self.ot_plan = self.clustering_matrix * (self.weights /
                                                 self.clustering_matrix.sum(1).reshape(-1,)).reshape(-1,1)
        self.reduced_weights = self.ot_plan.sum(dim=0)
        return self._obj()

    def _support_update(self):
        weights_matrix = self.ot_plan / self.ot_plan.sum(dim=0)
        barycenter_mean = torch.einsum('ij, ik -> jk', weights_matrix, self.means)
        barycenter_cov = torch.einsum('ij, ikl -> jkl', weights_matrix, self.covs)
        # diff = self.means.unsqueeze(0) - barycenter_mean.unsqueeze(1)
        # temp1 = weights_matrix.T
        # temp = temp1.unsqueeze(-1) * diff
        # barycenter_cov = torch.einsum("ijk,ijl->ikl", temp, diff) + barycenter_cov ### 有点问题
        self.reduced_means, self.reduced_covs = barycenter_mean, barycenter_cov

        self.cost_matrix = TensorKL(means=[self.means, self.reduced_means],
                           covs=[self.covs, self.reduced_covs])
        return self._obj()

    def iterative(self):
        import gc
        init_begin = time.time()
        self._initialize_parameter()
        # print(f'End initialization, cost {time.time() - init_begin}s')
        obj = torch.inf
        # for n_iter in tqdm(range(1, self.max_iter + 1), desc="Inner Iteration"):
        for n_iter in range(1, self.max_iter + 1):
            with torch.no_grad():
                proc_time = time.time()
                obj_current = self._weight_update()
                index = torch.where(self.ot_plan.sum(dim=0) != 0)
                self.new_n = index[0].shape[0]
                self.ot_plan = self.ot_plan.T[index].T
                self.reduced_means = self.reduced_means[index[0]]
                self.reduced_covs = self.reduced_covs[index[0]]
                self.reduced_weights = self.reduced_weights[index[0]]
                change = obj - obj_current
                self.cost_matrix = self.cost_matrix[:, index[0]]
                # print("weight update change", change)
                if abs(change) < self.tol:
                    self.converged_ = True
                    self.obj = obj
                    self.n_iter_ = n_iter
                    break
                if change < 0.0:
                    raise ValueError('Weight update: The objective function is increasing!')
                obj = obj_current
                obj_current = self._support_update()
                change = obj - obj_current
                # print("support update change", change)
                self.time_.append(time.time() - proc_time)
                if abs(change) < self.tol:
                    self.converged_ = True
                    self.obj = obj
                    self.n_iter_ = n_iter
                    break
                if change < 0.0:
                    raise ValueError('Support update: The objective function is increasing!')
                obj = obj_current
        #         torch.cuda.empty_cache()
        #         gc.collect()
        # torch.cuda.empty_cache()
        # gc.collect()
        self.cost_index = torch.argmin(self.cost_matrix, dim=0)
        del self.cost_matrix, self.ot_plan, self.clustering_matrix
        self.cost_matrix = None
        self.ot_plan = None
        self.clustering_matrix = None

        # 强制释放 CUDA 缓存
        torch.cuda.empty_cache()

        if not self.converged_:
            print('Algorithm did not converge. '
                  f'The final loss is {self._obj()}. '
                  'Try different init parameters, '
                  'or increase max_iter, tol ')
            # return False


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
        # n, d = self.all_means.shape
        # order = 1
        # for order_ in range(100):
        #     per_slice = math.ceil(n / 2 ** (d * order_))
        #     per_slice_n = int(self.new_n / 2 ** (d * order_))
        #     if per_slice * per_slice_n < 800000: # 3000的样子
        #         order = order_
        #         # print(f'KD Tree Depth: {order}')
        #         break
        max_points_per_node = 3000
        max_depth = int(np.log2(self.all_means.shape[0] / max_points_per_node))

        original_means = torch.arange(len(self.all_means))
        self.means_list, self.index_list = balanced_kd_split(self.all_means, original_means, max_depth=max_depth)
        self.covs_list = [self.all_covs[self.index_list[i]] for i in range(len(self.means_list))]
        weights_list = [self.all_weights[self.index_list[i]] for i in range(len(self.means_list))]
        self.weights_list = [weights_list[i] / np.sum(weights_list[i]) for i in range(len(self.means_list))]
        self.batch = len(self.means_list)
        print(f'Generated robust KD partitions: {2 ** max_depth}')

    def iterative(self):
        self.KDdevide()
        self.reduction_list = []
        left_num = self.new_n
        for i in tqdm(range(len(self.means_list)), desc="Outer Iteration: KD Tree"):
            per_num = int(left_num / (len(self.index_list) - i))
            left_num -= per_num
            reduction = GMM_Sampler(self.means_list[i],
                                    self.covs_list[i],
                                    self.weights_list[i],
                                    per_num,
                                    tol=self.tol,
                                    max_iter=self.max_iter,
                                    random_state=self.random_state)
            reduction.iterative()
            self.reduction_list.append(reduction)
            del reduction
            torch.cuda.empty_cache()



def gaussian_model_reduction(gaussians, ratio, random_seed=42):
    def loading(gaussians):
        means = gaussians.get_xyz.detach().cpu().numpy()
        cov_no_normalized = gaussians.get_covariance().detach().cpu().numpy()
        weights_no_normalized = gaussians.get_opacity.detach().cpu().numpy().reshape(-1, )
        weight = weights_no_normalized / np.sum(weights_no_normalized)
        cov = np.zeros((cov_no_normalized.shape[0], 3, 3), dtype=cov_no_normalized.dtype)
        for i in range(cov_no_normalized.shape[0]):
            temp = np.zeros((3, 3), dtype=cov_no_normalized.dtype)
            temp[0, :] = cov_no_normalized[i][:3]
            temp[1, 1:] = cov_no_normalized[i][3:-1]
            temp[2, 2] = cov_no_normalized[i][-1]
            cov[i] = temp + temp.T - np.diag(cov_no_normalized[i][[0, 3, 5]])
        return means, weight, cov
    means, weights, covs = loading(gaussians)
    downsample_num = int(ratio) if ratio > 1 else max(1, int(means.shape[0] * ratio))
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    kd_reduced_gaussian = KdGMR_CTD(means, covs, weights, downsample_num,
                                    random_state=random_seed)  ## transfer numpy into torch
    kd_reduced_gaussian.iterative()
    device = gaussians._xyz.device
    block_num = len(kd_reduced_gaussian.reduction_list)
    record_index_matrix = torch.zeros((0, 1), dtype=torch.int32).to(device)
    reduced_means = torch.zeros((0, 3), dtype=gaussians._xyz.dtype).to(device)
    reduced_scaling = torch.zeros((0, 3), dtype=gaussians._scaling.dtype).to(device)
    reduced_rotation = torch.zeros((0, 4), dtype=gaussians._rotation.dtype).to(device)
    for i in range(block_num):
        origin_index = kd_reduced_gaussian.index_list[i].to(device)
        inner_index = kd_reduced_gaussian.reduction_list[i].cost_index
        record_index = origin_index[inner_index].reshape(-1, 1)
        record_index_matrix = torch.vstack([record_index_matrix, record_index])

        reduced_means_ = kd_reduced_gaussian.reduction_list[i].reduced_means
        reduced_means = torch.vstack([reduced_means, reduced_means_])  ###

        reduced_covs_ = kd_reduced_gaussian.reduction_list[i].reduced_covs

        reduced_scaling_, reduced_rotation_, _ = decompose_covariance_torch(reduced_covs_)
        reduced_scaling = torch.vstack([reduced_scaling, reduced_scaling_])
        reduced_rotation = torch.vstack([reduced_rotation, reduced_rotation_])
    keep_indices = record_index_matrix.reshape(-1, )
    # keep_indices = torch.from_numpy(record_index_matrix).to("cuda")
    from scene import GaussianModel
    new_model = GaussianModel(sh_degree=3)

    with torch.no_grad():
        new_features_dc = gaussians._features_dc[keep_indices]
        new_model._features_dc = nn.Parameter(new_features_dc.clone())

        new_features_rest = gaussians._features_rest[keep_indices]
        new_model._features_rest = nn.Parameter(new_features_rest.clone())

        new_opacity = gaussians._opacity[keep_indices]
        new_model._opacity = nn.Parameter(new_opacity.clone())

        new_model._opacity = nn.Parameter(new_opacity.clone())
        new_model._xyz = nn.Parameter(reduced_means)
        new_model._scaling = nn.Parameter(
            torch.log(reduced_scaling))
        new_model._rotation = nn.Parameter(reduced_rotation)

    # return new_model
    return new_model
    print(f"saved {downsample_num} gaussians to {output_path}")
