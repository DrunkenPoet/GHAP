import time
import numpy as np
# from .utils import *
from torch import nn
from tqdm import tqdm
# from .distance import *
from typing import Tuple, List
from pytorch3d.transforms import matrix_to_quaternion
import torch
from scene import GaussianModel
torch.set_printoptions(precision=8)  # 设置显示8位小数


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


def Euclidean_dist(gm: GaussianModel, reduced_gm: GaussianModel, method: str):
    assert gm.max_sh_degree == reduced_gm.max_sh_degree, (f'Two Gaussian Models have different sh_degree: {gm.max_sh_degree }'
                                                          f' and {reduced_gm.max_sh_degree }')
    mus1, mus2 = gm.get_xyz, reduced_gm.get_xyz
    coordinate_dist = torch.cdist(mus1, mus2, p=2).square()

    cov1, cov2 = gm.get_covariance(), reduced_gm.get_covariance()
    cov1_ = torch.cat([cov1, cov1[:, [1, 2, 4]]], dim=1)
    cov2_ = torch.cat([cov2, cov2[:, [1,2,4]]], dim=1)
    cov_dist = torch.cdist(cov1_.unsqueeze(1), cov2_.unsqueeze(0), p=2).squeeze(1).square()
    if method == 'NEARST':
        breakpoint()
        return coordinate_dist + cov_dist
    elif method == 'MERGE':
        dim = gm.get_features.shape[1] * gm.get_features.shape[2]
        features1, features2 = gm.get_features.reshape(-1, dim), reduced_gm.get_features.reshape(-1, dim)
        feature_dist = torch.cdist(features1, features2, p=2).square()
        return coordinate_dist + cov_dist + feature_dist


def subsampling_gaussian_model(gm: GaussianModel, keep_indices):
    device = gm.get_xyz.device
    keep_indices.to(device)
    new_model = GaussianModel(sh_degree=3)
    with torch.no_grad():
        new_xyz = gm._xyz[keep_indices]
        new_model._xyz = nn.Parameter(new_xyz.clone())
        new_features_dc = gm._features_dc[keep_indices]
        new_model._features_dc = nn.Parameter(new_features_dc.clone())
        new_features_rest = gm._features_rest[keep_indices]
        new_model._features_rest = nn.Parameter(new_features_rest.clone())
        new_scaling = gm._scaling[keep_indices]
        new_model._scaling = nn.Parameter(new_scaling.clone())
        new_rotation = gm._rotation[keep_indices]
        new_model._rotation = nn.Parameter(new_rotation.clone())
        new_opacity = gm._opacity[keep_indices]
        new_model._opacity = nn.Parameter(new_opacity.clone())
    return new_model


def decompose_covariance(symmetric):
    covariance = torch.zeros((symmetric.shape[0], 3, 3), dtype=torch.float, device=symmetric.device)
    covariance[:, 0, :] = symmetric[:, :3]
    covariance[:, 1, 1] = symmetric[:, 3]
    covariance[:, 1, 2] = symmetric[:, 4]
    covariance[:, 2, 2] = symmetric[:, 5]
    covariance[:, 1, 0] = symmetric[:, 1]
    covariance[:, 2, 0] = symmetric[:, 2]
    covariance[:, 2, 1] = symmetric[:, 4]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
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


class GMM_Sampler_:
    def __init__(self,
                 gm: GaussianModel,
                 n,
                 method='NEARST',
                 tol=1e-5,
                 max_iter=100,
                 random_state=0,
                 init_method=None,
                 gm_init=None
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gaussian_model = gm
        self.weights = self.gaussian_model.get_opacity / torch.sum(self.gaussian_model.get_opacity)
        # self.weights.reshape(-1,)
        self.tol = tol
        self.max_iter = max_iter
        self.origin_n = self.weights.shape[0]
        self.new_n = int(n)
        self.random_state = random_state
        self.converged_ = False
        self.time_ = []
        self.init_method = init_method
        self.init_gm = gm_init
        self.method = method
        print(self.init_method)

    def _initialize_parameter(self):
        if self.init_method is None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            keep_indices = np.random.choice(self.origin_n, size=self.new_n, replace=False)
            keep_indices = torch.from_numpy(keep_indices)
            self.reduced_gm = subsampling_gaussian_model(self.gaussian_model, keep_indices)

        else:
            self.reduced_gm = self.init_gm

        self.cost_matrix = Euclidean_dist(self.gaussian_model, self.reduced_gm, self.method)
        breakpoint()
        print('cost_matrix')

    def _obj(self):
        return torch.sum(self.cost_matrix * self.ot_plan)

    def _weight_update(self):
        min_values, min_indices = torch.min(self.cost_matrix, dim=1)
        self.clustering_matrix = torch.Tensor(self.cost_matrix == min_values.unsqueeze(1))
        self.ot_plan = self.clustering_matrix * (self.weights /
                                                self.clustering_matrix.sum(dim=1).reshape(-1,1))
        self.reduced_weights = self.ot_plan.sum(dim=0)
        return self._obj()

    def _support_update(self):
        weights_matrix = self.ot_plan / self.ot_plan.sum(dim=0)
        barycenter_mean = torch.einsum('ij, ik -> jk', weights_matrix, self.gaussian_model.get_xyz)
        barycenter_cov = torch.einsum('ij, ik -> jk', weights_matrix, self.gaussian_model.get_covariance())
        self.reduced_gm._xyz = nn.Parameter(barycenter_mean)
        reduced_scaling, reduced_rotation, _ = decompose_covariance(barycenter_cov)
        self.reduced_gm._rotation = nn.Parameter(reduced_rotation)
        self.reduced_gm._scaling = nn.Parameter(torch.log(reduced_scaling))
        if self.method == 'MERGE':
            barycenter_f_dc = torch.einsum('ij, ilk -> jlk', weights_matrix, self.gaussian_model.get_features_dc)
            barycenter_f_rest = torch.einsum('ij, ilk -> jlk', weights_matrix, self.gaussian_model.get_features_rest)
            barycenter_opacity = torch.einsum('ij, ik -> jk', weights_matrix, self.gaussian_model._opacity)
            self.reduced_gm._features_rest = nn.Parameter(barycenter_f_rest)
            self.reduced_gm._features_dc = nn.Parameter(barycenter_f_dc)
            self.reduced_gm._opacity = nn.Parameter(barycenter_opacity)
        self.cost_matrix = Euclidean_dist(self.gaussian_model, self.reduced_gm, self.method)
        return self._obj()

    def iterative(self):
        self._initialize_parameter()
        obj = torch.inf
        for n_iter in range(1, self.max_iter + 1):
            with torch.no_grad():
                proc_time = time.time()
                obj_current = self._weight_update()
                index = torch.where(self.ot_plan.sum(dim=0) != 0)
                if self.new_n != index[0].shape[0]:
                    self.new_n = index[0].shape[0]
                    self.ot_plan = self.ot_plan.T[index].T
                    self.reduced_gm = subsampling_gaussian_model(self.reduced_gm, index[0])
                    self.cost_matrix = self.cost_matrix[:, index[0]]
                change = obj - obj_current
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
                self.time_.append(time.time() - proc_time)
                if abs(change) < self.tol:
                    self.converged_ = True
                    self.obj = obj
                    self.n_iter_ = n_iter
                    break
                if change < 0.0:
                    raise ValueError('Support update: The objective function is increasing!')
                obj = obj_current
        if self.method == 'NEARST':
            _index = torch.argmin(self.cost_matrix, dim=0).to(self.device)
            new_features_dc = self.gaussian_model._features_dc[_index]
            self.reduced_gm._features_dc = nn.Parameter(new_features_dc.clone())

            new_features_rest = self.gaussian_model._features_rest[_index]
            self.reduced_gm._features_rest = nn.Parameter(new_features_rest.clone())

            new_opacity = self.gaussian_model._opacity[_index]
            self.reduced_gm._opacity = nn.Parameter(new_opacity.clone())

        if not self.converged_:
            print('Algorithm did not converge. '
                  f'The final loss is {self._obj()}. '
                  'Try different init parameters, '
                  'or increase max_iter, tol ')
            # return False

class KdGMR_CTD_:
    def __init__(self, gm: GaussianModel, n, tol=1e-5, max_iter=100, random_state=0, method='NEARST'):
        self.device = gm.get_xyz.device
        self.gaussian_model_all = gm
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.new_n = n
        self.means = self.gaussian_model_all.get_xyz.detach().cpu().numpy()
        self.n = self.means.shape[0]
        self.index_list = None
        self.method = method

    def KDdevide(self):
        max_points_per_node = 1000
        max_depth = int(np.log2(self.n / max_points_per_node))
        _, self.index_list = balanced_kd_split(self.means, np.arange(self.n), max_depth=max_depth)

    def iterative(self):
        self.KDdevide()
        self.reduction_list = []
        left_num = self.new_n
        for i in tqdm(range(len(self.index_list)), desc="Batch Mixture Gaussian Reduction"):
            per_num = int(left_num / (len(self.index_list) - i))
            left_num -= per_num
            temp_gm = subsampling_gaussian_model(self.gaussian_model_all,
                                                 torch.from_numpy(self.index_list[i]).long().to(self.device))
            reduction = GMM_Sampler_(temp_gm,
                                     per_num,
                                     tol=self.tol,
                                     max_iter=self.max_iter,
                                     random_state=self.random_state,
                                     method=self.method)
            reduction.iterative()
            self.reduction_list.append(reduction)

        assert left_num == 0, f'Something wrong with Subsampling number. Left {left_num} points to be reduced!'
