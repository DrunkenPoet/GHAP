import numpy as np
from IPython.utils.io import temp_pyfile
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
from typing import Tuple, List
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import torch
from torch import nn
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid

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


def decompose_covariance(Sigma):
    """
    covs (N, 3, 3) into scaling (N, 3) and rotation (N, 4).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1
    scaling = np.sqrt(eigenvalues)

    rotation_matrix = eigenvectors
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    quaternion = np.roll(quaternion, 1)

    return scaling, quaternion, rotation_matrix


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



def construct_list_of_attributes(dc, rest, scaling, rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(dc.shape[1] * dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(rest.shape[1] * rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_ply(xyz, f_dc, f_rest, opacities, scaling, rotation, path, transfer=True):
    """using save_ply method from GaussianModel"""
    assert isinstance(xyz, np.ndarray) and f_dc.shape[2] == 3, f'type of xyz {type(xyz)}, shape of f_dc {f_dc.shape}'

    temp_gaussian = GaussianModel(sh_degree=3) #TODO: sh_degree
    temp_gaussian._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    temp_gaussian._features_dc = nn.Parameter(
        torch.tensor(f_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
    temp_gaussian._features_rest = nn.Parameter(
        torch.tensor(f_rest, dtype=torch.float, device="cuda").contiguous().requires_grad_(
            True))
    temp_gaussian._rotation = nn.Parameter(
        torch.tensor(rotation, dtype=torch.float, device="cuda").requires_grad_(True))
    if transfer == False:
        temp_gaussian._opacity = nn.Parameter(inverse_sigmoid(torch.tensor(opacities.reshape(-1,1), dtype=torch.float, device="cuda").requires_grad_(True)))
        temp_gaussian._scaling = nn.Parameter(
            torch.log(torch.tensor(scaling, dtype=torch.float, device="cuda").requires_grad_(True)))
    else:
        temp_gaussian._opacity = nn.Parameter(torch.tensor(opacities.reshape(-1,1), dtype=torch.float, device="cuda").requires_grad_(True))
        temp_gaussian._scaling = nn.Parameter(torch.tensor(scaling, dtype=torch.float, device="cuda").requires_grad_(True))

    temp_gaussian.save_ply(path)


from collections import deque
class RobustAutoSamplingTrigger:
    def __init__(self,
                 window_size=100,     # 检测窗口大小
                 cv_threshold=5.0,    # CV阈值（单位：%）
                 min_iter=1000,       # 最小触发迭代
                 cooldown=5000        # 触发冷却期
                ):
        self.loss_window = deque(maxlen=window_size)
        self.cv_threshold = cv_threshold / 100  # 转换为小数形式
        self.min_iter = min_iter
        self.cooldown = cooldown
        self.last_trigger = -np.inf
        self.ratio = -0.15
        self.times = 0

    def _calculate_cv(self):
        arr = np.array(self.loss_window)
        mu = np.mean(arr)
        if mu < 1e-6:  # 避免除以0
            return np.inf
        return np.std(arr) / mu

    def update(self, current_iter, loss_value):
        self.loss_window.append(loss_value)
        # 触发条件检查
        if len(self.loss_window) < self.loss_window.maxlen:
            return False
        if (current_iter - self.last_trigger) < self.cooldown:
            return False
        if current_iter < self.min_iter:
            return False
        # 计算CV
        cv = self._calculate_cv()
        return cv < self.cv_threshold

    def trigger_sampling(self, current_iter, loss_value):
        if self.times >= 4:
            return False
        if self.update(current_iter, loss_value):
            self.last_trigger = current_iter
            return True
        return False

    def updata_ratio(self):
        self.ratio += 0.2
        self.times += 1
