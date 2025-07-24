import time
import numpy as np
import torch


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
        self.cost_matrix = []

        if not self.converged_:
            print('Algorithm did not converge. '
                  f'The final loss is {self._obj()}. '
                  'Try different init parameters, '
                  'or increase max_iter, tol ')
            # return False