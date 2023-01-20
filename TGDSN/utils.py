import numpy as np
import torch
import random
import math
from torch import nn


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)  # 逐元素操作？

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, v2.t())  # 向量点乘

    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0

    for i in range(v1.shape[0]):
        for j in range(v2.shape[0]):
            if res[i][j] >= 0:
                res[i][j] = 1
            else:
                res[i][j] = 0

    return res


def recon_loss(batch_size, private_feature_cons, shared_feature_cons, A):
    recons_metrix_p = private_feature_cons
    recons_metrix_s = shared_feature_cons
    for i in range(batch_size):
        for j in range(14):
            for k in range(14):
                if (recons_metrix_p[i][j][k] >= 0):
                    recons_metrix_p[i][j][k] = 1
                else:
                    recons_metrix_p[i][j][k] = 0
        for j in range(14):
            for k in range(14):
                if (recons_metrix_s[i][j][k] >= 0):
                    recons_metrix_s[i][j][k] = 1
                else:
                    recons_metrix_s[i][j][k] = 0

        recons_metrix_p[i] = private_feature_cons[i] + shared_feature_cons[i]
        for j in range(14):
            for k in range(14):
                if (recons_metrix_p[i][j][k] >= 1):
                    recons_metrix_p[i][j][k] = 1

    A = np.expand_dims(A, 0).repeat(batch_size, axis=0)
    A = torch.tensor(A)
    # loss_fn1 = torch.nn.MSELoss(reduction='none')
    # loss1 = loss_fn1(a.float(), b.float())
    # print('loss_none:\n', loss1)

    loss_fn2 = torch.nn.MSELoss(reduction='sum')
    loss2 = loss_fn2(A, recons_metrix_p)

    # loss_fn3 = torch.nn.MSELoss(reduction='mean')
    # loss3 = loss_fn3(A.float(), recons_metrix_p.float())

    return loss2


def recon_loss_2(batch_size, feature_cons, A):
    A = np.expand_dims(A.cpu(), 0).repeat(batch_size, axis=0)
    A = torch.tensor(A)
    # loss_fn1 = torch.nn.MSELoss(reduction='none')
    # loss1 = loss_fn1(a.float(), b.float())
    # print('loss_none:\n', loss1)

    loss_fn2 = torch.nn.MSELoss(reduction='mean')
    loss2 = loss_fn2(A.cuda(), feature_cons.cuda())

    # loss_fn3 = torch.nn.MSELoss(reduction='mean')
    # loss3 = loss_fn3(A.float(), recons_metrix_p.float())

    return loss2


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def loop_iterable(iterable):
    while True:
        yield from iterable

def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
def scoring_func(error_arr):
    pos_error_arr = error_arr[error_arr >= 0] 
    neg_error_arr = error_arr[error_arr < 0]
    score = 0 
    for error in neg_error_arr:
            score = math.exp(-(error / 13)) - 1 + score 
    for error in pos_error_arr: 
            score = math.exp(error / 10) - 1 + score
    return score

def roll(x, shift: int, dim: int = -1, fill_pad: int = None):

    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim))), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim)))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([gap, x.index_select(dim, torch.arange(shift))], dim=dim)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat.cuda(),y.cuda()))
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



