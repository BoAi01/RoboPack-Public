import pdb

import torch
import numpy as np
import scipy

from torch.autograd import Function
# import emd
from torch import nn
import torch.nn.functional as F


class PositionLoss(torch.nn.Module):
    def __init__(self, loss_type, loss_weights, loss_by_n_points, N_list, object_weights=None):
        super(PositionLoss, self).__init__()

        self.loss_type = loss_type

        if "chamfer_emd" in self.loss_type:
            self.loss_weights = loss_weights

        self.loss_by_n_points = loss_by_n_points
        self.object_weights = object_weights
        self.N_list = N_list
        if object_weights is not None:
            assert len(object_weights) == len(N_list), f"object_weights {object_weights} should have the same length as N_list {N_list}"
        
        self.N_cum = np.cumsum([0] + self.N_list)

        self.chamfer = Chamfer()
        self.emd_cpu = EMDCPU()
        self.emd_cuda = EMDCPU()  # EMDCUDA()
        self.mse = MSE()
        
        print(f"loss type {loss_type} instantiated, with object losses weighted {object_weights}")

    # @profile
    def __call__(self, x, y, return_losses=False):
        # check self.N_cum is valid
        assert self.N_cum[-1] == x.shape[1], \
            f'x.shape is {x.shape} while self.N_cum is {self.N_cum}: x.shape[1] should equal to self.N_cum[-1]'

        x_list = [
            x[:, self.N_cum[i] : self.N_cum[i + 1]] for i in range(len(self.N_cum) - 1)
        ]
        y_list = [
            y[:, self.N_cum[i] : self.N_cum[i + 1]] for i in range(len(self.N_cum) - 1)
        ]

        loss = 0
        losses = []
        for i, (x, y, n) in enumerate(zip(x_list, y_list, self.N_list)):
            # compute loss for each object

            if self.loss_type == "mse":
                object_loss = self.mse(x, y)
                
                if self.object_weights is not None:
                    loss += object_loss * self.object_weights[i] 
                else:
                    loss += object_loss 
                
                # the loss recorded for each object is not affected
                # by the re-weighting procedure
                losses.append(object_loss)
            
            # elif self.loss_type == "chamfer":
            #     loss += self.chamfer(x, y)
            # elif self.loss_type == "emd_cpu":
            #     loss += self.emd_cpu(x, y)
            # elif self.loss_type == "emd_cuda":
            #     loss += self.emd_cuda(x, y)
            # elif "chamfer_emd" in self.loss_type:
            #     if self.loss_weights["chamfer"] > 0:
            #         chamfer_loss = self.chamfer(x, y)
            #         loss += self.loss_weights["chamfer"] * chamfer_loss

            #     if self.loss_weights["emd"] > 0:
            #         if "cpu" in self.loss_type:
            #             emd_loss = self.emd_cpu(x, y)
            #         else:
            #             emd_loss = self.emd_cuda(x, y)

            #         loss += self.loss_weights["emd"] * emd_loss
            else:
                raise NotImplementedError("Only MSE is supported now")

            if self.loss_by_n_points:
                loss = loss * (self.N_list[0] / n)

        if return_losses:
            return loss, losses
        else:
            return loss


class Chamfer:
    @staticmethod
    def compute(x, y, keep_dim=False):
        # x: [B, M, D]
        # y: [B, N, D]
        M = x.shape[1]
        N = y.shape[1]

        # x: [B, M, N, D]
        x_repeat = x[:, :, None, :].repeat(1, 1, N, 1)
        # y: [B, M, N, D]
        y_repeat = y[:, None, :, :].repeat(1, M, 1, 1)
        # dis: [B, M, N]
        dis_pos = torch.norm(x_repeat - y_repeat, dim=-1)

        dis_x_to_nearest_y = torch.min(dis_pos, dim=2)[0] 
        dis_y_to_nearest_x = torch.min(dis_pos, dim=1)[0]
        
        if keep_dim:
            return dis_x_to_nearest_y, dis_y_to_nearest_x
        else:
            return torch.mean(dis_x_to_nearest_y) + torch.mean(dis_y_to_nearest_x)
        
    # @profile
    def __call__(self, x, y):
        return self.compute(x, y)


class EMDCPU:
    # @profile
    def __call__(self, x, y):
        B = x.shape[0]

        x_ = x.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        y_ind_list = []
        for i in range(B):
            cost_matrix = scipy.spatial.distance.cdist(x_[i], y_[i])
            try:
                ind1, ind2 = scipy.optimize.linear_sum_assignment(
                    cost_matrix, maximize=False
                )
            except:
                print("Error in linear sum assignment!")

            y_ind_list.append(ind2)

        y_ind = np.stack(y_ind_list)
        batch_ind = torch.arange(B, device=x.device)[:, None]

        emd_pos = torch.mean(torch.norm(x - y[batch_ind, y_ind], dim=-1))

        return emd_pos


class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert(n == m)
        assert(xyz1.size()[0] == xyz2.size()[0])
        assert(n % 1024 == 0)
        assert(batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None


class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)


class EMDCUDA:
    def __init__(self):
        self.emd = emdModule()

    # @profile
    def __call__(self, x, y):
        B = x.shape[0]
        N = x.shape[1]

        y_min, _ = torch.min(y, dim=1, keepdim=True)
        y_max, _ = torch.max(y, dim=1, keepdim=True)

        x_norm = (x - y_min) / (y_max - y_min)
        y_norm = (y - y_min) / (y_max - y_min)

        if N % 1024 == 0:
            # 0.005, 50 for training
            _, assignment = self.emd(x_norm, y_norm, 0.002, 50)
        else:
            n_repeat = 2 if N > 1024 else 1024 // N + 1
            x_repeat = x_norm.repeat(1, n_repeat, 1)[:, : 1024 * (N // 1024 + 1)]
            y_repeat = y_norm.repeat(1, n_repeat, 1)[:, : 1024 * (N // 1024 + 1)]

            _, assignment = self.emd(x_repeat, y_repeat, 0.002, 50)
            assignment = torch.remainder(assignment[:, :N], N)

        assignment = assignment.to(dtype=torch.int64, device=x.device)
        batch_ind = torch.arange(B, device=x.device)[:, None]

        emd_pos = torch.mean(torch.norm(x - y[batch_ind, assignment], dim=-1))
        return emd_pos


class MSE:
    def __call__(self, x, y):
        mse_pos = F.mse_loss(x, y)

        return mse_pos
