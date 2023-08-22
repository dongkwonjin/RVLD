import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.utils import *

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll1 = nn.NLLLoss(reduce=True)
        self.nll2 = nn.NLLLoss(reduce=False)

    def forward(self, logits, labels, reduce=True):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score

        if reduce == True:
            loss = self.nll1(log_score, labels)
        else:
            loss = self.nll2(log_score, labels)
        return loss

def robust_l1(x):
    """Robust L1 metric."""
    return (x ** 2 + 0.001 ** 2) ** 0.5

def image_grads(input, stride=1):
    input_gh = input[:, :, stride:] - input[:, :, :-stride]
    input_gw = input[:, :, :, stride:] - input[:, :, :, :-stride]
    return input_gh, input_gw

class smooth_loss(nn.Module):
    def __init__(self, edge_constant=150.0):
        super(smooth_loss, self).__init__()

        self.edge_constant = edge_constant

    def forward(self, uv):
        flow_gx, flow_gy = image_grads(uv)
        loss = (torch.mean(robust_l1(flow_gx)) + torch.mean(robust_l1(flow_gy))) / 2.
        return loss

class Loss_Function(nn.Module):
    def __init__(self, cfg):
        super(Loss_Function, self).__init__()
        self.cfg = cfg

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()
        self.loss_nce = nn.CrossEntropyLoss()
        self.loss_score = nn.MSELoss()
        self.loss_focal = SoftmaxFocalLoss(gamma=2)
        self.loss_smooth = smooth_loss()

        self.sf = cfg.scale_factor['seg']
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]
        self.weight = [1e0, 1e1, 1e2, 1e2, 1e2, 1e2]

    def forward(self, out, gt, t, epoch):
        loss_dict = dict()

        # flow
        seg = gt[f'{t[:-1]}{int(t[-1]) + 1}']['seg_label'][self.sf[0]].type(torch.float)
        b, h, w = seg.shape
        seg = seg.view(b, 1, h, w)
        warped_gt_seg = F.grid_sample(seg, out[t]['grid'], mode='bilinear', padding_mode='zeros')
        loss_dict['warping_c'] = self.loss_mse(warped_gt_seg[:, 0], gt[t]['seg_label'][self.sf[0]].type(torch.float))

        # seg
        loss_dict['seg'] = self.loss_focal(out[t]['seg_map_logit'], gt[t]['seg_label'][self.sf[0]])

        # coeff
        exclude_map = gt[t]['seg_label'][self.sf[0]].unsqueeze(3) * gt[t]['visit'][self.sf[0]].unsqueeze(3)
        loss_dict['coeff_iou'] = self.compute_IoU_loss(out[t]['coeff_map'].permute(0, 2, 3, 1), gt[t]['coeff_label'][self.sf[0]],
                                                       exclude_map=exclude_map)

        l_sum = torch.FloatTensor([0.0]).cuda()
        for key in list(loss_dict):
            l_sum += loss_dict[key]
        loss_dict['sum'] = l_sum

        return loss_dict

    def compute_IoU_loss(self, out, gt, exclude_map):
        e1 = 0.1
        out_x_coords = self.coeff_to_x_coord_conversion(out)
        gt_x_coords = self.coeff_to_x_coord_conversion(gt)

        d1 = torch.min(out_x_coords + e1, gt_x_coords + e1) - torch.max(out_x_coords - e1, gt_x_coords - e1)
        d2 = torch.max(out_x_coords + e1, gt_x_coords + e1) - torch.min(out_x_coords - e1, gt_x_coords - e1)

        d1 = torch.sum(d1, dim=3)
        d2 = torch.sum(d2, dim=3)

        iou = (d1 / (d2 + 1e-9))
        iou_loss = torch.mean((1 - iou)[exclude_map[:, :, :, 0] == 1])
        return iou_loss

    def coeff_to_x_coord_conversion(self, coeff_map, mode=None):
        m = self.cfg.top_m
        b, h, w, _ = coeff_map.shape
        coeff_map = coeff_map.reshape(-1, m, 1)
        U = self.U.view(1, -1, m).expand(coeff_map.shape[0], -1, m)
        x_coord_map = torch.bmm(U, coeff_map)
        x_coord_map = x_coord_map.view(b, h, w, -1)
        return x_coord_map
