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

class Loss_Function(nn.Module):
    def __init__(self, cfg):
        super(Loss_Function, self).__init__()
        self.cfg = cfg

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()
        self.loss_nce = nn.CrossEntropyLoss()
        self.loss_score = nn.MSELoss()
        self.loss_focal = SoftmaxFocalLoss(gamma=2)

        self.sf = cfg.scale_factor['seg']
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]
        self.weight = [1e0, 1e2, 1e2, 1e2, 1e3, 1e3]

    def forward(self, out, gt):
        loss_dict = dict()

        exclude_map = gt['seg_label'][self.sf[0]].unsqueeze(3) * gt['visit'][self.sf[0]].unsqueeze(3)
        loss_dict['coeff_iou'] = self.compute_IoU_loss(out['coeff_map'].permute(0, 2, 3, 1), gt['coeff_label'][self.sf[0]],
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

        # gt_x_coords_img = gt_x_coords * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
        # invalid_mask = (gt_x_coords_img < 0) | (gt_x_coords_img >= self.cfg.width)

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
