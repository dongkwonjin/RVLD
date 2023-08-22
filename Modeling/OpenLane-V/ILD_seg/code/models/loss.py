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

        l_seg = self.loss_focal(out['seg_map_logit'], gt['seg_label'][self.sf[0]])

        loss_dict['sum'] = l_seg
        loss_dict['seg'] = l_seg

        return loss_dict

