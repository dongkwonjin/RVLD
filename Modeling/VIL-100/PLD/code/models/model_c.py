import cv2
import math

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from models.backbone import *
from libs.utils import *

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe = pe.view(1, d_model * 2, height, width)
    return pe

class Deformable_Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Deformable_Conv2d, self).__init__()
        self.deform_conv2d = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, offset, mask=None):
        x = self.deform_conv2d(x, offset, mask)
        return x


class conv1d_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv1d_relu, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

class conv_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv1d_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(conv1d_bn_relu, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

        self.sf = self.cfg.scale_factor['img']
        self.seg_sf = self.cfg.scale_factor['seg']

        self.c_feat = 64

        self.feat_embedding = torch.nn.Sequential(
            conv_bn_relu(1, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
        )

        self.regressor = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
        )

        kernel_size = 3
        self.offset_regression = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            torch.nn.Conv2d(self.c_feat, 2 * kernel_size * kernel_size, 1)
        )

        self.deform_conv2d = Deformable_Conv2d(in_channels=self.c_feat, out_channels=self.cfg.top_m,
                                               kernel_size=kernel_size, stride=1, padding=1)

        self.pe = positionalencoding2d(d_model=self.c_feat, height=self.cfg.height // self.seg_sf[0], width=self.cfg.width // self.seg_sf[0]).cuda()

    def forward_for_regression(self):
        b, _, _, _ = self.prob_map.shape
        feat_c = self.feat_embedding(self.prob_map.detach())
        feat_c = feat_c + self.pe.expand(b, -1, -1, -1)

        offset = self.offset_regression(feat_c)

        x = self.regressor(feat_c)
        coeff_map = self.deform_conv2d(x, offset)

        return {'coeff_map_init': coeff_map}
