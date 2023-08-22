import cv2
import math

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from models.backbone import *
from libs.utils import *

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
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

        # model
        self.encoder = resnet(layers=self.cfg.backbone, pretrained=True)
        backbone = self.cfg.backbone

        self.feat_squeeze1 = torch.nn.Sequential(
            conv_bn_relu(128, 64, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(64, self.c_feat, 3, padding=1),
        )
        self.feat_squeeze2 = torch.nn.Sequential(
            conv_bn_relu(256, 64, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(64, self.c_feat, 3, padding=1),
        )
        self.feat_squeeze3 = torch.nn.Sequential(
            conv_bn_relu(512, 64, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(64, self.c_feat, 3, padding=1),
        )

        self.feat_combine = torch.nn.Sequential(
            conv_bn_relu(self.c_feat * 3, 64, 3, padding=1, dilation=1),
            torch.nn.Conv2d(64, self.c_feat, 1),
        )

        self.feat_refine = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, padding=1, dilation=1)
        )

        self.classifier = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, 2, 1)
        )

    def forward_for_encoding(self, img):
        # Feature extraction
        feat1, feat2, feat3 = self.encoder(img)

        self.feat = dict()
        self.feat[self.sf[0]] = feat1
        self.feat[self.sf[1]] = feat2
        self.feat[self.sf[2]] = feat3

    def forward_for_squeeze(self):
        # Feature squeeze and concat
        x1 = self.feat_squeeze1(self.feat[self.sf[0]])
        x2 = self.feat_squeeze2(self.feat[self.sf[1]])
        x2 = torch.nn.functional.interpolate(x2, scale_factor=2, mode='bilinear')
        x3 = self.feat_squeeze3(self.feat[self.sf[2]])
        x3 = torch.nn.functional.interpolate(x3, scale_factor=4, mode='bilinear')
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x4 = self.feat_combine(x_concat)
        x4 = torch.nn.functional.interpolate(x4, scale_factor=2, mode='bilinear')
        self.img_feat = self.feat_refine(x4)

    def forward_for_classification(self):
        out = self.classifier(self.img_feat)
        self.prob_map = F.softmax(out, dim=1)

        return {'seg_map_logit_init': out,
                'seg_map_init': self.prob_map[:, 1:2]}
