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
        out = self.deform_conv2d(x, offset, mask)
        return out

class Conv_ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(Conv_ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += x
        out = self.relu(out)
        return out


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

        self.sf = self.cfg.scale_factor['img']
        self.seg_sf = self.cfg.scale_factor['seg']
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

        self.window_size = cfg.window_size
        self.c_feat = 64

        self.flow_estimator = torch.nn.Sequential(
            conv_relu(self.cfg.window_size**2 + self.c_feat, self.c_feat, kernel_size=1),
            conv_relu(self.c_feat, self.c_feat, kernel_size=1),
            conv_relu(self.c_feat, self.c_feat, kernel_size=3, stride=2, padding=1),
            conv_relu(self.c_feat, self.c_feat, kernel_size=3, stride=1, padding=2, dilation=2),
            conv_relu(self.c_feat, self.c_feat, kernel_size=3, stride=2, padding=1),
            conv_relu(self.c_feat, self.c_feat, kernel_size=3, stride=1, padding=2, dilation=2),
            Conv_ResBlock(self.c_feat, self.c_feat, kernel_size=3, stride=1, padding=1),
            Conv_ResBlock(self.c_feat, self.c_feat, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, 2, kernel_size=1),
        )

        self.feat_embed = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
        )

        self.feat_guide = torch.nn.Sequential(
            conv_bn_relu(1, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            torch.nn.Conv2d(self.c_feat, self.c_feat, 1),
        )

        self.feat_aggregator = torch.nn.Sequential(
            conv_bn_relu(self.c_feat * 3, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, self.c_feat, 1)
        )

        self.classifier = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, 2, 1)
        )
        self.grid_generator()

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


    def grid_generator(self):
        x = np.linspace(0, self.cfg.width // self.seg_sf[0] - 1, self.cfg.width // self.seg_sf[0])
        y = np.linspace(0, self.cfg.height // self.seg_sf[0] - 1, self.cfg.height // self.seg_sf[0])
        grid_xy = np.float32(np.meshgrid(x, y))
        _, h, w = grid_xy.shape
        self.grid_xy = to_tensor(grid_xy).permute(1, 2, 0).view(1, h, w, 2)
        self.grid_xy[:, :, :, 0] = (self.grid_xy[:, :, :, 0] / (self.cfg.width // self.seg_sf[0] - 1) - 0.5) * 2
        self.grid_xy[:, :, :, 1] = (self.grid_xy[:, :, :, 1] / (self.cfg.height // self.seg_sf[0] - 1) - 0.5) * 2

    def local_window_attention(self, query_data, key_data):
        b, c, h, w = key_data.shape
        query_data = query_data.permute(0, 2, 3, 1).reshape(-1, c, 1)
        key_data = F.unfold(key_data, kernel_size=(self.window_size, self.window_size), stride=1, padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size**2, h, w).permute(0, 3, 4, 1, 2)
        key_data = key_data.reshape(-1, c, self.window_size**2)

        correlation = torch.bmm(query_data.permute(0, 2, 1), key_data) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size**2).permute(0, 3, 1, 2)
        return sim_map

    def forward_for_flow_estimation(self, query_data, key_data):
        query_data = self.feat_embed(query_data)
        key_data = self.feat_embed(key_data)

        cost_v = self.local_window_attention(query_data, key_data)

        x = torch.cat((cost_v, query_data), dim=1)
        flow = self.flow_estimator(x)
        flow = torch.nn.functional.interpolate(flow, scale_factor=4, mode='bilinear')

        b = len(flow)
        grid = self.grid_xy.expand(b, -1, -1, -1) + flow.permute(0, 2, 3, 1)

        return flow, grid

    def forward_for_mask_generation(self, query_data, key_data):
        data_combined = torch.cat((query_data, key_data), dim=1)
        mask = self.mask_generator(data_combined)
        mask = torch.sigmoid(mask)
        return mask

    def forward_for_feat_aggregation(self, is_training=False):
        key_t = f't-{0}'
        query_img_feat = self.memory['img_feat'][key_t]
        query_probmap = self.memory['prob_map'][key_t][:, 1:]
        for t in range(1, self.cfg.num_t + 1):
            key_t = f't-{t}'
            key_img_feat = self.memory['img_feat'][key_t]
            key_probmap = self.memory['prob_map'][key_t][:, 1:]
            key_guide = self.memory['guide_cls'][key_t]

            flow_c, grid_c = self.forward_for_flow_estimation(query_img_feat, key_img_feat)
            aligned_key_guide = F.grid_sample(key_guide, grid_c, mode='bilinear', padding_mode='zeros')
            aligned_key_img_feat = F.grid_sample(key_img_feat, grid_c, mode='bilinear', padding_mode='zeros')
            aligned_key_probmap = F.grid_sample(key_probmap, grid_c, mode='bilinear', padding_mode='zeros')

            feat_guide = self.forward_for_guidance_feat(aligned_key_guide)

            self.img_feat = self.feat_aggregator(torch.cat((query_img_feat, aligned_key_img_feat, feat_guide), dim=1))

        return {'key_probmap': key_probmap,
                'key_guide': key_guide,
                'aligned_key_probmap': aligned_key_probmap,
                'aligned_key_guide': aligned_key_guide,
                'grid': grid_c}

    def forward_for_guidance_feat(self, guide_map):
        # data = torch.cat((prob_map, guide_map), dim=1)
        feat_guide = self.feat_guide(guide_map)
        return feat_guide

    def forward_for_classification(self):
        out = self.classifier(self.img_feat)
        self.prob_map = F.softmax(out, dim=1)
        return {'seg_map_logit': out,
                'seg_map': self.prob_map[:, 1:2]}

    def forward_for_regression(self):
        b, _, _, _ = self.prob_map.shape
        feat_c = self.feat_embedding(self.prob_map[:, 1:].detach())
        feat_c = feat_c + self.pe.expand(b, -1, -1, -1)
        offset = self.offset_regression(feat_c)
        x = self.regressor(feat_c)
        coeff_map = self.deform_conv2d(x, offset)

        return {'coeff_map': coeff_map}

