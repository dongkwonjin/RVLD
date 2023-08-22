import os
import cv2
import torch
import math

import numpy as np

from libs.utils import *

class Post_Processing(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sf = cfg.scale_factor['seg']

        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

    def draw_polyline_cv(self, pts, color=(255, 0, 0), s=1):
        out = np.ascontiguousarray(np.zeros(self.seg_map.shape, dtype=np.float32))
        out = cv2.polylines(out, pts, False, color, s)
        return out

    def measure_confidence_score(self, seg_map, lane_mask):
        lane_mask[:self.height_idx[0]] = 0
        score = np.sum(lane_mask * seg_map) / (np.sum(lane_mask) + 1e-8)
        return score

    def run_for_nms(self):
        seg_map = to_np(self.seg_map)
        coeff_map = self.coeff_map.clone()
        h, w = seg_map.shape

        if self.mode == 'init':
            nms_thresd = 0.6
        elif self.mode == 'f':
            nms_thresd = self.cfg.nms_thresd
        for i in range(self.cfg.max_lane_num * 2):
            idx_max = np.argmax(seg_map)
            if len(self.out['idx']) >= self.cfg.max_lane_num:
                break
            if seg_map[idx_max // w, idx_max % w] > nms_thresd:
                coeff = coeff_map[:, idx_max // w, idx_max % w]
                # removal
                x_coords = torch.matmul(self.U, coeff) * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
                x_coords = x_coords / (self.cfg.width - 1) * (self.cfg.width // self.sf[0] - 1)
                y_coords = to_tensor(self.cfg.py_coord).view(1, len(x_coords), 1) / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1)
                x_coords = x_coords.view(1, len(x_coords), 1)
                lane_coords = torch.cat((x_coords, y_coords), dim=2)
                lane_mask = self.draw_polyline_cv(np.int32(to_np(lane_coords)), color=(1, 1, 1), s=self.cfg.removal['lane_width'])
                lane_mask2 = self.draw_polyline_cv(np.int32(to_np(lane_coords)), color=(1, 1, 1), s=1)
                score = self.measure_confidence_score(seg_map, lane_mask2)
                seg_map[idx_max // w, idx_max % w] = 0
                if score >= 0.3:
                    self.out['idx'].append(int(idx_max))
                    self.out['coeff'].append(coeff.view(-1, 1))
                    seg_map = seg_map * (1 - lane_mask)
                    self.out['lane_pts'].append(np.int32(to_np(lane_coords)))
            else:

                break

    def run_for_coeff_to_x_coord_conversion(self):
        x_coords = list()

        coeff_results = self.out['coeff']
        if len(coeff_results) != 0:
            coeff_results = torch.cat(coeff_results, dim=1)
        if len(coeff_results) != 0:
            x_coords = torch.matmul(self.U, coeff_results)
            x_coords = x_coords * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
            x_coords = x_coords.permute(1, 0)

        self.x_coords = x_coords
        return {'x_coords': x_coords}

    def run_for_height_filtering(self):
        idxlist = to_np(torch.sum((self.seg_map > self.cfg.height_thresd), dim=1)).nonzero()[0]
        if len(idxlist) > 0:
            idx_ed = idxlist[0] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1)
            idx_st = idxlist[-1] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1)
            lane_idx_ed = np.argmin(np.abs(self.cfg.py_coord - idx_ed))
            lane_idx_st = np.argmin(np.abs(self.cfg.py_coord - idx_st))
            self.height_idx = [lane_idx_ed, lane_idx_st]
            return {'height_idx': [lane_idx_ed, lane_idx_st]}
        else:
            self.height_idx = None
            return {'height_idx': None}

    def measure_IoU(self, X1, X2):
        ep = 1e-7
        X = X1 + X2
        X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
        X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)
        iou = X_inter / (X_uni + ep)
        return iou

    def lane_mask_generation_for_training(self, data, gt):
        out = dict()
        out['guide_cls'] = list()
        out['guide_num'] = list()
        h, w = self.seg_map.shape
        for i in range(len(data)):
            if len(data[i]['lane_pts']) > 0:
                lane_mask = self.draw_polyline_cv(data[i]['lane_pts'], color=(1, 1, 1), s=1)
                lane_mask = cv2.dilate(lane_mask, kernel=(3, 3), iterations=1)

            else:
                lane_mask = np.zeros(shape=self.seg_map.shape, dtype=np.float32)
            height_idx = data[i]['height_idx']
            if height_idx is not None:
                h_idx1 = int(self.cfg.py_coord[np.minimum(height_idx[0] + 1, len(self.cfg.py_coord) - 1)] / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1))
                lane_mask[:h_idx1] = 0
            lane_mask = to_tensor(lane_mask).view(1, h, w)
            gt_lane_mask = gt['seg_label'][self.sf[0]][i].view(1, h, w).type(torch.float32)
            iou = self.measure_IoU(lane_mask, gt_lane_mask)
            if iou > 0.6:
                lane_mask = lane_mask
            else:
                case = random.randint(0, 1)
                if case == 0:
                    lane_mask = gt_lane_mask
                else:
                    lane_mask = torch.zeros(size=(1, h, w), dtype=torch.float32).cuda()
            case_neg = random.randint(0, 1)
            if case_neg == 1:
                lane_mask += gt['guide_mask_neg'][i].view(1, h, w)
            lane_mask_f = ((lane_mask) != 0).type(torch.float32)
            out['guide_cls'].append(lane_mask_f)

        lane_mask = torch.cat(out['guide_cls'])
        b, h, w = lane_mask.shape
        out['guide_cls'] = lane_mask.view(b, 1, h, w)
        return out

    def lane_mask_generation(self, data):
        out = dict()
        out['guide_cls'] = list()
        out['guide_num'] = list()

        for i in range(len(data)):
            if len(data[i]['lane_pts']) > 0:
                lane_mask = self.draw_polyline_cv(data[i]['lane_pts'], color=(1, 1, 1), s=1)
                lane_mask = cv2.dilate(lane_mask, kernel=(3, 3), iterations=1)

            else:
                lane_mask = np.zeros(shape=self.seg_map.shape, dtype=np.float32)
            height_idx = data[i]['height_idx']
            if height_idx is not None:
                h_idx1 = int(self.cfg.py_coord[np.minimum(height_idx[0] + 1, len(self.cfg.py_coord) - 1)] / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1))
                lane_mask[:h_idx1] = 0
            out['guide_cls'].append(lane_mask)
            out['guide_num'].append(len(data[i]['lane_pts']))

        lane_mask = to_tensor(np.array(out['guide_cls']))
        b, h, w = lane_mask.shape
        out['guide_cls'] = lane_mask.view(b, 1, h, w)
        return out

    def run_for_test(self, data, batch_idx):
        # results
        out_f = list()

        if self.mode == 'init':
            self.seg_map = data['seg_map_init'][batch_idx, 0]
            self.coeff_map = data['coeff_map_init'][batch_idx]
        elif self.mode == 'f':
            self.seg_map = data['seg_map'][0, 0]
            self.coeff_map = data['coeff_map'][0]

        self.out = dict()
        self.out['coeff'] = list()
        self.out['lane_pts'] = list()
        self.out['idx'] = list()
        self.out['height'] = list()

        self.out.update(self.run_for_height_filtering())
        self.run_for_nms()
        self.out.update(self.run_for_coeff_to_x_coord_conversion())

        out_f.append(self.out)

        return out_f

    def run_for_training(self, data):
        # results
        out_f = list()
        b = len(data['seg_map_init']) if self.mode == 'init' else len(data['seg_map'])

        for i in range(b):
            if self.mode == 'init':
                self.seg_map = data['seg_map_init'][i, 0]
                self.coeff_map = data['coeff_map_init'][i]
            elif self.mode == 'f':
                self.seg_map = data['seg_map'][i, 0]
                self.coeff_map = data['coeff_map'][i]

            self.out = dict()
            self.out['coeff'] = list()
            self.out['lane_pts'] = list()
            self.out['idx'] = list()
            self.out['height'] = list()

            self.out.update(self.run_for_height_filtering())
            self.run_for_nms()
            self.out.update(self.run_for_coeff_to_x_coord_conversion())

            out_f.append(self.out)

        return out_f

    def update(self, mode=None):
        self.mode = mode