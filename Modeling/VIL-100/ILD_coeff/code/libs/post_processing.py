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

    def draw_polyline_cv(self, data, color=(255, 0, 0), s=1):
        out = np.ascontiguousarray(np.zeros(self.seg_map.shape, dtype=np.uint8))
        pts = np.int32(data).reshape((-1, 1, 2))
        out = cv2.polylines(out, [pts], False, color, s)
        return out

    def measure_confidence_score(self, seg_map, lane_mask):
        lane_mask[:self.height_idx[0]] = 0
        lane_mask[self.height_idx[1] + 1:] = 0
        score = np.sum(lane_mask * seg_map) / np.sum(lane_mask)
        return score

    def run_for_nms(self):
        seg_map = to_np(self.seg_map)
        coeff_map = self.coeff_map.clone()
        h, w = seg_map.shape

        for i in range(self.cfg.max_lane_num * 2):
            idx_max = np.argmax(seg_map)
            if seg_map[idx_max // w, idx_max % w] > self.cfg.nms_thresd:
                coeff = coeff_map[:, idx_max // w, idx_max % w]
                # removal
                x_coords = torch.matmul(self.U, coeff) * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
                x_coords = x_coords / (self.cfg.width - 1) * (self.cfg.width // self.sf[0] - 1)
                y_coords = to_tensor(self.cfg.py_coord).view(1, len(x_coords), 1) / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1)
                x_coords = x_coords.view(1, len(x_coords), 1)
                lane_coords = torch.cat((x_coords, y_coords), dim=2)
                lane_mask = self.draw_polyline_cv(to_np(lane_coords), color=(1, 1, 1), s=self.cfg.removal['lane_width'])
                lane_mask2 = self.draw_polyline_cv(to_np(lane_coords), color=(1, 1, 1), s=1)
                score = self.measure_confidence_score(seg_map, lane_mask2)
                seg_map[idx_max // w, idx_max % w] = 0
                if score >= 0.2:
                    self.out['idx'].append(int(idx_max))
                    self.out['coeff'].append(coeff.view(-1, 1))
                    seg_map = seg_map * (1 - lane_mask)
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

    def run_for_vp_detection(self):
        if len(self.x_coords) > 1:
            x_coords = to_np(self.x_coords)
            vp_idx = np.argmin(np.mean(np.abs(np.mean(x_coords, axis=0, keepdims=True) - x_coords), axis=0))
        else:
            vp_idx = 0

        return {'vp_idx': vp_idx}

    def run_for_height_filtering(self):
        idxlist = to_np(torch.sum((self.seg_map > self.cfg.height_thresd), dim=1)).nonzero()[0]
        if len(idxlist) > 0:
            idx_ed = idxlist[0] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1)
            idx_st = idxlist[-1] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1)
            lane_idx_ed = np.argmin(np.abs(self.cfg.py_coord - idx_ed))
            lane_idx_st = np.argmin(np.abs(self.cfg.py_coord - idx_st)) + 1
            self.height_idx = [lane_idx_ed, lane_idx_st]
            return {'height_idx': [lane_idx_ed, lane_idx_st]}
        else:
            self.height_idx = None
            return {'height_idx': None}

    def run(self, data):
        # results
        out_f = list()
        b = len(data['seg_map'])

        for i in range(b):
            self.seg_map = data['seg_map'][i, 0]
            self.coeff_map = data['coeff_map'][i]
            self.batch_idx = i
            self.out = dict()
            self.out['coeff'] = list()
            self.out['idx'] = list()
            self.out['height'] = list()

            self.out.update(self.run_for_height_filtering())
            self.run_for_nms()
            self.out.update(self.run_for_coeff_to_x_coord_conversion())


            out_f.append(self.out)

        return out_f

    def update(self, batch, mode):
        self.mode = mode
        self.img_name = batch['img_name'][0]
