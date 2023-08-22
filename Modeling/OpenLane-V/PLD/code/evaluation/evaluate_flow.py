import os
import cv2
import torch
import math

import numpy as np

from libs.utils import *

class Evaluation_Flow(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sf = cfg.scale_factor['seg']

    # def measure_IoU(self, X1, X2):
    #     ep = 1e-7
    #     X = X1 + X2
    #     X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
    #     X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)
    #     iou = X_inter / (X_uni + ep)
    #     return iou

    def run_for_fscore(self):
        table = (self.b_map + 1) * (self.seg_map_gt + 2)
        self.results['tp'] += list(to_np(torch.sum(table == 6, dim=(1, 2))))
        self.results['tn'] += list(to_np(torch.sum(table == 2, dim=(1, 2))))
        self.results['fp'] += list(to_np(torch.sum(table == 4, dim=(1, 2))))
        self.results['fn'] += list(to_np(torch.sum(table == 3, dim=(1, 2))))

    def measure(self):
        ep = 1e-7
        results = load_pickle(f'{self.cfg.dir["out"]}/{self.mode}/pickle/eval_seg_results')

        tp = np.sum(np.float32(results['tp']))
        fp = np.sum(np.float32(results['fp']))
        fn = np.sum(np.float32(results['fn']))
        precision = tp / (tp + fp + ep)
        recall = tp / (tp + fn + ep)
        fscore = 2 * precision * recall / (precision + recall + ep)

        print(f'\nprecision {precision}, recall {recall}, fscore {fscore}\n')
        return {'seg_precision': precision, 'seg_recall': recall, 'seg_fscore': fscore}

    def init(self):
        self.results = dict()
        self.results['tp'] = list()
        self.results['tn'] = list()
        self.results['fp'] = list()
        self.results['fn'] = list()
        self.results['precision'] = 0
        self.results['recall'] = 0
        self.results['fscore'] = 0

    def update(self, batch, out, mode):
        self.mode = mode
        try:
            self.seg_map = out['seg_map']
        except:
            self.seg_map = out['seg_map_init']
        self.b_map = (self.seg_map > self.cfg.prob_thresd).type(torch.long)[:, 0]
        self.seg_map_gt = batch['seg_label'][self.sf[0]]