import cv2
import torch

import numpy as np

from scipy import interpolate
from sklearn.linear_model import LinearRegression

from libs.utils import *

class LaneEval(object):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.lr = LinearRegression()
        self.pixel_thresh = 20
        self.pt_thresh = 0.85

    def get_angle(self, xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            self.lr.fit(ys[:, None], xs)
            k = self.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    def line_accuracy(self, pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    def bench(self, pred, gt_x, gt_y):
        angles = [self.get_angle(np.array(gt_x[i]), np.array(gt_y[i])) for i in range(len(gt_x))]
        threshs = [self.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        idx = 0
        for x_gts, y_gts, thresh in zip(gt_x, gt_y, threshs):
            accs = []
            for i in range(len(self.f_interp1d)):
                f = self.f_interp1d[i]
                y_min = np.min(self.pred[i][:, 1])
                y_max = np.max(self.pred[i][:, 1])
                check = (y_min <= y_gts) * (y_gts <= y_max)
                x_pred = np.zeros(check.shape[0], dtype=np.float32) - 2
                x_pred[check] = f(y_gts[check])
                accs.append(self.line_accuracy(np.array(x_pred), np.array(x_gts), thresh))

            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < self.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
            idx += 1
        fp = len(pred) - matched
        if len(gt_x) > 8 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt_x) > 8:
            s -= min(line_accs)
        return s / max(min(8.0, len(gt_x)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt_x), 8.) , 1.)

    def load_gt_data(self):
        data = load_pickle(f'{self.cfg.dir["pre0_test"]}/{self.file_name}')
        self.gt_x = list()
        self.gt_y = list()
        for i in range(len(data['lane'])):
            if len(data['lane'][i]['points']) == 0:
                continue
            self.gt_x.append(np.array(data['lane'][i]['points'])[:, 0])
            self.gt_y.append(np.array(data['lane'][i]['points'])[:, 1])

    def load_pred_data(self):
        self.pred = load_pickle(f'{self.cfg.dir["out"]}/{self.mode}/results/acc/{self.file_name}')
        self.f_interp1d = list()
        for i in range(len(self.pred)):
            x = self.pred[i][:, 0]
            y = self.pred[i][:, 1]
            self.f_interp1d.append(interpolate.interp1d(y, x, kind='cubic'))

    def measure_accuracy(self, mode='test'):
        self.mode = mode

        datalist = load_pickle(f'{self.cfg.dir["out"]}/{mode}/pickle/datalist')
        num = len(datalist)
        total_acc, total_fp, total_fn = 0, 0, 0
        results = list()
        for i in range(num):
            self.file_name = datalist[i]

            self.load_gt_data()
            self.load_pred_data()

            acc, fp, fn = self.bench(self.pred, self.gt_x, self.gt_y)
            total_acc += acc
            total_fp += fp
            total_fn += fn

            results.append(acc)

        total_acc = total_acc / num
        total_fp = total_fp / num
        total_fn = total_fn / num

        save_pickle(path=f'{self.cfg.dir["out"]}/{self.mode}/results/acc/results', data=results)

        print('---------Performance %s---------\n'
              'acc %5f / fp %5f/ fn %5f' % ("VIL-100", total_acc, total_fp, total_fn))

        return {'acc': total_acc, 'fp': total_fp, 'fn': total_fn}
