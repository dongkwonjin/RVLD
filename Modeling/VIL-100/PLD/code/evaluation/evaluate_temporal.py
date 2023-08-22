# from https://github.com/lucastabelini/LaneATT
import os
import argparse
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon

from libs.utils import *

def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(255, 255, 255), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    eps = 1e-10
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / ((x | y).sum() + eps)
    return ious

def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in xs]
    ys = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious

def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T

def culane_metric(pred, anno, shape, width=30, iou_threshold=0.5, official=True, img_shape=(590, 1640, 3)):
    if len(pred) == 0:
        return 0, 0, len(anno), np.zeros(len(pred)), np.zeros(len(pred), dtype=bool)
    if len(anno) == 0:
        return 0, len(pred), 0, np.zeros(len(pred)), np.zeros(len(pred), dtype=bool)
    interp_pred = np.array([interp(list(dict.fromkeys(pred_lane)), n=5) for pred_lane in pred], dtype=object)  # (4, 50, 2)
    interp_anno = np.array([interp(list(dict.fromkeys(anno_lane)), n=5) for anno_lane in anno], dtype=object)  # (4, 50, 2)

    if official:
        ious = discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=shape)
    else:
        ious = continuous_cross_iou(interp_pred, interp_anno, width=width, img_shape=shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    tp = int((ious[row_ind, col_ind] > iou_threshold).sum())
    fp = len(pred) - tp
    fn = len(anno) - tp
    pred_ious = np.zeros(len(pred))
    pred_ious[row_ind] = ious[row_ind, col_ind]
    return tp, fp, fn, pred_ious, pred_ious > iou_threshold

def culane_metric2(pred, anno, shape, width=30, iou_threshold=0.5, official=True, img_shape=(590, 1640, 3)):
    if len(pred) == 0:
        return [], [], []
    if len(anno) == 0:
        return [], [], []
    interp_pred = np.array([interp(list(dict.fromkeys(pred_lane)), n=5) for pred_lane in pred], dtype=object)  # (4, 50, 2)
    interp_anno = np.array([interp(list(dict.fromkeys(anno_lane)), n=5) for anno_lane in anno], dtype=object)  # (4, 50, 2)

    if official:
        ious = discrete_cross_iou(interp_anno, interp_pred, width=width, img_shape=shape)
    else:
        ious = continuous_cross_iou(interp_anno, interp_pred, width=width, img_shape=shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    return row_ind, col_ind, ious


def load_culane_img_data(path):
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data

def load_culane_data(data_dir, file_list_path):
    with open(file_list_path, 'r') as file_list:
        filepaths = [
            # os.path.join(data_dir, line[1 if line[0] == '/' else 0:].rstrip().replace('.jpg', '.lines.txt'))
            os.path.join(data_dir, line[1 if line[0] == '/' else 0:].rstrip() + '.lines.txt')
            for line in file_list.readlines()
        ]

    data = []
    for path in tqdm(filepaths):
        img_data = load_culane_img_data(path)
        data.append(img_data)

    return data

class LaneEval_Temporal(object):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def settings(self, test_mode, iou):
        self.lane_width = 30
        self.official = True
        self.sequential = False
        self.iou = iou
        self.num_t = 1
        self.iou_threshold = iou

        self.test_mode = test_mode
        self.out_dir = f'{self.cfg.dir["out"]}/{test_mode}'
        self.anno_dir = f'{self.cfg.dir["dataset"]}/txt/anno_txt/'
        self.pred_dir = f'{self.out_dir}/results/iou/txt/pred_txt'
        self.list_path = f'{self.pred_dir}/datalist.txt'

        datalist = load_pickle(f'{self.out_dir}/pickle/datalist')
        temp = [x + '\n' for x in datalist]
        with open(self.list_path, 'w') as g:
            g.writelines(temp)

        if os.path.exists(f'{self.out_dir}/pickle/datalist_video.pickle') == False:
            self.get_video_datalist(datalist)

        self.Ns = 0
        self.Nj = 0
        self.Nm = 0

    def get_video_datalist(self, datalist):
        datalist_video = dict()
        for data_name in datalist:
            dir_name = os.path.dirname(data_name)
            file_name = os.path.basename(data_name)

            if dir_name not in datalist_video.keys():
                datalist_video[dir_name] = list()
            datalist_video[dir_name].append(data_name)

        save_pickle(f'{self.out_dir}/pickle/datalist_video', data=datalist_video)

    def measure_IoU(self, mode, iou=0.5):
        print('culane laneatt metric evaluation start!')
        self.settings(mode, iou)
        results = self.eval_predictions()
        header = '=' * 20 + 'Results ({})'.format(os.path.basename(self.list_path)) + '=' * 20
        print(header)
        for metric, value in results.items():
            if isinstance(value, float):
                print('{}: {:.4f}'.format(metric, value))
            else:
                print('{}: {}'.format(metric, value))
        print('=' * len(header))
        print('culane laneatt metric evaluation done!')
        return results

    def eval_predictions(self):
        print('List: {}'.format(self.list_path))
        print('Loading prediction data...')
        predictions = load_culane_data(self.pred_dir, self.list_path)
        print('Loading annotation data...')
        annotations = load_culane_data(self.anno_dir, self.list_path)
        print('Calculating metric {}...'.format('sequentially' if self.sequential else 'in parallel'))
        img_shape = load_pickle(f'{self.out_dir}/results/iou/shape_list')
        if self.sequential:
            results_s = t_map(partial(culane_metric2, width=self.lane_width, iou_threshold=self.iou, official=self.official, img_shape=img_shape), predictions, annotations, img_shape)
        else:
            results_s = p_map(partial(culane_metric2, width=self.lane_width, iou_threshold=self.iou, official=self.official, img_shape=img_shape), predictions, annotations, img_shape)
        save_pickle(path=f'{self.pred_dir}/results', data=results_s)

        # video metric
        results_s = load_pickle(f'{self.pred_dir}/results')
        datalist_video = load_pickle(f'{self.out_dir}/pickle/datalist_video')
        video_list = list(datalist_video)
        idx = 0
        results_t = list()
        for video_name in tqdm(video_list):
            for j in range(len(datalist_video[video_name])):
                self.shape = img_shape[idx]
                if j == 0:
                    self.forward_for_dict_initialization()
                    self.data['pred']['t-0'] = predictions[idx]
                    self.data['anno']['t-0'] = annotations[idx]
                    results = results_s[idx]
                    self.data['results']['t-0'] = results
                    self.memory_t += 1
                else:
                    self.forward_for_dict_memorization()
                    self.data['pred']['t-0'] = predictions[idx]
                    self.data['anno']['t-0'] = annotations[idx]
                    results = results_s[idx]
                    self.data['results']['t-0'] = results
                    self.memory_t += 1

                    self.matching_lane_instance()

                    results_t.append(self.metric_per_inter_frame())
                idx += 1
                # if idx % 100 == 0:
                #     print(f'{idx} == > {datalist_video[video_name][j]} done')

        save_pickle(path=f'{self.pred_dir}/results_t', data=results_t)
        results_t = load_pickle(f'{self.pred_dir}/results_t')

        total_ns = sum(ns for ns, _, _ in results_t)
        total_nj = sum(nj for _, nj, _ in results_t)
        total_nm = sum(nm for _, _, nm in results_t)
        R_s = float(total_ns) / (total_ns + total_nj + total_nm)
        R_j = float(total_nj) / (total_ns + total_nj + total_nm)
        R_m = float(total_nm) / (total_ns + total_nj + total_nm)
        print(f'Ns: {total_ns}, Nj: {total_nj}, Nm: {total_nm}, Rs: {R_s}, Rj: {R_j}, Rm: {R_m}')
        print(f'total N_s N_j N_m ==> {self.Ns} {self.Nj} {self.Nm}')
        return {'Ns': total_ns, 'Nj': total_nj, 'Nm': total_nm, 'Rs': R_s, 'Rj': R_j, 'Rm': R_m}

    def metric_per_inter_frame(self):
        Ns = 0
        Nj = 0
        Nm = 0
        for t in range(self.num_t):
            anno_match_row_ind = self.match_data[t]['row_ind']  # anno at t
            anno_match_col_ind = self.match_data[t]['col_ind']  # anno at t-1

            row_ind1, col_ind1, ious1 = self.data['results']['t-0']
            row_ind2, col_ind2, ious2 = self.data['results'][f't-{t+1}']

            for i in range(len(anno_match_row_ind)):
                current_anno_idx = anno_match_row_ind[i]
                tmp_idx1 = (row_ind1 == current_anno_idx).nonzero()[0]
                if len(tmp_idx1) != 0:
                    current_pred_idx = col_ind1[tmp_idx1]
                    iou1 = ious1[current_anno_idx, current_pred_idx]
                else:
                    iou1 = 0

                prev_anno_idx = anno_match_col_ind[i]
                tmp_idx2 = (row_ind2 == prev_anno_idx).nonzero()[0]
                if len(tmp_idx2) != 0:
                    prev_pred_idx = col_ind2[tmp_idx2]
                    iou2 = ious2[prev_anno_idx, prev_pred_idx]
                else:
                    iou2 = 0
                if ((iou1 > self.iou_threshold) and (iou2 < self.iou_threshold)) or ((iou1 < self.iou_threshold) and (iou2 > self.iou_threshold)):
                    Nj += 1
                    self.Nj += 1
                elif (iou1 < self.iou_threshold) and (iou2 < self.iou_threshold):
                    Nm += 1
                    self.Nm += 1
                else:
                    Ns += 1
                    self.Ns += 1

        return Ns, Nj, Nm

    def matching_lane_instance(self, width=30, iou_threshold=0.5):
        self.match_data = list()
        for t in range(self.num_t):
            anno_ref = self.data['anno']['t-0']
            anno_tgt = self.data['anno'][f't-{t+1}']

            interp_ref = np.array([interp(list(dict.fromkeys(pred_lane)), n=5) for pred_lane in anno_ref], dtype=object)  # (4, 50, 2)
            interp_tgt = np.array([interp(list(dict.fromkeys(anno_lane)), n=5) for anno_lane in anno_tgt], dtype=object)  # (4, 50, 2)

            ious = discrete_cross_iou(interp_ref, interp_tgt, width=width, img_shape=self.shape)
            row_ind, col_ind = linear_sum_assignment(1 - ious)
            check = ious[row_ind, col_ind] > iou_threshold
            valid_row_ind = row_ind[check]
            valid_col_ind = col_ind[check]
            match_data = dict()
            match_data['row_ind'] = valid_row_ind
            match_data['col_ind'] = valid_col_ind
            self.match_data.append(match_data)

    def forward_for_dict_initialization(self):
        self.keylist = ['pred', 'anno', 'results']
        self.data = dict()
        for key in self.keylist:
            self.data[key] = dict()
        self.memory_t = 0

    def forward_for_dict_memorization(self):
        for i in range(self.memory_t - 1, -1, -1):
            for key in self.keylist:
                self.data[key][f't-{i+1}'] = self.data[key][f't-{i}']

        for key in self.keylist:
            self.data[key].pop('t-0')
        if self.memory_t >= self.cfg.num_t:
            self.memory_t -= 1

    def metric_per_frame(self, width=30):
        pred = self.data['pred']['t-0']
        anno = self.data['anno']['t-0']
        if len(pred) == 0:
            return [], [], [], 0, 0, len(anno)
        if len(anno) == 0:
            return [], [], [], 0, len(pred), 0
        interp_pred = np.array([interp(list(dict.fromkeys(pred_lane)), n=5) for pred_lane in pred], dtype=object)  # (4, 50, 2)
        interp_anno = np.array([interp(list(dict.fromkeys(anno_lane)), n=5) for anno_lane in anno], dtype=object)  # (4, 50, 2)

        ious = discrete_cross_iou(interp_anno, interp_pred, width=width, img_shape=self.shape)

        row_ind, col_ind = linear_sum_assignment(1 - ious)

        return row_ind, col_ind, ious