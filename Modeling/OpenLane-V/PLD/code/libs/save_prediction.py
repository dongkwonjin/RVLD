import cv2
import shutil
import json
import torch

import numpy as np

from libs.utils import *

class Save_Prediction_for_eval_iou(object):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def make_file(self, name):
        txt_save_path = f'{self.output_dir}/{name}.lines.txt'
        self.txt_save_path = txt_save_path
        save_dir, _ = os.path.split(txt_save_path)
        mkdir(save_dir)

    def write_data(self):
        with open(self.txt_save_path, 'w') as f:
            for i in range(len(self.pred_lane)):
                for j in range(self.pred_lane[i].shape[0]):
                    f.write('%.3f %.3f ' % (self.pred_lane[i][j][0], self.pred_lane[i][j][1]))
                f.write('\n')

    def rescale_pts(self, pts):
        pts[:, :, 0] = pts[:, :, 0] / (self.cfg.width - 1) * (self.cfg.org_width - 1)
        pts[:, :, 1] = pts[:, :, 1] / (self.cfg.height - 1) * (self.cfg.org_height - self.cfg.crop_size - 1) + self.cfg.crop_size
        return pts

    def get_2D_lane_points(self, px_coord):
        lane_pts = list()
        if len(px_coord) != 0:
            px_coord = to_np(px_coord)
            py_coord = np.repeat(self.py_coord, len(px_coord), 0)
            px_coord = px_coord[:, :, np.newaxis]
            py_coord = py_coord[:, :, np.newaxis]
            lane_pts = np.concatenate((px_coord, py_coord), axis=2)
            lane_pts = self.rescale_pts(lane_pts)

        self.pred_lane = list()
        for i in range(len(lane_pts)):
            height_idx = 0
            if self.use_height:
                height_idx = self.height_idx
            self.pred_lane.append(lane_pts[i, height_idx[0]:])

    def load_pred_data(self):
        data = load_pickle(f'{self.cfg.dir["out"]}/{self.test_mode}/pickle/{self.file_name}')
        self.out = data['out']
        self.out_x_coord = self.out[self.key[0]]

        self.height_idx = self.out['height_idx']

        self.get_2D_lane_points(self.out_x_coord)

    def write_pred_data(self):
        self.make_file(self.file_name)
        self.write_data()

    def run(self):
        self.datalist = load_pickle(f'{self.cfg.dir["out"]}/{self.test_mode}/pickle/datalist')

        for i in range(len(self.datalist)):
            self.file_name = self.datalist[i].replace('.jpg', '')
            self.load_pred_data()
            self.write_pred_data()

    def settings(self, key, test_mode='test', use_height=True):
        self.key = key
        self.test_mode = test_mode
        self.use_height = use_height

        self.py_coord = np.float32(self.cfg.py_coord).reshape(1, -1)

        self.output_dir = f'{self.cfg.dir["out"]}/{test_mode}/results/iou/txt/pred_txt'

        # self.shape_list = list()
        mkdir(self.output_dir)
