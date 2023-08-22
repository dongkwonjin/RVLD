import cv2
import numpy as np

from libs.utils import *

class Visualize(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)
        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = dict()

    def update_data(self, data, name):
        self.show[name] = data

    def draw_polyline(self, data, name, ref_name='img', color=(255, 0, 0), s=1):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        pts = np.int32(data).reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], False, color, s, lineType=cv2.LINE_AA)
        self.show[name] = img

    def display_imglist(self, dir_name, file_name, list):
        disp = self.line
        for i in range(len(list)):
            if list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, disp)

    def update_datalist(self, img, img_name, label, dir_name, file_name, img_idx):
        self.update_data(img, 'img')
        self.update_data(img_name, 'img_name')
        self.update_data(label, 'label')

        self.dir_name = dir_name
        self.file_name = file_name
        self.img_idx = img_idx

        self.show['img_overlap'] = np.copy(self.show['img'])
        self.show['label_overlap'] = np.copy(self.show['label'])

    def draw_lanes_for_datalist(self, lane_pts):
        self.draw_polyline(data=lane_pts, name='img_overlap', ref_name='img_overlap', color=(0, 255, 0))
        self.draw_polyline(data=lane_pts, name='label_overlap', ref_name='label_overlap', color=(0, 255, 0))

    def save_datalist(self, error_list):
        if error_list[0] == False:
            dir_name = f'{self.cfg.dir["out"]}/display/{self.dir_name}/'
            file_name = f'{self.file_name}.jpg'
            self.display_imglist(dir_name, file_name, list=['img', 'img_overlap', 'label_overlap'])
        if error_list[0] == True:
            dir_name = f'{self.cfg.dir["out"]}/display_error/{self.dir_name}/'
            file_name = f'{self.file_name}.jpg'
            self.display_imglist(dir_name, file_name, list=['img', 'img_overlap', 'label_overlap'])