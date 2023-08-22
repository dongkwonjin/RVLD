import cv2
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np

from libs.utils import *

class Visualize_cv(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255
        self.show = {}

        self.sf = cfg.scale_factor['seg']
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

    def update_image(self, img, name='img'):
        img = to_np(img.permute(1, 2, 0))
        img = np.uint8((img * self.cfg.std + self.cfg.mean) * 255)[:, :, [2, 1, 0]]
        self.show[name] = img

    def update_label(self, label, name='label'):
        label = to_np(label)
        label = np.repeat(np.expand_dims(np.uint8(label != 0) * 255, axis=2), 3, 2)
        self.show[name] = label

    def update_data(self, data, name=None):
        self.show[name] = data

    def update_image_name(self, img_name):
        self.show['img_name'] = img_name

    def b_map_to_rgb_image(self, data):
        data = np.repeat(np.uint8(to_np2(data.permute(1, 2, 0) * 255)), 3, 2)
        data = cv2.resize(data, (self.cfg.width, self.cfg.height))
        return data

    def draw_text(self, pred, label, name, ref_name='img', color=(255, 0, 0)):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        cv2.rectangle(img, (1, 1), (250, 120), color, 1)
        cv2.putText(img, 'pred : ' + str(pred), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, 'label : ' + str(label), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        self.show[name] = img

    def draw_polyline_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        pts = np.int32(data).reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], False, color, s, lineType=cv2.LINE_AA)
        self.show[name] = img

    def display_imglist(self, path, list):
        # boundary line
        if self.show[list[0]].shape[0] != self.line.shape[0]:
            self.line = np.zeros((self.show[list[0]].shape[0], 3, 3), dtype=np.uint8)
            self.line[:, :, :] = 255
        disp = self.line

        for i in range(len(list)):
            if list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)

        mkdir(os.path.dirname(path))
        cv2.imwrite(path, disp)

    def display_for_train(self, batch, out, batch_idx):
        self.update_image(batch['img'][0], name='img')
        self.update_image_name(batch['img_name'][0])

        self.show['seg_map'] = self.b_map_to_rgb_image(out['seg_map'][0])
        self.show['seg_map_gt'] = self.b_map_to_rgb_image(batch['seg_label'][self.sf[0]][0:1])

        save_namelist = ['img', 'seg_map', 'seg_map_gt']
        # save result
        self.display_imglist(path=f'{self.cfg.dir["out"]}/train/display/{str(batch_idx)}.jpg',
                             list=save_namelist)

    def display_for_test(self, batch, out, batch_idx, mode):
        # img
        self.update_image(batch['img'][batch_idx], name='img')
        self.update_image_name(batch['img_name'][batch_idx])

        # label
        self.update_label(batch['org_label'][batch_idx], name='org_label')

        # output
        self.show['seg_map'] = self.b_map_to_rgb_image(out['seg_map'][batch_idx])

        self.show['label_overlap'] = self.show['org_label']
        self.show['seg_overlap'] = self.show['seg_map']
        self.show['img_overlap'] = self.show['img']

        # save result
        dirname = os.path.dirname(self.show["img_name"])
        filename = os.path.basename(self.show["img_name"])
        self.display_imglist(path=f'{self.cfg.dir["out"]}/{mode}/display/{dirname}/{filename}.jpg',
                             list=['img', 'seg_map', 'org_label'])

    def disp_errorlist(self, datalist, mode):
        for i in range(len(datalist)):
            dirname = os.path.dirname(datalist[i])
            filename = os.path.basename(datalist[i])
            src = f'{self.cfg.dir["out"]}/{mode}/display/{dirname}/{filename}.jpg'
            tgt = f'{self.cfg.dir["out"]}/{mode}/display_errorlist/{dirname}/{filename}.jpg'
            mkdir(os.path.dirname(tgt))
            shutil.copy(src, tgt)