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

    def coeff_map_to_xy_coord_map_conversion(self, coeff_map):
        top_m = self.cfg.top_m
        _, h, w = coeff_map.shape
        coeff_map = coeff_map.view(top_m, -1, 1).permute(1, 0, 2)
        U = self.U.view(1, -1, top_m).expand(coeff_map.shape[0], -1, top_m)
        x_coord_map = torch.bmm(U, coeff_map) * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
        y_coord_map = to_tensor(self.cfg.py_coord).view(1, -1, 1).expand(h * w, -1, 1)
        lane_coord_map = torch.cat((x_coord_map, y_coord_map), dim=2)
        return lane_coord_map

    def coeff_map_to_xy_coord_map_conversion2(self, coeff_map):
        top_m = self.cfg.top_m
        _, n = coeff_map.shape
        coeff_map = coeff_map.view(top_m, -1, 1).permute(1, 0, 2)
        U = self.U.view(1, -1, top_m).expand(coeff_map.shape[0], -1, top_m)
        x_coord_map = torch.bmm(U, coeff_map) * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
        y_coord_map = to_tensor(self.cfg.py_coord).view(1, -1, 1).expand(n, -1, 1)
        lane_coord_map = torch.cat((x_coord_map, y_coord_map), dim=2)
        return lane_coord_map

    def overlap_lane_coord_map(self, lane_coord_map, seg_map):
        idx = (seg_map.view(-1) > 0.6)

        pos_lane_coord_map = lane_coord_map[idx == True]
        pos_lane_coord_map = to_np2(pos_lane_coord_map)
        pos_out_map = np.zeros((self.cfg.height, self.cfg.width), dtype=np.int64)
        for i in range(0, len(pos_lane_coord_map), 3):
            lane_coord = pos_lane_coord_map[i]
            self.show['temp'] = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
            self.draw_polyline_cv(lane_coord, name='temp', ref_name='temp', color=1, s=1)
            pos_out_map += self.show['temp']
        pos_out_map = (pos_out_map != 0)
        pos_out_map = np.uint8(to_3D_np(pos_out_map) * 255)
        return pos_out_map

    def overlap_lane_coord_map2(self, lane_coord_map):
        pos_lane_coord_map = lane_coord_map
        pos_lane_coord_map = to_np2(pos_lane_coord_map)
        pos_out_map = np.zeros((self.cfg.height, self.cfg.width), dtype=np.int64)
        for i in range(len(pos_lane_coord_map)):
            lane_coord = pos_lane_coord_map[i]
            self.show['temp'] = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
            self.draw_polyline_cv(lane_coord, name='temp', ref_name='temp', color=1, s=1)
            pos_out_map += self.show['temp']
        pos_out_map = (pos_out_map != 0)
        pos_out_map = np.uint8(to_3D_np(pos_out_map) * 255)
        return pos_out_map

    def draw_selected_lane_coords(self, x_coords, height_idx=None, color=(0, 255, 0), s=2):
        self.show['temp'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        if len(x_coords) != 0:
            for i in range(len(x_coords)):

                if height_idx is None:
                    x_coord = x_coords[i]
                    y_coord = to_tensor(self.cfg.py_coord).view(1, -1, 1)
                else:
                    x_coord = x_coords[i][height_idx[0]:height_idx[1]]
                    y_coord = to_tensor(self.cfg.py_coord[height_idx[0]:height_idx[1]]).view(1, -1, 1)

                lane_coords = to_np(torch.cat((x_coord.view(1, len(x_coord), 1), y_coord), dim=2))
                self.draw_polyline_cv(lane_coords, name='temp', ref_name='temp', color=(255, 255, 255), s=2)
                self.draw_polyline_cv(lane_coords, name='label_overlap', ref_name='label_overlap', color=color, s=s)
                self.draw_polyline_cv(lane_coords, name='img_overlap', ref_name='img_overlap', color=color, s=s)
                self.draw_polyline_cv(lane_coords, name='seg_overlap', ref_name='seg_overlap', color=color, s=s)
        return self.show['temp']

    def display_for_train(self, batch, out, batch_idx):
        self.update_image(batch['img'][0], name='img')
        self.update_image_name(batch['img_name'][0])

        lane_coord_map = self.coeff_map_to_xy_coord_map_conversion(out['coeff_map'][0])

        self.show['coeff_pos_map'] = self.overlap_lane_coord_map(lane_coord_map, out['seg_map'][0])

        self.show['seg_map'] = self.b_map_to_rgb_image(out['seg_map'][0])
        self.show['seg_map_gt'] = self.b_map_to_rgb_image(batch['seg_label'][self.sf[0]][0:1])

        save_namelist = ['img', 'coeff_pos_map', 'lane_mask', 'seg_map', 'seg_map_gt']
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
        self.show['coeff_cls_map'] = self.draw_selected_lane_coords(out['x_coords'], height_idx=out['height_idx'], color=(0, 255, 0), s=2)

        # save result
        dirname = os.path.dirname(self.show["img_name"])
        filename = os.path.basename(self.show["img_name"])
        self.display_imglist(path=f'{self.cfg.dir["out"]}/{mode}/display/{dirname}/{filename}.jpg',
                             list=['img', 'img_overlap', 'seg_map', 'seg_overlap', 'label_overlap', 'org_label'])

    def disp_errorlist(self, datalist, mode):
        for i in range(len(datalist)):
            dirname = os.path.dirname(datalist[i])
            filename = os.path.basename(datalist[i])
            src = f'{self.cfg.dir["out"]}/{mode}/display/{dirname}/{filename}.jpg'
            tgt = f'{self.cfg.dir["out"]}/{mode}/display_errorlist/{dirname}/{filename}.jpg'
            mkdir(os.path.dirname(tgt))
            shutil.copy(src, tgt)