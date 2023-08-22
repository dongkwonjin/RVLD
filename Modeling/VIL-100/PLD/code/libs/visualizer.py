import cv2
import math

import matplotlib.pyplot as plt
import numpy as np

from libs.utils import *

class Visualize_cv(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255
        self.line_h = np.zeros((3, cfg.width * 4 + 3*5, 3), dtype=np.uint8)
        self.line_h[:, :, :] = 255
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

    def save_data(self, path, data):
        mkdir(os.path.dirname(path))
        cv2.imwrite(path, data)

    def concat_imglist(self, list):
        # boundary line
        if self.show[list[0]].shape[0] != self.line.shape[0]:
            self.line = np.zeros((self.show[list[0]].shape[0], 3, 3), dtype=np.uint8)
            self.line[:, :, :] = 255
        disp = self.line

        for i in range(len(list)):
            if list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)
        return disp

    def display_single_results(self, path):
        results = cv2.imread(path.replace(self.cfg.dir["out"], self.cfg.dir["model2"]).replace('val/display', 'test/display'))
        st = self.cfg.width * 1 + 3 * 2
        ed = st + self.cfg.width * 2 + 3 * 1

        st2 = self.cfg.width * 4 + 3 * 5
        ed2 = st2 + self.cfg.width * 1 + 3 * 0

        self.show['single'] = np.concatenate((results[:, st:ed], results[:, st2:ed2]), axis=1)

    def display_clrnet_results(self, path):
        results = cv2.imread(path)
        self.show['clrnet'] = results

    def coeff_map_to_xy_coord_map_conversion(self, coeff_map):
        top_m = self.cfg.top_m
        _, h, w = coeff_map.shape
        coeff_map = coeff_map.view(top_m, -1, 1).permute(1, 0, 2)
        U = self.U.view(1, -1, top_m).expand(coeff_map.shape[0], -1, top_m)
        x_coord_map = torch.bmm(U, coeff_map) * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
        y_coord_map = to_tensor(self.cfg.py_coord).view(1, -1, 1).expand(h*w, -1, 1)
        lane_coord_map = torch.cat((x_coord_map, y_coord_map), dim=2)
        return lane_coord_map

    def overlap_lane_coord_map(self, lane_coord_map, seg_map):
        idx = (seg_map.view(-1) > 0.5)

        pos_lane_coord_map = lane_coord_map[idx == True]
        pos_lane_coord_map = to_np2(pos_lane_coord_map)
        pos_out_map = np.zeros((self.cfg.height, self.cfg.width), dtype=np.int64)
        for i in range(0, len(pos_lane_coord_map), 3):
            lane_coord = pos_lane_coord_map[i]
            self.show['temp'] = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
            self.draw_polyline_cv(lane_coord, name='temp', ref_name='temp', color=1, s=2)
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

    def display_for_model_single(self, batch, out):
        self.update_image(batch['img'][0], name='img')
        self.update_image_name(batch['img_name'][0])

        # lane_coord_map = self.coeff_map_to_xy_coord_map_conversion(out['coeff_map'][0])
        # self.show['coeff_pos_map'] = self.overlap_lane_coord_map(lane_coord_map, out['seg_map'][0])
        self.show['seg_map'] = self.b_map_to_rgb_image(out['seg_map'][0])
        self.show['seg_map_gt'] = self.b_map_to_rgb_image(batch['seg_label'][self.sf[0]][0:1])

    def display_for_train_per_frame(self, batch, out):
        self.update_image(batch['img'][0], name='img')
        self.update_image_name(batch['img_name'][0])

        # if 'mask' in out.keys():
        #     self.show['mask'] = self.b_map_to_rgb_image(out['mask'][0])
        # else:
        #     self.show['mask'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        # if 'aligned' in out.keys():
        #     self.show['aligned'] = self.b_map_to_rgb_image(out['aligned'][0])
        # else:
        #     self.show['aligned'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)

        try:
            lane_coord_map = self.coeff_map_to_xy_coord_map_conversion(out['coeff_map'][0])
            self.show['coeff_pos_map'] = self.overlap_lane_coord_map(lane_coord_map, out['seg_map'][0])
        except:
            lane_coord_map = self.coeff_map_to_xy_coord_map_conversion(out['coeff_map_init'][0])
            self.show['coeff_pos_map'] = self.overlap_lane_coord_map(lane_coord_map, out['seg_map_init'][0])

        self.show['seg_map_init'] = self.b_map_to_rgb_image(out['seg_map_init'][0])
        if 'key_guide' in out.keys():
            self.show['key_guide'] = self.b_map_to_rgb_image(out['key_guide'][0])
        else:
            self.show['key_guide'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        if 'seg_map' in out.keys():
            self.show['seg_map'] = self.b_map_to_rgb_image(out['seg_map'][0])
        else:
            self.show['seg_map'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        if 'aligned_key_guide' in out.keys():
            self.show['aligned_key_guide'] = self.b_map_to_rgb_image(out['aligned_key_guide'][0])
        else:
            self.show['aligned_key_guide'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        # if 'aligned_key_guide_f' in out.keys():
        #     self.show['aligned_key_guide_f'] = self.b_map_to_rgb_image(out['aligned_key_guide_f'][0])
        # else:
        #     self.show['aligned_key_guide_f'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        if 'aligned_key_probmap' in out.keys():
            self.show['aligned_key_probmap'] = self.b_map_to_rgb_image(out['aligned_key_probmap'][0])
        else:
            self.show['aligned_key_probmap'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)

        self.show['seg_map_gt'] = self.b_map_to_rgb_image(batch['seg_label'][self.sf[0]][0:1])

    def display_for_train(self, batch, out, batch_idx):
        save_namelist = ['img', 'coeff_pos_map', 'key_guide', 'aligned_key_guide', 'aligned_key_probmap', 'seg_map_init', 'seg_map', 'seg_map_gt']
        disp = []
        for t in range(self.cfg.clip_length, -1, -1):
            key_t = f't-{t}'
            self.show[key_t] = dict()
            self.display_for_train_per_frame(batch[key_t], out[key_t])
            disp.append(self.concat_imglist(list=save_namelist))

        disp = np.concatenate(disp, axis=0)
        self.save_data(path=f'{self.cfg.dir["out"]}/train/display/{batch_idx}.jpg', data=disp)


    def display_for_test(self, batch, out, prev_frame_num, batch_idx, mode):
        # img
        self.update_image(batch['img'][batch_idx], name='img')
        self.update_image_name(batch['img_name'][batch_idx])

        # label
        self.update_label(batch['org_label'][batch_idx], name='org_label')

        # output
        # self.show['seg_map_init'] = self.b_map_to_rgb_image(out['seg_map_init'][0])
        if prev_frame_num != 0:
            self.show['seg_map'] = self.b_map_to_rgb_image(out['seg_map'][0])
        else:
            self.show['seg_map'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        if 'key_guide' in out.keys():
            self.show['key_guide'] = self.b_map_to_rgb_image(out['key_guide'][0])
        else:
            self.show['key_guide'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        if prev_frame_num != 0:
            self.show['aligned_key_guide'] = self.b_map_to_rgb_image(out['aligned_key_guide'][0])
        else:
            self.show['aligned_key_guide'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        # if 'key_probmap' in out.keys():
        #     self.show['key_probmap'] = self.b_map_to_rgb_image(out['aligned_key_probmap'][0])
        # else:
        #     self.show['key_probmap'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        #
        if 'aligned_key_probmap' in out.keys():
            self.show['aligned_key_probmap'] = self.b_map_to_rgb_image(out['aligned_key_probmap'][0])
        else:
            self.show['aligned_key_probmap'] = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)

        self.show['label_overlap'] = self.show['org_label']
        self.show['seg_overlap'] = self.show['seg_map']
        self.show['img_overlap'] = self.show['img']
        self.show['coeff_cls_map'] = self.draw_selected_lane_coords(out['x_coords'], height_idx=out['height_idx'], color=(0, 255, 0), s=2)

        # save result
        dirname = os.path.dirname(self.show["img_name"])
        filename = os.path.basename(self.show["img_name"])
        self.display_single_results(path=f'{self.cfg.dir["out"]}/{mode}/display/{dirname}/{filename}.jpg')
        # self.display_clrnet_results(path=f'{self.cfg.dir["clrnet"]}/{dirname}/{filename}.jpg')
        self.display_imglist(path=f'{self.cfg.dir["out"]}/{mode}/display/{dirname}/{filename}.jpg',
                             list=['img', 'single', 'aligned_key_guide', 'img_overlap', 'clrnet',
                                   'seg_map', 'seg_overlap', 'label_overlap', 'org_label'])
        # self.display_imglist(path=f'{self.cfg.dir["out"]}/{mode}/display/{dirname}/{filename}.jpg',
        #                      list=['img', 'single', 'img_overlap', 'clrnet',
        #                            'seg_map', 'seg_overlap', 'label_overlap'])


        if 'mask' in out.keys():
            self.show.pop('mask')
