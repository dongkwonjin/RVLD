import cv2
import torch
import torch.nn.functional as F

from libs.utils import *

from scipy.optimize import curve_fit
from scipy import interpolate

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.dataloader = dict_DB['dataloader']
        self.visualizer = dict_DB['visualizer']

    def get_lane_points(self, idx):
        if self.flip_idx == 0:
            lane_pts = self.lane_pts[idx][0]
        else:
            lane_pts = self.lane_pts[idx][0]
            lane_pts[:, 0] = self.org_width - lane_pts[:, 0]

        return to_np2(lane_pts)

    def rescale_pts(self, data):
        data_r = np.copy(data)
        data_r[:, 0] = data_r[:, 0] / (self.org_width - 1) * (self.cfg.width - 1)
        data_r[:, 1] = data_r[:, 1] / (self.org_height - 1) * (self.cfg.height + self.cfg.crop_size - 1) - self.cfg.crop_size

        return data_r

    def check_one_to_one_mapping(self, data):
        dy = (data[:, 1][1:] - data[:, 1][:-1])
        c1 = np.sum(dy > 0)
        c2 = np.sum(dy <= 0)

        if c1 * c2 != 0:
            self.is_error_case['one_to_one'] = True
            print(f'error case: not one-to-one mapping! {self.img_name}')

    def interp_extrap(self, lane_pts):
        if lane_pts[0, 1] > lane_pts[-1, 1]:
            lane_pts = np.flip(lane_pts, axis=0)
        if self.cfg.mode_interp == 'spline':
            f = interpolate.InterpolatedUnivariateSpline(lane_pts[:, 1], lane_pts[:, 0], k=1)
            new_x_pts = f(self.cfg.py_coord)
        elif self.cfg.mode_interp == 'splrep':
            f = interpolate.splrep(lane_pts[:, 1], lane_pts[:, 0], k=1, s=20)
            new_x_pts = interpolate.splev(self.cfg.py_coord, f)
        else:
            f = interpolate.interp1d(lane_pts[:, 1], lane_pts[:, 0], kind=self.cfg.mode_interp, fill_value='extrapolate')
            new_x_pts = f(self.cfg.py_coord)

        new_lane_pts = np.concatenate((new_x_pts.reshape(-1, 1), self.cfg.py_coord.reshape(-1, 1)), axis=1)
        return new_lane_pts

    def get_lane_component(self):
        out = {'x_coord': [],
               'org_lane': []}

        for i in range(len(self.lane_pts)):
            self.is_error_case['one_to_one'] = False
            self.is_error_case['fitting'] = False

            lane_pts = self.get_lane_points(i)
            lane_pts = self.rescale_pts(lane_pts)

            # check
            self.check_one_to_one_mapping(lane_pts)

            # remove duplicate pts
            unique_idx = np.sort(np.unique(lane_pts[:, 1], return_index=True)[1])
            lane_pts = lane_pts[unique_idx]
            unique_idx = np.sort(np.unique(lane_pts[:, 0], return_index=True)[1])
            lane_pts = lane_pts[unique_idx]

            # interpolation & extrapolation
            try:
                new_lane_pts = self.interp_extrap(lane_pts)
            except:
                print(f'error case: fiiting! {self.img_name}')
                self.is_error_case['fitting'] = True

            # visualize
            if self.cfg.display_all == True and self.is_error_case['fitting'] == False:
                self.visualizer.draw_lanes_for_datalist(new_lane_pts)

            if self.is_error_case['one_to_one'] + self.is_error_case['fitting'] == 0:
                out['x_coord'].append(new_lane_pts[:, 0])
                out['org_lane'].append(lane_pts)
            else:
                self.is_error_case['total'] = True

        return out

    def get_flipped_data(self, pre_out):
        out = {'x_coord': [],
               'org_lane': []}
        for i in range(len(pre_out['x_coord']) - 1, -1, -1):
            x_coord = self.cfg.width - 1 - pre_out['x_coord'][i]
            org_lane = np.copy(pre_out['org_lane'][i])
            org_lane[:, 0] = self.org_width - org_lane[:, 0]
            out['x_coord'].append(x_coord)
            out['org_lane'].append(org_lane)

        return out

    def run_flip(self):
        for i in range(0, 2):  # 1: horizontal flip
            self.flip_idx = i

            if i == 1:
                self.img = self.img.flip(2)
                self.label = self.label.flip(1)

            self.visualizer.update_datalist(self.img, self.img_name, self.label, self.dir_name, self.file_name, self.img_idx)

            if i == 0:
                self.out_f.append(self.get_lane_component())
            elif self.is_error_case['total'] == False:
                self.out_f.append(self.get_flipped_data(self.out_f[0]))

            if i == 0 and self.cfg.display_all == True:
                self.visualizer.save_datalist([self.is_error_case['total']])

    def update_batch_data(self, batch, i):
        self.img = batch['img'][0].cuda()
        self.label = batch['label'][0].cuda()
        self.img_name = batch['img_name'][0].replace('.jpg', '')
        self.org_height = batch['org_h']
        self.org_width = batch['org_w']
        self.img_idx = i
        self.dir_name = os.path.dirname(self.img_name)
        self.file_name = os.path.basename(self.img_name)

        self.lane_pts = batch['lane_pts']

        self.out_f = list()
        self.is_error_case = dict()
        self.is_error_case['one_to_one'] = False
        self.is_error_case['fitting'] = False
        self.is_error_case['short'] = False
        self.is_error_case['total'] = False

    def init(self):
        self.datalist = list()
        self.datalist_error = list()

    def run(self):
        print('start')

        self.init()

        for i, batch in enumerate(self.dataloader):
            self.update_batch_data(batch, i)
            self.run_flip()

            # save pickle
            if self.cfg.save_pickle == True:
                save_pickle(f'{self.cfg.dir["out"]}/pickle/{self.img_name}', data=self.out_f)
                if self.is_error_case['total'] == False:
                    self.datalist.append(self.img_name)
                else:
                    self.datalist_error.append(self.img_name)

            print(f'image {i} ===> {self.img_name} clear')

        if self.cfg.save_pickle == True:
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist', self.datalist)
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_error', self.datalist_error)
