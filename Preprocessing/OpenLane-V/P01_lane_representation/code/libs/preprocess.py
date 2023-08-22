import cv2
import torch
import torch.nn.functional as F

from libs.utils import *

from scipy.optimize import curve_fit
from scipy import interpolate

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.visualizer = dict_DB['visualizer']

    def get_lane_points(self, idx):
        lane_pts = self.lane_pts[idx]
        return lane_pts

    def rescale_pts(self, data):
        data_r = np.copy(data)
        data_r[:, 0] = data_r[:, 0] / (self.cfg.org_width - 1) * (self.cfg.width - 1)
        data_r[:, 1] = (data_r[:, 1] - self.cfg.crop_size) / (self.cfg.org_height - 1 - self.cfg.crop_size) * (self.cfg.height - 1)

        return data_r

    def check_one_to_one_mapping(self, data):
        dy = np.diff(data[:, 1])
        c1 = np.sum(dy > 0)
        c2 = np.sum(dy <= 0)

        if c1 * c2 != 0:
            self.is_error_case['one_to_one'] = True
            print(f'error case: not one-to-one mapping! {self.img_name}')

    def check_lane_y_pts(self, data):
        dy = np.diff(data[:, 1])
        idx = (dy <= 0).nonzero()[0]
        return idx

    def interp_extrap(self, lane_pts):
        if self.cfg.mode_interp == 'spline':
            f = interpolate.InterpolatedUnivariateSpline(lane_pts[:, 1], lane_pts[:, 0], k=1)
            new_x_pts = f(self.cfg.py_coord)
        elif self.cfg.mode_interp == 'splrep':
            f = interpolate.splrep(lane_pts[:, 1], lane_pts[:, 0], k=1, s=5)
            new_x_pts = interpolate.splev(self.cfg.py_coord, f)
        else:
            f = interpolate.interp1d(lane_pts[:, 1], lane_pts[:, 0], kind=self.cfg.mode_interp, fill_value='extrapolate')
            new_x_pts = f(self.cfg.py_coord)

        new_lane_pts = np.concatenate((new_x_pts.reshape(-1, 1), self.cfg.py_coord.reshape(-1, 1)), axis=1)
        return new_lane_pts

    def get_lane_component(self):
        out = {'x_coord': []}

        for i in range(len(self.lane_pts)):
            self.is_error_case['one_to_one'] = False
            self.is_error_case['fitting'] = False

            lane_pts = self.get_lane_points(i)
            lane_pts = self.rescale_pts(lane_pts)

            if lane_pts[0, 1] > lane_pts[-1, 1]:
                lane_pts = np.flip(lane_pts, axis=0)
            # remove duplicate pts
            lane_pts[:, 1] = np.round(lane_pts[:, 1])
            unique_idx = np.sort(np.unique(lane_pts[:, 1], return_index=True)[1])
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
            else:
                self.is_error_case['total'] = True

        return out

    def load_img_data(self):
        img = cv2.imread(f'{self.cfg.dir["dataset"]}/images/{self.cfg.datalist_mode}/{self.datalist[self.img_idx]}.jpg')
        img = img[self.cfg.crop_size:]
        img = cv2.resize(img, (self.cfg.width, self.cfg.height))

        seg_label = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        seg_label = np.ascontiguousarray(seg_label)

        for i in range(len(self.lane_pts)):
            pts_org = np.float32(self.lane_pts[i])
            pts = np.copy(pts_org)
            pts[:, 0] = pts_org[:, 0] / (self.org_width - 1) * (self.cfg.width - 1)
            pts[:, 1] = (pts_org[:, 1] - self.cfg.crop_size) / (self.org_height - self.cfg.crop_size - 1) * (self.cfg.height - 1)
            pts = np.int32(pts)
            seg_label = cv2.polylines(seg_label, [pts], False, (255, 255, 255), 1, 4)

        self.img = img
        self.seg_label = seg_label

    def load_data(self, idx):
        data = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/label/{self.cfg.datalist_mode}/{self.datalist[idx]}')
        self.img_name = self.datalist[idx].replace('.jpg', '')
        self.org_height = self.cfg.org_height
        self.org_width = self.cfg.org_width
        self.img_idx = idx
        self.dir_name = os.path.dirname(self.img_name)
        self.file_name = os.path.basename(self.img_name)

        self.lane_pts = data['lanes']

        self.out_f = list()
        self.is_error_case = dict()
        self.is_error_case['one_to_one'] = False
        self.is_error_case['fitting'] = False
        self.is_error_case['short'] = False
        self.is_error_case['total'] = False

        if self.cfg.display_all == True:
            self.load_img_data()

    def get_flipped_data(self, pre_out):
        out = {'x_coord': [],
               'org_lane': []}
        for i in range(len(pre_out['x_coord']) - 1, -1, -1):
            x_coord = self.cfg.width - 1 - pre_out['x_coord'][i]
            out['x_coord'].append(x_coord)

        return out

    def run_flip(self):
        for i in range(0, 2):  # 1: horizontal flip
            self.flip_idx = i

            if i == 1 and self.cfg.data_flip == False:
                break

            if i == 1 and self.cfg.display_all == True:
                self.img = cv2.flip(self.img, 1)
                self.seg_label = cv2.flip(self.seg_label, 1)

            if self.cfg.display_all == True:
                self.visualizer.update_datalist(self.img, self.img_name, self.seg_label, self.dir_name, self.file_name, self.img_idx)

            if i == 0:
                self.out_f.append(self.get_lane_component())
            elif self.is_error_case['total'] == False:
                self.out_f.append(self.get_flipped_data(self.out_f[0]))

            # visualizer

            if self.cfg.display_all == True and i == 0:
                self.visualizer.save_datalist([self.is_error_case['total']])

    def init(self):
        self.datalist_out = list()
        self.datalist_out_error = list()


    def run(self):
        print('start')
        self.init()
        self.datalist = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/list/datalist_{self.cfg.datalist_mode}')
        for i in range(len(self.datalist)):
            self.load_data(i)
            print(f'image {i} ===> {self.img_name} load')

            self.run_flip()

            # save pickle
            if self.cfg.save_pickle == True:
                save_pickle(f'{self.cfg.dir["out"]}/pickle/{self.img_name}', data=self.out_f)
                if self.is_error_case['total'] == False:
                    self.datalist_out.append(self.img_name)
                else:
                    self.datalist_out_error.append(self.img_name)

            print(f'image {i} ===> {self.img_name} clear')

        if self.cfg.save_pickle == True:
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist', self.datalist_out)
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_error', self.datalist_out_error)

