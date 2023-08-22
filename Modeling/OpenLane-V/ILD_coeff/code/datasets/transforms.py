import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from scipy import interpolate

from libs.utils import *

class Transforms(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.U = to_np(load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m])
        self.U_t = self.U.transpose(1, 0)

    def settings(self):
        if self.cfg.mode_transform == 'basic':
            transforms = self.basic_transforms(self.cfg.height, self.cfg.width)
        elif self.cfg.mode_transform == 'complex':
            transforms = self.complex_transforms(self.cfg.height, self.cfg.width)
        else:
            transforms = self.custom_transforms(self.cfg.height, self.cfg.width)

        transforms_for_test = self.transforms_for_test(self.cfg.height, self.cfg.width)

        img_transforms = []
        for aug in transforms:
            p = aug['p']
            if aug['name'] != 'OneOf':
                img_transforms.append(
                    iaa.Sometimes(p=p, then_list=getattr(iaa, aug['name'])(**aug['parameters'])))
            else:
                img_transforms.append(
                    iaa.Sometimes(p=p, then_list=iaa.OneOf([getattr(iaa, aug_['name'])(**aug_['parameters']) for aug_ in aug['transforms']])))

        img_transforms_for_test = []
        for aug in transforms_for_test:
            p = aug['p']
            img_transforms_for_test.append(
                iaa.Sometimes(p=p, then_list=getattr(iaa, aug['name'])(**aug['parameters'])))

        self.transform = iaa.Sequential(img_transforms)
        self.transform_for_test = iaa.Sequential(img_transforms_for_test)

    def lane_to_linestrings(self, data):
        lane = list()
        for i in range(len(data)):
            pts = data[i]
            lane.append(LineString(pts))
        return lane

    def linestrings_to_lanes(self, data):
        lanes = []
        for pts in data:
            lanes.append(pts.coords)

        return lanes

    def process(self, img_org, anno):
        img_org = np.uint8(img_org)
        line_strings_org = self.lane_to_linestrings(anno)
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        img_new, line_strings = self.transform(image=img_org, line_strings=line_strings_org)
        anno_new = self.linestrings_to_lanes(line_strings)
        return img_new, anno_new

    def process_for_test(self, img_org, anno):
        img_org = np.uint8(img_org)
        line_strings_org = self.lane_to_linestrings(anno)
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        img_new, line_strings = self.transform_for_test(image=img_org, line_strings=line_strings_org)
        anno_new = self.linestrings_to_lanes(line_strings)
        return img_new, anno_new

    def check_one_to_one_mapping(self, data):
        dy = (data[:, 1][1:] - data[:, 1][:-1])
        c1 = np.sum(dy > 0)
        c2 = np.sum(dy <= 0)

        if c1 * c2 != 0:
            self.is_error_case['one_to_one'] = True
            # print(f'error case: not one-to-one mapping! {self.img_name}')

    def init_error_case(self, img_name):
        self.img_name = img_name
        self.is_error_case = dict()
        self.is_error_case['one_to_one'] = False
        self.is_error_case['fitting'] = False
        self.is_error_case['short'] = False
        self.is_error_case['iou'] = False
        self.is_error_case['total'] = False

    def approximate_lanes(self, lane_pts):
        if len(lane_pts) != 0:
            x_pts = np.float32(lane_pts).transpose(1, 0)
            c = np.matmul(self.U_t, (x_pts - (self.cfg.width - 1) / 2) / ((self.cfg.width - 1) / 2))
            approx_x_pts = np.matmul(self.U, c) * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
            return {'c': c.transpose(1, 0), 'approx_lanes': approx_x_pts.transpose(1, 0)}
        else:
            return {'c': [], 'approx_lanes': []}

    def get_lane_components(self, lanes):
        out = list()
        for i in range(len(lanes)):
            lane_pts = lanes[i]
            self.is_error_case['one_to_one'] = False
            self.is_error_case['fitting'] = False

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
                self.is_error_case['fitting'] = True

            if self.is_error_case['one_to_one'] + self.is_error_case['fitting'] == 0:
                out.append(new_lane_pts[:, 0])
            else:
                self.is_error_case['total'] = True
                break
        return {'extended_lanes': out}

    def interp_extrap(self, lane_pts):
        if lane_pts[0, 1] > lane_pts[-1, 1]:
            lane_pts = np.flip(lane_pts, axis=0)
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

    ### Option
    def custom_transforms(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            # dict(name='OneOf',
            #      transforms=[
            #          dict(name='MotionBlur', parameters=dict(k=(3, 5))),
            #          dict(name='MedianBlur', parameters=dict(k=(3, 5)))
            #      ],
            #      p=0.2),
            # dict(name='Affine',
            #      parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
            #                                             y=(-0.1, 0.1)),
            #                      rotate=(-10, 10),
            #                      scale=(0.8, 1.2)),
            #      p=0.7),
            # dict(name='Resize',
            #      parameters=dict(size=dict(height=img_h, width=img_w)),
            #      p=1.0),
        ]
        return transform

    def transforms_for_test(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ]
        return transform

    def basic_transforms(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ]
        return transform

    def complex_transforms(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            # dict(name='OneOf',
            #      transforms=[
            #          dict(name='MotionBlur', parameters=dict(k=(3, 5))),
            #          dict(name='MedianBlur', parameters=dict(k=(3, 5)))
            #      ],
            #      p=0.2),
            # dict(name='Affine',
            #      parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
            #                                             y=(-0.1, 0.1)),
            #                      rotate=(-10, 10),
            #                      scale=(0.8, 1.2)),
            #      p=0.7),
            # dict(name='Resize',
            #      parameters=dict(size=dict(height=img_h, width=img_w)),
            #      p=1.0),
        ]
        return transform