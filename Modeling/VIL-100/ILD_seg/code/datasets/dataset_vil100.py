import cv2

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

from datasets.transforms import *
from libs.utils import *

class Dataset_Train(Dataset):
    def __init__(self, cfg, update=None):
        self.cfg = cfg
        self.datalist = load_pickle(f'{self.cfg.dir["pre2"]}/datalist')

        # image transform
        self.transform = Transforms(cfg)
        self.transform.settings()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def cropping(self, img, lanes):
        img = img[self.cfg.crop_size:]
        for i in range(len(lanes)):
            lanes[i][:, 1] -= self.cfg.crop_size
        return img, lanes

    def get_data_org(self, idx):
        img = Image.open(f'{self.cfg.dir["dataset"]}/JPEGImages/{self.datalist[idx]}.jpg').convert('RGB')
        anno = load_pickle(f'{self.cfg.dir["pre0_train"]}/{self.datalist[idx]}')
        return img, anno

    def get_data_aug(self, img, anno):
        img_new, anno_new = self.transform.process(img, anno['lane'])
        img_new, anno_new = self.cropping(img_new, anno_new)

        img_new = Image.fromarray(img_new)
        img_new = self.to_tensor(img_new)
        self.org_width, self.org_height = img.size

        return {'img': self.normalize(img_new),
                'img_rgb': img_new,
                'lanes': anno_new,
                'org_h': self.org_height, 'org_w': self.org_width}

    def get_data_preprocessed(self, data):
        # initialize error case
        self.transform.init_error_case(data['img_name'])

        # preprocessing
        out = dict()
        out.update(self.transform.get_lane_components(data['lanes']))
        out.update(self.transform.approximate_lanes(out['extended_lanes']))
        out['is_error_case'] = self.transform.is_error_case['total']
        return out

    def get_downsampled_label_seg(self, lanes, idx, sf):
        for s in sf:
            lane_pts = np.copy(lanes)
            lane_pts[:, 0] = lanes[:, 0] / (self.cfg.width - 1) * (self.cfg.width // s - 1)
            lane_pts[:, 1] = lanes[:, 1] / (self.cfg.height - 1) * (self.cfg.height // s - 1)

            self.label['seg_label'][s] = cv2.polylines(self.label['seg_label'][s], [np.int32(lane_pts)], False, idx + 1, self.cfg.lane_width['seg'])
            self.label['visit'][s] += (self.label['seg_label'][s] == idx + 1)

    def get_downsampled_label_coeff(self, data, idx, sf):
        for s in sf:
            self.label['coeff_label'][s][self.label['seg_label'][s] == (idx + 1), :] = data


    def get_label(self, data):
        out = dict()
        self.label = dict()
        self.label['org_label'] = np.ascontiguousarray(np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8))
        self.label['seg_label'] = dict()
        self.label['coeff_label'] = dict()
        self.label['visit'] = dict()

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.ascontiguousarray(np.zeros((self.cfg.height // s, self.cfg.width // s), dtype=np.uint8))
            self.label['coeff_label'][s] = np.zeros((self.cfg.height // s, self.cfg.width // s, self.cfg.top_m), dtype=np.float32)
            self.label['visit'][s] = np.zeros((self.cfg.height // s, self.cfg.width // s), dtype=np.float32)

        for i in range(len(data['lanes'])):
            lane_pts = data['lanes'][i]
            self.label['org_label'] = cv2.polylines(self.label['org_label'], [np.int32(lane_pts)], False, i+1, self.cfg.lane_width['org'], lineType=cv2.LINE_AA)
            self.get_downsampled_label_seg(lane_pts, i, self.cfg.scale_factor['seg'])

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = cv2.dilate(self.label['seg_label'][s], kernel=(3, 3), iterations=1)
            self.label['visit'][s] = np.float32(self.label['visit'][s] > 1)
            self.label['visit'][s] = cv2.dilate(self.label['visit'][s], kernel=(3, 3), iterations=1)

        for i in range(len(data['c'])):
            self.get_downsampled_label_coeff(data['c'][i], i, self.cfg.scale_factor['seg'])

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.int64(self.label['seg_label'][s] != 0)
            self.label['visit'][s] = np.float32(self.label['visit'][s] == 0)
            self.label['coeff_label'][s] = self.label['coeff_label'][s] * np.expand_dims(self.label['visit'][s], axis=2)
        self.label['org_label'] = np.float32(self.label['org_label'] != 1)

        out.update(self.label)

        return out

    def remove_dict_keys(self, data):
        data.pop('lanes')
        data.pop('extended_lanes')
        data.pop('approx_lanes')
        data.pop('c')
        return data

    def __getitem__(self, idx):
        out = dict()
        out['img_name'] = self.datalist[idx]
        img, anno = self.get_data_org(idx)
        out.update(self.get_data_aug(img, anno))
        out.update(self.get_data_preprocessed(out))
        out.update(self.get_label(out))
        out = self.remove_dict_keys(out)

        return out

    def __len__(self):
        return len(self.datalist)

class Dataset_Test(Dataset):
    def __init__(self, cfg, mode=None):
        self.cfg = cfg
        self.mode = 'train' if 'val' in mode else 'test'
        self.datalist = load_pickle(f'{self.cfg.dir[f"pre0_{self.mode}"]}/datalist')

        if cfg.sampling == True:
            sampling = np.arange(0, len(self.datalist), cfg.sampling_step)
            self.datalist = np.array(self.datalist)[sampling].tolist()

        # image transform
        self.transform = Transforms(cfg)
        self.transform.settings()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def cropping(self, img, lanes):
        img = img[self.cfg.crop_size:]
        for i in range(len(lanes)):
            lanes[i][:, 1] -= self.cfg.crop_size
        return img, lanes

    def get_data_org(self, idx):
        img = Image.open(f'{self.cfg.dir["dataset"]}/JPEGImages/{self.datalist[idx]}.jpg').convert('RGB')
        anno = load_pickle(f'{self.cfg.dir[f"pre0_{self.mode}"]}/{self.datalist[idx]}')
        return img, anno

    def get_data_aug(self, img, anno):
        img_new, anno_new = self.transform.process_for_test(img, anno['lane'])
        img_new, anno_new = self.cropping(img_new, anno_new)

        img_new = Image.fromarray(img_new)
        img_new = self.to_tensor(img_new)
        self.org_width, self.org_height = img.size

        return {'img': self.normalize(img_new),
                'img_rgb': img_new,
                'lanes': anno_new,
                'org_h': self.org_height, 'org_w': self.org_width}

    def get_downsampled_label_seg(self, lanes, idx, sf):
        for s in sf:
            lane_pts = np.copy(lanes)
            lane_pts[:, 0] = lanes[:, 0] / (self.cfg.width - 1) * (self.cfg.width // s - 1)
            lane_pts[:, 1] = lanes[:, 1] / (self.cfg.height - 1) * (self.cfg.height // s - 1)

            self.label['seg_label'][s] = cv2.polylines(self.label['seg_label'][s], [np.int32(lane_pts)], False, 1, self.cfg.lane_width['seg'])

    def get_label(self, data):
        out = dict()

        self.label = dict()
        self.label['org_label'] = np.ascontiguousarray(np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8))
        self.label['seg_label'] = dict()

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.ascontiguousarray(np.zeros((self.cfg.height // s, self.cfg.width // s), dtype=np.float32))

        for i in range(len(data['lanes'])):
            lane_pts = data['lanes'][i]
            self.label['org_label'] = cv2.polylines(self.label['org_label'], [np.int32(lane_pts)], False, 1, self.cfg.lane_width['org'], lineType=cv2.LINE_AA)
            self.get_downsampled_label_seg(lane_pts, i, self.cfg.scale_factor['seg'])

        for s in self.cfg.scale_factor['seg']:
            if self.cfg.lane_width['mode'] == 'gaussian':
                self.label['seg_label'][s] = cv2.GaussianBlur(self.label['seg_label'][s], self.cfg.lane_width['kernel'],
                                                              sigmaX=self.cfg.lane_width['sigmaX'], sigmaY=self.cfg.lane_width['sigmaY'])
            else:
                self.label['seg_label'][s] = cv2.dilate(self.label['seg_label'][s], kernel=self.cfg.lane_width['kernel'], iterations=1)

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.int64(self.label['seg_label'][s] != 0)

        self.label['org_label'] = np.float32(self.label['org_label'] != 0)

        out.update(self.label)

        return out

    def remove_dict_keys(self, data):
        data.pop('lanes')
        return data

    def __getitem__(self, idx):
        out = dict()
        out['img_name'] = self.datalist[idx]
        img, anno = self.get_data_org(idx)
        out.update(self.get_data_aug(img, anno))
        out.update(self.get_label(out))
        out = self.remove_dict_keys(out)

        return out

    def __len__(self):
        return len(self.datalist)