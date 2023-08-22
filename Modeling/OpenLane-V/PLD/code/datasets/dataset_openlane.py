import cv2

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

from datasets.transforms import *
from libs.utils import *

class Dataset_Train(Dataset):
    def __init__(self, cfg, update=None):
        self.cfg = cfg
        self.datalist_video = load_pickle(f'{self.cfg.dir["pre3_train"]}/datalist_{3}')
        self.datalist = list(self.datalist_video)
        if update == True:
            err = load_pickle(f'{self.cfg.dir["model1"]}/val_for_training_set/pickle/error')
            datalist = load_pickle(f'{self.cfg.dir["model1"]}/val_for_training_set/pickle/datalist')
            idx_sorted = np.argsort(err)[::-1]
            idx_sorted = idx_sorted[:int(len(idx_sorted) * 0.3)]
            errorlist = list(np.array(datalist)[idx_sorted])
            errorlist = sorted(list(set(errorlist).intersection(set(self.datalist))))
            ratio = int(np.round(len(self.datalist) / (len(errorlist))))
            print(f'val for training set ratio : {ratio}, total num : {len(self.datalist)}, error num : {len(errorlist)}')
            datalist = self.datalist + (errorlist * ratio)
            np.random.shuffle(datalist)
            self.datalist = datalist[:len(self.datalist)]

        # image transform
        self.transform = Transforms(cfg)
        self.transform.settings()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def cropping(self, img, lanes):
        img = img.crop((0, self.cfg.crop_size, int(img.size[0]), int(img.size[1])))
        for i in range(len(lanes['lanes'])):
            if len(lanes['lanes'][i]) == 0:
                continue
            lanes['lanes'][i][:, 1] -= self.cfg.crop_size
            if self.flip == 1:
                lanes['lanes'][i][:, 0] = (self.cfg.org_width - 1) - lanes['lanes'][i][:, 0]
        if self.flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return img, lanes

    def get_data_org(self, img_name):
        img = Image.open(f'{self.cfg.dir["dataset"]}/images/training/{img_name}.jpg').convert('RGB')
        anno = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/label/training/{img_name}')
        img, anno = self.cropping(img, anno)
        return img, anno

    def get_data_aug(self, img, anno):
        img_new, anno_new = self.transform.process(img, anno['lanes'])

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

        for i in range(len(data['lanes'])):
            self.get_downsampled_label_coeff(data['c'][i], i, self.cfg.scale_factor['seg'])

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.int64(self.label['seg_label'][s] != 0)
            self.label['visit'][s] = np.float32(self.label['visit'][s] == 0)
            self.label['coeff_label'][s] = self.label['coeff_label'][s] * np.expand_dims(self.label['visit'][s], axis=2)
        self.label['org_label'] = np.float32(self.label['org_label'] != 1)

        out.update(self.label)

        return out

    def get_guidance_data(self, data):
        guide_mask_neg = np.zeros((self.cfg.height // self.cfg.scale_factor['seg'][0], self.cfg.width // self.cfg.scale_factor['seg'][0]), dtype=np.float32)
        if len(data['lanes']) > 1:
            sf = self.cfg.scale_factor['seg']
            lanes = data['lanes']
            sorted_idx = np.argsort(np.array(lanes)[:, -1, 0])
            ref_idx = random.randint(0, len(sorted_idx) - 2)
            lane1 = lanes[sorted_idx[ref_idx]]
            lane2 = lanes[sorted_idx[ref_idx + 1]]
            case = random.randint(0, 1)
            if case == 0:
                w = random.uniform(0.1, 0.3)
            else:
                w = random.uniform(0.7, 0.9)
            guide_lane = np.copy(lane1)
            guide_lane[:, 0] = lane1[:, 0] * w + lane2[:, 0] * (1 - w)
            for s in sf:
                guide_lane[:, 0] = guide_lane[:, 0] / (self.cfg.width - 1) * (self.cfg.width // s - 1)
                guide_lane[:, 1] = guide_lane[:, 1] / (self.cfg.height - 1) * (self.cfg.height // s - 1)
            guide_mask_neg = cv2.polylines(guide_mask_neg, [np.int32(guide_lane)], False, 1, self.cfg.lane_width['seg'])
            guide_mask_neg = cv2.dilate(guide_mask_neg, kernel=(3, 3), iterations=1)

        return {'guide_mask_neg': guide_mask_neg}

    def remove_dict_keys(self, data):
        data['lane_num'] = len(data['lanes'])
        data.pop('lanes')
        data.pop('extended_lanes')
        data.pop('approx_lanes')
        data.pop('c')

        return data

    def __getitem__(self, idx):
        out = dict()
        t_frame = self.datalist[idx]
        self.flip = random.randint(0, 1)
        reverse = random.randint(0, 1)
        if reverse == 0:
            datalist_video = sorted(random.sample(self.datalist_video[t_frame], self.cfg.clip_length + 1), reverse=True)
        else:
            datalist_video = sorted(random.sample(self.datalist_video[t_frame], self.cfg.clip_length + 1), reverse=False)
        for i in range(self.cfg.clip_length + 1):
            img_name = datalist_video[i]
            img, anno = self.get_data_org(img_name)

            t = f't-{i}'
            out[t] = dict()
            out[t]['img_name'] = img_name
            out[t].update(self.get_data_aug(img, anno))
            out[t].update(self.get_data_preprocessed(out[t]))
            out[t].update(self.get_label(out[t]))
            out[t].update(self.get_guidance_data(out[t]))
            out[t] = self.remove_dict_keys(out[t])

        return out

    def __len__(self):
        return len(self.datalist)

class Dataset_Test(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datalist = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/list/datalist_validation')
        self.datalist_video = load_pickle(f'{self.cfg.dir["pre3_test"]}/datalist_{3}')

        if cfg.sampling == True:
            if cfg.sampling_mode == 'video':
                datalist_out = list()
                datalist_video_out = dict()
                self.datalist_video = load_pickle(f'{self.cfg.dir["pre3_test"]}/datalist_{3}')
                self.datalist_split_video = load_pickle(f'{self.cfg.dir["pre3_test"]}/datalist_split_video')
                sampling = np.arange(0, len(self.datalist_split_video), cfg.sampling_step)
                datalist_video = np.array(list(self.datalist_split_video))[sampling].tolist()
                for i in range(len(datalist_video)):
                    video_name = datalist_video[i]
                    datalist_out += self.datalist_split_video[video_name]
                for i in range(len(datalist_out)):
                    datalist_video_out[datalist_out[i]] = self.datalist_video[datalist_out[i]]
                self.datalist_video = datalist_video_out
                self.datalist = list(self.datalist_video)

        # image transform
        self.transform = Transforms(cfg)
        self.transform.settings()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def cropping(self, img, lanes):
        img = img.crop((0, self.cfg.crop_size, int(img.size[0]), int(img.size[1])))
        for i in range(len(lanes['lanes'])):
            if len(lanes['lanes'][i]) == 0:
                continue
            lanes['lanes'][i][:, 1] -= self.cfg.crop_size
        return img, lanes

    def get_data_org(self, idx):
        img = Image.open(f'{self.cfg.dir["dataset"]}/images/validation/{self.datalist[idx]}.jpg').convert('RGB')
        anno = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/label/validation/{self.datalist[idx]}')
        img, anno = self.cropping(img, anno)
        return img, anno

    def get_data_aug(self, img, anno):
        img_new, anno_new = self.transform.process_for_test(img, anno['lanes'])

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
        out['prev_num'] = len(self.datalist_video[self.datalist[idx]]) - 1
        img, anno = self.get_data_org(idx)
        out.update(self.get_data_aug(img, anno))
        out.update(self.get_label(out))
        out = self.remove_dict_keys(out)

        return out

    def __len__(self):
        return len(self.datalist)
