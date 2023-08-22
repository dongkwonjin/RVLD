import cv2

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from libs.utils import *

class Dataset_Train(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datalist = load_pickle(cfg.dir['pre0'] + 'datalist')

        self.transform = transforms.Compose([transforms.Resize((cfg.height + cfg.crop_size, cfg.width), interpolation=2), transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, idx, flip=0):
        img = Image.open(f'{self.cfg.dir["dataset"]}/JPEGImages/{self.datalist[idx]}.jpg').convert('RGB')
        self.org_width, self.org_height = img.size
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.transform(img)
        img = img[:, self.cfg.crop_size:, :]

        return {'img': self.normalize(img),
                'img_rgb': img,
                'org_h': self.org_height, 'org_w': self.org_width}

    def get_label(self, idx):
        data = load_pickle(f'{self.cfg.dir["pre0"]}/{self.datalist[idx]}')

        seg_label = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
        seg_label = np.ascontiguousarray(seg_label)

        lane_pts = list()
        for i in range(len(data['lane'])):
            pts_org = np.float32(data['lane'][i]['points'])
            lane_pts.append(pts_org)
            pts = np.copy(pts_org)
            pts[:, 0] = pts_org[:, 0] / (self.org_width - 1) * (self.cfg.width - 1)
            pts[:, 1] = pts_org[:, 1] / (self.org_height - 1) * (self.cfg.height + self.cfg.crop_size - 1) - self.cfg.crop_size
            pts = np.int32(pts).reshape((-1, 1, 2))
            seg_label = cv2.polylines(seg_label, [pts], False, 1, 4)

        return {'label': np.float32(seg_label),
                'lane_pts': lane_pts}

    def __getitem__(self, idx):
        out = dict()
        out['img_name'] = self.datalist[idx]
        out.update(self.get_image(idx))
        out.update(self.get_label(idx))

        return out

    def __len__(self):
        return len(self.datalist)
