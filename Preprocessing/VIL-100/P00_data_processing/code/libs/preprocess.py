import cv2
import ast

import torch
import torch.nn.functional as F

from libs.utils import *

class Preprocessing(object):

    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

    def run(self):
        print('start')
        with open(f'{self.cfg.dir["dataset"]}/data/{self.cfg.datalist_mode}.txt', 'r') as f:
            datalist = f.read().splitlines()
        datalist = sorted(datalist)
        self.datalist = list()
        for i in range(len(datalist)):
            img_name = datalist[i].replace(" ", "")
            json_name = f'{img_name.replace("/JPEGImages", "")}.json'
            json_file_path = f'{self.cfg.dir["dataset"]}/Json{json_name}'
            with open(json_file_path, 'r') as j:
                data = json.loads(j.read())

            if len(data['annotations']['lane']) == 2:
                a = 1
            img_path = f'{self.cfg.dir["dataset"]}/{img_name}'
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            out = dict()
            out['height'] = h
            out['width'] = w
            out['lane'] = data['annotations']['lane']
            if self.cfg.save_pickle == True:
                path = f'{self.cfg.dir["out"]}/pickle/{img_name.replace("/JPEGImages/", "").replace(".jpg", "")}'
                save_pickle(path=path, data=out)

            self.datalist.append(img_name.replace("/JPEGImages/", "").replace(".jpg", ""))

            print('i : {}, : image name : {} done'.format(i, img_name))

        if self.cfg.save_pickle == True:
            path = f'{self.cfg.dir["out"]}/pickle/datalist'
            save_pickle(path, data=self.datalist)