import cv2
import math

import torch
import torch.nn.functional as F

from libs.utils import *

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

    def run(self):
        print('start')

        datalist_org = load_pickle(f'{self.cfg.dir["pre0"]}/datalist')
        if self.cfg.datalist_mode == 'train':
            datalist = load_pickle(f'{self.cfg.dir["pre2"]}/datalist')
        else:
            datalist = datalist_org

        datalist_ex = sorted(list(set(datalist_org) - set(datalist)))
        print(f'The number of excluded data : {len(datalist_ex)}')

        datalist_out = dict()
        for i in range(len(datalist)):
            name = datalist[i]
            dirname = os.path.dirname(name)
            filename = os.path.basename(name)

            datalist_out[name] = list()
            datalist_out[name].append(name)
            update = filename
            for t in range(self.cfg.clip_length * 3):
                prev_filename = str(int(update) - 3).zfill(5)
                prev_name = f'{dirname}/{prev_filename}'
                update = prev_filename
                if '-' in prev_filename:
                    continue
                if prev_name not in datalist:
                    continue
                if len(datalist_out[name]) == self.cfg.clip_length + 1:
                    break
                datalist_out[name].append(prev_name)
            if len(datalist_out[name]) < self.cfg.num_t + 1 and self.cfg.datalist_mode == 'train':
                datalist_out.pop(name)
            print(f'{i} ==> {name} done')

        print(f'The number of datalist_org: {len(datalist_org)}')
        print(f'The number of datalist_pre2: {len(datalist)}')
        print(f'The number of datalist_video: {len(datalist_out)}')

        if self.cfg.save_pickle == True:
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_{self.cfg.clip_length}', data=datalist_out)