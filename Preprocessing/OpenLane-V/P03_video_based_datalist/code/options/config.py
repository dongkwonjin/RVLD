import os
import torch

import numpy as np

class Config(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_system()
        self.setting_for_path()
        self.setting_for_image_param()
        self.setting_for_dataloader()
        self.setting_for_visualization()
        self.setting_for_save()
        # --------preprocessing-------- #
        self.setting_for_video_based()
        # --------others-------- #

    def setting_for_system(self):
        self.gpu_id = "0"
        self.seed = 123
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        torch.backends.cudnn.deterministic = True

    def setting_for_path(self):
        self.pc = 'main'
        self.dir = dict()
        self.setting_for_dataset_path()  # dataset path

        self.dir['proj'] = os.path.dirname(os.getcwd())
        self.dir['head_proj'] = os.path.dirname(self.dir['proj'])
        self.dir['pre2'] = f'{self.dir["head_proj"]}/P02_SVD/output_{self.datalist_mode}/pickle'
        self.dir['out'] = f'{os.getcwd().replace("code", "output")}_{self.datalist_mode}'

    def setting_for_dataset_path(self):
        self.dataset = 'openlane'  # ['vil100']
        self.datalist_mode = 'training'  # ['training', 'testing', 'validation']

        # ------------------- need to modify -------------------
        self.dir['dataset'] = '--dataset path'
        # ------------------------------------------------------

    def setting_for_image_param(self):
        self.org_height = 1280
        self.org_width = 1920
        self.height = 384
        self.width = 640
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_size = 480

    def setting_for_dataloader(self):
        self.num_workers = 4
        self.batch_size = 1
        self.data_flip = False

    def setting_for_visualization(self):
        self.display_all = False

    def setting_for_save(self):
        self.save_pickle = True

    def setting_for_video_based(self):
        self.num_t = 3  # use previous {} frames
        self.clip_length = 3
