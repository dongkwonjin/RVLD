import os
import torch

class Config(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_system()
        self.setting_for_path()
        self.setting_for_image_param()
        self.setting_for_save()

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
        self.dir['out'] = f'{os.getcwd().replace("code", "output")}_{self.datalist_mode}'

    def setting_for_dataset_path(self):
        self.dataset = 'vil100'  # ['vil100']
        self.datalist_mode = 'train'  # ['train', 'test']

        # ------------------- need to modify -------------------
        self.dir['dataset'] = '--dataset_dir'
        # ------------------------------------------------------

    def setting_for_image_param(self):
        self.height = 384
        self.width = 640
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_size = 180

    def setting_for_save(self):
        self.save_pickle = True
