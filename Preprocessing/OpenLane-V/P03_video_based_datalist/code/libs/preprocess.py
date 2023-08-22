import cv2
import torch
import torch.nn.functional as F

from libs.utils import *

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

    def get_video_datalist_for_processing(self):
        print('start')

        if self.cfg.datalist_mode == 'training':
            datalist = load_pickle(f'{self.cfg.dir["pre2"]}/datalist')
        else:
            datalist = []

        path = f'{self.cfg.dir["dataset"]}/OpenLane-V/list/datalist_video_{self.cfg.datalist_mode}'
        datalist_video = load_pickle(path)
        save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_split_video', data=datalist_video)

        datalist_out = dict()
        video_list = list(datalist_video)
        num = 0
        for i in range(len(video_list)):
            video_name = video_list[i]
            file_list = datalist_video[video_name]
            for j in range(len(file_list)):
                name = file_list[j]
                dirname = os.path.dirname(name)
                filename = os.path.basename(name)

                if (name not in datalist) and self.cfg.datalist_mode == 'training':
                    print(f'exclude {name}')
                    continue

                datalist_out[name] = list()
                datalist_out[name].append(name)
                for t in range(1, self.cfg.clip_length * 3):
                    if j - t < 0:
                        break
                    prev_filename = file_list[j-t]
                    if len(datalist_out[name]) == self.cfg.clip_length + 1:
                        break
                    datalist_out[name].append(prev_filename)

                    if (prev_filename not in datalist) and self.cfg.datalist_mode == 'training':
                        print(f'exclude {name}')
                        break

                if (name not in datalist) and self.cfg.datalist_mode == 'training':
                    datalist_out.pop(name)
                    continue

                if len(datalist_out[name]) < self.cfg.clip_length + 1 and self.cfg.datalist_mode == 'training':
                    datalist_out.pop(name)
                print(f'{num} ==> {filename} done')
                num += 1

        print(f'The number of datalist_video: {len(datalist_out)}')
        if self.cfg.save_pickle == True:
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_{self.cfg.clip_length}', data=datalist_out)

    def run(self):
        print('start')

        self.get_video_datalist_for_processing()
