import cv2
import torch
import torch.nn.functional as F

from libs.utils import *

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.visualizer = dict_DB['visualizer']

    def get_lane_mask(self, px_coord, sf=4, s=3):

        temp = np.zeros((self.cfg.height // sf, self.cfg.width // sf), dtype=np.float32)
        temp = np.ascontiguousarray(temp)

        x = px_coord / (self.cfg.width - 1) * (self.cfg.width // sf - 1)
        y = self.cfg.py_coord / (self.cfg.height - 1) * (self.cfg.height // sf - 1)

        xy_coord = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        xy_coord = np.int32(xy_coord).reshape((-1, 1, 2))

        lane_mask = cv2.polylines(temp, [xy_coord], False, 1, s)
        return to_tensor(lane_mask).unsqueeze(0)

    def measure_IoU(self, X1, X2):
        X = X1 + X2

        X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
        X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)

        iou = X_inter / X_uni

        return iou

    def compute_approximation_error(self, px_coord_ap, px_coord):
        for i in range(px_coord_ap.shape[1]):
            lane_mask1 = self.get_lane_mask(to_np(px_coord_ap[:, i]))
            lane_mask2 = self.get_lane_mask(to_np(px_coord[:, i]))

            iou = self.measure_IoU(lane_mask1, lane_mask2)

            if iou < self.cfg.thresd_iou:
                print('approximation error : IoU : {}'.format(iou))
                self.is_error_case = True
                break

    def approximate_lane(self):
        out = self.data[self.flip_idx]

        px_coord = torch.FloatTensor([]).cuda()
        for i in range(len(out['x_coord'])):
            x_coord = out['x_coord'][i]
            if len(x_coord) == 0:
                print('empty size!')
                continue
            px_coord = torch.cat((px_coord, to_tensor(x_coord[self.cfg.sample_idx]).view(-1, 1)), dim=1)

        if px_coord.shape[0] == 0:
            return {'px_coord_ap': [], 'c': []}

        U = self.U[:, :self.cfg.top_m]
        U_t = self.U[:, :self.cfg.top_m].permute(1, 0)
        px_coord = px_coord.type(torch.float)

        c = torch.matmul(U_t, (px_coord - (self.cfg.width - 1) / 2) / ((self.cfg.width - 1) / 2))
        px_coord_ap = torch.matmul(U, c) * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
        py_coord = to_tensor(self.cfg.py_coord).type(torch.float)

        self.compute_approximation_error(px_coord_ap, px_coord)

        if self.cfg.display_all == True and self.is_error_case == True:
            self.visualizer.draw_lanes_for_datalist(px_coord, px_coord_ap, py_coord)

        return {'c': to_np(c.permute(1, 0))}

    def construct_lane_matrix(self):
        datalist = load_pickle(f'{self.cfg.dir["pre1"]}/datalist')

        self.mat = torch.FloatTensor([]).cuda()
        for i in range(len(datalist)):
            img_name = datalist[i]
            data = load_pickle(f'{self.cfg.dir["pre1"]}/{img_name}')

            if i % 2 == 0:
                continue

            for j in range(0, 2):  # 1: horizontal flip
                if j == 1 and self.cfg.data_flip == False:
                    continue
                for k in range(len(data[j]['x_coord'])):
                    if len(data[j]['x_coord'][k]) == 0:
                        continue
                    x_coord = data[j]['x_coord'][k]
                    x_data = to_tensor(x_coord)
                    if torch.max(x_data) >= self.cfg.thresd_max_x or torch.min(x_data) <= self.cfg.thresd_min_x:
                        continue
                    self.mat = torch.cat((self.mat, x_data.view(-1, 1)), dim=1)

            print('%d done!' % i)

        if self.cfg.save_pickle == True:
            save_pickle(f'{self.cfg.dir["out"]}/pickle/matrix', data=self.mat)

    def do_SVD(self):
        U, S, V = torch.svd((self.mat.cpu() - (self.cfg.width - 1) / 2) / ((self.cfg.width - 1) / 2))
        self.U = U.type(torch.float32).cuda()
        self.S = S.type(torch.float32).cuda()
        self.V = V.type(torch.float32).cuda()

        if self.cfg.save_pickle == True:
            save_pickle(f'{self.cfg.dir["out"]}/pickle/U', data=self.U)
            save_pickle(f'{self.cfg.dir["out"]}/pickle/S', data=self.S)


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

            if i == 0 and self.cfg.display_all == True:
                self.visualizer.update_datalist(self.img, self.img_name, self.seg_label, self.dir_name, self.file_name, self.img_idx)

            self.out_f.append(self.approximate_lane())

            # visualize

            if self.cfg.display_all == True and i == 0:
                self.visualizer.save_datalist([self.is_error_case])

    def run(self):
        print('start')
        self.init()
        self.construct_lane_matrix()
        self.mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/matrix')
        if self.cfg.node_sampling == True:
            self.mat = self.mat[self.cfg.sample_idx]

        self.do_SVD()
        self.U = load_pickle(f'{self.cfg.dir["out"]}/pickle/U')
        self.S = load_pickle(f'{self.cfg.dir["out"]}/pickle/S')

        self.datalist = load_pickle(f'{self.cfg.dir["pre1"]}/datalist')
        for i in range(len(self.datalist)):
            self.load_data(i)
            print(f'image {i} ===> {self.img_name} load')
            self.run_flip()
            # save pickle
            if self.cfg.save_pickle == True:
                save_pickle(f'{self.cfg.dir["out"]}/pickle/{self.img_name}', data=self.out_f)
                if self.is_error_case == False:
                    self.datalist_out.append(self.img_name)
                else:
                    self.datalist_out_error.append(self.img_name)

            print(f'image {i} ===> {self.img_name} clear')

        if self.cfg.save_pickle == True:
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist', self.datalist_out)
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_error', data=self.datalist_out_error)


    def init(self):
        self.datalist_out = list()
        self.datalist_out_error = list()


    def load_img_data(self):
        img = cv2.imread(f'{self.cfg.dir["dataset"]}/images/{self.cfg.datalist_mode}/{self.datalist[self.img_idx]}.jpg')
        img = img[self.cfg.crop_size:]
        img = cv2.resize(img, (self.cfg.width, self.cfg.height))

        seg_label = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        seg_label = np.ascontiguousarray(seg_label)

        self.img = img
        self.seg_label = seg_label

    def load_data(self, idx):
        data_org = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/label/{self.cfg.datalist_mode}/{self.datalist[idx]}')
        data = load_pickle(f'{self.cfg.dir["pre1"]}/{self.datalist[idx]}')
        self.img_name = self.datalist[idx].replace('.jpg', '')
        self.org_height = self.cfg.org_height
        self.org_width = self.cfg.org_width
        self.img_idx = idx
        self.dir_name = os.path.dirname(self.img_name)
        self.file_name = os.path.basename(self.img_name)

        self.lane_pts_org = data_org['lanes']
        self.data = data

        self.out_f = list()
        self.is_error_case = False

        if self.cfg.display_all == True:
            self.load_img_data()
