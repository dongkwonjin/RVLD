import numpy as np
from libs.utils import *

class Test_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.testloader = dict_DB['testloader']
        self.post_process = dict_DB['post_process']
        self.save_pred_for_eval_iou = dict_DB['save_pred_for_eval_iou']
        self.eval_iou_laneatt = dict_DB['eval_iou_laneatt']
        self.eval_seg = dict_DB['eval_seg']
        self.visualizer = dict_DB['visualizer']

    def init_data(self):
        self.result = {'out': {}, 'gt': {}, 'name': []}
        self.datalist = []
        self.eval_seg.init()

    def batch_to_cuda(self, batch):
        for name in list(batch):
            if torch.is_tensor(batch[name]):
                batch[name] = batch[name].cuda()
            elif type(batch[name]) is dict:
                for key in batch[name].keys():
                    batch[name][key] = batch[name][key].cuda()
        return batch

    def run(self, model_s, model, mode='val'):
        self.init_data()

        with torch.no_grad():
            model_s.eval()
            model.eval()
            for i, batch in enumerate(self.testloader):  # load batch data
                batch = self.batch_to_cuda(batch)

                # model
                out = dict()
                model_s.forward_for_encoding(batch['img'])
                model_s.forward_for_squeeze()
                out.update(model_s.forward_for_classification())
                model.prob_map = model_s.prob_map[:, 1:]
                out.update(model.forward_for_regression())


                self.post_process.update(batch, mode)
                out_post = self.post_process.run(out)

                self.eval_seg.update(batch, out, mode)
                self.eval_seg.run_for_miou()
                self.eval_seg.run_for_fscore()

                for j in range(len(out_post)):
                    # visualize
                    if self.cfg.disp_test_result == True:
                        out.update(out_post[j])
                        self.visualizer.display_for_test(batch=batch, out=out, batch_idx=j, mode=mode)

                    # record output data
                    self.result['out']['x_coords'] = out_post[j]['x_coords']
                    self.result['out']['height_idx'] = out_post[j]['height_idx']
                    self.result['name'] = batch['img_name'][j]

                    if self.cfg.save_pickle == True:
                        save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/{batch["img_name"][j].replace(".jpg", "")}', data=self.result)

                self.datalist += batch['img_name']

                if i % 50 == 1:
                    print(f'image {i} ---> {batch["img_name"][0]} done!')

        if self.cfg.save_pickle == True:
            save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/datalist', data=self.datalist)
            save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/eval_seg_results', data=self.eval_seg.results)


        # evaluation
        if 'training' in mode:
            return self.evaluation_for_training_set(mode)
        else:
            return self.evaluation(mode)

    def evaluation(self, mode):
        metric = dict()

        try:
            metric.update(self.eval_seg.measure())
        except:
            print('e')

        if self.cfg.do_eval_iou_laneatt == True:
            self.save_pred_for_eval_iou.settings(key=['x_coords'], test_mode=mode, use_height=True)
            self.save_pred_for_eval_iou.run()
            metric.update(self.eval_iou_laneatt.measure_IoU(mode, self.cfg.iou_thresd['laneatt']))

        return metric

    def evaluation_for_training_set(self, mode):
        datalist = load_pickle(f'{self.cfg.dir["out"]}/{mode}/pickle/datalist')
        results = load_pickle(f'{self.cfg.dir["out"]}/{mode}/pickle/eval_seg_results')
        err = list()
        for i in range(len(results['fp'])):
            err.append(results['fp'][i] + results['fn'][i])
        err = np.float32(err)
        idx_sorted = np.argsort(err)[::-1]
        idx_sorted = idx_sorted[:int(len(idx_sorted) * 0.1)]
        error_list = sorted(list(np.array(datalist)[idx_sorted]))
        save_pickle(path=f'{self.cfg.dir["out"]}/val_for_training_set/pickle/errorlist', data=error_list)
        save_pickle(path=f'{self.cfg.dir["out"]}/val_for_training_set/pickle/error', data=err)
        try:
            self.visualizer.disp_errorlist(error_list, mode)
        except:
            print('done!')
