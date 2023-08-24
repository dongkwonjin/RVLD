import torch

from datasets.dataset_openlane import *

from libs.save_model import *
from libs.utils import *
from libs.utils import _init_fn

class Train_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']

        self.model_s = dict_DB['model_s']
        self.model_c = dict_DB['model_c']
        self.model = dict_DB['model']

        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualizer = dict_DB['visualizer']

        self.test_process = dict_DB['test_process']
        self.post_process = dict_DB['post_process']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']
        self.iteration = dict_DB['iteration']
        self.batch_iteration = dict_DB['batch_iteration']

        self.cfg.iteration['validation'] = len(self.dataloader) // 4
        self.cfg.iteration['record'] = self.cfg.iteration['validation'] // 10

        self.vm = dict_DB['video_memory']

    def batch_to_cuda(self, batch):
        for name in list(batch):
            if torch.is_tensor(batch[name]):
                batch[name] = batch[name].cuda()
            elif type(batch[name]) is dict:
                for key in batch[name].keys():
                    batch[name][key] = batch[name][key].cuda()
        return batch

    def finetune_model(self):
        val1 = True
        val2 = False

        for param in self.model.regressor.parameters():
            param.requires_grad = val1
        for param in self.model.offset_regression.parameters():
            param.requires_grad = val1
        for param in self.model.deform_conv2d.parameters():
            param.requires_grad = val1
        for param in self.model.classifier.parameters():
            param.requires_grad = val2
        if val1 == False:
            self.model.regressor.eval()
            self.model.offset_regression.eval()
            self.model.deform_conv2d.eval()
        if val2 == False:
            self.model.classifier.eval()

    def training(self):
        loss_t = dict()
        # train start
        self.model_s.eval()
        self.model_c.eval()
        self.model.train()
        print('train start')
        logger('train start\n', self.logfile)

        self.finetune_model()

        for i, batch in enumerate(self.dataloader):
            # load data
            for t in range(self.cfg.clip_length + 1):
                key_t = f't-{t}'
                batch[key_t] = self.batch_to_cuda(batch[key_t])

            # model
            out = dict()
            loss_per_clip = dict()
            for t in range(self.cfg.clip_length, -1, -1):
                key_t = f't-{t}'
                out[key_t] = dict()
                if self.cfg.clip_length - t == 0:
                    self.vm.forward_for_dict_initialization()
                else:
                    self.vm.forward_for_dict_memorization()

                self.model.clip_idx = self.cfg.clip_length - t

                self.vm.forward_for_dict_initialization_per_frame(t='t-0')
                self.model_s.forward_for_encoding(batch[key_t]['img'])
                self.model_s.forward_for_squeeze()
                out[key_t].update(self.model_s.forward_for_classification())
                self.model_c.prob_map = self.model_s.prob_map[:, 1:]
                out[key_t].update(self.model_c.forward_for_regression())

                self.vm.forward_for_dict_update_per_frame(self.model_s, mode='intra')
                if self.cfg.clip_length - t >= self.cfg.num_t:
                    self.model = self.vm.forward_for_dict_transfer(self.model)

                    out[key_t].update(self.model.forward_for_feat_aggregation(is_training=True))
                    out[key_t].update(self.model.forward_for_classification())
                    out[key_t].update(self.model.forward_for_regression())
                    self.vm.forward_for_dict_update_per_frame(self.model, mode='update')

                    # loss
                    loss_per_clip[key_t] = self.loss_fn(out=out, gt=batch, t=key_t, epoch=self.epoch)

                # lane mask guide
                self.post_process.mode = ('f' if self.cfg.clip_length - t >= self.cfg.num_t else 'init')
                out_post = self.post_process.run_for_training(out[key_t])
                out[key_t].update(self.post_process.lane_mask_generation_for_training(out_post, batch[key_t]))
                self.vm.data['guide_cls']['t-0'] = out[key_t]['guide_cls']

            # total loss for a video clip
            loss = dict()
            loss['sum'] = torch.FloatTensor([0.0]).cuda()
            for t in range(self.cfg.clip_length - 1, -1, -1):
                key_t = f't-{t}'
                loss['sum'] += loss_per_clip[key_t]['sum']
            loss['sum'] /= self.cfg.clip_length

            # optimize
            self.optimizer.zero_grad()
            loss['sum'].backward()
            self.optimizer.step()

            for l_name in loss_per_clip['t-0']:
                if l_name not in loss_t.keys():
                    loss_t[l_name] = 0
                if l_name in loss_per_clip['t-0'].keys():
                    loss_t[l_name] += loss_per_clip['t-0'][l_name].item()

            if i % self.cfg.disp_step == 0:
                print(f'epoch : {self.epoch}, batch_iteration {self.batch_iteration}, iteration : {self.iteration}, iter-per-epoch : {i} ==> {batch["t-0"]["img_name"][0]}')
                self.visualizer.display_for_train(batch, out, i)

                logger(f'epoch {self.epoch} batch_iteration {self.batch_iteration} iteration {self.iteration} iter-per-epoch : {i} Loss_{"t-0"} : {round(loss_per_clip["t-0"]["sum"].item(), 4)}, ', self.logfile)
                logger(f'||| {batch["t-0"]["img_name"][0]}\n', self.logfile)

            self.iteration += self.cfg.batch_size['img']
            self.batch_iteration += 1
            if (self.batch_iteration % self.cfg.iteration['record']) == 0 or (self.batch_iteration % self.cfg.iteration['validation']) == 0:
                # save model
                self.ckpt = {'epoch': self.epoch,
                             'iteration': self.iteration,
                             'batch_iteration': self.batch_iteration,
                             'model': self.model,
                             'optimizer': self.optimizer,
                             'val_result': self.val_result}

                save_model(checkpoint=self.ckpt, param='checkpoint_final', path=self.cfg.dir['weight'])

            if (self.batch_iteration % self.cfg.iteration['record']) == 0:
                # logger
                logger('\nAverage Loss : ', self.logfile)
                print('\nAverage Loss : ', end='')
                for l_name in loss_t:
                    logger(f'{l_name} : {round(loss_t[l_name] / (i + 1), 6)}, ', self.logfile)
                for l_name in loss_t:
                    print(f'{l_name} : {round(loss_t[l_name] / (i + 1), 6)}, ', end='')
                print('\n')

            if self.batch_iteration % self.cfg.iteration['validation'] == 0 and self.epoch >= self.cfg.epoch_update1:
                self.test()
                self.model.train()
                self.finetune_model()

            self.scheduler.step(self.batch_iteration)

    def test(self):
        metric = self.test_process.run(self.model_s, self.model_c, self.model, mode='val')

        logger(f'\nEpoch {self.ckpt["epoch"]} Iteration {self.ckpt["iteration"]} ==> Validation result', self.logfile)
        print(f'\nEpoch {self.ckpt["epoch"]} Iteration {self.ckpt["iteration"]}')
        for key in metric.keys():
            logger(f'{key} {metric[key]}\n', self.logfile)
            print(f'{key} {metric[key]}\n')

        namelist = ['seg_fscore', 'F1']
        for name in namelist:
            model_name = f'checkpoint_max_{name}_{self.cfg.dataset_name}'
            self.val_result[name] = save_model_max(self.ckpt, self.cfg.dir['weight'], self.val_result[name], metric[name], logger, self.logfile, model_name)

    def update_dataloader(self):
        if self.epoch == self.cfg.epoch_update2 and self.cfg.clip_length <= 1:
            print(f'{self.epoch} : update dataloader : {self.cfg.clip_length} --> {self.cfg.clip_length + 1}')
            self.cfg.clip_length += 1
            dataset = Dataset_Train(cfg=self.cfg, update=False)
            trainloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.cfg.batch_size['img'],
                                                      shuffle=True,
                                                      num_workers=self.cfg.num_workers,
                                                      worker_init_fn=_init_fn)
            self.dataloader = trainloader

    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch

            print(f'\nepoch {epoch}\n')
            logger(f'\nepoch {epoch}\n', self.logfile)
            self.update_dataloader()
            self.training()
