import torch

from datasets.dataset_vil100 import *

from libs.save_model import *
from libs.utils import *
from libs.utils import _init_fn

class Train_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']

        self.model = dict_DB['model']

        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualizer = dict_DB['visualizer']

        self.test_process = dict_DB['test_process']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']
        self.iteration = dict_DB['iteration']
        self.batch_iteration = dict_DB['batch_iteration']

        self.cfg.iteration['validation'] = len(self.dataloader)
        self.cfg.iteration['record'] = self.cfg.iteration['validation'] // 4

    def batch_to_cuda(self, batch):
        for name in list(batch):
            if torch.is_tensor(batch[name]):
                batch[name] = batch[name].cuda()
            elif type(batch[name]) is dict:
                for key in batch[name].keys():
                    batch[name][key] = batch[name][key].cuda()
        return batch

    def training(self):
        loss_t = dict()

        # train start
        self.model.train()
        print('train start')
        logger('train start\n', self.logfile)
        for i, batch in enumerate(self.dataloader):
            # load data
            batch = self.batch_to_cuda(batch)

            # model
            out = dict()
            self.model.forward_for_encoding(batch['img'])
            self.model.forward_for_squeeze()
            out.update(self.model.forward_for_classification())

            # loss
            loss = self.loss_fn(
                out=out,
                gt=batch
            )

            # optimize
            self.optimizer.zero_grad()
            loss['sum'].backward()
            self.optimizer.step()

            for l_name in loss:
                if l_name not in loss_t.keys():
                    loss_t[l_name] = 0
                loss_t[l_name] += loss[l_name].item()

            if i % self.cfg.disp_step == 0:
                print(f'epoch : {self.epoch}, batch_iteration {self.batch_iteration}, iteration : {self.iteration}, iter per epoch : {i} ==> {batch["img_name"][0]}')
                self.visualizer.display_for_train(batch, out, i)
                for l_name in loss:
                    logger(f'epoch {self.epoch} batch_iteration {self.batch_iteration} iteration {self.iteration} iter per epoch : {i} Loss_{l_name} : {round(loss[l_name].item(), 4)}, ', self.logfile)
                logger(f'||| {batch["img_name"][0]}\n', self.logfile)

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

            if self.epoch > 100 or (self.epoch + 1) % 1 == 0:
                if self.batch_iteration % self.cfg.iteration['validation'] == 0:
                    self.test()
                    self.model.train()

            self.scheduler.step(self.batch_iteration)

    def test(self):
        metric = self.test_process.run(self.model, mode='val')

        logger(f'\nEpoch {self.ckpt["epoch"]} Iteration {self.ckpt["iteration"]} ==> Validation result', self.logfile)
        print(f'\nEpoch {self.ckpt["epoch"]} Iteration {self.ckpt["iteration"]}')
        for key in metric.keys():
            logger(f'{key} {metric[key]}\n', self.logfile)
            print(f'{key} {metric[key]}\n')

        namelist = ['seg_fscore']
        for name in namelist:
            model_name = f'checkpoint_max_{name}_{self.cfg.dataset_name}'
            self.val_result[name] = save_model_max(self.ckpt, self.cfg.dir['weight'], self.val_result[name], metric[name], logger, self.logfile, model_name)

    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch

            print(f'\nepoch {epoch}\n')
            logger(f'\nepoch {epoch}\n', self.logfile)

            self.training()
