from datasets.dataset_openlane import *
from libs.visualizer import *
from evaluation.evaluate import Evaluation
from libs.post_processing import *
from libs.save_prediction import *
from evaluation.evaluate_iou_laneatt import LaneEval_CULane_LaneATT

from libs.utils import _init_fn
from libs.load_model import *

def prepare_dataloader(cfg, dict_DB):
    # train dataloader
    dataset = Dataset_Train(cfg=cfg)
    trainloader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=cfg.batch_size['img'],
                                              shuffle=True,
                                              num_workers=cfg.num_workers,
                                              worker_init_fn=_init_fn)
    dict_DB['trainloader'] = trainloader

    # test dataloader
    dataset = Dataset_Test(cfg=cfg, mode=cfg.run_mode)
    testloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=cfg.batch_size['img'] * 2,
                                             shuffle=False,
                                             num_workers=cfg.num_workers,
                                             pin_memory=False)
    dict_DB['testloader'] = testloader
    return dict_DB

def prepare_model(cfg, dict_DB):
    if 'val' in cfg.run_mode:
        dict_DB = load_model_for_test(cfg, dict_DB)
    elif 'test' in cfg.run_mode:
        dict_DB = load_model_for_test(cfg, dict_DB)
    elif 'train' in cfg.run_mode:
        dict_DB = load_model_for_train(cfg, dict_DB)

    return dict_DB

def prepare_post_processing(cfg, dict_DB):
    dict_DB['post_process'] = Post_Processing(cfg=cfg)
    dict_DB['save_pred_for_eval_iou'] = Save_Prediction_for_eval_iou(cfg=cfg)
    return dict_DB

def prepare_visualization(cfg, dict_DB):
    dict_DB['visualizer'] = Visualize_cv(cfg=cfg)
    return dict_DB

def prepare_evaluation(cfg, dict_DB):
    dict_DB['eval_seg'] = Evaluation(cfg=cfg)
    dict_DB['eval_iou_laneatt'] = LaneEval_CULane_LaneATT(cfg=cfg)
    return dict_DB

def prepare_training(cfg, dict_DB):
    logfile = f'{cfg.dir["out"]}/train/log/logfile.txt'
    mkdir(path=f'{cfg.dir["out"]}/train/log/')
    if cfg.run_mode == 'train' and cfg.resume == True:
        rmfile(path=logfile)
        val_result = dict()
        val_result['acc'] = 0
        val_result['F1'] = 0
        val_result['f'] = 0
        val_result['miou'] = 0
        val_result['err'] = 999

        val_result['seg_miou'] = 0
        val_result['seg_fscore'] = 0
        dict_DB['val_result'] = val_result
        dict_DB['epoch'] = 0
        dict_DB['iteration'] = 0
        dict_DB['batch_iteration'] = 0

        record_config(cfg, logfile)

    dict_DB['logfile'] = logfile
    return dict_DB


