import os

from options.config import Config
from options.args import *
from tools.test import *
from tools.train import *
from libs.prepare import *

def main_eval(cfg, dict_DB):
    # eval option
    test_process = Test_Process(cfg, dict_DB)
    test_process.evaluation(mode='test')

def main_val_for_training_set(cfg, dict_DB):
    # test option
    test_process = Test_Process(cfg, dict_DB)
    test_process.run(dict_DB['model'], mode='val_for_training_set')

def main_test(cfg, dict_DB):
    # test option
    test_process = Test_Process(cfg, dict_DB)
    test_process.run(dict_DB['model'], mode='test')

def main_train(cfg, dict_DB):
    # train option
    dict_DB['test_process'] = Test_Process(cfg, dict_DB)
    train_process = Train_Process(cfg, dict_DB)
    train_process.run()

def main():
    # Config
    cfg = Config()
    cfg = parse_args_default(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    torch.backends.cudnn.deterministic = True

    # prepare
    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_dataloader(cfg, dict_DB)
    dict_DB = prepare_model(cfg, dict_DB)
    dict_DB = prepare_evaluation(cfg, dict_DB)
    dict_DB = prepare_training(cfg, dict_DB)

    if 'val_for_training_set' in cfg.run_mode:
        main_val_for_training_set(cfg, dict_DB)
    elif 'test' in cfg.run_mode:
        main_test(cfg, dict_DB)
    elif 'train' in cfg.run_mode:
        main_train(cfg, dict_DB)
    elif 'eval' in cfg.run_mode:
        main_eval(cfg, dict_DB)


if __name__ == '__main__':
    main()
