import argparse

def parse_args(cfg):
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--run_mode', type=str, default='train', help='run mode (train, test)')
    parser.add_argument('--pre_dir', type=str, default='--root/preprocessed/DATASET_NAME/', help='preprocessed data dir')
    parser.add_argument('--dataset_dir', default=None, help='dataset dir')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg

# Example on my PC env
# --------------------------------------------------------
def parse_args_default(cfg):
    root = '/media/dkjin/4fefb28c-5de9-4abd-a935-aa2d61392048'
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--run_mode', type=str, default='train', help='run mode (train, test, test_paper)')
    parser.add_argument('--pre_dir', type=str, default=f'{root}/Work/Current/Lane_detection/Project_02/P07_github/preprocessed/OpenLane-V', help='preprocessed data dir')
    parser.add_argument('--dataset_dir', default='/home/dkjin/Project/Dataset/OpenLane', help='dataset')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg
# --------------------------------------------------------

def args_to_config(cfg, args):
    if args.dataset_dir is not None:
        cfg.dir['dataset'] = args.dataset_dir
    if args.pre_dir is not None:
        cfg.dir['head_pre'] = args.pre_dir
        cfg.dir['pre2'] = cfg.dir['pre2'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre3_train'] = cfg.dir['pre3_train'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre3_test'] = cfg.dir['pre3_test'].replace('--preprocessed data path', args.pre_dir)

    return cfg
