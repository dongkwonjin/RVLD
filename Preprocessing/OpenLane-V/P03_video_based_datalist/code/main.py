from options.config import Config
from options.args import *
from libs.preprocess import *

def run_preprocessing(cfg, dict_DB):
    preprocess = Preprocessing(cfg, dict_DB)
    preprocess.run()

def main():

    # option
    cfg = Config()
    cfg = parse_args(cfg)

    # prepare
    dict_DB = dict()

    # run
    run_preprocessing(cfg, dict_DB)

if __name__ == '__main__':
    main()