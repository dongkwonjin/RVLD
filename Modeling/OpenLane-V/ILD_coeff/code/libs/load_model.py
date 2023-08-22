import torch
from models.model import Model
from models.model_s import Model as Model_Single
from models.loss import *

def load_model_for_test(cfg, dict_DB):
    dict_DB['model_s'] = load_pretrained_model_single(cfg)
    if cfg.param_name == 'trained_last':
        checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_final')
    elif cfg.param_name == 'max':
        checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_max_F1_{cfg.dataset_name}')
    model = Model(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    dict_DB['model'] = model
    return dict_DB

def load_model_for_train(cfg, dict_DB):
    dict_DB['model_s'] = load_pretrained_model_single(cfg)
    model = Model(cfg=cfg)
    model.cuda()

    if cfg.optim['mode'] == 'adam_w':
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=cfg.optim['lr'],
                                      weight_decay=cfg.optim['weight_decay'],
                                      betas=cfg.optim['betas'], eps=cfg.optim['eps'])
    elif cfg.optim['mode'] == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=cfg.optim['lr'],
                                     weight_decay=cfg.optim['weight_decay'])

    cfg.optim['milestones'] = list(np.arange(0, len(dict_DB['trainloader']) * cfg.epochs, len(dict_DB['trainloader']) * 4))[1:]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=cfg.optim['milestones'],
                                                     gamma=cfg.optim['gamma'])

    if cfg.resume == False:
        checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_final')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=cfg.optim['milestones'],
                                                         gamma=cfg.optim['gamma'],
                                                         last_epoch=checkpoint['batch_iteration'])
        dict_DB['epoch'] = checkpoint['epoch']
        dict_DB['iteration'] = checkpoint['iteration']
        dict_DB['batch_iteration'] = checkpoint['batch_iteration']
        dict_DB['val_result'] = checkpoint['val_result']

    loss_fn = Loss_Function(cfg)

    dict_DB['model'] = model
    dict_DB['optimizer'] = optimizer
    dict_DB['scheduler'] = scheduler
    dict_DB['loss_fn'] = loss_fn

    return dict_DB

def load_pretrained_model_single(cfg):
    checkpoint = torch.load(f'{cfg.dir["pretrained_weight"]}/checkpoint_max_seg_fscore_{cfg.dataset_name}')
    model = Model_Single(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model
