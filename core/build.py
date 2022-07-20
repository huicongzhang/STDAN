#!/usr/bin/python


import os
import sys
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.packing
import models
from models.STDAN_Stack import STDAN_Stack
from datetime import datetime as dt
from tensorboardX import SummaryWriter
from core.train import train
from core.test import test
import logging
from losses.multi_loss import *
from utils import log
def  bulid_net(cfg,args,output_dir):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use

    
    log_dir      = output_dir % 'logs'
    ckpt_dir     = output_dir % 'checkpoints'
    
    
    
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    train_transforms = utils.data_transforms.Compose([
        # utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE),
        utils.data_transforms.RandomVerticalFlip(),
        utils.data_transforms.RandomHorizontalFlip(),
        # utils.data_transforms.RandomColorChannel(),
        utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
        utils.data_transforms.ToTensor(),
    ])

    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME]()
    # dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME]()
    # dataset_loader = data.Data(args)
    # Set up networks

    deblurnet = models.__dict__[cfg.NETWORK.DEBLURNETARCH].__dict__[cfg.NETWORK.DEBLURNETARCH]()

    log.info('%s Parameters in %s: %d.' % (dt.now(), cfg.NETWORK.DEBLURNETARCH,
                                                utils.network_utils.count_parameters(deblurnet)))

    # Initialize weights of networks
    # deblurnet.apply()

    # Set up solver
    base_params = []
    motion_branch_params = []
    attention_params = []
    # pretrain_params = []
    # ['reference_points', 'sampling_offsets']
    for name,param in deblurnet.named_parameters():
        if 'reference_points' in name or 'sampling_offsets' in name:
            if param.requires_grad == True:
                attention_params.append(param)
        # elif "spynet" in name or "flow_pwc" in name or "flow_net" in name:
        elif "motion_branch" in name or "motion_out" in name:
            if param.requires_grad == True:
                
                motion_branch_params.append(param)
            
                # param.requires_grad = False
        else:
            if param.requires_grad == True:
                
                base_params.append(param)
    
    optim_param = [
            {'params':base_params,'initial_lr':cfg.TRAIN.LEARNING_RATE,"lr":cfg.TRAIN.LEARNING_RATE},
            {'params':motion_branch_params,'initial_lr':cfg.TRAIN.LEARNING_RATE,"lr":cfg.TRAIN.LEARNING_RATE},
            {'params':attention_params,'initial_lr':cfg.TRAIN.LEARNING_RATE*0.01,"lr":cfg.TRAIN.LEARNING_RATE*0.01},
        ]
    # a =  filter(lambda p: p.requires_grad, deblurnet.parameters())
    deblurnet_solver = torch.optim.Adam(optim_param,lr=cfg.TRAIN.LEARNING_RATE,
                                         betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))

    

    # Load pretrained model if exists
    init_epoch       = 0
    Best_Epoch       = -1
    Best_Img_PSNR    = 0
    
    
    if cfg.NETWORK.PHASE in ['resume']:
        log.info(' %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        
        checkpoint = torch.load(os.path.join(cfg.CONST.WEIGHTS),map_location='cpu')
        # net_state_dict = deblurnet.state_dict()
        deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
        # deblurnet.load_state_dict(checkpoint['deblurnet_state_dict'])
        # state_dict = checkpoint['deblurnet_state_dict']
        # net_state_dict = deblurnet.state_dict()
        deblurnet_solver.load_state_dict(checkpoint['deblurnet_solver_state_dict'])
        for state in deblurnet_solver.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        # deblurnet_lr_scheduler.load_state_dict(checkpoint['deblurnet_lr_scheduler'])
        init_epoch = checkpoint['epoch_idx']
        Best_Img_PSNR = checkpoint['Best_Img_PSNR']
        Best_Epoch = checkpoint['Best_Epoch']
        
    
    
    elif cfg.NETWORK.PHASE in ['test']:
        log.info(' %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        
        checkpoint = torch.load(os.path.join(cfg.CONST.WEIGHTS),map_location='cpu')
        
        weights = {}
        for k,v in checkpoint.items():
            if 'flow_net' not in k:
                weights.update({k.replace('module.','').replace('Defattn1.','MMA.').replace('Defattn3.','MSA.'):v})
       
        deblurnet.load_state_dict(weights)
        
        init_epoch = 0
        Best_Img_PSNR = 0
        Best_Epoch = 0
        
        log.info('{0} Recover complete. Current epoch #{1}, Best_Img_PSNR = {2} at epoch #{3}.' \
              .format(dt.now(), init_epoch, Best_Img_PSNR, Best_Epoch))
       
    
    deblurnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(deblurnet_solver,
                                                                milestones=cfg.TRAIN.LR_MILESTONES,
                                                                gamma=cfg.TRAIN.LR_DECAY,last_epoch=(init_epoch))
    
    if torch.cuda.is_available():
        deblurnet = torch.nn.DataParallel(deblurnet).cuda()
    


   
    
    if cfg.NETWORK.PHASE in ["train","resume"]:
        train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
        test_writer  = SummaryWriter(os.path.join(log_dir, 'test'))
    else:
        train_writer = None
        test_writer  = None
    
    """ if cfg.CONST.PACKING == True and cfg.NETWORK.PHASE in ['train']:
        utils.packing.packing(os.path.join(code_dir,"code.tar"),".") """
        
    log.info(' Output_dir： {0}'.format(output_dir[:-2]))
    
    if cfg.NETWORK.PHASE in ['train','resume']:
        train(cfg, init_epoch, dataset_loader, train_transforms, test_transforms,
                              deblurnet, deblurnet_solver, deblurnet_lr_scheduler,
                              ckpt_dir, train_writer, test_writer,
                              Best_Img_PSNR, Best_Epoch)
    else:
        
        test(cfg, init_epoch,Best_Img_PSNR,ckpt_dir,dataset_loader, test_transforms, deblurnet, deblurnet_solver,test_writer)
        