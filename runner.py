#!/usr/bin/python


from numpy.lib.utils import info
from utils import log
import matplotlib
""" from visualizer import get_local
get_local.activate() """
# from models import deformable_transformer
import os

import torch
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

# Fix problem: possible deadlock in dataloader
# import cv2
# cv2.setNumThreads(0)

from argparse import ArgumentParser
# from pprint import pprint
from datetime import datetime as dt
from config import cfg
# from models.deformable_detr import DeformableDETR
# from models.deformable_transformer import DeformableTransformer

# torch.manual_seed(1)

def get_args_from_command_line():

    parser = ArgumentParser(description='Parser of Runner of Network')
    parser.add_argument('--gpu_id', dest='gpu_id', help='GPU device id to use [cuda]', type=str)
    parser.add_argument('--phase', dest='phase', help='phase of CNN', type=str)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', type=str)
    parser.add_argument('--data_path', dest='data_path', help='Set dataset root_path', type=str)
    parser.add_argument('--data_name', dest='data_name', help='Set dataset root_path', type=str)
    parser.add_argument('--out_path', dest='out_path', help='Set output path')
    parser.add_argument('--packing', dest='packing', help='set packing')
    
    
    args = parser.parse_args()
    return args

def main():
    
    # Get args from command line
    
    args = get_args_from_command_line()
    args.n_sequence = 5
    args.n_frames_per_video = 100
    # print(args.packing)
    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.phase is not None:
        cfg.NETWORK.PHASE = args.phase
        if cfg.NETWORK.PHASE == "test":
            cfg.CONST.TEST_BATCH_SIZE = 1
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
    """ if args.packing is not None:
        cfg.CONST.PACKING = args.packing """
    if args.data_name is not None:
        
        cfg.DATASET.DATASET_NAME = args.data_name
    if args.data_path is not None:
        cfg.DIR.DATASET_ROOT = args.data_path
        if cfg.DATASET.DATASET_NAME == 'DVD':
            cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/VideoDeblur.json'
            cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/input/%s.jpg')
            cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/GT/%s.jpg')
        if cfg.DATASET.DATASET_NAME == 'GOPRO':
            cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/GoproDeblur.json'
            cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/blur_gamma/%s.png')
            cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/sharp/%s.png')
        if cfg.DATASET.DATASET_NAME in ['BSD_1ms8ms','BSD_2ms16ms','BSD_3ms24ms']:

            cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/Blur/RGB/%s.png')
            cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/Sharp/RGB/%s.png')
            if cfg.DATASET.DATASET_NAME == 'BSD_1ms8ms':
                cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/BSD_1ms8msDeblur.json'
            elif cfg.DATASET.DATASET_NAME == 'BSD_2ms16ms':
                cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/BSD_2ms16msDeblur.json'
            elif cfg.DATASET.DATASET_NAME == 'BSD_3ms24ms':
                cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/BSD_3ms24msDeblur.json'
        
        
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    """ if args.packing is not None:
        cfg.CONST.PACKING = args.packing """
    
    # #####################################################################################
    
    # #####################################################################################

    
    
    if cfg.NETWORK.PHASE == 'resume':
        output_dir = os.path.join(cfg.DIR.OUT_PATH,"train",cfg.CONST.WEIGHTS.split("/")[-3  ], '%s')
        # print_log    = os.path.join("exp_log", cfg.CONST.WEIGHTS.split("/")[0]+".log")
    elif  cfg.NETWORK.PHASE == "test":
        output_dir = os.path.join(cfg.DIR.OUT_PATH,"test",cfg.NETWORK.DEBLURNETARCH + "_" + cfg.DATASET.DATASET_NAME, '%s')
        print_log    = output_dir % 'print.log'
        dir_exit = os.path.join(cfg.DIR.OUT_PATH,"test",cfg.NETWORK.DEBLURNETARCH + "_" + cfg.DATASET.DATASET_NAME)
        if not os.path.exists(dir_exit):
            os.makedirs(dir_exit)
    else:
        output_dir = os.path.join(cfg.DIR.OUT_PATH,"train",dt.now().isoformat() + '_' + cfg.NETWORK.DEBLURNETARCH + "_" + cfg.NETWORK.TAG, '%s')
        # print_log    = os.path.join("exp_log", dt.now().isoformat() + '_' + cfg.NETWORK.DEBLURNETARCH + "_" + cfg.NETWORK.TAG +".log")
        log_dir      = output_dir % 'logs'
        ckpt_dir     = output_dir % 'checkpoints'
        # code_dir     = output_dir % 'code'
        print_log    = output_dir % 'print.log'
        data_log_dirs = [log_dir,ckpt_dir]
        for dir_exit in data_log_dirs:
            if not os.path.exists(dir_exit):
                os.makedirs(dir_exit)
    # print_log    = os.path.join("exp_log", cfg.CONST.WEIGHTS.split("/")[0]+".log")
    
    
    

    
    log.setFileHandler(print_log,mode="w")
    # Print config
    # print('Use config:')
    # pprint(cfg)
    log.info('Use config:')
    log.info(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str and not cfg.CONST.DEVICE == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    from core.build import bulid_net
    import torch
    log.info('CUDA DEVICES NUMBER: '+ str(torch.cuda.device_count()))
    
    # Setup Network & Start train/test process
    bulid_net(cfg,args,output_dir)

if __name__ == '__main__':

    main()
