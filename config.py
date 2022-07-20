#!/usr/bin/python

from pickle import FALSE
from easydict import EasyDict as edict
import os
import socket

__C     = edict()
cfg     = __C

#
# Common
#
__C.CONST                               = edict()
__C.CONST.DEVICE                        = '1'                   # gpu_ids
__C.CONST.NUM_WORKER                    = 8                               # number of data workers
__C.CONST.WEIGHTS                       = 'weights/DVD_release.pth' # data weights path
__C.CONST.TRAIN_BATCH_SIZE              = 8
__C.CONST.TEST_BATCH_SIZE               = 4
# __C.CONST.PACKING                       = True
#
# Dataset
#
__C.DATASET                             = edict()
__C.DATASET.DATASET_NAME                = 'DVD'       # available options:  'DVD','GOPRO','BSD_1ms8ms','BSD_2ms16ms','BSD_3ms24ms'

#
# logs and checkpoint Directories
#
__C.DIR                                 = edict()
__C.DIR.OUT_PATH = './exp_log'                          # logs path

#
# please set the DATASET_ROOT to your path
#
if cfg.DATASET.DATASET_NAME == 'DVD':
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/VideoDeblur.json'
    __C.DIR.DATASET_ROOT = '/home/hczhang/datasets/DeepVideoDeblurring_Dataset/quantitative_datasets'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/input/%s.jpg')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/GT/%s.jpg')
# real
elif cfg.DATASET.DATASET_NAME == 'DVD_Real':
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/DVD_Real.json'
    __C.DIR.DATASET_ROOT = '/home/hczhang/datasets/DeepVideoDeblurring_Dataset/qualitative_datasets'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/input/%s.jpg')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/input/%s.jpg')
elif cfg.DATASET.DATASET_NAME == 'GOPRO':
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/GoproDeblur.json'
    __C.DIR.DATASET_ROOT = '/home/hczhang/datasets/GOPRO_Large'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/blur_gamma/%s.png')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/sharp/%s.png')
elif cfg.DATASET.DATASET_NAME == 'BSD_1ms8ms':
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/BSD_1ms8msDeblur.json'
    __C.DIR.DATASET_ROOT = '/home/hczhang/datasets/BSD/BSD_1ms8ms'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/Blur/RGB/%s.png')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/Sharp/RGB/%s.png')
elif cfg.DATASET.DATASET_NAME == 'BSD_2ms16ms':
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/BSD_2ms16msDeblur.json'
    __C.DIR.DATASET_ROOT = '/home/hczhang/datasets/BSD/BSD_2ms16ms'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/Blur/RGB/%s.png')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/Sharp/RGB/%s.png')
elif cfg.DATASET.DATASET_NAME == 'BSD_3ms24ms':
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/BSD_3ms24msDeblur.json'
    __C.DIR.DATASET_ROOT = '/home/hczhang/datasets/BSD/BSD_3ms24ms'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/Blur/RGB/%s.png')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/Sharp/RGB/%s.png')

#
# data augmentation
#
__C.DATA                                = edict()
__C.DATA.STD                            = [255.0, 255.0, 255.0]
__C.DATA.MEAN                           = [0.0, 0.0, 0.0]
__C.DATA.CROP_IMG_SIZE                  = [256, 256]              # Crop image size: height, width
__C.DATA.GAUSSIAN                       = [0, 1e-4]               # mu, std_var
__C.DATA.COLOR_JITTER                   = [0.2, 0.15, 0.3, 0.1]   # brightness, contrast, saturation, hue
__C.DATA.TRAIN_SEQ_LENGTH               = 5
__C.DATA.FRAME_LENGTH                   = 5
__C.DATA.TEST_SEQ_LENGTH                = 5


#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.DEBLURNETARCH               = 'STDAN_Stack'             
__C.NETWORK.PHASE                       = 'test'                 # available options: 'train', 'test', 'resume'
__C.NETWORK.TAG                         = "DVD"                  # logs folder tag

#
# Training
#

__C.TRAIN                               = edict()
__C.TRAIN.USE_PERCET_LOSS               = False
__C.TRAIN.NUM_EPOCHES                   = 1200                    # maximum number of epoches
__C.TRAIN.LEARNING_RATE                 = 1e-4
__C.TRAIN.LR_MILESTONES                 = [400,600,800,1000]   
__C.TRAIN.LR_DECAY                      = 0.5                   # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM                      = 0.9
__C.TRAIN.BETA                          = 0.999
__C.TRAIN.BIAS_DECAY                    = 0.0                    # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY                  = 0.0                    # regularization of weight, default: 0
__C.TRAIN.PRINT_FREQ                    = 100                    # print step
__C.TRAIN.SAVE_FREQ                     = 10                     # weights will be overwritten every save_freq epoch

#
# Testing options
#
__C.TEST                                = edict()
# __C.TEST.VISUALIZATION_NUM              = 10
__C.TEST.PRINT_FREQ                     = 5
if __C.NETWORK.PHASE == 'test':
    __C.CONST.TEST_BATCH_SIZE           = 1
