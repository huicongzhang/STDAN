import os
from turtle import backward
from utils import log, util
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import torchvision
import random

from losses.multi_loss import *
from time import time

from core.test import test
# from models.VGG19 import VGG19
# from utils.network_utils import flow2rgb
from tqdm import tqdm
from models.submodules import warp
def warp_loss(frames_list,flow_forwards,flow_backwards):
    n, t, c, h, w = frames_list.size()
    
    forward_loss = 0
    backward_loss = 0
    for idx in [[0,1,2],[1,2,3],[2,3,4],[1,2,3]]:
        frames = frames_list[:,idx,:,:,:]
        for flow_forward,flow_backward in zip(flow_forwards,flow_backwards):
            frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            backward_frames = warp(frames_1,flow_backward.reshape(-1, 2, h, w))
            forward_frames = warp(frames_2,flow_forward.reshape(-1, 2, h, w))
            forward_loss += l1Loss(forward_frames,frames_1)
            backward_loss += l1Loss(backward_frames,frames_2)
    return (0.5*forward_loss + 0.5*backward_loss)/len(flow_forwards)

def train(cfg, init_epoch, dataset_loader, train_transforms, val_transforms,
                                  deblurnet, deblurnet_solver, deblurnet_lr_scheduler,
                                  ckpt_dir, train_writer, val_writer,
                                  Best_Img_PSNR, Best_Epoch):

        # init_epoch = init_epoch -1
    # n_itr = 8.649*1000
    n_itr = 0
    # Training loop
    
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Set up data loader
        train_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, train_transforms),
            batch_size=cfg.CONST.TRAIN_BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)
        # train_data_loader = dataset_loader.loader_train

        
        # Tick / tock
        epoch_start_time = time()
        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        deblur_mse_losses = utils.network_utils.AverageMeter()
        warp_mse_losses = utils.network_utils.AverageMeter()
        
        deblur_losses = utils.network_utils.AverageMeter()
        
        img_PSNRs_iter1 = utils.network_utils.AverageMeter()
        img_PSNRs_iter2 = utils.network_utils.AverageMeter()
        
        # Adjust learning rate
        
        

        batch_end_time = time()
        


        log.info(' learning rate: {0}'.format(deblurnet_lr_scheduler.get_last_lr()))
        tqdm_train = tqdm(train_data_loader)
        tqdm_train.set_description('[TRAIN] [Epoch {0}/{1}]'.format(epoch_idx,cfg.TRAIN.NUM_EPOCHES))
        for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_train):
            # Measure data time
            data_time.update(time() - batch_end_time)
            # Get data from data loader
            seq_blur  = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_blur]
            seq_clear = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_clear]
            
            # switch models to training mode
            deblurnet.train()

            # Train the model
            
            input_seq = torch.cat(seq_blur,1)
            gt_seq = torch.cat(seq_clear,1)
            
            b,t,c,h,w = gt_seq.shape
            recons_1, recons_2, recons_3, out,flow_fowards,flow_backwards = deblurnet(input_seq)

            output_img = torch.cat([recons_1, recons_2, recons_3, out],dim=1)
            
            down_simple_gt = F.interpolate(gt_seq.reshape(-1,c,h,w), size=(h//4, w//4),mode='bilinear', align_corners=True).reshape(b,t,c,h//4,w//4)
            warploss = warp_loss(down_simple_gt, flow_fowards, flow_backwards)*0.05 
            warp_mse_losses.update(warploss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            
            t_gt_seq = torch.cat([gt_seq[:,1,:,:,:],gt_seq[:,2,:,:,:],gt_seq[:,3,:,:,:],gt_seq[:,2,:,:,:]],dim=1)
            deblur_mse_loss = l1Loss(output_img, t_gt_seq)
            deblur_mse_losses.update(deblur_mse_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            
            deblur_loss = deblur_mse_loss + warploss  
            deblur_losses.update(deblur_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
        
            img_PSNR = util.calc_psnr(out.detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_iter1.update(img_PSNR, cfg.CONST.TRAIN_BATCH_SIZE)
   
            img_PSNR = util.calc_psnr(recons_2.detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_iter2.update(img_PSNR, cfg.CONST.TRAIN_BATCH_SIZE)
            
            
            deblurnet_solver.zero_grad()
            deblur_loss.backward()
            deblurnet_solver.step()
            

            
            n_itr = n_itr + 1

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            # t.set_postfix()
            tqdm_train.set_postfix_str("  DeblurLoss {0} [{1}, {2}] PSNR_itr1 {3} PSNR_itr2 {4}".format(
                                        deblur_losses, deblur_mse_losses, warp_mse_losses,img_PSNRs_iter2,img_PSNRs_iter1))
            
        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EpochPSNR_TRAIN', img_PSNRs_iter1.avg, epoch_idx)
        deblurnet_lr_scheduler.step()
        
        epoch_end_time = time()
        log.info('[TRAIN] [Epoch {0}/{1}]\t EpochTime {2}\t itr1 {3} itr2 {4}'
              .format(epoch_idx, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, img_PSNRs_iter2.avg,img_PSNRs_iter1.avg))
        
       
        

        if epoch_idx%5 == 0:
            test_img_PSNR,Best_Img_PSNR = test(cfg, epoch_idx, Best_Img_PSNR,ckpt_dir,dataset_loader, val_transforms, deblurnet,deblurnet_solver,val_writer)
            
         
        
        if epoch_idx%cfg.TRAIN.SAVE_FREQ == 0:
            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % ((epoch_idx) % (10*cfg.TRAIN.SAVE_FREQ)  )), \
                                                      epoch_idx, deblurnet,deblurnet_solver, \
                                                      Best_Img_PSNR, Best_Epoch)
            
        
        utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'latest-ckpt.pth.tar'), \
                                                      epoch_idx, deblurnet, deblurnet_solver,\
                                                      Best_Img_PSNR, Best_Epoch)
        

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()