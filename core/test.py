from torch import gt
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multi_loss import *
from utils import util
import torchvision

import numpy as np
import scipy.io as io
from utils import log

from time import time
# from visualizer import get_local
# from utils.imgio_gen import visulize_attention_ratio
from utils.util import ssim_calculate
from tqdm import tqdm
import pandas as pd
# from mmflow.datasets import visualize_flow, write_flow
from models.submodules import warp
def warp_loss(frames_list,flow_forwards,flow_backwards):
    n, t, c, h, w = frames_list.size()
    
    forward_loss = 0
    backward_loss = 0
    for flag,idx in enumerate([[1,2,3]]):
        frames = frames_list[:,idx,:,:,:]
        # for flow_forward,flow_backward in zip(flow_forwards,flow_backwards):
        flow_forward = flow_forwards
        flow_backward = flow_backwards
        # flow_forward = torch.zeros_like(flow_forwards)
        # flow_backward = torch.zeros_like(flow_backwards)
        frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
        frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
        backward_frames = warp(frames_1,flow_backward.reshape(-1, 2, h, w))
        forward_frames = warp(frames_2,flow_forward.reshape(-1, 2, h, w))
        forward_loss += l1Loss(forward_frames,frames_1)
        backward_loss += l1Loss(backward_frames,frames_2)
    return (0.5*forward_loss + 0.5*backward_loss)
def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def test(cfg, epoch_idx, Best_Img_PSNR,ckpt_dir,dataset_loader, test_transforms, deblurnet, deblurnet_solver,test_writer):
    # Set up data loader
    test_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, test_transforms),
        batch_size=cfg.CONST.TEST_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)
   
    # test_data_loader = dataset_loader.loader_test
    # seq_num = len(test_data_loader)
    # Batch average meterics
    # batch_time = utils.network_utils.AverageMeter()
    test_time = utils.network_utils.AverageMeter()
    data_time = utils.network_utils.AverageMeter()
    img_PSNRs_iter2 = utils.network_utils.AverageMeter()
    img_ssims_iter1 = utils.network_utils.AverageMeter()
    img_ssims_iter2 = utils.network_utils.AverageMeter()
    warp_mse_losses_iter1 = utils.network_utils.AverageMeter()
    warp_mse_losses_iter2 = utils.network_utils.AverageMeter()
    # img_PSNRs_mid = utils.network_utils.AverageMeter()
    img_PSNRs_iter1 = utils.network_utils.AverageMeter()
    batch_end_time = time()
    # test_psnr = dict()
    # g_names= 'init'
    deblurnet.eval()
    tqdm_test = tqdm(test_data_loader)
    tqdm_test.set_description('[TEST] [Epoch {0}/{1}]'.format(epoch_idx,cfg.TRAIN.NUM_EPOCHES))
    
    for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_test):
        data_time.update(time() - batch_end_time)

        seq_blur = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_blur]
        seq_clear = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_clear]
        # seq_len = len(seq_blur)
        # Switch models to training mode
        

        
            
        # if name[0] == "IMG_0055.00077":
        
        with torch.no_grad():
            input_seq = []
            gt_seq = []
            # input_seq = [seq_blur[0] for i in range((cfg.DATA.FRAME_LENGTH-1)//2)]
            input_seq += seq_blur
            # input_seq += [seq_blur[-1] for i in range((cfg.DATA.FRAME_LENGTH-1)//2)]
            input_seq = torch.cat(input_seq,1)
            gt_seq = torch.cat(seq_clear,1)
            b,t,c,h,w = gt_seq.shape
            # np.save("sharp.npy",gt_seq.data.cpu().numpy())
            torch.cuda.synchronize()
            test_time_start = time()
            recons_1, recons_2, recons_3, out,flow_forwards,flow_backwards = deblurnet(input_seq)
            # output_img = torch.cat([recons_1, recons_2, recons_3, out],dim=1)
            # output_img_one,output_img = deblurnet(input_seq)
            torch.cuda.synchronize()
            test_time.update((time() - test_time_start)/t)

            """ down_simple_gt = F.interpolate(gt_seq.reshape(-1,c,h,w), size=(h//4, w//4),mode='bilinear', align_corners=True).reshape(b,t,c,h//4,w//4)
            warploss = warp_loss(down_simple_gt, flow_forwards[1], flow_backwards[1])
            warp_mse_losses_iter1.update(warploss.item(), cfg.CONST.TEST_BATCH_SIZE)
            warploss = warp_loss(down_simple_gt, flow_forwards[-1], flow_backwards[-1])
            warp_mse_losses_iter2.update(warploss.item(), cfg.CONST.TEST_BATCH_SIZE) """
            # t_gt_seq = torch.cat([gt_seq[:,1,:,:,:],gt_seq[:,2,:,:,:],gt_seq[:,3,:,:,:],gt_seq[:,2,:,:,:]],dim=1)
            # img_PSNR =  PSNR(output_img[:,-1,:,:,:], t_gt_seq[:,-1,:,:,:])
            # img_PSNR =  util.calc_psnr(output_img[:,-1,:,:,:], t_gt_seq[:,-1,:,:,:])
            # img_PSNR_tt =  PSNR(output_img, t_gt_seq)
            img_PSNR2 = util.calc_psnr(out.detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_iter2.update(img_PSNR2, cfg.CONST.TEST_BATCH_SIZE)
            # img_PSNR = PSNR(output_img[:,:-1,:,:,:].contiguous().view(b*3,c,h,w), t_gt_seq[:,:-1,:,:,:].contiguous().view(b*3,c,h,w))
            # img_PSNR = util.calc_psnr()
            img_PSNR = util.calc_psnr(recons_2.detach(),gt_seq[:,2,:,:,:].detach())
            # img_PSNR_tt2 = PSNR(output_img[:,:-1,:,:,:].contiguous().view(b,3*c,h,w), t_gt_seq[:,:-1,:,:,:].contiguous().view(b,3*c,h,w))
            img_PSNRs_iter1.update(img_PSNR, cfg.CONST.TEST_BATCH_SIZE)
            batch_end_time = time()
            
            # log.info('[TEST] [Ech {0}/{1}][Seq {2} {3}/{4}] RT {5} DT {6}\t imgPSNR_iter1 {7} imgPSNR {8}'
                        # .format(epoch_idx, cfg.TRAIN.NUM_EPOCHES, name, seq_idx+1, seq_num, test_time ,data_time, img_PSNRs_iter1,img_PSNRs_iter2))
            # log.info("[TEST] [{0} {1}]".format(img_PSNRs_iter1,img_PSNRs_iter2))
            
            # cfg.NETWORK.PHASE == 'test':
            if cfg.NETWORK.PHASE == 'test':
                output_image = out.cpu().detach()*255
                gt_image = gt_seq[:,2,:,:,:].cpu().detach()*255
                output_image = output_image[0].permute(1,2,0)
                gt_image = gt_image[0].permute(1,2,0)
                output_image_it1 = recons_2.cpu().detach()*255
                output_image_it1 = output_image_it1[0].permute(1,2,0)
                img_ssims_iter1.update(ssim_calculate(output_image_it1.numpy(),gt_image.numpy()),cfg.CONST.TEST_BATCH_SIZE)
                img_ssims_iter2.update(ssim_calculate(output_image.numpy(),gt_image.numpy()),cfg.CONST.TEST_BATCH_SIZE)
                tqdm_test.set_postfix_str('RT {0} DT {1} imgPSNR_iter1 {2} imgPSNR_iter2 {3} ssim_it1 {4} ssim_it2 {5}'
                        .format(test_time ,data_time, img_PSNRs_iter1,img_PSNRs_iter2,img_ssims_iter1,img_ssims_iter2))
            else:
                tqdm_test.set_postfix_str('RT {0} DT {1} imgPSNR_iter1 {2} imgPSNR_iter2 {3}'
                        .format(test_time ,data_time, img_PSNRs_iter1,img_PSNRs_iter2))     
            
            
    # Output testing results
    if cfg.NETWORK.PHASE == 'test':

        log.info('============================ TEST RESULTS ============================')
        log.info('[TEST] Total_Mean_PSNR:itr1:{0},itr2:{1},best:{2},ssim_it1 {3},ssim_it2 {4}'.format(img_PSNRs_iter1.avg,img_PSNRs_iter2.avg,Best_Img_PSNR,img_ssims_iter1.avg,img_ssims_iter2.avg))
        
       
    else:
        # Output val results
        log.info('============================ TEST RESULTS ============================')
        
        

        # Add testing results to TensorBoard
        test_writer.add_scalar('EpochPSNR_TEST', img_PSNRs_iter2.avg, epoch_idx)
        if img_PSNRs_iter2.avg  >= Best_Img_PSNR:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            Best_Img_PSNR = img_PSNRs_iter2.avg
            Best_Epoch = epoch_idx
            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                                                      epoch_idx, deblurnet,deblurnet_solver, \
                                                      Best_Img_PSNR, Best_Epoch)
        log.info('[TEST] Total_Mean_PSNR:itr1:{0},itr2:{1},best:{2}'.format(img_PSNRs_iter1.avg,img_PSNRs_iter2.avg,Best_Img_PSNR))
        
        # test_writer.add_scalar(cfg.NETWORK.DEBLURNETARCH + '/EpochPSNR_TEST', img_PSNRs_mid.avg, epoch_idx + 1)
        return img_PSNRs_iter2.avg,Best_Img_PSNR