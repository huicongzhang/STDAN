#!/usr/bin/python


import os
import sys
import torch
import numpy as np
from datetime import datetime as dt
import math
from torch import chunk, nn,einsum, select
from config import cfg
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import cv2
from torch.autograd import Variable
def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def init_weights_xavier(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)
def default_init_weights(module, scale=0.1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale*0.1
def init_weights_kaiming(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(file_path, epoch_idx, deblurnet, deblurnet_solver,Best_Img_PSNR, Best_Epoch):
    # print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'Best_Img_PSNR': Best_Img_PSNR,
        'Best_Epoch': Best_Epoch,
        'deblurnet_state_dict': deblurnet.state_dict(),
        'deblurnet_solver_state_dict': deblurnet_solver.state_dict(),
    }
    torch.save(checkpoint, file_path)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_weight_parameters(model):
    return [param for name, param in model.named_parameters() if ('weight' in name)]

def get_bias_parameters(model):
    return [param for name, param in model.named_parameters() if ('bias' in name)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.5f} ({:.5f})'.format(self.val, self.avg)

'''input Tensor: 2 H W'''
def flow2rgb(flowmap):
    assert(isinstance(flowmap, torch.Tensor))
    global args
    _, H, W = flowmap.shape
    rgb = torch.ones((3,H,W))
    normalized_flow_map = flowmap / (flowmap.max())
    rgb[0] += normalized_flow_map[0]
    rgb[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb[2] += normalized_flow_map[1]

    return rgb.clamp(0,1)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = grid.to(x.device)
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, padding_mode='border',align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    # mask = torch.ones(x.size(), device=x.device, requires_grad=False)
    mask = nn.functional.grid_sample(mask, vgrid,align_corners=True )

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    output = output * mask

    return output



