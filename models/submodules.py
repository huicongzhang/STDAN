#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>


from os import terminal_size
from time import thread_time_ns
from utils import log
from warnings import simplefilter
import torch.nn as nn
import torch
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import numpy as np
from config import cfg
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, rand
from models.ops.modules.ms_deform_attn import MSDeformAttn, MSDeformAttn_Fusion
# from models.position_encoding import PositionEmbeddingSine
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
# from utils.network_utils import warp
from torch.autograd import Variable
def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True,act=nn.LeakyReLU(0.1,inplace=True)):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        act
    )

def upconv(in_channels, out_channels,act=nn.LeakyReLU(0.1,inplace=True)):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        act
    )

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias,act=nn.LeakyReLU(0.1,inplace=True)):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            # nn.LeakyReLU(0.1,inplace=True),
            act,
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out

class DeformableAttnBlock(nn.Module):
    def __init__(self,n_heads=4,n_levels=3,n_points=4,d_model=32):
        super().__init__()
        self.n_levels = n_levels
        
        self.defor_attn = MSDeformAttn(d_model=d_model,n_levels=3,n_heads=n_heads,n_points=n_points)
        self.feed_forward = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.emb_qk = nn.Conv2d(3*d_model+8, 3*d_model, kernel_size=3, padding=1)
        self.emb_v = nn.Conv2d(3*d_model, 3*d_model, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.1,inplace=True)

        
        self.feedforward = nn.Sequential(
            nn.Conv2d(2*d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            )
        self.act = nn.LeakyReLU(0.1,inplace=True)
        
        
    def get_reference_points(self,spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    def get_valid_ratio(self,mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    def preprocess(self,srcs):
        bs,t,c,h,w = srcs.shape
        masks = [torch.zeros((bs,h,w)).bool().to(srcs.device) for _ in range(t)]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lv1 in range(t):
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        return spatial_shapes,valid_ratios
    def forward(self,frame,srcframe,flow_forward,flow_backward):
        b,t,c,h,w = frame.shape
        # bs,t,c,h,w = frame.shape
        warp_fea01 = warp(frame[:,0],flow_backward[:,0])
        warp_fea21 = warp(frame[:,2],flow_forward[:,1])


        qureys = self.act(self.emb_qk(torch.cat([warp_fea01,frame[:,1],warp_fea21,flow_forward.reshape(b,-1,h,w),flow_backward.reshape(b,-1,h,w)],1))).reshape(b,t,c,h,w)
        
        value = self.act(self.emb_v(frame.reshape(b,t*c,h,w)).reshape(b,t,c,h,w))
        
        
        spatial_shapes,valid_ratios = self.preprocess(value)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes,valid_ratios,device=value.device)
        
        
        
        output = self.defor_attn(qureys,reference_points,value,spatial_shapes,level_start_index,None,flow_forward,flow_backward)
        
        output = self.feed_forward(output)
        output = output.reshape(b,t,c,h,w) + frame



        
        tseq_encoder_0 = torch.cat([output.reshape(b*t,c,h,w),srcframe.reshape(b*t,c,h,w)],1)
        output = output.reshape(b*t,c,h,w) + self.feedforward(tseq_encoder_0)
        return output.reshape(b,t,c,h,w),srcframe
class DeformableAttnBlock_FUSION(nn.Module):
    def __init__(self,n_heads=4,n_levels=3,n_points=4,d_model=32):
        super().__init__()
        self.n_levels = n_levels
        
        self.defor_attn = MSDeformAttn_Fusion(d_model=d_model,n_levels=3,n_heads=n_heads,n_points=n_points)
        self.feed_forward = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.emb_qk = nn.Conv2d(3*d_model+4, 3*d_model, kernel_size=3, padding=1)
        self.emb_v = nn.Conv2d(3*d_model, 3*d_model, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.1,inplace=True)

        
        self.feedforward = nn.Sequential(
            nn.Conv2d(2*d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            )
        self.act = nn.LeakyReLU(0.1,inplace=True)
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1,inplace=True)
            )
        
    def get_reference_points(self,spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    def get_valid_ratio(self,mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    def preprocess(self,srcs):
        bs,t,c,h,w = srcs.shape
        masks = [torch.zeros((bs,h,w)).bool().to(srcs.device) for _ in range(t)]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lv1 in range(t):
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        return spatial_shapes,valid_ratios
    def forward(self,frame,srcframe,flow_forward,flow_backward):
        b,t,c,h,w = frame.shape
        # bs,t,c,h,w = frame.shape
        warp_fea01 = warp(frame[:,0],flow_backward[:,0])
        warp_fea21 = warp(frame[:,2],flow_forward[:,1])


        qureys = self.act(self.emb_qk(torch.cat([warp_fea01,frame[:,1],warp_fea21,flow_forward[:,1],flow_backward[:,0]],1))).reshape(b,t,c,h,w)
        
        value = self.act(self.emb_v(frame.reshape(b,t*c,h,w)).reshape(b,t,c,h,w))
        
        
        spatial_shapes,valid_ratios = self.preprocess(value)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes[0].reshape(1,2),valid_ratios,device=value.device)
        
        
        
        output = self.defor_attn(qureys,reference_points,value,spatial_shapes,level_start_index,None,flow_forward,flow_backward)
        
        output = self.feed_forward(output)
        output = output.reshape(b,c,h,w) + frame[:,1]



        
        tseq_encoder_0 = torch.cat([output,srcframe[:,1]],1)
        output = output.reshape(b,c,h,w) + self.feedforward(tseq_encoder_0)
        output = self.fusion(output)
        return output
def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).reshape(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).reshape(-1, 1).repeat(1, W)
        xx = xx.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(x.device)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border',align_corners=True)
        # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        # mask = nn.functional.grid_sample(mask, vgrid,align_corners=True )

        # mask[mask < 0.999] = 0
        # mask[mask > 0] = 1

        # output = output * mask

        return output

