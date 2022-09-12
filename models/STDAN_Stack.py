import torch
import torch.nn as nn
from models.model import STDAN
# from models.model import flow_pwc
from utils import util
import torch.nn.functional as F
from mmflow.ops import Warp



class STDAN_Stack(nn.Module):

    def __init__(self, in_channels=3, n_sequence=5, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=True, load_recons_net=False, flow_pretrain_fn='/home/hczhang/CODE/CDVD-TSP/pretrain_models/network-default.pytorch', recons_pretrain_fn='',
                 is_mask_filter=True, device='cuda'):
        super(STDAN_Stack, self).__init__()
        # print("Creating CDVD-TSP Net")

        self.n_sequence = n_sequence
        self.device = device

        assert n_sequence == 5, "Only support args.n_sequence=5; but get args.n_sequence={}".format(n_sequence)

        self.is_mask_filter = is_mask_filter
        print('Is meanfilter image when process mask:', 'True' if is_mask_filter else 'False')
        extra_channels = 1
        print('Select mask mode: concat, num_mask={}'.format(extra_channels))

        self.recons_net = STDAN.STDAN(in_channels=in_channels, n_sequence=3, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat,
                                                    extra_channels=extra_channels)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))
    def down_size(self,frame):
        _,_,h,w = frame.shape
        frame = F.interpolate(frame, size=(h//4, w//4),mode='bilinear', align_corners=True)
        return frame
    def forward(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        concated = torch.stack([frame_list[0], frame_list[1], frame_list[2]], dim=1)
        recons_1,flow_forward_recons_1,flow_backward_recons_1 = self.recons_net(concated)

        
        concated = torch.stack([frame_list[1], frame_list[2], frame_list[3]], dim=1)
        recons_2, flow_forward_recons_2,flow_backward_recons_2 = self.recons_net(concated)

        
        concated = torch.stack([frame_list[2], frame_list[3], frame_list[4]], dim=1)
        recons_3, flow_forward_recons_3,flow_backward_recons_3 = self.recons_net(concated)

        
        concated = torch.stack([recons_1, recons_2, recons_3], dim=1) 

        out, flow_forward_out,flow_backward_out = self.recons_net(concated)
        
        flow_forwards = [flow_forward_recons_1,flow_forward_recons_2,flow_forward_recons_3,flow_forward_out]
        flow_backwards = [flow_backward_recons_1,flow_backward_recons_2,flow_backward_recons_3,flow_backward_out]
        return recons_1, recons_2, recons_3, out,flow_forwards,flow_backwards
