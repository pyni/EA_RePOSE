from scipy import spatial
import torch.nn as nn
import torch
from lib.config import cfg
from lib.networks.rdopt.util import rot_vec_to_mat


def crop_input(inp, bbox):
    bs = inp.shape[0]
    inp_r = []
    for i in range(bs):
        inp_r_ = inp[i, :, bbox[i, 0]:bbox[i, 2], bbox[i, 1]:bbox[i, 3]]
        inp_r.append(inp_r_)
    inp_r = torch.stack(inp_r, dim=0)
    return inp_r



class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

    def forward(self, batch):


        loss,_,_ =   self.net(batch['inp'].cuda(), batch['K'].cuda(),
                          batch['x_ini'].cuda(), batch['bbox'].cuda(),
                          batch['x2s'].cuda(), batch['x4s'].cuda(), batch['x8s'].cuda(), batch['xfc'].cuda() ,batch['R'].cuda(),batch['t'].cuda(),None,batch['mask']) # inp, K, x_ini, bbox, x2s, x4s, x8s, xfc




        scalar_stats = {}

        scalar_stats.update({'loss': loss.mean()})
        image_stats = {}

        return loss, scalar_stats, image_stats
