import torch
from lib.config import cfg, args
import numpy as np
import os





def run_reinforcetraining():
    import time
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network, load_model

    from lib.train import optimizer
    from lib.train import recorder
    from lib.train import scheduler

    #############################################
    from pathlib import Path
    from copy import deepcopy
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from torch import nn
    from tqdm.auto import  tqdm  as  tqdmauto
 

    episodes = 20000
 
 
    recordtrain = recorder.Recorder(cfg)
    network = make_network(cfg).cuda()
    optimizertrain = optimizer.make_optimizer(cfg, network)
    schedulertrain = scheduler.make_lr_scheduler(cfg, optimizertrain)
    scheduler.set_lr_scheduler(cfg, schedulertrain)

    network = make_network(cfg).cuda()
    pretrained_model = torch.load('./bestresult/'+cfg.cls_type+'.pth')
    network.load_state_dict(pretrained_model, strict=False)

    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    outputs = []

    tot_elapsed_time = 0.0
    tot_valid_cnt = 0
    forbiden = []  
 

    class DeepQNetwork(nn.Module):
        def __init__(self):
            super(DeepQNetwork, self).__init__()
            #   Create layers
            self.fc1 = nn.Linear(in_features=49, out_features=128)
            self.fc2 = nn.Linear(in_features=128, out_features=64)
            self.out = nn.Linear(in_features=64, out_features=3)

        def forward(self, state):
            #   Implement forward pass
            state = torch.as_tensor(state, dtype=torch.float32)
            state = torch.nn.functional.relu(self.fc1(state))
            state = torch.nn.functional.relu(self.fc2(state))
            Q = self.out(state)

            return Q
 
 

 
    buffer = ReplayBuffer(mem_size=10000, state_shape=(49,))
 
    batch_size = 64
    NETWORK_UPDATE_FREQUENCY = 1
    NETWORK_SYNC_FREQUENCY = 2000/4
    gamma = 0.99
    replay_buffer_size = 0  
    if 1:
        q_network = DeepQNetwork()
        checkpoint = torch.load('./bestresult_dqn/'+cfg.cls_type+'.pth')
        q_network.load_state_dict(checkpoint)
 
        epsilon = 0.08
        step_count = 0
        total_rewards = []

        ff= open('/hpctmp/pyni/weighttraining/action_logfortest.txt', 'w')

        step_count = 0
       # with tqdmauto(range(episodes)) as pbar:
        for epi in range(episodes):

            outputs = []
            for batchi, batch in enumerate(tqdm.tqdm(data_loader)):
                inp = batch['inp'].cuda()
                K = batch['K'].cuda()
                x_ini = batch['x_ini'].cuda()
                bbox = batch['bbox'].cuda()
 

                _,output ,elapsed_time,epsilon,q_network, buffer,rewards,actions,successornot=network(inp, K, x_ini, bbox, batch['R'].cuda(), batch['t'].cuda(),
                                                  int(batch['img_id'][0]), batch['mask'], buffer, evaluator, batch,
                                                  False,q_network, epi,epsilon)
                ff.write(str(actions)+'\n')
                tot_elapsed_time += elapsed_time
                tot_valid_cnt += 1
                outputs.append(output)
 
            for i, batch in enumerate(tqdm.tqdm(data_loader)):
  
                     output = outputs[i]
                     evaluator.evaluate(output, batch)

            print('Average FPS:', 1000 / (tot_elapsed_time / tot_valid_cnt))
            evaluator.summarize()
            ff.close()




def run_linemod():
    from lib.datasets.linemod import linemod_to_coco
    linemod_to_coco.linemod_to_coco(cfg)


if __name__ == '__main__':

    globals()['run_' + args.type]()
