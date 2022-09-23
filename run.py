import torch
from lib.config import cfg, args
import numpy as np
import os






def run_evaluate():
    import time
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network,load_model

    from lib.train import optimizer
    from lib.train import recorder
    from lib.train import scheduler
    recordtrain=recorder.Recorder(cfg)
    network = make_network(cfg).cuda()
    optimizertrain=optimizer.make_optimizer(cfg, network)
    schedulertrain=scheduler.make_lr_scheduler(cfg,optimizertrain)
    scheduler.set_lr_scheduler(cfg,schedulertrain)



    network = make_network(cfg).cuda()

    pretrained_model = torch.load('./bestresult/'+cfg.cls_type+'/'+cfg.cls_type+'.pth')
    network.load_state_dict(pretrained_model['net'], strict=False)



    network.eval()
    data_loader= make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)






    outputs = []
    tot_elapsed_time = 0.0
    tot_valid_cnt = 0

    print('Start inference...')
    with torch.inference_mode():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):

                inp = batch['inp'].cuda()
                K = batch['K'].cuda()
                x_ini = batch['x_ini'].cuda()
                bbox = batch['bbox'].cuda()


                if    int(batch['img_id'][0])  :

                    _,output ,elapsed_time= network(inp, K, x_ini, bbox  )

                    tot_elapsed_time += elapsed_time
                    tot_valid_cnt += 1
                    outputs.append(output)
                else:
                    outputs.append([])

    print('Start computing ADD(-S) metrics...')
    for i, batch in enumerate(tqdm.tqdm(data_loader)):

            output = outputs[i]

            evaluator.evaluate(output, batch)

    print('Average FPS:', 1000 / (tot_elapsed_time / tot_valid_cnt))
    evaluator.summarize()






def run_linemod():
    from lib.datasets.linemod import linemod_to_coco
    linemod_to_coco.linemod_to_coco(cfg)


if __name__ == '__main__':

    globals()['run_' + args.type]()
