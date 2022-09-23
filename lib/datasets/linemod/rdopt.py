import os
import random
import math

import cv2
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
from scipy import ndimage
from lib.utils.linemod import linemod_config
from lib.utils.pvnet import (pvnet_data_utils, pvnet_pose_utils,
                             visualize_utils)
from lib.config import cfg
import random as rd
from transforms3d.euler import mat2euler, euler2mat
from transforms3d.quaternions import mat2quat, quat2mat, qmult
class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split, transforms=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        cgmodel_dir = os.path.join(cfg['cgmodel_dir'], cfg.model,
                                   f'{cfg.model}.ply')
        self.model = pvnet_data_utils.get_ply_model(cgmodel_dir)

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfg

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]
        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        path = path.replace('benchwise', 'benchvise')
        mask_path = anno['mask_path'].replace('benchwise', 'benchvise')
        inp = Image.open(path)
        w, h = inp.size
        K = np.array(anno['K']).astype(np.float32)
        pose = np.array(anno['pose']).astype(np.float32)
        R = pose[:, :3]
        R = cv2.Rodrigues(R)[0].reshape(3)
        t = np.expand_dims(pose[:, 3], axis=1)
        cls = anno['cls']
        if cls == 'benchwise':
            cls = 'benchvise'
        cls_idx = linemod_config.linemod_cls_names.index(cls) + 1
        mask = pvnet_data_utils.read_linemod_mask(mask_path, anno['type'],
                                                  cls_idx)


        dataset = cfg.train.dataset if self.split == 'train' else cfg.test.dataset
        directory = f'cache/{dataset}/{cfg.model}'

        filename = f'{directory}/{img_id}.npy'
        result = np.load(filename, allow_pickle=True).item()
        bbox = result['bbox']
        x_ini = result['x_ini']

        filename = f'{directory}/{img_id}_features.npz'
        features = np.load(filename, allow_pickle=True)

        inp = np.asarray(inp).astype(np.uint8)


        return inp, K, R, t, x_ini, bbox, features

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, K, R, t, x_ini, bbox, features = self.read_data(img_id)

        if self._transforms is not None:
            img = self._transforms(img)

        ret = {
            'inp': img,
            'K': K,
            'x_ini': x_ini,
            'bbox': bbox,
            'R': R,
            't': t,
            'img_id': img_id,
            'x2s': features['x2s'][0],
            'x4s': features['x4s'][0],
            'x8s': features['x8s'][0],
            'xfc': features['xfc'][0],
        }

        return ret

    def __len__(self):
        return len(self.img_ids)


import pickle as pk




@torch.jit.script
def rot_vec_to_mat(vec):
    bs = vec.shape[0]

    theta = torch.norm(vec, dim=1)
    wx = vec[:, 0] / theta
    wy = vec[:, 1] / theta
    wz = vec[:, 2] / theta

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    R = torch.zeros((bs, 3, 3), device=vec.device)
    R[:, 0, 0] = costheta + wx * wx * (1 - costheta)
    R[:, 1, 0] = wz * sintheta + wx * wy * (1 - costheta)
    R[:, 2, 0] = -wy * sintheta + wx * wz * (1 - costheta)
    R[:, 0, 1] = wx * wy * (1 - costheta) - wz * sintheta
    R[:, 1, 1] = costheta + wy * wy * (1 - costheta)
    R[:, 2, 1] = wx * sintheta + wy * wz * (1 - costheta)
    R[:, 0, 2] = wy * sintheta + wx * wz * (1 - costheta)
    R[:, 1, 2] = -wx * sintheta + wy * wz * (1 - costheta)
    R[:, 2, 2] = costheta + wz * wz * (1 - costheta)

    return R




class Dataset_train(data.Dataset):
    def __init__(self, ann_file, data_root, split, transforms=None ):
        super(Dataset_train, self).__init__()


        self.data_root = data_root
        self.split = split

        cgmodel_dir = os.path.join(cfg['cgmodel_dir'], cfg.model,
                                   f'{cfg.model}.ply')
        self.model = pvnet_data_utils.get_ply_model(cgmodel_dir)
        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfg
        ff = open('./linemod_posecnn_results.pkl', 'rb')
        if cfg.model == 'cam':
            cfg.model = 'camera'
        self.posecnn = pk.load(ff)[cfg.model]
        self.mode =  cfg.mode
        self.b2b_RT=np.load('./cache/blender2bop_RT.npy',allow_pickle=True).tolist()


    def se3_q2m(self, se3_q):
        assert se3_q.size == 7
        se3_mx = np.zeros((3, 4))
        quat = se3_q[:4]
        R = quat2mat(quat)
        se3_mx[:, :3] = R
        se3_mx[:, 3] = se3_q[4:]
        return se3_mx

    def rot_mat_to_vec(self, matrix):
        bs = matrix.shape[0]

        # Axes.
        axis = torch.zeros((bs, 3))
        axis[:, 0] = matrix[:, 2, 1] - matrix[:, 1, 2]
        axis[:, 1] = matrix[:, 0, 2] - matrix[:, 2, 0]
        axis[:, 2] = matrix[:, 1, 0] - matrix[:, 0, 1]

        # Angle.
        norm = torch.norm(axis[:, 1:], dim=1).unsqueeze(1)
        r = torch.cat((axis[:, 0].unsqueeze(1), norm), dim=1)
        r = torch.norm(r, dim=1).unsqueeze(1)
        t = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
        t = t.unsqueeze(1)
        theta = torch.atan2(r, t - 1)

        # Normalise the axis.
        axis = axis / r

        # Return the data.
        return theta * axis

    def read_data(self, img_id):

        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        anno = self.coco.loadAnns(ann_ids)[0]
        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        path = path.replace('benchwise', 'benchvise')
 
        inp = Image.open(path)
        w, h = inp.size
        K = np.array(anno['K']).astype(np.float32)
        pose = np.array(anno['pose']).astype(np.float32)
        R = pose[:, :3]
        R = cv2.Rodrigues(R)[0].reshape(3)

        t = np.expand_dims(pose[:, 3], axis=1)
        cls = anno['cls']
        if cls == 'benchwise':
            cls = 'benchvise'


        dataset = cfg.train.dataset if self.split == 'train' else cfg.test.dataset


        directory = f'cache/{dataset}/{cfg.model}'

        filename = f'{directory}/{img_id}.npy'

        result = np.load(filename, allow_pickle=True).item()

        bbox = result['bbox']


        if self.mode=='PVNET':
              x_ini = result['x_ini']

        elif self.mode=='PoseCNN':
            x_ini = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
            x_ini_matrix =   self.se3_q2m(self.posecnn[img_id-1]['pose'])

            R_ = x_ini_matrix[:, :3]
            R_final=R_@np.linalg.inv(np.array(self.b2b_RT[cfg.model][:3,:3].T))
            t_final=x_ini_matrix[:3,3:]+R_@np.array(self.b2b_RT[cfg.model][:3,3:] )


            R_ = cv2.Rodrigues(R_final)[0].reshape(3)

            t_ =t_final

            x_ini[:, :3] = torch.tensor(R_)
            x_ini[:, 3:] =  torch.tensor(t_).view( 1, 3).unsqueeze(0)
            x_ini=x_ini.squeeze()



        inp = np.asarray(inp).astype(np.uint8)



        return inp, K, R, t, x_ini, bbox

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, K, R, t, x_ini, bbox  = self.read_data(img_id)

        if self._transforms is not None:
            img = self._transforms(img)


        ret = {
            'inp': img,
            'K': K,
            'x_ini': x_ini,
            'bbox': bbox,
            'R': R,
            't': t,
            'img_id': img_id,
        }


        return ret

    def __len__(self):
        return len(self.img_ids)



