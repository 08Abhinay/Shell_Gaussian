# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import json
from pathlib import Path

import torch
import numpy as np

from render import util

from .dataset import Dataset

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

def _load_img(path):
    img = util.load_image_raw(path)
    if img.ndim == 2:
        img = img[..., np.newaxis]  # (H, W) -> (H, W, 1)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        if img.shape[-1] >= 3:
            img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

def _load_invdepth(path):
    invdepth = np.load(path).astype(np.float32)
    if invdepth.ndim == 2:
        invdepth = invdepth[..., np.newaxis]
    elif invdepth.shape[-1] > 1:
        invdepth = invdepth[..., :1]
    return torch.tensor(invdepth, dtype=torch.float32)

def _resolve_invdepth_path(base_dir, frame, key='invdepth_path', folder='invdepth'):
    if key in frame:
        return os.path.join(base_dir, frame[key])

    rel = Path(frame['file_path'])
    parts = list(rel.parts)
    if 'image' in parts:
        parts[parts.index('image')] = folder
    else:
        parts.insert(0, folder)
    parts[-1] = Path(parts[-1]).with_suffix('.npy').name
    return os.path.join(base_dir, *parts)

class DatasetNERF(Dataset):
    def __init__(self, cfg_path, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)

        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_images = len(self.cfg['frames'])

        # Determine resolution & aspect ratio
        self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        if self.FLAGS.local_rank == 0:
            print("DatasetNERF: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy   = util.fovx_to_fovy(cfg['frames'][idx]['camera_angle_x'], self.aspect)
        proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image data and modelview matrix
        frame  = cfg['frames'][idx]
        img    = _load_img(os.path.join(self.base_dir, frame['file_path']))
        mask   = _load_img(os.path.join(self.base_dir, frame['file_path']).replace('/image/', '/mask/').replace('.jpg', '.png'))
        img    = torch.cat([img, mask[:,:,:1]], dim=-1)
        mv     = torch.linalg.inv(torch.tensor(frame['transform_matrix'], dtype=torch.float32))
        mv     = mv @ util.rotate_x(-np.pi / 2)
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv
        invdepth = None
        invdepth_second = None
        if self.FLAGS.use_depth:
            invdepth_path = _resolve_invdepth_path(self.base_dir, frame)
            if not os.path.exists(invdepth_path):
                raise FileNotFoundError(f"Depth is enabled but inverse-depth target is missing: {invdepth_path}")
            invdepth = _load_invdepth(invdepth_path)
        if self.FLAGS.use_depth_2nd_layer:
            invdepth_second_path = _resolve_invdepth_path(
                self.base_dir,
                frame,
                key='invdepth_second_path',
                folder='invdepth_second',
            )
            if not os.path.exists(invdepth_second_path):
                raise FileNotFoundError(f"Second-layer depth is enabled but target is missing: {invdepth_second_path}")
            invdepth_second = _load_invdepth(invdepth_second_path)

        return (
            img[None, ...],
            mv[None, ...],
            mvp[None, ...],
            campos[None, ...],
            None if invdepth is None else invdepth[None, ...],
            None if invdepth_second is None else invdepth_second[None, ...],
        ) # Add batch dimension

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        iter_res = self.FLAGS.train_res
        
        img      = []
        fovy     = util.fovx_to_fovy(self.cfg['frames'][itr % self.n_images]['camera_angle_x'], self.aspect)

        if self.FLAGS.pre_load:
            img, mv, mvp, campos, invdepth, invdepth_second = self.preloaded_data[itr % self.n_images]
        else:
            img, mv, mvp, campos, invdepth, invdepth_second = self._parse_frame(self.cfg, itr % self.n_images)

        # Resize image to training resolution if needed
        if img.shape[1] != iter_res[0] or img.shape[2] != iter_res[1]:
            img = torch.nn.functional.interpolate(
                img.permute(0, 3, 1, 2),  # NHWC -> NCHW
                size=(iter_res[0], iter_res[1]),
                mode='bilinear',
                align_corners=False,
                antialias=True,
            ).permute(0, 2, 3, 1)  # NCHW -> NHWC

        if invdepth is not None and (invdepth.shape[1] != iter_res[0] or invdepth.shape[2] != iter_res[1]):
            invdepth = torch.nn.functional.interpolate(
                invdepth.permute(0, 3, 1, 2),
                size=(iter_res[0], iter_res[1]),
                mode='bilinear',
                align_corners=False,
            ).permute(0, 2, 3, 1)
        if invdepth_second is not None and (invdepth_second.shape[1] != iter_res[0] or invdepth_second.shape[2] != iter_res[1]):
            invdepth_second = torch.nn.functional.interpolate(
                invdepth_second.permute(0, 3, 1, 2),
                size=(iter_res[0], iter_res[1]),
                mode='bilinear',
                align_corners=False,
            ).permute(0, 2, 3, 1)

        sample = {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
        if invdepth is not None:
            sample['invdepth'] = invdepth
        if invdepth_second is not None:
            sample['invdepth_second'] = invdepth_second
        return sample
