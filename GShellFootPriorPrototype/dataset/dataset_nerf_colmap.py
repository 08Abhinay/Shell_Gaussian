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
try:
    import cv2
except ImportError:
    cv2 = None

from render import util

from .dataset import Dataset

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

_FAST_LDR_EXTENSIONS = {'.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.tif', '.tiff', '.webp'}


def _load_img_raw(path):
    ext = Path(path).suffix.lower()
    if cv2 is not None and ext in _FAST_LDR_EXTENSIONS:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.ndim == 3 and img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            return img
    return util.load_image_raw(path)


def _load_img(path):
    img = _load_img_raw(path)
    if img.ndim == 2:
        img = img[..., np.newaxis]  # (H, W) -> (H, W, 1)
    if img.dtype != np.float32: # LDR image
        img = np.ascontiguousarray(img.astype(np.float32, copy=False) / 255.0)
        img = torch.from_numpy(img)
        if img.shape[-1] >= 3:
            img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.from_numpy(np.ascontiguousarray(img))
    return img

class DatasetNERF(Dataset):
    _preload_cache = {}

    @classmethod
    def _preload_cache_key(cls, cfg_path, FLAGS):
        return (
            os.path.realpath(cfg_path),
            tuple(float(v) for v in FLAGS.cam_near_far),
        )

    def __init__(self, cfg_path, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)

        # Load config / transforms
        with open(cfg_path, 'r') as fh:
            self.cfg = json.load(fh)
        self.n_images = len(self.cfg['frames'])
        self.preloaded_data = None

        cache_key = None
        if self.FLAGS.pre_load:
            cache_key = self._preload_cache_key(cfg_path, self.FLAGS)
            self.preloaded_data = self._preload_cache.get(cache_key)

        # Determine resolution & aspect ratio before preloading so projection math is ready.
        if self.preloaded_data is not None:
            self.resolution = self.preloaded_data[0][0].shape[1:3]
        else:
            first_img = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path']))
            self.resolution = first_img.shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        # Pre-load from disc to avoid slow image parsing.
        if self.FLAGS.pre_load and self.preloaded_data is None:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data.append(self._parse_frame(self.cfg, i))
            self._preload_cache[cache_key] = self.preloaded_data

        if self.FLAGS.local_rank == 0:
            print("DatasetNERF: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy   = util.fovx_to_fovy(cfg['frames'][idx]['camera_angle_x'], self.aspect)
        proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image data and modelview matrix
        img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']))
        mask   = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']).replace('/image/', '/mask/').replace('.jpg', '.png'))
        img    = torch.cat([img, mask[:,:,:1]], dim=-1)
        mv     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        mv     = mv @ util.rotate_x(-np.pi / 2)
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        iter_res = self.FLAGS.train_res
        
        img      = []
        fovy     = util.fovx_to_fovy(self.cfg['frames'][itr % self.n_images]['camera_angle_x'], self.aspect)

        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            img, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)

        # Resize image to training resolution if needed
        if img.shape[1] != iter_res[0] or img.shape[2] != iter_res[1]:
            img = torch.nn.functional.interpolate(
                img.permute(0, 3, 1, 2),  # NHWC -> NCHW
                size=(iter_res[0], iter_res[1]),
                mode='bilinear',
                align_corners=False,
                antialias=True,
            ).permute(0, 2, 3, 1)  # NCHW -> NHWC

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
