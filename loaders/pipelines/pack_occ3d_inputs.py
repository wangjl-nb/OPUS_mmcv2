from typing import Sequence

import numpy as np
import torch

from mmengine.structures import BaseDataElement
from mmdet3d.registry import TRANSFORMS

try:
    from mmdet3d.structures import Det3DDataSample
except Exception:  # pragma: no cover
    Det3DDataSample = BaseDataElement


@TRANSFORMS.register_module()
class PackOcc3DInputs:
    def __init__(self, meta_keys: Sequence[str] = ()):
        self.meta_keys = tuple(meta_keys)

    def _to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return data

    def _pack_imgs(self, imgs):
        if isinstance(imgs, list):
            img_tensors = []
            for img in imgs:
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                if img.ndim == 3:
                    img = img.permute(2, 0, 1)
                img_tensors.append(img)
            return torch.stack(img_tensors, dim=0)
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        if imgs.ndim == 3:
            imgs = imgs.permute(2, 0, 1)
        return imgs

    def __call__(self, results):
        inputs = {}
        if 'img' in results:
            inputs['img'] = self._pack_imgs(results['img'])
        if 'points' in results:
            points = results['points']
            if hasattr(points, 'tensor'):
                points = points.tensor
            inputs['points'] = self._to_tensor(points)

        data_sample = Det3DDataSample()
        if 'voxel_semantics' in results:
            voxel_semantics = self._to_tensor(results['voxel_semantics'])
            data_sample.voxel_semantics = voxel_semantics.long()
        if 'mask_camera' in results:
            mask_camera = self._to_tensor(results['mask_camera'])
            data_sample.mask_camera = mask_camera.bool()
        if 'mask_lidar' in results:
            mask_lidar = self._to_tensor(results['mask_lidar'])
            data_sample.mask_lidar = mask_lidar.bool()

        meta = {}
        for key in self.meta_keys:
            if key in results:
                meta[key] = results[key]
        if meta:
            data_sample.set_metainfo(meta)

        return dict(inputs=inputs, data_samples=data_sample)
