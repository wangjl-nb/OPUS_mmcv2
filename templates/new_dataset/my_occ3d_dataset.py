"""Template dataset for Occ3D-style training/evaluation.

Copy this file to loaders/my_occ3d_dataset.py, then:
1) rename class name,
2) update @DATASETS.register_module name usage in config,
3) adapt get_data_info() if your raw meta schema differs.
"""

import os.path as osp

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmdet3d.registry import DATASETS

from loaders.geometry import quaternion_to_matrix, transform_matrix
from loaders.utils import compose_ego2img


@DATASETS.register_module()
class MyOcc3DDataset(BaseDataset):
    def __init__(self,
                 ann_file,
                 data_root,
                 pipeline,
                 modality,
                 classes=None,
                 occ_root=None,
                 dataset_cfg=None,
                 test_mode=False,
                 **kwargs):
        self.modality = modality
        self.occ_root = occ_root
        self.dataset_cfg = dataset_cfg or {}
        kwargs.setdefault('serialize_data', False)
        metainfo = dict(classes=classes) if classes is not None else None
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            metainfo=metainfo,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        data = load(self.ann_file)
        if isinstance(data, dict) and 'infos' in data:
            return data['infos']
        if isinstance(data, dict) and 'data_list' in data:
            return data['data_list']
        if isinstance(data, list):
            return data
        raise TypeError(f'Unsupported annotation format: {type(data)}')

    def get_data_info(self, index):
        info = self.data_list[index]

        ego2global_translation = np.asarray(info['ego2global_translation'])
        ego2global_rotation = quaternion_to_matrix(info['ego2global_rotation'])
        lidar2ego_translation = np.asarray(info['lidar2ego_translation'])
        lidar2ego_rotation = quaternion_to_matrix(info['lidar2ego_rotation'])

        ego2lidar = transform_matrix(
            lidar2ego_translation,
            info['lidar2ego_rotation'],
            inverse=True)

        input_dict = dict(
            sample_idx=index,
            sample_token=info['token'],
            scene_name=info['scene_name'],
            timestamp=info['timestamp'] / 1e6,
            ego2lidar=ego2lidar,
            ego2obj=ego2lidar,
            ego2occ=np.eye(4, dtype=np.float32),
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
        )

        if self.modality.get('use_lidar', False):
            lidar_path = info['lidar_path']
            input_dict.update(dict(
                pts_filename=lidar_path,
                lidar_points=info.get('lidar_points', {'lidar_path': lidar_path}),
                lidar_sweeps=dict(
                    prev=info.get('lidar_sweeps', []),
                    next=[]),
            ))

        if self.modality.get('use_camera', False):
            img_paths = []
            img_timestamps = []
            ego2img = []

            cam_types = list(info['cams'].keys())
            for cam_name in cam_types:
                cam_info = info['cams'][cam_name]
                img_paths.append(osp.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)
                ego2img.append(compose_ego2img(
                    ego2global_translation,
                    ego2global_rotation,
                    cam_info['sensor2global_translation'],
                    cam_info['sensor2global_rotation'].T,
                    cam_info['cam_intrinsic'],
                ))

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                ego2img=ego2img,
                cam_sweeps=dict(
                    prev=info.get('cam_sweeps', []),
                    next=[]),
                cam_types=cam_types,
                num_views=len(cam_types),
            ))

        if not self.test_mode:
            # For occupancy-only training, placeholder ann_info is acceptable.
            input_dict['ann_info'] = dict(
                gt_bboxes_3d=np.array([[[]]]),
                gt_labels_3d=np.array([[[]]]),
                gt_names=np.array([[[]]]),
            )

        return input_dict
