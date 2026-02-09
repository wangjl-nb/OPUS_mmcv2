import os
import os.path as osp
import warnings

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import mkdir_or_exist
from mmdet3d.registry import DATASETS
from tqdm import tqdm

from .geometry import quaternion_to_matrix, transform_matrix
from .utils import compose_ego2img


@DATASETS.register_module()
class TartangroundOcc3DDataset(BaseDataset):
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
            self.data_infos = data['infos']
        elif isinstance(data, dict) and 'data_list' in data:
            self.data_infos = data['data_list']
        elif isinstance(data, list):
            self.data_infos = data
        else:
            raise TypeError(f'Unsupported annotation format: {type(data)}')
        return self.data_infos

    def collect_cam_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1

        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def collect_lidar_sweeps(self, index, into_past=20, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            curr_index = curr_index - 1

        all_sweeps_next = []
        curr_index = index + 1
        last_timestamp = self.data_infos[index]['timestamp']
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps'][::-1]
            if curr_sweeps and curr_sweeps[0]['timestamp'] == last_timestamp:
                curr_sweeps = curr_sweeps[1:]
            if not curr_sweeps:
                curr_index = curr_index + 1
                continue
            all_sweeps_next.extend(curr_sweeps)
            curr_index = curr_index + 1
            last_timestamp = all_sweeps_next[-1]['timestamp']

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']

        ego2global_rotation_mat = quaternion_to_matrix(ego2global_rotation)
        lidar2ego_rotation_mat = quaternion_to_matrix(lidar2ego_rotation)
        ego2lidar = transform_matrix(
            lidar2ego_translation, lidar2ego_rotation, inverse=True)

        input_dict = dict(
            sample_token=info['token'],
            scene_name=info['scene_name'],
            timestamp=info['timestamp'] / 1e6,
            ego2lidar=ego2lidar,
            ego2obj=ego2lidar,
            ego2occ=np.eye(4),
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        if self.modality['use_lidar']:
            lidar_sweeps_prev, lidar_sweeps_next = self.collect_lidar_sweeps(index)
            input_dict.update(dict(
                pts_filename=info['lidar_path'],
                lidar_points=info.get('lidar_points', {'lidar_path': info['lidar_path']}),
                lidar_sweeps={'prev': lidar_sweeps_prev, 'next': lidar_sweeps_next},
            ))

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            ego2img = []
            cam_types = list(info['cams'].keys())

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)
                ego2img.append(
                    compose_ego2img(
                        ego2global_translation,
                        ego2global_rotation_mat,
                        cam_info['sensor2global_translation'],
                        cam_info['sensor2global_rotation'].T,
                        cam_info['cam_intrinsic']
                    )
                )

            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                ego2img=ego2img,
                cam_sweeps={'prev': cam_sweeps_prev, 'next': cam_sweeps_next},
                cam_types=cam_types,
                num_views=len(cam_types),
            ))

        if not self.test_mode:
            annos = dict(
                gt_bboxes_3d=np.array([[[]]]),
                gt_labels_3d=np.array([[[]]]),
                gt_names=np.array([[[]]]))
            input_dict['ann_info'] = annos

        input_dict['sample_idx'] = index
        return input_dict

    def _build_metric(self, eval_kwargs):
        from .metrics.occ3d_metric import Occ3DMetric

        occ_io_cfg = self.dataset_cfg.get('occ_io', {})
        metric_cfg = self.dataset_cfg.get('metric', {})
        ray_cfg = self.dataset_cfg.get('ray', {})

        return Occ3DMetric(
            ann_file=self.ann_file,
            occ_root=eval_kwargs.get('occ_root', self.occ_root or osp.join(self.data_root, 'gts')),
            empty_label=eval_kwargs.get(
                'empty_label', self.dataset_cfg.get('empty_label', metric_cfg.get('empty_label', 79))),
            use_camera_mask=eval_kwargs.get(
                'use_camera_mask', metric_cfg.get('use_camera_mask', True)),
            compute_rayiou=eval_kwargs.get(
                'compute_rayiou', metric_cfg.get('compute_rayiou', True)),
            pc_range=eval_kwargs.get('pc_range', self.dataset_cfg.get('pc_range', None)),
            voxel_size=eval_kwargs.get('voxel_size', self.dataset_cfg.get('voxel_size', None)),
            class_names=eval_kwargs.get('class_names', self.dataset_cfg.get('class_names', None)),
            miou_num_workers=eval_kwargs.get(
                'miou_num_workers', metric_cfg.get('miou_num_workers', 0)),
            occ_path_template=eval_kwargs.get(
                'occ_path_template', occ_io_cfg.get('path_template', '{scene_name}/{token}/labels.npz')),
            semantics_key=eval_kwargs.get(
                'semantics_key', occ_io_cfg.get('semantics_key', 'semantics')),
            mask_camera_key=eval_kwargs.get(
                'mask_camera_key', occ_io_cfg.get('mask_camera_key', 'mask_camera')),
            mask_lidar_key=eval_kwargs.get(
                'mask_lidar_key', occ_io_cfg.get('mask_lidar_key', 'mask_lidar')),
            ray_num_workers=eval_kwargs.get('ray_num_workers', ray_cfg.get('num_workers', 8)),
            ray_cfg=eval_kwargs.get('ray_cfg', ray_cfg),
        )

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        warnings.warn(
            'TartangroundOcc3DDataset.evaluate is deprecated; please use mmengine '
            'evaluators (Occ3DMetric) via val_evaluator/test_evaluator.',
            DeprecationWarning,
            stacklevel=2,
        )
        metric = self._build_metric(eval_kwargs)
        packed_results = [
            dict(pred=pred, sample_idx=i)
            for i, pred in enumerate(occ_results)
        ]
        return metric.compute_metrics(packed_results)

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, f'{sample_token}.npz')
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        print('\nFinished.')
