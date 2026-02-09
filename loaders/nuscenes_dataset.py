import os
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmdet3d.registry import DATASETS
from .geometry import quaternion_to_matrix


@DATASETS.register_module()
class CustomNuScenesDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 data_root,
                 pipeline,
                 modality,
                 classes=None,
                 test_mode=False,
                 **kwargs):
        self.modality = modality
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

    def collect_sweeps(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
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
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = quaternion_to_matrix(ego2global_rotation)
        lidar2ego_rotation = quaternion_to_matrix(lidar2ego_rotation)

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
