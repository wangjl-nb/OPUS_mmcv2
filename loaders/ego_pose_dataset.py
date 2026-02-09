import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import quaternion_to_matrix, transform_matrix

np.set_printoptions(precision=3, suppress=True)


class EgoPoseDataset(Dataset):
    def __init__(self, data_infos, max_origins=8, origin_xy_bound=39.0):
        super().__init__()

        self.data_infos = data_infos
        self.max_origins = max_origins
        self.origin_xy_bound = float(origin_xy_bound)
        self.scene_frames = {}

        for info in data_infos:
            scene_token = self.get_scene_token(info)
            if scene_token not in self.scene_frames:
                self.scene_frames[scene_token] = []
            self.scene_frames[scene_token].append(info)

    def __len__(self):
        return len(self.data_infos)

    def get_scene_token(self, info):
        if 'scene_token' in info:
            scene_name = info['scene_token']
        elif 'scene_name' in info:
            scene_name = info['scene_name']
        else:
            scene_name = info['occ_path'].split('occupancy/')[-1].split('/')[0]
        return scene_name

    def get_ego_from_lidar(self, info):
        return transform_matrix(
            np.array(info['lidar2ego_translation']),
            info['lidar2ego_rotation'])

    def get_global_pose(self, info, inverse=False):
        global_from_ego = transform_matrix(
            np.array(info['ego2global_translation']),
            info['ego2global_rotation'])
        ego_from_lidar = transform_matrix(
            np.array(info['lidar2ego_translation']),
            info['lidar2ego_rotation'])
        pose = global_from_ego.dot(ego_from_lidar)
        if inverse:
            pose = np.linalg.inv(pose)
        return pose

    def __getitem__(self, idx):
        info = self.data_infos[idx]

        ref_sample_token = info.get('token', info.get('sample_token'))
        ref_lidar_from_global = self.get_global_pose(info, inverse=True)
        ref_ego_from_lidar = self.get_ego_from_lidar(info)

        scene_token = self.get_scene_token(info)
        scene_frame = self.scene_frames[scene_token]
        ref_index = scene_frame.index(info)

        output_origin_list = []
        for curr_index in range(len(scene_frame)):
            if curr_index == ref_index:
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                global_from_curr = self.get_global_pose(scene_frame[curr_index], inverse=False)
                ref_from_curr = ref_lidar_from_global.dot(global_from_curr)
                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)

            origin_tf_pad = np.ones([4], dtype=np.float32)
            origin_tf_pad[:3] = origin_tf
            origin_tf = np.dot(ref_ego_from_lidar[:3], origin_tf_pad.T).T

            if np.abs(origin_tf[0]) < self.origin_xy_bound and np.abs(origin_tf[1]) < self.origin_xy_bound:
                output_origin_list.append(origin_tf)

        if not output_origin_list:
            output_origin_list.append(np.zeros(3, dtype=np.float32))

        if self.max_origins is not None and len(output_origin_list) > self.max_origins:
            select_idx = np.round(
                np.linspace(0, len(output_origin_list) - 1, self.max_origins)
            ).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))
        return (ref_sample_token, output_origin_tensor)
