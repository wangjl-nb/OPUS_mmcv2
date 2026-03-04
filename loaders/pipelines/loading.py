import copy
import os
import cv2
import mmcv
import torch
import numpy as np
import os.path as osp
from PIL import Image
from numpy.linalg import inv
from mmengine.dist import get_dist_info
from mmengine.fileio import FileClient, get
from mmengine.utils import check_file_exist
try:
    from mmdet3d.structures.points import BasePoints
    from mmdet3d.structures.points import get_points_type
except Exception:  # pragma: no cover
    from mmdet3d.core.points import BasePoints
    from mmdet3d.core.points import get_points_type
from mmdet3d.registry import TRANSFORMS
from ..utils import compose_ego2img

DEFAULT_CAM_TYPES = [
    'CAM_LEFT', 'CAM_BACK', 'CAM_FRONT',
    'CAM_BOTTOM', 'CAM_TOP', 'CAM_RIGHT'
]

DEFAULT_OCC3D_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

DEFAULT_OCCUPANCY_CLASS_NAMES = [
    'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]


def _infer_cam_types_from_sweeps(sweeps):
    for sweep in sweeps:
        if isinstance(sweep, dict) and sweep:
            return list(sweep.keys())
    return []


def _resolve_cam_types(results, configured_cam_types=None):
    if 'cam_types' in results and results['cam_types']:
        return list(results['cam_types'])

    if configured_cam_types:
        return list(configured_cam_types)

    cam_sweeps = results.get('cam_sweeps', {})
    if isinstance(cam_sweeps, dict):
        inferred = _infer_cam_types_from_sweeps(cam_sweeps.get('prev', []))
        if inferred:
            return inferred
        inferred = _infer_cam_types_from_sweeps(cam_sweeps.get('next', []))
        if inferred:
            return inferred

    sweeps = results.get('sweeps', {})
    if isinstance(sweeps, dict):
        inferred = _infer_cam_types_from_sweeps(sweeps.get('prev', []))
        if inferred:
            return inferred
        inferred = _infer_cam_types_from_sweeps(sweeps.get('next', []))
        if inferred:
            return inferred

    num_views = results.get('num_views', None)
    if num_views is None and isinstance(results.get('img', None), list):
        num_views = len(results['img'])
    if num_views is not None:
        return [f'CAM_{i}' for i in range(int(num_views))]
    return list(DEFAULT_CAM_TYPES)


def _resolve_num_views(results, cam_types):
    if cam_types:
        return int(len(cam_types))
    if 'num_views' in results and results['num_views'] is not None:
        return int(results['num_views'])
    if isinstance(results.get('img', None), list) and results['img']:
        return int(len(results['img']))
    return int(len(DEFAULT_CAM_TYPES))


@TRANSFORMS.register_module(force=True)
class LoadMultiViewImageFromFiles:
    """Load multi-view images and support configurable decoding backend."""

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 imdecode_backend=None,
                 backend_args=None,
                 num_views=5,
                 num_ref_frames=-1,
                 test_mode=False,
                 set_default_scale=True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        self.num_views = num_views
        self.num_ref_frames = num_ref_frames
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale

    def _select_ref_frames(self, results):
        if self.num_ref_frames <= 0:
            return results
        init_choice = np.array([0], dtype=np.int64)
        num_views = int(results.get('num_views', self.num_views) or self.num_views)
        num_frames = len(results['img_filename']) // num_views - 1
        if num_frames == 0:
            choices = np.random.choice(1, self.num_ref_frames, replace=True)
        elif num_frames >= self.num_ref_frames:
            if self.test_mode:
                choices = np.arange(num_frames - self.num_ref_frames, num_frames) + 1
            else:
                choices = np.random.choice(num_frames, self.num_ref_frames, replace=False) + 1
        elif num_frames > 0 and num_frames < self.num_ref_frames:
            if self.test_mode:
                base_choices = np.arange(num_frames) + 1
                random_choices = np.random.choice(
                    num_frames, self.num_ref_frames - num_frames, replace=True) + 1
                choices = np.concatenate([base_choices, random_choices])
            else:
                choices = np.random.choice(num_frames, self.num_ref_frames, replace=True) + 1
        else:
            raise NotImplementedError
        choices = np.concatenate([init_choice, choices])
        select_filename = []
        for choice in choices:
            select_filename += results['img_filename'][choice * num_views:
                                                       (choice + 1) * num_views]
        results['img_filename'] = select_filename
        for key in ['cam2img', 'lidar2cam']:
            if key in results:
                select_results = []
                for choice in choices:
                    select_results += results[key][choice * num_views:(choice + 1) * num_views]
                results[key] = select_results
        for key in ['ego2global']:
            if key in results:
                select_results = []
                for choice in choices:
                    select_results += [results[key][choice]]
                results[key] = select_results
        for key in ['lidar2cam']:
            if key in results:
                for choice_idx in range(1, len(choices)):
                    pad_prev_ego2global = np.eye(4)
                    prev_ego2global = results['ego2global'][choice_idx]
                    pad_prev_ego2global[:prev_ego2global.shape[0], :prev_ego2global.shape[1]] = prev_ego2global
                    pad_cur_ego2global = np.eye(4)
                    cur_ego2global = results['ego2global'][0]
                    pad_cur_ego2global[:cur_ego2global.shape[0], :cur_ego2global.shape[1]] = cur_ego2global
                    cur2prev = np.linalg.inv(pad_prev_ego2global).dot(pad_cur_ego2global)
                    for result_idx in range(choice_idx * num_views,
                                            (choice_idx + 1) * num_views):
                        results[key][result_idx] = results[key][result_idx].dot(cur2prev)
        return results

    def __call__(self, results):
        if 'img_filename' in results:
            results = self._select_ref_frames(results)
            filename = results['img_filename']
        elif 'images' in results:
            filename, cam2img, lidar2cam = [], [], []
            for _, cam_item in results['images'].items():
                filename.append(cam_item['img_path'])
                if 'cam2img' in cam_item:
                    cam2img.append(cam_item['cam2img'])
                if 'lidar2cam' in cam_item:
                    lidar2cam.append(cam_item['lidar2cam'])
            results['filename'] = filename
            if cam2img:
                results['cam2img'] = cam2img
                results['ori_cam2img'] = copy.deepcopy(results['cam2img'])
            if lidar2cam:
                results['lidar2cam'] = lidar2cam
            results['img_filename'] = filename
        else:
            raise KeyError('Results must contain "img_filename" or "images".')

        img_bytes = [get(name, backend_args=self.backend_args) for name in filename]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type, backend=self.imdecode_backend)
            for img_byte in img_bytes
        ]
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
            imgs = [mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        num_views = results.get('num_views', None)
        if num_views is None or int(num_views) <= 0:
            num_views = len(results['img']) if self.num_ref_frames <= 0 else self.num_views
        results['num_views'] = int(num_views)
        results['num_ref_frames'] = self.num_ref_frames
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'imdecode_backend={self.imdecode_backend}, '
        repr_str += f'num_views={self.num_views}, '
        repr_str += f'num_ref_frames={self.num_ref_frames}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str


@TRANSFORMS.register_module()
class LoadOcc3DFromFile:

    def __init__(self,
                 occ_root,
                 ignore_class_names=None,
                 path_template='{scene_name}/{token}/labels.npz',
                 semantics_key='semantics',
                 mask_camera_key='mask_camera',
                 mask_lidar_key='mask_lidar',
                 class_names=None,
                 empty_label=None):
        self.occ_root = occ_root
        self.ignore_class_names = ignore_class_names or []
        self.path_template = path_template
        self.semantics_key = semantics_key
        self.mask_camera_key = mask_camera_key
        self.mask_lidar_key = mask_lidar_key
        self.occ_class_names = class_names or list(DEFAULT_OCC3D_CLASS_NAMES)
        self.empty_label = len(self.occ_class_names) - 1 if empty_label is None else int(empty_label)

    def _build_occ_path(self, results):
        fmt = dict(results)
        if 'sample_token' in results:
            fmt.setdefault('token', results['sample_token'])
        if 'token' in results:
            fmt.setdefault('sample_token', results['token'])
        return osp.join(self.occ_root, self.path_template.format(**fmt))

    def __call__(self, results):
        occ_file = self._build_occ_path(results)
        occ_labels = np.load(occ_file)

        semantics = np.array(occ_labels[self.semantics_key], copy=True)
        mask_shape = semantics.shape
        mask_lidar = occ_labels[self.mask_lidar_key].astype(np.bool_)             if self.mask_lidar_key in occ_labels else np.ones(mask_shape, dtype=np.bool_)
        mask_camera = occ_labels[self.mask_camera_key].astype(np.bool_)             if self.mask_camera_key in occ_labels else np.ones(mask_shape, dtype=np.bool_)

        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        if self.ignore_class_names:
            for class_id, class_name in enumerate(self.occ_class_names):
                if class_id == self.empty_label:
                    continue
                if class_name not in self.ignore_class_names:
                    continue
                semantics[semantics == class_id] = self.empty_label

        results['voxel_semantics'] = np.ascontiguousarray(semantics)
        return results


@TRANSFORMS.register_module()
class LoadOccupancyFromFile:

    def __init__(self,
                 occ_root,
                 ignore_class_names=None,
                 path_template='scene_{scene_token}/occupancy/{lidar_token}.npy',
                 pc_range=None,
                 voxel_size=None,
                 src_class_names=None):
        self.occ_root = occ_root
        self.ignore_class_names = ignore_class_names or ['noise']
        self.path_template = path_template
        self.pc_range = np.array(pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3])
        self.voxel_size = np.array(voxel_size if voxel_size is not None else [0.2, 0.2, 0.2])
        self.occ_class_names = src_class_names or list(DEFAULT_OCCUPANCY_CLASS_NAMES)

    def _build_occ_path(self, results):
        fmt = dict(results)
        if 'sample_token' in results:
            fmt.setdefault('token', results['sample_token'])
        if 'token' in results:
            fmt.setdefault('sample_token', results['token'])
        return osp.join(self.occ_root, self.path_template.format(**fmt))

    def __call__(self, results):
        occ_file = self._build_occ_path(results)
        occ_labels = np.load(occ_file)
        coors, labels = occ_labels[:, :3], occ_labels[:, 3].astype(np.int64)

        curr_class_names = [n for n in self.occ_class_names if n not in self.ignore_class_names]
        empty_label = len(curr_class_names)

        class_to_idx = {name: idx for idx, name in enumerate(curr_class_names)}
        label_mapper = np.full(len(self.occ_class_names), empty_label, dtype=np.int64)
        for idx, name in enumerate(self.occ_class_names):
            if name in class_to_idx:
                label_mapper[idx] = class_to_idx[name]
        labels = label_mapper[labels]

        scene_size = self.pc_range[3:] - self.pc_range[:3]
        voxel_num = (scene_size / self.voxel_size).astype(np.int64)
        semantics = np.full(voxel_num, empty_label, dtype=np.uint8)
        semantics[coors[:, 2], coors[:, 1], coors[:, 0]] = labels
        results['voxel_semantics'] = np.ascontiguousarray(semantics)
        return results


@TRANSFORMS.register_module()
class LoadMultiViewImageFromMultiSweeps:
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 train_interval=[4, 8],
                 test_interval=6,
                 force_offline=False,
                 cam_types=None,
                 imdecode_backend='turbojpeg'):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        self.force_offline = force_offline
        self.cam_types = list(cam_types) if cam_types is not None else None

        self.train_interval = train_interval
        self.test_interval = test_interval

        mmcv.use_backend(imdecode_backend)

    def _get_cam_types(self, results):
        return _resolve_cam_types(results, self.cam_types)

    def _pick_sweep(self, sweeps, idx, cam_types):
        sweep_idx = min(idx, len(sweeps) - 1)
        sweep = sweeps[sweep_idx]
        if len(sweep.keys()) < len(cam_types) and sweep_idx > 0:
            sweep = sweeps[sweep_idx - 1]
        sensors = [sensor for sensor in cam_types if sensor in sweep]
        if not sensors:
            sensors = list(sweep.keys())
        return sweep, sensors

    def load_offline(self, results):
        cam_types = self._get_cam_types(results)
        num_views = _resolve_num_views(results, cam_types)

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(num_views):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['cam_sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['cam_sweeps']['prev'])
                choices = list(range(len(results['cam_sweeps']['prev']))) + [
                    len(results['cam_sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['cam_sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep, sensors = self._pick_sweep(results['cam_sweeps']['prev'], idx, cam_types)
                for sensor in sensors:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'].T,
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def load_online(self, results):
        assert self.test_mode

        cam_types = self._get_cam_types(results)
        num_views = _resolve_num_views(results, cam_types)

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(num_views):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep, sensors = self._pick_sweep(results['cam_sweeps']['prev'], idx, cam_types)
                for sensor in sensors:
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'].T,
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode and (not self.force_offline):
            return self.load_online(results)
        return self.load_offline(results)


@TRANSFORMS.register_module()
class LoadMultiViewImageFromMultiSweepsFuture:
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 cam_types=None):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        self.cam_types = list(cam_types) if cam_types is not None else None

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = _resolve_cam_types(results, self.cam_types)
        num_views = _resolve_num_views(results, cam_types)

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(num_views):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]
            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]
                if len(sweep.keys()) < len(cam_types) and sweep_idx > 0:
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]
                sensors = [sensor for sensor in cam_types if sensor in sweep] or list(sweep.keys())
                for sensor in sensors:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'].T,
                        sweep[sensor]['cam_intrinsic'],
                    ))

        if len(results['cam_sweeps']['next']) == 0:
            for _ in range(self.next_sweeps_num):
                for j in range(num_views):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]
            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['next']) - 1)
                sweep = results['cam_sweeps']['next'][sweep_idx]
                if len(sweep.keys()) < len(cam_types) and sweep_idx > 0:
                    sweep = results['cam_sweeps']['next'][sweep_idx - 1]
                sensors = [sensor for sensor in cam_types if sensor in sweep] or list(sweep.keys())
                for sensor in sensors:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'].T,
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results


'''
This func loads previous and future frames in interleaved order,
e.g. curr, prev1, next1, prev2, next2, prev3, next3...
'''
@TRANSFORMS.register_module()
class LoadMultiViewImageFromMultiSweepsFutureInterleave:
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 cam_types=None):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        self.cam_types = list(cam_types) if cam_types is not None else None

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = _resolve_cam_types(results, self.cam_types)
        num_views = _resolve_num_views(results, cam_types)

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        results_prev = dict(img=[], img_timestamp=[], filename=[], ego2img=[])
        results_next = dict(img=[], img_timestamp=[], filename=[], ego2img=[])

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(num_views):
                    results_prev['img'].append(results['img'][j])
                    results_prev['img_timestamp'].append(results['img_timestamp'][j])
                    results_prev['filename'].append(results['filename'][j])
                    results_prev['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]
            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]
                if len(sweep.keys()) < len(cam_types) and sweep_idx > 0:
                    sweep = results['sweeps']['prev'][sweep_idx - 1]
                sensors = [sensor for sensor in cam_types if sensor in sweep] or list(sweep.keys())
                for sensor in sensors:
                    results_prev['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_prev['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_prev['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results_prev['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'].T,
                        sweep[sensor]['cam_intrinsic'],
                    ))

        if len(results['sweeps']['next']) == 0:
            for _ in range(self.next_sweeps_num):
                for j in range(num_views):
                    results_next['img'].append(results['img'][j])
                    results_next['img_timestamp'].append(results['img_timestamp'][j])
                    results_next['filename'].append(results['filename'][j])
                    results_next['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]
            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['next']) - 1)
                sweep = results['sweeps']['next'][sweep_idx]
                if len(sweep.keys()) < len(cam_types) and sweep_idx > 0:
                    sweep = results['sweeps']['next'][sweep_idx - 1]
                sensors = [sensor for sensor in cam_types if sensor in sweep] or list(sweep.keys())
                for sensor in sensors:
                    results_next['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_next['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_next['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results_next['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'].T,
                        sweep[sensor]['cam_intrinsic'],
                    ))

        assert len(results_prev['img']) % num_views == 0
        assert len(results_next['img']) % num_views == 0

        for i in range(len(results_prev['img']) // num_views):
            for j in range(num_views):
                results['img'].append(results_prev['img'][i * num_views + j])
                results['img_timestamp'].append(results_prev['img_timestamp'][i * num_views + j])
                results['filename'].append(results_prev['filename'][i * num_views + j])
                results['ego2img'].append(results_prev['ego2img'][i * num_views + j])

            for j in range(num_views):
                results['img'].append(results_next['img'][i * num_views + j])
                results['img_timestamp'].append(results_next['img_timestamp'][i * num_views + j])
                results['filename'].append(results_next['filename'][i * num_views + j])
                results['ego2img'].append(results_next['ego2img'][i * num_views + j])

        return results


@TRANSFORMS.register_module()
class LoadPointsFromMultiViewDepth:
    """Build pseudo-LiDAR points from selected multi-view depth maps.

    The transform keeps the point tensor format compatible with the existing
    LiDAR pipeline: ``[x, y, z, intensity, time]`` in LiDAR coordinates.
    """

    def __init__(self,
                 cam_types=None,
                 depth_key='depth_path',
                 coord_type='LIDAR',
                 load_dim=5,
                 use_dim=[0, 1, 2, 3, 4],
                 time_dim=4,
                 sample_stride=7,
                 max_points_total=560000,
                 depth_min=0.1,
                 depth_max=80.0,
                 coord_convention='tartanair_ned',
                 intensity_value=0.0,
                 strict_depth_exist=False,
                 fallback_depth_from_image_path=True,
                 history_dynamic_extrinsics=True,
                 dynamic_extrinsics_fallback='static'):
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['LIDAR', 'DEPTH', 'CAMERA']
        assert coord_convention in ['tartanair_ned', 'opencv']
        assert dynamic_extrinsics_fallback in ['static', 'skip', 'raise'], \
            f'Unsupported dynamic_extrinsics_fallback: {dynamic_extrinsics_fallback}'

        self.cam_types = list(cam_types) if cam_types is not None else None
        self.depth_key = depth_key
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.time_dim = time_dim
        self.sample_stride = max(int(sample_stride), 1)
        self.max_points_total = int(max_points_total) if max_points_total is not None else 0
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.coord_convention = coord_convention
        self.intensity_value = float(intensity_value)
        self.strict_depth_exist = bool(strict_depth_exist)
        self.fallback_depth_from_image_path = bool(fallback_depth_from_image_path)
        self.history_dynamic_extrinsics = bool(history_dynamic_extrinsics)
        self.dynamic_extrinsics_fallback = dynamic_extrinsics_fallback

    def _path_keys(self, path):
        if not isinstance(path, str) or not path:
            return []
        keys = set()
        norm_path = osp.normpath(path)
        keys.add(norm_path)
        keys.add(osp.normpath(osp.abspath(path)))
        try:
            keys.add(osp.normpath(osp.relpath(path)))
        except Exception:
            pass
        return list(keys)

    def _register_cam_info(self, lookup, cam_name, cam_info):
        if not isinstance(cam_info, dict):
            return
        image_path = cam_info.get('data_path', None)
        for key in self._path_keys(image_path):
            if key not in lookup:
                lookup[key] = (cam_name, cam_info)

    def _build_cam_lookup(self, results):
        lookup = {}
        cam_types = _resolve_cam_types(results, self.cam_types)

        cams = results.get('cams', {})
        if isinstance(cams, dict):
            for cam_name in cam_types:
                if cam_name in cams:
                    self._register_cam_info(lookup, cam_name, cams[cam_name])
            for cam_name, cam_info in cams.items():
                self._register_cam_info(lookup, cam_name, cam_info)

        cam_sweeps = results.get('cam_sweeps', {})
        if isinstance(cam_sweeps, dict):
            for side in ['prev', 'next']:
                sweeps = cam_sweeps.get(side, [])
                if not isinstance(sweeps, list):
                    continue
                for sweep in sweeps:
                    if not isinstance(sweep, dict):
                        continue
                    for cam_name, cam_info in sweep.items():
                        self._register_cam_info(lookup, cam_name, cam_info)
        elif isinstance(cam_sweeps, list):
            for sweep in cam_sweeps:
                if not isinstance(sweep, dict):
                    continue
                for cam_name, cam_info in sweep.items():
                    self._register_cam_info(lookup, cam_name, cam_info)

        return lookup, cam_types

    def _depth_candidates_from_image(self, image_path):
        if not isinstance(image_path, str) or not image_path:
            return []
        path = osp.normpath(image_path)
        parent = osp.dirname(path)
        dirname = osp.basename(parent)
        stem, _ = osp.splitext(osp.basename(path))

        if not stem.endswith('_depth'):
            stem = stem + '_depth'

        if dirname.startswith('image_'):
            depth_dir = osp.join(osp.dirname(parent), dirname.replace('image_', 'depth_', 1))
        else:
            depth_dir = parent
        return [
            osp.join(depth_dir, stem + '.png'),
            osp.join(depth_dir, stem + '.npy'),
        ]

    def _resolve_depth_path(self, cam_info, image_path):
        candidates = []
        depth_path = cam_info.get(self.depth_key, None) if isinstance(cam_info, dict) else None
        if isinstance(depth_path, str) and depth_path:
            candidates.append(depth_path)
        if self.fallback_depth_from_image_path:
            candidates.extend(self._depth_candidates_from_image(image_path))

        checked = set()
        for cand in candidates:
            if not isinstance(cand, str) or not cand:
                continue
            for key in self._path_keys(cand):
                if key in checked:
                    continue
                checked.add(key)
                if osp.exists(key):
                    return key
        return candidates[0] if candidates else None

    def _load_depth(self, depth_path):
        if depth_path is None:
            return None
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_rgba is None:
                return None
            depth = depth_rgba.view('<f4').squeeze()
        if depth is None:
            return None
        if depth.ndim > 2:
            depth = np.squeeze(depth)
        return depth.astype(np.float32)

    def _depth_to_points_camera(self, depth, cam_intrinsic):
        if depth is None or depth.ndim != 2:
            return np.zeros((0, 3), dtype=np.float32)

        h, w = depth.shape
        v_idx = np.arange(0, h, self.sample_stride, dtype=np.int32)
        u_idx = np.arange(0, w, self.sample_stride, dtype=np.int32)
        if v_idx.size == 0 or u_idx.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        depth_sub = depth[np.ix_(v_idx, u_idx)]
        u, v = np.meshgrid(u_idx.astype(np.float32) + 0.5,
                           v_idx.astype(np.float32) + 0.5)

        valid = np.isfinite(depth_sub)
        valid &= (depth_sub > self.depth_min) & (depth_sub < self.depth_max)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        d = depth_sub[valid]
        u = u[valid]
        v = v[valid]

        intrinsic = np.asarray(cam_intrinsic, dtype=np.float32)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy

        if self.coord_convention == 'tartanair_ned':
            pts_cam = np.stack([d, x, y], axis=1)
        else:
            pts_cam = np.stack([x, y, d], axis=1)
        return pts_cam.astype(np.float32)

    def _as_rotation_matrix(self, rotation):
        rot = np.asarray(rotation)
        if rot.shape == (3, 3):
            return rot.astype(np.float64)

        quat = np.asarray(rotation, dtype=np.float64).reshape(-1)
        if quat.size != 4:
            raise ValueError(f'Unsupported rotation shape: {rot.shape}')
        norm = np.linalg.norm(quat)
        if norm <= 0:
            raise ValueError('Quaternion rotation has zero norm')
        w, x, y, z = quat / norm
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    def _as_translation(self, translation):
        trans = np.asarray(translation, dtype=np.float64).reshape(-1)
        if trans.size != 3:
            raise ValueError(f'Unsupported translation shape: {np.asarray(translation).shape}')
        return trans

    def _compose_row_transform(self, rot_ab, trans_ab, rot_bc, trans_bc):
        rot_ac = rot_bc @ rot_ab
        trans_ac = trans_ab @ rot_bc.T + trans_bc
        return rot_ac, trans_ac

    def _invert_row_transform(self, rot_ab, trans_ab):
        rot_ba = rot_ab.T
        trans_ba = -trans_ab @ rot_ab
        return rot_ba, trans_ba

    def _get_static_sensor2lidar(self, cam_info):
        rot = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float32)
        trans = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float32).reshape(-1)
        if rot.shape != (3, 3):
            raise ValueError(f'Invalid sensor2lidar_rotation shape: {rot.shape}')
        if trans.size != 3:
            raise ValueError(f'Invalid sensor2lidar_translation shape: {trans.shape}')
        return rot, trans

    def _get_dynamic_sensor2lidar(self, cam_info, results):
        rot_eg = self._as_rotation_matrix(results['ego2global_rotation'])
        trans_eg = self._as_translation(results['ego2global_translation'])
        rot_le = self._as_rotation_matrix(results['lidar2ego_rotation'])
        trans_le = self._as_translation(results['lidar2ego_translation'])

        rot_lg, trans_lg = self._compose_row_transform(rot_le, trans_le, rot_eg, trans_eg)
        rot_gl, trans_gl = self._invert_row_transform(rot_lg, trans_lg)

        rot_sg = self._as_rotation_matrix(cam_info['sensor2global_rotation'])
        trans_sg = self._as_translation(cam_info['sensor2global_translation'])
        rot_sl, trans_sl = self._compose_row_transform(rot_sg, trans_sg, rot_gl, trans_gl)

        return rot_sl.astype(np.float32), trans_sl.astype(np.float32)

    def _resolve_sensor2lidar(self, cam_info, results, use_dynamic):
        if not use_dynamic:
            return self._get_static_sensor2lidar(cam_info)

        try:
            return self._get_dynamic_sensor2lidar(cam_info, results)
        except Exception:
            if self.dynamic_extrinsics_fallback == 'raise':
                raise
            if self.dynamic_extrinsics_fallback == 'skip':
                return None, None
            return self._get_static_sensor2lidar(cam_info)

    def _camera_to_lidar(self, pts_cam, cam_info, results=None, use_dynamic=False):
        if pts_cam.shape[0] == 0:
            return pts_cam
        rot, trans = self._resolve_sensor2lidar(cam_info, results, use_dynamic)
        if rot is None or trans is None:
            return None
        return pts_cam @ rot.T + trans[None, :]

    def __call__(self, results):
        filenames = results.get('filename', [])
        img_timestamps = results.get('img_timestamp', [])
        if not isinstance(filenames, list):
            filenames = list(filenames)
        if not isinstance(img_timestamps, list):
            img_timestamps = list(img_timestamps)

        cam_lookup, cam_types = self._build_cam_lookup(results)
        num_views = _resolve_num_views(results, cam_types)
        base_timestamp = float(results.get('timestamp', 0.0))

        point_chunks = []
        for idx, image_path in enumerate(filenames):
            match = None
            for key in self._path_keys(image_path):
                if key in cam_lookup:
                    match = cam_lookup[key]
                    break
            if match is None:
                if self.strict_depth_exist:
                    raise FileNotFoundError(f'No camera metadata matched for image: {image_path}')
                continue

            _, cam_info = match
            depth_path = self._resolve_depth_path(cam_info, image_path)
            if depth_path is None or (not osp.exists(depth_path)):
                if self.strict_depth_exist:
                    raise FileNotFoundError(f'No depth file for image: {image_path}')
                continue

            depth = self._load_depth(depth_path)
            if depth is None:
                if self.strict_depth_exist:
                    raise FileNotFoundError(f'Failed to load depth file: {depth_path}')
                continue

            pts_cam = self._depth_to_points_camera(depth, cam_info['cam_intrinsic'])
            if pts_cam.shape[0] == 0:
                continue
            use_dynamic = self.history_dynamic_extrinsics and (idx >= num_views)
            pts_lidar = self._camera_to_lidar(
                pts_cam,
                cam_info,
                results=results,
                use_dynamic=use_dynamic)
            if pts_lidar is None:
                continue

            if idx < num_views:
                time_delta = 0.0
            else:
                img_ts = float(img_timestamps[idx]) if idx < len(img_timestamps) else base_timestamp
                time_delta = max(base_timestamp - img_ts, 0.0)
                if abs(time_delta) < 1e-6:
                    time_delta = 0.0

            intensity = np.full((pts_lidar.shape[0], 1), self.intensity_value, dtype=np.float32)
            time_col = np.full((pts_lidar.shape[0], 1), time_delta, dtype=np.float32)
            pts = np.concatenate([pts_lidar.astype(np.float32), intensity, time_col], axis=1)
            point_chunks.append(pts)

        if point_chunks:
            points = np.concatenate(point_chunks, axis=0)
        else:
            points = np.zeros((0, self.load_dim), dtype=np.float32)

        if self.max_points_total > 0 and points.shape[0] > self.max_points_total:
            choice = np.random.choice(points.shape[0], self.max_points_total, replace=False)
            points = points[choice]

        points = points[:, self.use_dim]
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        return results


@TRANSFORMS.register_module()
class LoadMapAnythingExtraFromDepth:
    """Construct mapanything_extra views from depth + intrinsics + camera poses.

    This transform is intended to run after image augmentations so that depth and
    intrinsics are synchronized with the image-domain transform.
    """

    def __init__(self,
                 cam_types=None,
                 depth_key='depth_path',
                 fallback_depth_from_image_path=True,
                 strict=True,
                 min_valid_depth_ratio=1e-4,
                 depth_nonnegative_eps=-1e-6,
                 apply_ida_to_depth=True,
                 apply_ida_to_intrinsics=True,
                 filter_depth_by_pcrange=False,
                 point_cloud_range=None,
                 rotation_atol=1e-3,
                 rotation_rtol=1e-3):
        self.cam_types = list(cam_types) if cam_types is not None else None
        self.depth_key = depth_key
        self.fallback_depth_from_image_path = bool(fallback_depth_from_image_path)
        self.strict = bool(strict)
        self.min_valid_depth_ratio = float(min_valid_depth_ratio)
        self.depth_nonnegative_eps = float(depth_nonnegative_eps)
        self.apply_ida_to_depth = bool(apply_ida_to_depth)
        self.apply_ida_to_intrinsics = bool(apply_ida_to_intrinsics)
        self.filter_depth_by_pcrange = bool(filter_depth_by_pcrange)
        self.point_cloud_range = self._parse_point_cloud_range(point_cloud_range) \
            if self.filter_depth_by_pcrange else None
        self.rotation_atol = float(rotation_atol)
        self.rotation_rtol = float(rotation_rtol)

    def _path_keys(self, path):
        if not isinstance(path, str) or not path:
            return []
        keys = set()
        norm_path = osp.normpath(path)
        keys.add(norm_path)
        keys.add(osp.normpath(osp.abspath(path)))
        try:
            keys.add(osp.normpath(osp.relpath(path)))
        except Exception:
            pass
        return list(keys)

    def _register_cam_info(self, lookup, cam_name, cam_info):
        if not isinstance(cam_info, dict):
            return
        image_path = cam_info.get('data_path', None)
        for key in self._path_keys(image_path):
            if key not in lookup:
                lookup[key] = (cam_name, cam_info)

    def _build_cam_lookup(self, results):
        lookup = {}
        cam_types = _resolve_cam_types(results, self.cam_types)

        cams = results.get('cams', {})
        if isinstance(cams, dict):
            for cam_name in cam_types:
                if cam_name in cams:
                    self._register_cam_info(lookup, cam_name, cams[cam_name])
            for cam_name, cam_info in cams.items():
                self._register_cam_info(lookup, cam_name, cam_info)

        cam_sweeps = results.get('cam_sweeps', {})
        if isinstance(cam_sweeps, dict):
            for side in ['prev', 'next']:
                sweeps = cam_sweeps.get(side, [])
                if not isinstance(sweeps, list):
                    continue
                for sweep in sweeps:
                    if not isinstance(sweep, dict):
                        continue
                    for cam_name, cam_info in sweep.items():
                        self._register_cam_info(lookup, cam_name, cam_info)
        elif isinstance(cam_sweeps, list):
            for sweep in cam_sweeps:
                if not isinstance(sweep, dict):
                    continue
                for cam_name, cam_info in sweep.items():
                    self._register_cam_info(lookup, cam_name, cam_info)
        return lookup

    def _depth_candidates_from_image(self, image_path):
        if not isinstance(image_path, str) or not image_path:
            return []
        path = osp.normpath(image_path)
        parent = osp.dirname(path)
        dirname = osp.basename(parent)
        stem, _ = osp.splitext(osp.basename(path))
        if not stem.endswith('_depth'):
            stem = stem + '_depth'
        if dirname.startswith('image_'):
            depth_dir = osp.join(osp.dirname(parent), dirname.replace('image_', 'depth_', 1))
        else:
            depth_dir = parent
        return [
            osp.join(depth_dir, stem + '.png'),
            osp.join(depth_dir, stem + '.npy'),
        ]

    def _resolve_depth_path(self, cam_info, image_path):
        candidates = []
        depth_path = cam_info.get(self.depth_key, None) if isinstance(cam_info, dict) else None
        if isinstance(depth_path, str) and depth_path:
            candidates.append(depth_path)
        if self.fallback_depth_from_image_path:
            candidates.extend(self._depth_candidates_from_image(image_path))

        checked = set()
        for cand in candidates:
            if not isinstance(cand, str) or not cand:
                continue
            for key in self._path_keys(cand):
                if key in checked:
                    continue
                checked.add(key)
                if osp.exists(key):
                    return key
        return candidates[0] if candidates else None

    def _load_depth(self, depth_path):
        if depth_path is None:
            return None
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_rgba is None:
                return None
            depth = depth_rgba.view('<f4').squeeze()
        if depth is None:
            return None
        if depth.ndim > 2:
            depth = np.squeeze(depth)
        if depth.ndim != 2:
            return None
        return depth.astype(np.float32)

    @staticmethod
    def _as_rotation_matrix(rotation):
        rot = np.asarray(rotation)
        if rot.shape == (3, 3):
            return rot.astype(np.float64)

        quat = np.asarray(rotation, dtype=np.float64).reshape(-1)
        if quat.size != 4:
            raise ValueError(f'Unsupported rotation shape: {rot.shape}')
        norm = np.linalg.norm(quat)
        if norm <= 0:
            raise ValueError('Quaternion rotation has zero norm')
        w, x, y, z = quat / norm
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    def _validate_rotation_matrix(self, rotation, context):
        rot = self._as_rotation_matrix(rotation)
        if not np.all(np.isfinite(rot)):
            raise ValueError(f'{context}: rotation contains non-finite values')
        ortho = rot.T @ rot
        if not np.allclose(ortho, np.eye(3), rtol=self.rotation_rtol, atol=self.rotation_atol):
            raise ValueError(f'{context}: rotation is not orthonormal')
        det = float(np.linalg.det(rot))
        if (not np.isfinite(det)) or det <= 0:
            raise ValueError(f'{context}: rotation determinant must be positive, got {det}')
        return rot

    @staticmethod
    def _image_hw(img):
        if isinstance(img, np.ndarray):
            if img.ndim < 2:
                raise ValueError(f'Unsupported image ndim={img.ndim}')
            return int(img.shape[0]), int(img.shape[1])
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                if img.shape[0] in (1, 3):
                    return int(img.shape[1]), int(img.shape[2])
                return int(img.shape[0]), int(img.shape[1])
            if img.dim() == 2:
                return int(img.shape[0]), int(img.shape[1])
        raise TypeError(f'Unsupported image type for shape inference: {type(img)}')

    def _apply_ida_to_depth(self, depth, ida_aug_params):
        if not self.apply_ida_to_depth:
            return depth
        if ida_aug_params is None:
            return depth

        resize_dims = ida_aug_params.get('resize_dims', None)
        crop = ida_aug_params.get('crop', None)
        flip = bool(ida_aug_params.get('flip', False))
        rotate = float(ida_aug_params.get('rotate', 0.0))
        if resize_dims is None or crop is None:
            if self.strict:
                raise KeyError('ida_aug_params must contain resize_dims and crop')
            return depth

        pil_depth = Image.fromarray(depth.astype(np.float32), mode='F')
        pil_depth = pil_depth.resize(tuple(int(v) for v in resize_dims), resample=Image.BILINEAR)
        pil_depth = pil_depth.crop(tuple(int(v) for v in crop))
        if flip:
            pil_depth = pil_depth.transpose(method=Image.FLIP_LEFT_RIGHT)
        if abs(rotate) > 1e-6:
            pil_depth = pil_depth.rotate(rotate, resample=Image.BILINEAR)
        return np.array(pil_depth, dtype=np.float32)

    def _apply_ida_to_intrinsics(self, intrinsics, ida_aug_params):
        if not self.apply_ida_to_intrinsics:
            return intrinsics
        if ida_aug_params is None:
            return intrinsics

        ida_mat = ida_aug_params.get('ida_mat', None)
        if ida_mat is None:
            if self.strict:
                raise KeyError('ida_aug_params must contain ida_mat')
            return intrinsics
        ida_mat = np.asarray(ida_mat, dtype=np.float64)
        if ida_mat.shape != (4, 4):
            raise ValueError(f'ida_aug_params["ida_mat"] must be 4x4, got {ida_mat.shape}')
        ida_affine = ida_mat[:3, :3]
        return (ida_affine @ intrinsics.astype(np.float64)).astype(np.float32)

    def _find_cam_info(self, cam_lookup, image_path):
        for key in self._path_keys(image_path):
            if key in cam_lookup:
                return cam_lookup[key][1]

        if isinstance(image_path, str) and image_path:
            basename = osp.basename(image_path)
            matched = []
            for key, (_, cam_info) in cam_lookup.items():
                if isinstance(key, str) and key.endswith(basename):
                    matched.append(cam_info)
            if len(matched) == 1:
                return matched[0]
        return None

    def _validate_intrinsics(self, intrinsics, context):
        if intrinsics.shape != (3, 3):
            raise ValueError(f'{context}: intrinsics must be 3x3, got {intrinsics.shape}')
        if not np.all(np.isfinite(intrinsics)):
            raise ValueError(f'{context}: intrinsics contain non-finite values')
        if abs(float(intrinsics[0, 0])) <= 1e-6 or abs(float(intrinsics[1, 1])) <= 1e-6:
            raise ValueError(
                f'{context}: intrinsics fx/fy magnitude must be positive, '
                f'got fx={intrinsics[0, 0]}, fy={intrinsics[1, 1]}')

    def _validate_depth(self, depth, context):
        if depth.ndim != 2:
            raise ValueError(f'{context}: depth must be 2D HxW, got shape={depth.shape}')
        if not np.all(np.isfinite(depth)):
            raise ValueError(f'{context}: depth contains non-finite values')
        if depth.size == 0:
            raise ValueError(f'{context}: empty depth map')

        min_depth = float(np.min(depth))
        if min_depth < self.depth_nonnegative_eps:
            raise ValueError(
                f'{context}: depth has negative values (min={min_depth}, '
                f'eps={self.depth_nonnegative_eps})')

        valid_mask = depth > 0.0
        valid_ratio = float(np.mean(valid_mask))
        if valid_ratio < self.min_valid_depth_ratio:
            raise ValueError(
                f'{context}: valid depth ratio too low ({valid_ratio:.6f} < {self.min_valid_depth_ratio:.6f})')

    @staticmethod
    def _parse_point_cloud_range(point_cloud_range):
        if point_cloud_range is None:
            raise ValueError(
                'point_cloud_range must be provided when filter_depth_by_pcrange=True')
        pcr = np.asarray(point_cloud_range, dtype=np.float32).reshape(-1)
        if pcr.size != 6:
            raise ValueError(
                f'point_cloud_range must contain 6 values, got shape={pcr.shape}')
        if not np.all(np.isfinite(pcr)):
            raise ValueError('point_cloud_range contains non-finite values')
        if not (pcr[0] < pcr[3] and pcr[1] < pcr[4] and pcr[2] < pcr[5]):
            raise ValueError(
                f'point_cloud_range min/max invalid: {pcr.tolist()}')
        return pcr

    def _filter_depth_by_point_cloud_range(self, depth, intrinsics, camera_pose, context):
        if not self.filter_depth_by_pcrange:
            return depth

        valid_depth = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid_depth):
            return np.zeros_like(depth, dtype=np.float32)

        v, u = np.nonzero(valid_depth)
        z = depth[v, u].astype(np.float64)

        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        if abs(fx) <= 1e-6 or abs(fy) <= 1e-6:
            raise ValueError(
                f'{context}: invalid intrinsics for unprojection, fx={fx}, fy={fy}')

        x = (u.astype(np.float64) - cx) * z / fx
        y = (v.astype(np.float64) - cy) * z / fy
        points_cam = np.stack([x, y, z], axis=1)

        rot = np.asarray(camera_pose[:3, :3], dtype=np.float64)
        trans = np.asarray(camera_pose[:3, 3], dtype=np.float64)
        points_world = (rot @ points_cam.T).T + trans[None, :]

        pcr = self.point_cloud_range
        in_range = (
            (points_world[:, 0] >= float(pcr[0])) & (points_world[:, 0] <= float(pcr[3])) &
            (points_world[:, 1] >= float(pcr[1])) & (points_world[:, 1] <= float(pcr[4])) &
            (points_world[:, 2] >= float(pcr[2])) & (points_world[:, 2] <= float(pcr[5]))
        )

        filtered = np.zeros_like(depth, dtype=np.float32)
        if np.any(in_range):
            keep_v = v[in_range]
            keep_u = u[in_range]
            filtered[keep_v, keep_u] = depth[keep_v, keep_u]
        return filtered

    def _build_camera_pose(self, cam_info, context):
        if not isinstance(cam_info, dict):
            raise TypeError(f'{context}: cam_info must be dict, got {type(cam_info)}')
        rotation = cam_info.get('sensor2global_rotation', None)
        translation = cam_info.get('sensor2global_translation', None)
        if rotation is None or translation is None:
            raise KeyError(f'{context}: missing sensor2global rotation/translation')

        rot = self._validate_rotation_matrix(rotation, context=context)
        trans = np.asarray(translation, dtype=np.float64).reshape(-1)
        if trans.size != 3:
            raise ValueError(
                f'{context}: sensor2global_translation must have 3 values, got shape={trans.shape}')
        if not np.all(np.isfinite(trans)):
            raise ValueError(f'{context}: sensor2global_translation contains non-finite values')

        cam2world = np.eye(4, dtype=np.float32)
        cam2world[:3, :3] = rot.astype(np.float32)
        cam2world[:3, 3] = trans.astype(np.float32)
        return cam2world

    def __call__(self, results):
        filenames = results.get('filename', [])
        imgs = results.get('img', [])
        if not isinstance(filenames, list):
            filenames = list(filenames)
        if not isinstance(imgs, list):
            imgs = list(imgs)
        if len(filenames) != len(imgs):
            raise ValueError(
                f'filename/img length mismatch: len(filename)={len(filenames)} vs len(img)={len(imgs)}')
        if len(filenames) == 0:
            raise ValueError('Empty filename list: cannot build mapanything_extra')

        cam_lookup = self._build_cam_lookup(results)
        if self.strict and len(cam_lookup) == 0:
            raise ValueError('No camera metadata found in results (expected cams/cam_sweeps)')

        ida_aug_params = results.get('ida_aug_params', None)
        views = []
        for idx, (image_path, img) in enumerate(zip(filenames, imgs)):
            context = f'view[{idx}] image={image_path}'
            cam_info = self._find_cam_info(cam_lookup, image_path)
            if cam_info is None:
                raise FileNotFoundError(f'{context}: failed to match camera metadata by path')

            depth_path = self._resolve_depth_path(cam_info, image_path)
            if depth_path is None or (not osp.exists(depth_path)):
                raise FileNotFoundError(f'{context}: depth file does not exist ({depth_path})')
            depth = self._load_depth(depth_path)
            if depth is None:
                raise FileNotFoundError(f'{context}: failed to decode depth from {depth_path}')

            depth = self._apply_ida_to_depth(depth, ida_aug_params)
            target_hw = self._image_hw(img)
            if depth.shape != target_hw:
                raise ValueError(
                    f'{context}: depth/image shape mismatch after augmentation, '
                    f'depth={depth.shape}, image={target_hw}')
            depth = np.ascontiguousarray(depth, dtype=np.float32)

            intrinsics = np.asarray(cam_info.get('cam_intrinsic', None), dtype=np.float32)
            self._validate_intrinsics(intrinsics, context=context)
            intrinsics = self._apply_ida_to_intrinsics(intrinsics, ida_aug_params)
            self._validate_intrinsics(intrinsics, context=context)

            camera_pose = self._build_camera_pose(cam_info, context=context)
            if not np.all(np.isfinite(camera_pose)):
                raise ValueError(f'{context}: camera pose contains non-finite values')

            depth = self._filter_depth_by_point_cloud_range(
                depth=depth,
                intrinsics=intrinsics,
                camera_pose=camera_pose,
                context=context)
            self._validate_depth(depth, context=context)

            views.append(dict(
                depth_z=depth,
                intrinsics=np.ascontiguousarray(intrinsics, dtype=np.float32),
                camera_poses=np.ascontiguousarray(camera_pose, dtype=np.float32),
                is_metric_scale=True,
            ))

        if len(views) != len(imgs):
            raise ValueError(
                f'mapanything_extra views length mismatch: expected {len(imgs)}, got {len(views)}')
        results['mapanything_extra'] = dict(views=views)
        return results


# Repalce LoadPointsFromMultiSweeps in mmdet3d to adapt sparsebev data
@TRANSFORMS.register_module(force=True)
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        sweeps = results['lidar_sweeps']['prev']
        if self.pad_empty_sweeps and len(sweeps) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(sweeps) <= self.sweeps_num:
                choices = np.arange(len(sweeps))
            else:
                choices = np.arange(self.sweeps_num)

            for idx in choices:
                sweep = sweeps[idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@TRANSFORMS.register_module()
class LiDARToOccSpace:
    
    def __call__(self, results):
        points = results['points']
        ego2lidar, ego2occ = results['ego2lidar'], results['ego2occ']

        lidar2ego = torch.tensor(inv(ego2lidar)).float()
        lidar2occ = torch.tensor(ego2occ @ lidar2ego.numpy()).float()
        ones = torch.ones_like(points.tensor[..., :1])
        pts = torch.cat([points.tensor[..., :3], ones], dim=1).transpose(0, 1)
        pts = torch.matmul(lidar2occ, pts).transpose(0, 1)[...,:3]

        points.tensor = torch.cat([pts, points.tensor[..., 3:]], dim=1)
        results['points'] = points
        results['ego2lidar'] = ego2occ.copy()
        return results


@TRANSFORMS.register_module()
class ObjectToOccSpace:
    
    def __call__(self, results):
        ego2occ, ego2obj = results['ego2occ'], results['ego2obj']
        if np.array_equal(ego2occ, ego2obj):
            del results['ego2obj']
            return results
        
        boxes = results['gt_bboxes_3d']
        matrix = torch.from_numpy(ego2occ @ inv(ego2obj)).float()

        ctr, dims, yaw = boxes.center, boxes.dims, boxes.yaw
        velo = boxes.tensor[:, 7:9] if boxes.box_dim > 7 else torch.zeros_like(ctr[:, :2])

        ones = torch.ones_like(ctr[:, :1])
        ctr_ = torch.cat([ctr, ones], dim=-1)
        ctr_ = (matrix @ ctr_.unsqueeze(-1)).squeeze(-1)[:, :3]

        rot_matrix = matrix[:3, :3]
        yaw_ = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        yaw_ = (rot_matrix @ yaw_.unsqueeze(-1)).squeeze(-1)
        yaw_ = torch.atan2(yaw_[:, [1]], yaw_[:, [0]])

        velo_ = torch.cat([velo, torch.zeros_like(velo[:, :1])], dim=-1)
        velo_ = (rot_matrix @ velo_.unsqueeze(-1)).squeeze(-1)[:, :2]

        box_tensor = torch.cat([ctr_, dims, yaw_, velo_], dim=-1)
        boxes = type(boxes)(box_tensor, box_dim=boxes.box_dim, with_yaw=boxes.with_yaw)
        results['gt_bboxes_3d'] = boxes
        del results['ego2obj'] # objects and occupancy are in the same space now
        return results
