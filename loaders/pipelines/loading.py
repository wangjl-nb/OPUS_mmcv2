import copy
import os
import mmcv
import torch
import numpy as np
import os.path as osp
from numpy.linalg import inv
from mmengine.dist import get_dist_info
from mmengine.fileio import FileClient, get
from mmengine.utils import check_file_exist
try:
    from mmdet3d.structures.points import BasePoints
except Exception:  # pragma: no cover
    from mmdet3d.core.points import BasePoints
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
