import os.path as osp
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import BaseMetric
from mmengine.fileio import load
from mmengine.registry import METRICS

from models.utils import sparse2dense
from ..ego_pose_dataset import EgoPoseDataset
from ..old_metrics import Metric_mIoU_Occ3D_Custom, Metric_mIoU_Occupancy
from ..ray_metrics import main_custom as calc_rayiou_custom


def _load_infos(ann_file):
    data = load(ann_file)
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, list):
        return data
    raise TypeError(f'Unsupported annotation format: {type(data)}')


@METRICS.register_module()
class Occ3DMetric(BaseMetric):
    default_prefix = 'occ'

    def __init__(self,
                 ann_file,
                 occ_root,
                 empty_label=17,
                 use_camera_mask=True,
                 compute_rayiou=True,
                 pc_range=None,
                 voxel_size=None,
                 class_names=None,
                 miou_num_workers=0,
                 collect_device='cpu',
                 prefix=None,
                 collect_dir=None):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.ann_file = ann_file
        self.occ_root = occ_root
        self.empty_label = empty_label
        self.use_camera_mask = use_camera_mask
        self.compute_rayiou = compute_rayiou
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.class_names = class_names
        self.miou_num_workers = miou_num_workers

    def process(self, data_batch, data_samples):
        batch_samples = data_batch.get('data_samples', [])
        for pred, sample in zip(data_samples, batch_samples):
            meta = getattr(sample, 'metainfo', {}) if sample is not None else {}
            self.results.append(dict(
                pred=pred,
                sample_idx=meta.get('sample_idx', None),
                sample_token=meta.get('sample_token', None),
                scene_name=meta.get('scene_name', None),
            ))

    def compute_metrics(self, results):
        data_infos = _load_infos(self.ann_file)
        ordered = [None] * len(data_infos)
        for item in results:
            idx = item.get('sample_idx', None)
            if idx is not None and 0 <= idx < len(ordered):
                ordered[idx] = item
        if any(v is None for v in ordered):
            ordered = results

        metric = Metric_mIoU_Occ3D_Custom(
            num_classes=self.empty_label + 1,
            class_names=self.class_names,
            empty_label=self.empty_label,
            use_image_mask=self.use_camera_mask)
        occ_preds = []
        occ_gts = []

        def _compute_hist_for_sample(sample):
            occ_pred, occ_labels, mask_lidar, mask_camera = sample
            if self.use_camera_mask:
                masked_gt = occ_labels[mask_camera]
                masked_pred = occ_pred[mask_camera]
            else:
                masked_gt = occ_labels
                masked_pred = occ_pred

            pred = masked_pred.astype(np.int64).ravel()
            gt = masked_gt.astype(np.int64).ravel()
            num_classes = metric.num_classes
            k = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
            return np.bincount(
                num_classes * gt[k] + pred[k], minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

        hist_list = []
        for idx, info in enumerate(data_infos):
            result_dict = ordered[idx]['pred']
            scene_name = info['scene_name']
            token = info['token']
            occ_file = osp.join(self.occ_root, scene_name, token, 'labels.npz')
            occ_infos = np.load(occ_file)
            occ_labels = occ_infos['semantics']
            mask_lidar = occ_infos['mask_lidar'].astype(np.bool_)
            mask_camera = occ_infos['mask_camera'].astype(np.bool_)

            occ_pred, _ = sparse2dense(
                result_dict['occ_loc'],
                result_dict['sem_pred'],
                dense_shape=occ_labels.shape,
                empty_value=self.empty_label)

            occ_preds.append(occ_pred)
            occ_gts.append(occ_labels)
            hist_list.append((occ_pred, occ_labels, mask_lidar, mask_camera))

        if self.miou_num_workers and self.miou_num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.miou_num_workers) as executor:
                for hist in executor.map(_compute_hist_for_sample, hist_list):
                    metric.hist += hist
            metric.cnt = len(hist_list)
        else:
            for occ_pred, occ_labels, mask_lidar, mask_camera in hist_list:
                metric.add_batch(occ_pred, occ_labels, mask_lidar, mask_camera)

        metrics = {'mIoU': metric.count_miou()}

        if not self.compute_rayiou:
            return metrics

        data_loader = DataLoader(
            EgoPoseDataset(data_infos),
            batch_size=1,
            shuffle=False,
            num_workers=8,
        )
        sample_tokens = [info['token'] for info in data_infos]
        lidar_origins = []
        ray_occ_preds = []
        ray_occ_gts = []

        for batch in data_loader:
            token = batch[0][0]
            output_origin = batch[1]

            data_id = sample_tokens.index(token)
            info = data_infos[data_id]

            scene_name = info['scene_name']
            occ_file = osp.join(self.occ_root, scene_name, token, 'labels.npz')
            occ_infos = np.load(occ_file)
            gt_semantics = occ_infos['semantics']

            pred = ordered[data_id]['pred']
            sem_pred = torch.from_numpy(pred['sem_pred'])
            occ_loc = torch.from_numpy(pred['occ_loc'].astype(np.int64))
            occ_size = list(gt_semantics.shape)
            dense_sem_pred, _ = sparse2dense(
                occ_loc, sem_pred, dense_shape=occ_size, empty_value=self.empty_label)
            dense_sem_pred = dense_sem_pred.squeeze(0).numpy()

            lidar_origins.append(output_origin)
            ray_occ_preds.append(dense_sem_pred)
            ray_occ_gts.append(gt_semantics)

        ray_metrics = calc_rayiou_custom(
            ray_occ_preds,
            ray_occ_gts,
            lidar_origins,
            pc_range=self.pc_range,
            voxel_size=self.voxel_size,
            class_names=self.class_names,
            empty_label=self.empty_label)
        metrics.update(ray_metrics)
        return metrics


@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    default_prefix = 'occ'

    def __init__(self,
                 ann_file,
                 occ_root,
                 empty_label=16,
                 collect_device='cpu',
                 prefix=None,
                 collect_dir=None):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.ann_file = ann_file
        self.occ_root = occ_root
        self.empty_label = empty_label

    def process(self, data_batch, data_samples):
        batch_samples = data_batch.get('data_samples', [])
        for pred, sample in zip(data_samples, batch_samples):
            meta = getattr(sample, 'metainfo', {}) if sample is not None else {}
            self.results.append(dict(
                pred=pred,
                sample_idx=meta.get('sample_idx', None),
                scene_token=meta.get('scene_token', None),
                lidar_token=meta.get('lidar_token', None),
            ))

    def compute_metrics(self, results):
        data_infos = _load_infos(self.ann_file)
        ordered = [None] * len(data_infos)
        for item in results:
            idx = item.get('sample_idx', None)
            if idx is not None and 0 <= idx < len(ordered):
                ordered[idx] = item
        if any(v is None for v in ordered):
            ordered = results

        metric = Metric_mIoU_Occupancy()
        occ_class_names = [
            'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'
        ]
        ignore_class_names = ['noise']
        pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3])
        voxel_size = np.array([0.2, 0.2, 0.2])
        voxel_num = ((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int64)

        for idx, info in enumerate(data_infos):
            result_dict = ordered[idx]['pred']
            scene_token = info['scene_token']
            lidar_token = info['lidar_token']
            occ_file = osp.join(self.occ_root, f'scene_{scene_token}', 'occupancy', f'{lidar_token}.npy')
            occ_labels = np.load(occ_file)
            coors, labels = occ_labels[:, :3], occ_labels[:, 3]
            occ_labels, _ = sparse2dense(coors[:, ::-1], labels, voxel_num,
                                         empty_value=len(occ_class_names))
            mask = occ_labels != 0

            curr_class_names = [n for n in occ_class_names if n not in ignore_class_names]
            curr_bg_class_idx = len(curr_class_names)
            label_mapper = [curr_class_names.index(n) if n in curr_class_names else curr_bg_class_idx
                            for n in occ_class_names] + [curr_bg_class_idx]
            label_mapper = np.array(label_mapper)
            occ_labels = label_mapper[occ_labels]

            occ_pred, _ = sparse2dense(
                result_dict['occ_loc'], result_dict['sem_pred'], voxel_num, self.empty_label)
            metric.add_batch(occ_pred, occ_labels, mask)

        return {'mIoU': metric.count_miou()}
