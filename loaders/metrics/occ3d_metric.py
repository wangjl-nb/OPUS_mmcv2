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


def _ordered_results(results, total_size):
    ordered = [None] * total_size
    for item in results:
        idx = item.get('sample_idx', None)
        if idx is not None and 0 <= idx < total_size:
            ordered[idx] = item
    if any(v is None for v in ordered):
        return list(results)
    return ordered


def _info_with_aliases(info):
    info = dict(info)
    if 'token' in info:
        info.setdefault('sample_token', info['token'])
    if 'sample_token' in info:
        info.setdefault('token', info['sample_token'])
    return info


def _build_occ_path(root, path_template, info):
    info = _info_with_aliases(info)
    return osp.join(root, path_template.format(**info))


def _safe_voxel_size(voxel_size):
    if voxel_size is None:
        return None
    voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(-1)
    if voxel_size.size == 1:
        return float(voxel_size[0])
    if voxel_size.size == 3:
        if np.max(voxel_size) - np.min(voxel_size) < 1e-8:
            return float(voxel_size[0])
        return voxel_size.tolist()
    raise ValueError(f'Unsupported voxel_size shape: {voxel_size.shape}')


def _compute_hist(gt, pred, num_classes, mask=None):
    gt = np.asarray(gt, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)

    if mask is not None:
        mask = np.asarray(mask, dtype=np.bool_)
        gt = gt[mask]
        pred = pred[mask]

    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    return np.bincount(
        num_classes * gt[valid] + pred[valid], minlength=num_classes**2
    ).reshape(num_classes, num_classes)


def _compute_miou(hist, empty_label):
    hist = np.asarray(hist, dtype=np.float64)
    denom = hist.sum(1) + hist.sum(0) - np.diag(hist)
    iou = np.divide(
        np.diag(hist), denom,
        out=np.full(hist.shape[0], np.nan, dtype=np.float64),
        where=denom > 0,
    )

    if empty_label is not None and 0 <= empty_label < hist.shape[0]:
        valid_indices = [i for i in range(hist.shape[0]) if i != empty_label]
        mean_iou = np.nanmean(iou[valid_indices])
    else:
        mean_iou = np.nanmean(iou)
    return round(float(mean_iou) * 100.0, 2)


def _compute_occupied_iou(hist, empty_label):
    hist = np.asarray(hist)
    if hist.ndim != 2 or hist.shape[0] != hist.shape[1]:
        return np.nan
    num_classes = hist.shape[0]
    if empty_label is None or empty_label < 0 or empty_label >= num_classes:
        return np.nan

    occ_mask = np.ones(num_classes, dtype=bool)
    occ_mask[empty_label] = False
    tp = hist[np.ix_(occ_mask, occ_mask)].sum()
    fp = hist[empty_label, occ_mask].sum()
    fn = hist[occ_mask, empty_label].sum()
    denom = tp + fp + fn
    if denom <= 0:
        return np.nan
    return float(tp) / float(denom)


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
                 occ_path_template='{scene_name}/{token}/labels.npz',
                 semantics_key='semantics',
                 mask_camera_key='mask_camera',
                 mask_lidar_key='mask_lidar',
                 ray_num_workers=8,
                 ray_cfg=None,
                 collect_device='cpu',
                 prefix=None,
                 collect_dir=None):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.ann_file = ann_file
        self.occ_root = occ_root
        self.empty_label = int(empty_label)
        self.use_camera_mask = use_camera_mask
        self.compute_rayiou = compute_rayiou
        self.pc_range = pc_range
        self.voxel_size = _safe_voxel_size(voxel_size)
        self.class_names = class_names
        self.miou_num_workers = miou_num_workers

        self.occ_path_template = occ_path_template
        self.semantics_key = semantics_key
        self.mask_camera_key = mask_camera_key
        self.mask_lidar_key = mask_lidar_key

        self.ray_num_workers = ray_num_workers
        self.ray_cfg = ray_cfg or {}

    def process(self, data_batch, data_samples):
        batch_samples = data_batch.get('data_samples', []) if isinstance(data_batch, dict) else []
        for idx, pred in enumerate(data_samples):
            sample = batch_samples[idx] if idx < len(batch_samples) else None
            meta = getattr(sample, 'metainfo', {}) if sample is not None else {}
            self.results.append(dict(
                pred=pred,
                sample_idx=meta.get('sample_idx', None),
                sample_token=meta.get('sample_token', None),
                scene_name=meta.get('scene_name', None),
            ))

    def compute_metrics(self, results):
        data_infos = _load_infos(self.ann_file)
        ordered = _ordered_results(results, len(data_infos))

        num_classes = self.empty_label + 1
        if self.class_names is not None:
            num_classes = max(num_classes, len(self.class_names))

        hist = np.zeros((num_classes, num_classes), dtype=np.float64)
        dense_preds = [None] * len(data_infos)
        dense_gts = [None] * len(data_infos)

        hist_inputs = []
        for idx, info in enumerate(data_infos):
            if idx >= len(ordered):
                break
            result_dict = ordered[idx]['pred']
            occ_file = _build_occ_path(self.occ_root, self.occ_path_template, info)
            occ_infos = np.load(occ_file)

            occ_labels = np.asarray(occ_infos[self.semantics_key], dtype=np.int64)
            mask_camera = np.asarray(
                occ_infos[self.mask_camera_key], dtype=np.bool_) if self.mask_camera_key in occ_infos \
                else np.ones_like(occ_labels, dtype=np.bool_)
            mask_lidar = np.asarray(
                occ_infos[self.mask_lidar_key], dtype=np.bool_) if self.mask_lidar_key in occ_infos \
                else np.ones_like(occ_labels, dtype=np.bool_)

            occ_pred, _ = sparse2dense(
                result_dict['occ_loc'],
                result_dict['sem_pred'],
                dense_shape=occ_labels.shape,
                empty_value=self.empty_label)
            occ_pred = np.asarray(occ_pred, dtype=np.int64)

            dense_preds[idx] = occ_pred
            dense_gts[idx] = occ_labels

            mask = mask_camera if self.use_camera_mask else None
            hist_inputs.append((occ_labels, occ_pred, mask_lidar, mask_camera, mask))

        def _hist_worker(args):
            occ_labels, occ_pred, _mask_lidar, _mask_camera, mask = args
            return _compute_hist(occ_labels, occ_pred, num_classes, mask=mask)

        if self.miou_num_workers and self.miou_num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.miou_num_workers) as executor:
                for sample_hist in executor.map(_hist_worker, hist_inputs):
                    hist += sample_hist
        else:
            for item in hist_inputs:
                hist += _hist_worker(item)

        metrics = {
            'mIoU': _compute_miou(hist, self.empty_label),
            'IoU': round(_compute_occupied_iou(hist, self.empty_label) * 100, 2),
        }

        if not self.compute_rayiou:
            return metrics

        max_origins = self.ray_cfg.get('max_origins', 8)
        origin_xy_bound = self.ray_cfg.get('origin_xy_bound', 39.0)
        data_loader = DataLoader(
            EgoPoseDataset(data_infos, max_origins=max_origins, origin_xy_bound=origin_xy_bound),
            batch_size=1,
            shuffle=False,
            num_workers=self.ray_num_workers,
        )

        token_to_idx = {
            _info_with_aliases(info).get('token'): i
            for i, info in enumerate(data_infos)
        }

        lidar_origins = []
        ray_occ_preds = []
        ray_occ_gts = []

        for batch_idx, batch in enumerate(data_loader):
            token = batch[0][0]
            output_origin = batch[1]

            data_id = token_to_idx.get(token, batch_idx)
            if data_id >= len(dense_preds):
                continue
            if dense_preds[data_id] is None or dense_gts[data_id] is None:
                continue

            lidar_origins.append(output_origin)
            ray_occ_preds.append(dense_preds[data_id])
            ray_occ_gts.append(dense_gts[data_id])

        if ray_occ_preds and ray_occ_gts and lidar_origins:
            ray_metrics = calc_rayiou_custom(
                ray_occ_preds,
                ray_occ_gts,
                lidar_origins,
                pc_range=self.pc_range,
                voxel_size=self.voxel_size,
                class_names=self.class_names,
                empty_label=self.empty_label,
                ray_cfg=self.ray_cfg)
            metrics.update(ray_metrics)

        return metrics


@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    default_prefix = 'occ'

    def __init__(self,
                 ann_file,
                 occ_root,
                 empty_label=16,
                 occ_path_template='scene_{scene_token}/occupancy/{lidar_token}.npy',
                 src_class_names=None,
                 ignore_class_names=None,
                 class_names=None,
                 pc_range=None,
                 voxel_size=None,
                 collect_device='cpu',
                 prefix=None,
                 collect_dir=None):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.ann_file = ann_file
        self.occ_root = occ_root
        self.empty_label = int(empty_label)

        self.occ_path_template = occ_path_template
        self.src_class_names = src_class_names or [
            'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'
        ]
        self.ignore_class_names = ignore_class_names or ['noise']
        self.class_names = class_names

        self.pc_range = np.asarray(
            pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3],
            dtype=np.float32)
        self.voxel_size = np.asarray(
            voxel_size if voxel_size is not None else [0.2, 0.2, 0.2],
            dtype=np.float32)

    def process(self, data_batch, data_samples):
        batch_samples = data_batch.get('data_samples', []) if isinstance(data_batch, dict) else []
        for idx, pred in enumerate(data_samples):
            sample = batch_samples[idx] if idx < len(batch_samples) else None
            meta = getattr(sample, 'metainfo', {}) if sample is not None else {}
            self.results.append(dict(
                pred=pred,
                sample_idx=meta.get('sample_idx', None),
                scene_token=meta.get('scene_token', None),
                lidar_token=meta.get('lidar_token', None),
            ))

    def compute_metrics(self, results):
        data_infos = _load_infos(self.ann_file)
        ordered = _ordered_results(results, len(data_infos))

        scene_size = self.pc_range[3:] - self.pc_range[:3]
        voxel_num = (scene_size / self.voxel_size).astype(np.int64)

        raw_empty_label = len(self.src_class_names)
        mapper = np.full(raw_empty_label + 1, self.empty_label, dtype=np.int64)

        active_class_names = [
            name for name in self.src_class_names
            if name not in self.ignore_class_names
        ]
        class_to_idx = {name: idx for idx, name in enumerate(active_class_names)}
        ignore_label_ids = []
        for src_idx, src_name in enumerate(self.src_class_names):
            if src_name in class_to_idx:
                mapper[src_idx] = class_to_idx[src_name]
            else:
                ignore_label_ids.append(src_idx)
        mapper[raw_empty_label] = self.empty_label

        num_classes = max(self.empty_label + 1, np.max(mapper) + 1)
        hist = np.zeros((num_classes, num_classes), dtype=np.float64)

        for idx, info in enumerate(data_infos):
            if idx >= len(ordered):
                break
            result_dict = ordered[idx]['pred']

            occ_file = _build_occ_path(self.occ_root, self.occ_path_template, info)
            occ_labels = np.load(occ_file)
            coors, labels = occ_labels[:, :3], occ_labels[:, 3].astype(np.int64)

            raw_dense = np.full(voxel_num, raw_empty_label, dtype=np.int64)
            raw_dense[coors[:, 2], coors[:, 1], coors[:, 0]] = labels

            mask = np.ones_like(raw_dense, dtype=np.bool_)
            for ignore_label in ignore_label_ids:
                mask &= raw_dense != ignore_label

            gt_dense = mapper[raw_dense]
            pred_dense, _ = sparse2dense(
                result_dict['occ_loc'],
                result_dict['sem_pred'],
                voxel_num,
                self.empty_label)
            pred_dense = np.asarray(pred_dense, dtype=np.int64)

            hist += _compute_hist(gt_dense, pred_dense, num_classes, mask=mask)

        return {
            'mIoU': _compute_miou(hist, self.empty_label),
            'IoU': round(_compute_occupied_iou(hist, self.empty_label) * 100, 2),
        }
