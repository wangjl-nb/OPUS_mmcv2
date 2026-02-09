import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import copy
import importlib
import json
import os.path as osp
from datetime import datetime

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS


TRAIN_USED_FIELDS = [
    'inputs.img',
    'inputs.points',
    'data_samples.voxel_semantics',
    'data_samples.mask_camera',
    'data_samples.metainfo.ego2img',
    'data_samples.metainfo.ego2occ',
    'data_samples.metainfo.ego2lidar',
    'data_samples.metainfo.img_shape',
]


def write_ply(path, xyz, rgb=None, labels=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'xyz must have shape [N, 3], got {xyz.shape}')

    n = xyz.shape[0]
    cols = [xyz]
    fmt = ['%.4f', '%.4f', '%.4f']
    header_props = [
        'property float x',
        'property float y',
        'property float z',
    ]

    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.shape != (n, 3):
            raise ValueError(f'rgb must have shape [N, 3], got {rgb.shape}')
        cols.append(rgb)
        fmt += ['%d', '%d', '%d']
        header_props += [
            'property uchar red',
            'property uchar green',
            'property uchar blue',
        ]

    if labels is not None:
        labels = np.asarray(labels, dtype=np.int32).reshape(-1, 1)
        if labels.shape[0] != n:
            raise ValueError(f'labels must have length N={n}, got {labels.shape[0]}')
        cols.append(labels)
        fmt += ['%d']
        header_props += ['property int label']

    data = np.concatenate(cols, axis=1)
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        for prop in header_props:
            f.write(f'{prop}\n')
        f.write('end_header\n')
        np.savetxt(f, data, fmt=' '.join(fmt))


def build_palette(num_classes):
    # Deterministic pseudo-random palette, keeps label colors stable across runs.
    rng = np.random.default_rng(seed=2026)
    palette = rng.integers(0, 256, size=(max(1, num_classes), 3), dtype=np.uint8)
    palette[0] = np.array([180, 180, 180], dtype=np.uint8)
    return palette


def voxel_indices_to_xyz(indices, pc_range, voxel_size):
    indices = indices.astype(np.float32)
    x = (indices[:, 0] + 0.5) * voxel_size[0] + pc_range[0]
    y = (indices[:, 1] + 0.5) * voxel_size[1] + pc_range[1]
    z = (indices[:, 2] + 0.5) * voxel_size[2] + pc_range[2]
    return np.stack([x, y, z], axis=1)


def points_to_rgb(points):
    # Colorize raw points by height (z) to make structure easier to inspect.
    z = points[:, 2]
    z_min = float(np.min(z)) if z.size > 0 else 0.0
    z_max = float(np.max(z)) if z.size > 0 else 1.0
    denom = max(z_max - z_min, 1e-6)
    t = np.clip((z - z_min) / denom, 0.0, 1.0)
    rgb = np.stack([
        (255.0 * t),
        (255.0 * (1.0 - np.abs(2.0 * t - 1.0))),
        (255.0 * (1.0 - t)),
    ], axis=1)
    return rgb.astype(np.uint8)


def parse_indices(args, dataset_len):
    if args.sample_indices:
        indices = [int(i) for i in args.sample_indices]
    else:
        start = max(0, args.start_idx)
        end = min(dataset_len, start + args.num_samples)
        indices = list(range(start, end))

    if args.random_pick > 0:
        rng = np.random.default_rng(args.seed)
        if args.random_pick >= dataset_len:
            return list(range(dataset_len))
        indices = rng.choice(dataset_len, size=args.random_pick, replace=False).tolist()

    valid = [i for i in indices if 0 <= i < dataset_len]
    return sorted(set(valid))


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def register_modules_safely():
    from mmdet3d.utils import register_all_modules

    try:
        register_all_modules(init_default_scope=True)
    except KeyError as exc:
        # Custom loader names may collide with mmdet3d defaults.
        if 'LoadMultiViewImageFromFiles' not in str(exc):
            raise
        init_default_scope('mmdet3d')

    importlib.import_module('models')
    importlib.import_module('loaders')


def build_dataset_from_cfg(cfg, split):
    dataloader_key = f'{split}_dataloader'
    if dataloader_key not in cfg:
        raise KeyError(f'Config missing {dataloader_key}')

    dataset_cfg = copy.deepcopy(cfg[dataloader_key].dataset)
    for step in dataset_cfg.pipeline:
        if step.get('type') == 'LoadMultiViewImageFromMultiSweeps':
            step['force_offline'] = True
    return DATASETS.build(dataset_cfg)


def export_sample(dataset, cfg, index, save_dir, args, palette):
    packed = dataset[index]
    inputs = packed['inputs']
    data_sample = packed['data_samples']
    meta = dict(data_sample.metainfo)

    sample_idx = int(meta.get('sample_idx', index))
    sample_token = str(meta.get('sample_token', sample_idx))
    scene_name = str(meta.get('scene_name', 'unknown_scene'))
    prefix = f'{sample_idx:06d}_{scene_name}_{sample_token}'

    points = tensor_to_numpy(inputs['points'])
    point_xyz = points[:, :3]
    if args.max_points > 0 and point_xyz.shape[0] > args.max_points:
        choose = np.random.default_rng(args.seed + index).choice(
            point_xyz.shape[0], size=args.max_points, replace=False)
        point_xyz = point_xyz[choose]
    point_rgb = points_to_rgb(point_xyz)
    points_path = osp.join(save_dir, f'{prefix}_train_points.ply')
    write_ply(points_path, point_xyz, rgb=point_rgb)

    voxel_semantics = tensor_to_numpy(getattr(data_sample, 'voxel_semantics'))
    mask_camera = tensor_to_numpy(getattr(data_sample, 'mask_camera'))
    mask_lidar = tensor_to_numpy(getattr(data_sample, 'mask_lidar')) \
        if hasattr(data_sample, 'mask_lidar') else None

    num_classes = int(cfg.model['pts_bbox_head']['num_classes'])
    empty_label = int(cfg.model['pts_bbox_head'].get('empty_label', num_classes))
    pc_range = np.asarray(cfg.point_cloud_range, dtype=np.float32)
    voxel_size = np.asarray(cfg.voxel_size, dtype=np.float32)

    valid = voxel_semantics != empty_label
    if args.mask_mode == 'camera':
        valid = np.logical_and(valid, mask_camera.astype(np.bool_))
    elif args.mask_mode == 'lidar' and mask_lidar is not None:
        valid = np.logical_and(valid, mask_lidar.astype(np.bool_))

    occ_indices = np.argwhere(valid)
    occ_labels = voxel_semantics[valid].astype(np.int32)

    if args.max_voxels > 0 and occ_indices.shape[0] > args.max_voxels:
        choose = np.random.default_rng(args.seed + 999 + index).choice(
            occ_indices.shape[0], size=args.max_voxels, replace=False)
        occ_indices = occ_indices[choose]
        occ_labels = occ_labels[choose]

    occ_xyz = voxel_indices_to_xyz(occ_indices, pc_range, voxel_size)
    occ_rgb = palette[np.clip(occ_labels, 0, palette.shape[0] - 1)]
    occ_path = osp.join(save_dir, f'{prefix}_train_occ_gt.ply')
    write_ply(occ_path, occ_xyz, rgb=occ_rgb, labels=occ_labels)

    np.savez_compressed(
        osp.join(save_dir, f'{prefix}_train_metas.npz'),
        ego2img=np.asarray(meta.get('ego2img', []), dtype=np.float32),
        ego2occ=np.asarray(meta.get('ego2occ', []), dtype=np.float32),
        ego2lidar=np.asarray(meta.get('ego2lidar', []), dtype=np.float32),
    )

    summary = {
        'sample_idx': sample_idx,
        'sample_token': sample_token,
        'scene_name': scene_name,
        'used_training_fields': TRAIN_USED_FIELDS,
        'point_count_saved': int(point_xyz.shape[0]),
        'occ_count_saved': int(occ_xyz.shape[0]),
        'img_shape': list(getattr(inputs.get('img'), 'shape', [])),
        'points_shape': list(getattr(inputs.get('points'), 'shape', [])),
        'voxel_semantics_shape': list(voxel_semantics.shape),
        'mask_mode': args.mask_mode,
        'files': {
            'points_ply': osp.basename(points_path),
            'occ_ply': osp.basename(occ_path),
            'metas_npz': f'{prefix}_train_metas.npz',
        },
    }
    with open(osp.join(save_dir, f'{prefix}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Export train/val/test samples to 3D point clouds using training-used fields')
    parser.add_argument('--config', required=True, help='Config path')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--save-dir', default='train_field_clouds', help='Output root directory')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of sequential samples')
    parser.add_argument('--start-idx', type=int, default=0, help='Start index for sequential export')
    parser.add_argument('--sample-indices', nargs='+', type=int,
                        help='Explicit dataset indices to export')
    parser.add_argument('--random-pick', type=int, default=0,
                        help='Randomly choose N samples (overrides --num-samples)')
    parser.add_argument('--max-points', type=int, default=300000,
                        help='Max raw lidar points to save per sample')
    parser.add_argument('--max-voxels', type=int, default=300000,
                        help='Max occupancy voxels to save per sample')
    parser.add_argument('--mask-mode', choices=['none', 'camera', 'lidar'], default='camera',
                        help='Mask type for occupancy voxels')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--override', nargs='+', action=DictAction, help='Config override')
    args = parser.parse_args()

    np.random.seed(args.seed)

    register_modules_safely()

    cfg = Config.fromfile(args.config)
    if args.override is not None:
        cfg.merge_from_dict(args.override)

    dataset = build_dataset_from_cfg(cfg, args.split)
    indices = parse_indices(args, len(dataset))
    if not indices:
        raise ValueError('No valid sample indices selected.')

    run_name = osp.splitext(osp.basename(args.config))[0]
    run_name = f'{run_name}_{args.split}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    save_dir = osp.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    num_classes = int(cfg.model['pts_bbox_head']['num_classes'])
    palette = build_palette(num_classes)

    all_summaries = []
    for idx in indices:
        summary = export_sample(dataset, cfg, idx, save_dir, args, palette)
        all_summaries.append(summary)
        print(
            f"[Export] idx={summary['sample_idx']} token={summary['sample_token']} "
            f"points={summary['point_count_saved']} occ={summary['occ_count_saved']}")

    with open(osp.join(save_dir, 'export_manifest.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f'[Done] Saved {len(all_summaries)} samples to: {save_dir}')


if __name__ == '__main__':
    main()
