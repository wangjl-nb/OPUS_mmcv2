import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import importlib
import json
import os.path as osp
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset


def quaternion_to_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f'Quaternion must have 4 elements, got {q.shape[0]}')

    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError('Zero-norm quaternion is not valid')

    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def build_ego2global(info):
    if info is None:
        return np.eye(4, dtype=np.float64)

    translation = info.get('ego2global_translation', None)
    rotation = info.get('ego2global_rotation', None)
    if translation is None or rotation is None:
        return np.eye(4, dtype=np.float64)

    rotation_arr = np.asarray(rotation)
    if rotation_arr.shape == (3, 3):
        rotation_mat = rotation_arr.astype(np.float64)
    else:
        rotation_mat = quaternion_to_matrix(rotation)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_mat
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def ensure_ego2occ(meta):
    matrix = np.asarray(meta.get('ego2occ', np.eye(4)), dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.size == 16:
        return matrix.reshape(4, 4)
    raise ValueError(f'Invalid ego2occ shape: {matrix.shape}')


def occ_idx_to_xyz(occ_loc, pc_range, voxel_size):
    occ_loc = np.asarray(occ_loc, dtype=np.float64)
    return (occ_loc + 0.5) * voxel_size[None, :] + pc_range[None, :3]


def xyz_to_occ_idx(xyz, pc_range, voxel_size):
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.floor((xyz - pc_range[None, :3]) / voxel_size[None, :]).astype(np.int64)


def _inside_voxel_bounds(coords, voxel_shape):
    return (
        (coords[:, 0] >= 0) & (coords[:, 0] < voxel_shape[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < voxel_shape[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < voxel_shape[2])
    )


def map_occ_between_frames(occ_loc, transform_src_occ_to_tgt_occ, pc_range, voxel_size, voxel_shape):
    if occ_loc.size == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.bool_)

    src_xyz = occ_idx_to_xyz(occ_loc, pc_range, voxel_size)
    src_xyz_h = np.concatenate(
        [src_xyz, np.ones((src_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    tgt_xyz = (transform_src_occ_to_tgt_occ @ src_xyz_h.T).T[:, :3]

    tgt_occ_idx = xyz_to_occ_idx(tgt_xyz, pc_range, voxel_size)
    inside = _inside_voxel_bounds(tgt_occ_idx, voxel_shape)
    return tgt_occ_idx[inside], inside


def _coords_to_flat_ids(coords, voxel_shape):
    yz = voxel_shape[1] * voxel_shape[2]
    return coords[:, 0] * yz + coords[:, 1] * voxel_shape[2] + coords[:, 2]


def _flat_ids_to_coords(flat_ids, voxel_shape):
    yz = voxel_shape[1] * voxel_shape[2]
    x = flat_ids // yz
    rem = flat_ids % yz
    y = rem // voxel_shape[2]
    z = rem % voxel_shape[2]
    return np.stack([x, y, z], axis=1).astype(np.int64)


def vote_voxel_labels(
        mapped_coords_list,
        mapped_labels_list,
        current_coords,
        current_labels,
        num_classes,
        voxel_shape):
    all_coords = []
    all_labels = []
    for coords, labels in zip(mapped_coords_list, mapped_labels_list):
        if coords.size == 0 or labels.size == 0:
            continue
        all_coords.append(coords)
        all_labels.append(labels)

    if not all_coords and current_coords.size > 0 and current_labels.size > 0:
        all_coords.append(np.asarray(current_coords, dtype=np.int64))
        all_labels.append(np.asarray(current_labels, dtype=np.int64))

    if not all_coords:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    merged_coords = np.concatenate(all_coords, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0).astype(np.int64, copy=False)

    valid = (merged_labels >= 0) & (merged_labels < num_classes)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    merged_coords = merged_coords[valid]
    merged_labels = merged_labels[valid]

    voxel_ids = _coords_to_flat_ids(merged_coords, voxel_shape)
    pair_keys = voxel_ids * int(num_classes) + merged_labels
    unique_pair_keys, pair_counts = np.unique(pair_keys, return_counts=True)

    pair_voxel_ids = unique_pair_keys // int(num_classes)
    pair_labels = unique_pair_keys % int(num_classes)

    if current_coords.size > 0 and current_labels.size > 0:
        current_labels = np.asarray(current_labels, dtype=np.int64)
        current_valid = (current_labels >= 0) & (current_labels < num_classes)
        current_coords = current_coords[current_valid]
        current_labels = current_labels[current_valid]

        if current_coords.size > 0:
            current_voxel_ids = _coords_to_flat_ids(current_coords, voxel_shape)
            current_pair_keys = np.unique(current_voxel_ids * int(num_classes) + current_labels)
            current_pair_keys.sort()

            pos = np.searchsorted(current_pair_keys, unique_pair_keys)
            in_current = np.zeros(unique_pair_keys.shape[0], dtype=np.bool_)
            valid_pos = pos < current_pair_keys.shape[0]
            if np.any(valid_pos):
                in_current[valid_pos] = (
                    current_pair_keys[pos[valid_pos]] == unique_pair_keys[valid_pos]
                )
        else:
            in_current = np.zeros(unique_pair_keys.shape[0], dtype=np.bool_)
    else:
        in_current = np.zeros(unique_pair_keys.shape[0], dtype=np.bool_)

    scores = 2 * pair_counts.astype(np.int64) + in_current.astype(np.int64)

    # Lexicographic order: voxel asc -> score desc -> class asc.
    order = np.lexsort((pair_labels, -scores, pair_voxel_ids))
    sorted_voxel_ids = pair_voxel_ids[order]

    keep_first = np.ones(order.shape[0], dtype=np.bool_)
    keep_first[1:] = sorted_voxel_ids[1:] != sorted_voxel_ids[:-1]
    winner_index = order[keep_first]

    winner_voxel_ids = pair_voxel_ids[winner_index]
    winner_labels = pair_labels[winner_index].astype(np.int64)
    winner_coords = _flat_ids_to_coords(winner_voxel_ids, voxel_shape)
    return winner_coords, winner_labels


def sparse_to_dense(winner_coords, winner_labels, voxel_shape, empty_label, dtype):
    semantics = np.full(voxel_shape, empty_label, dtype=dtype)
    if winner_coords.size > 0:
        semantics[
            winner_coords[:, 0],
            winner_coords[:, 1],
            winner_coords[:, 2]
        ] = winner_labels.astype(dtype, copy=False)
    return semantics


def choose_dense_dtype(empty_label, num_classes):
    max_value = max(int(empty_label), int(num_classes) - 1)
    return np.uint8 if max_value <= np.iinfo(np.uint8).max else np.uint16


def build_palette(num_classes):
    rng = np.random.default_rng(seed=2026)
    palette = rng.integers(0, 256, size=(max(1, num_classes), 3), dtype=np.uint8)
    palette[0] = np.array([180, 180, 180], dtype=np.uint8)
    return palette


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
        if n > 0:
            np.savetxt(f, data, fmt=' '.join(fmt))


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    if hasattr(data, 'to'):
        return data.to(device)
    return data


def normalize_inference_inputs(inputs):
    if not isinstance(inputs, dict):
        return inputs

    img = inputs.get('img')
    if isinstance(img, list) and img and torch.is_tensor(img[0]):
        inputs['img'] = torch.stack(img, dim=0)

    return inputs


def load_occ_infos(base_dataset, scene_name, token):
    if scene_name is None or token is None:
        return None

    occ_root = getattr(base_dataset, 'occ_root', None)
    if not occ_root:
        data_root = getattr(base_dataset, 'data_root', None)
        if data_root is None:
            return None
        occ_root = osp.join(data_root, 'gts')

    dataset_cfg = getattr(base_dataset, 'dataset_cfg', {}) or {}
    occ_io_cfg = dataset_cfg.get('occ_io', {})
    occ_template = occ_io_cfg.get('path_template', '{scene_name}/{token}/labels.npz')
    occ_rel_path = occ_template.format(scene_name=scene_name, token=str(token))
    occ_file = osp.join(occ_root, occ_rel_path)
    if not osp.exists(occ_file):
        return None

    with np.load(occ_file) as occ_raw:
        return {k: occ_raw[k] for k in occ_raw.files}


def get_mask_camera(data_sample, base_dataset):
    if hasattr(data_sample, 'mask_camera'):
        mask_camera = getattr(data_sample, 'mask_camera')
        if torch.is_tensor(mask_camera):
            mask_camera = mask_camera.detach().cpu().numpy()
        else:
            mask_camera = np.asarray(mask_camera)
        return mask_camera.astype(np.bool_), None

    scene_name = data_sample.metainfo.get('scene_name', None)
    sample_token = data_sample.metainfo.get('sample_token', None)
    occ_infos = load_occ_infos(base_dataset, scene_name, sample_token)
    if occ_infos is None or 'mask_camera' not in occ_infos:
        return None, occ_infos
    return occ_infos['mask_camera'].astype(np.bool_), occ_infos


def apply_dense_mask(occ_loc, labels, dense_mask):
    if dense_mask is None or occ_loc.size == 0:
        return occ_loc, labels

    occ_idx = occ_loc.astype(np.int64)
    inside = (
        (occ_idx[:, 0] >= 0) & (occ_idx[:, 0] < dense_mask.shape[0]) &
        (occ_idx[:, 1] >= 0) & (occ_idx[:, 1] < dense_mask.shape[1]) &
        (occ_idx[:, 2] >= 0) & (occ_idx[:, 2] < dense_mask.shape[2])
    )
    keep = np.zeros(occ_idx.shape[0], dtype=np.bool_)
    if np.any(inside):
        valid_idx = occ_idx[inside]
        keep[inside] = dense_mask[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]

    return occ_loc[keep], labels[keep]


def get_data_infos(base_dataset):
    if hasattr(base_dataset, 'data_infos'):
        return base_dataset.data_infos
    if hasattr(base_dataset, 'data_list'):
        return base_dataset.data_list
    return None


def parse_args():
    try:
        from mmengine.config import DictAction
    except Exception:  # pragma: no cover
        DictAction = None

    parser = argparse.ArgumentParser(description='Multi-frame voxel aggregation inference')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--weights', required=True, help='Path to checkpoint')
    parser.add_argument('--save-dir', type=str, default='demo_outputs', help='Output directory')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='Dataset split')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--max-samples', type=int, default=-1, help='Max samples to run')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Split dataset into N shards for multi-process/multi-GPU inference')
    parser.add_argument('--shard-id', type=int, default=0,
                        help='Shard index in [0, num_shards-1]')
    parser.add_argument('--history-frames', type=int, default=5,
                        help='Number of previous frames used for aggregation')
    parser.add_argument('--output-format', choices=['ply', 'npz'], default='ply',
                        help='Prediction output format')
    parser.add_argument('--max-voxels', type=int, default=300000,
                        help='Max voxels saved to PLY per sample; <=0 keeps all')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--sample-indices', type=int, nargs='*', default=None,
                        help='Explicit dataset indices to run. If set, max-samples is ignored.')
    parser.add_argument('--random-sample', action='store_true',
                        help='Randomly choose max-samples from the split.')
    parser.add_argument('--random-train-sample', action='store_true',
                        help='Run inference on one random sample from train split')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic algorithms')
    if DictAction is None:
        parser.add_argument('--override', nargs='+', default=None,
                            help='Override config (requires mmengine DictAction at runtime)')
    else:
        parser.add_argument('--override', nargs='+', action=DictAction, help='Override config')
    return parser.parse_args()


def save_prediction(
        winner_coords,
        winner_labels,
        sample_idx,
        token,
        args,
        work_dir,
        pc_range,
        voxel_size,
        voxel_shape,
        empty_label,
        palette,
        dense_dtype):
    non_empty_voxels = int(winner_coords.shape[0])

    if args.output_format == 'npz':
        semantics = sparse_to_dense(
            winner_coords,
            winner_labels,
            voxel_shape,
            empty_label=empty_label,
            dtype=dense_dtype,
        )
        out_file = f'{sample_idx:0>6}_{token}_pred_voxel.npz'
        np.savez_compressed(osp.join(work_dir, out_file), semantics=semantics)
        return out_file, non_empty_voxels, non_empty_voxels

    save_coords = winner_coords
    save_labels = winner_labels
    if args.max_voxels > 0 and save_coords.shape[0] > args.max_voxels:
        rng = np.random.default_rng(args.seed + sample_idx)
        choice = rng.choice(save_coords.shape[0], size=args.max_voxels, replace=False)
        save_coords = save_coords[choice]
        save_labels = save_labels[choice]

    save_xyz = occ_idx_to_xyz(save_coords, pc_range, voxel_size).astype(np.float32)
    save_labels = save_labels.astype(np.int64, copy=False)
    save_rgb = palette[np.clip(save_labels, 0, palette.shape[0] - 1)]

    out_file = f'{sample_idx:0>6}_{token}_pred_voxel.ply'
    write_ply(
        osp.join(work_dir, out_file),
        save_xyz,
        rgb=save_rgb,
        labels=save_labels,
    )
    return out_file, non_empty_voxels, int(save_coords.shape[0])


def main():
    args = parse_args()

    from mmengine.config import Config
    from mmengine.dataset import DefaultSampler, pseudo_collate
    from mmengine.runner import load_checkpoint, set_random_seed
    from mmdet3d.registry import DATASETS, MODELS
    from mmdet3d.utils import register_all_modules

    if args.history_frames < 0:
        raise ValueError('--history-frames must be >= 0')

    from mmengine.registry import init_default_scope
    try:
        register_all_modules(init_default_scope=True)
    except KeyError as exc:
        if 'LoadMultiViewImageFromFiles' not in str(exc):
            raise
        # Fallback when custom transforms were registered before mmdet3d.
        init_default_scope('mmdet3d')

    cfg = Config.fromfile(args.config)
    if args.override is not None:
        cfg.merge_from_dict(args.override)

    run_name = osp.splitext(osp.split(args.config)[-1])[0]
    run_name += '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    work_dir = osp.join(args.save_dir, run_name)
    os.makedirs(work_dir, exist_ok=True)

    importlib.import_module('models')
    importlib.import_module('loaders')

    set_random_seed(args.seed, deterministic=args.deterministic)
    cudnn.benchmark = not args.deterministic

    if args.split == 'test':
        dataset_cfg = cfg.test_dataloader.dataset
    elif args.split == 'train':
        dataset_cfg = cfg.train_dataloader.dataset
    else:
        dataset_cfg = cfg.val_dataloader.dataset

    for p in dataset_cfg.pipeline:
        if p['type'] == 'LoadMultiViewImageFromMultiSweeps':
            p['force_offline'] = True

    dataset = DATASETS.build(dataset_cfg)
    base_dataset = dataset
    if hasattr(base_dataset, 'full_init'):
        base_dataset.full_init()

    if args.sample_indices is not None and len(args.sample_indices) > 0:
        if args.random_sample or args.random_train_sample:
            raise ValueError('--sample-indices cannot be combined with random sampling flags')
        total_count = len(dataset)
        indices = [i for i in args.sample_indices if 0 <= i < total_count]
        if len(indices) == 0:
            raise ValueError('No valid --sample-indices after bounds check.')
        dataset = Subset(dataset, indices)
        args.max_samples = -1
        print(f'[MultiFrameInference] Using explicit indices: {indices}')
    elif args.random_sample:
        if args.random_train_sample:
            raise ValueError('--random-sample cannot be combined with --random-train-sample')
        if args.max_samples <= 0:
            raise ValueError('--random-sample requires --max-samples > 0')
        total_count = len(dataset)
        count = min(args.max_samples, total_count)
        if count <= 0:
            raise ValueError('Empty dataset; cannot sample.')
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(total_count, size=count, replace=False).tolist()
        dataset = Subset(dataset, indices)
        args.max_samples = count
        print(f'[MultiFrameInference] Randomly sampled indices: {indices}')

    if args.random_train_sample:
        if args.split != 'train':
            raise ValueError('--random-train-sample can only be used with --split train')
        rng = np.random.default_rng(args.seed)
        rand_index = int(rng.integers(0, len(dataset)))
        print(f'[MultiFrameInference] Selected random train index: {rand_index}/{len(dataset)}')
        dataset = Subset(dataset, [rand_index])
        args.batch_size = 1
        if args.max_samples < 0 or args.max_samples > 1:
            args.max_samples = 1

    if args.num_shards < 1:
        raise ValueError('--num-shards must be >= 1')
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError('--shard-id must satisfy 0 <= shard_id < num_shards')
    if args.num_shards > 1:
        shard_indices = list(range(args.shard_id, len(dataset), args.num_shards))
        print(
            f'[MultiFrameInference] Using shard {args.shard_id}/{args.num_shards}, '
            f'samples: {len(shard_indices)}/{len(dataset)}. '
            'History is computed within this shard only.')
        dataset = Subset(dataset, shard_indices)

    sampler = DefaultSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=pseudo_collate,
        persistent_workers=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.weights, map_location=device, strict=False)
    model.to(device)
    model.eval()

    pc_range = np.asarray(cfg.point_cloud_range, dtype=np.float64)
    voxel_size = np.asarray(cfg.voxel_size, dtype=np.float64)
    num_classes = int(cfg.model['pts_bbox_head']['num_classes'])
    empty_label = int(cfg.model['pts_bbox_head'].get('empty_label', num_classes))

    dense_dtype = None
    palette = None
    if args.output_format == 'npz':
        dense_dtype = choose_dense_dtype(empty_label, num_classes)
    else:
        palette = build_palette(max(num_classes, empty_label + 1))

    scene_size = pc_range[3:] - pc_range[:3]
    voxel_shape = tuple(np.round(scene_size / voxel_size).astype(np.int64).tolist())

    data_infos = get_data_infos(base_dataset)
    if data_infos is None:
        raise RuntimeError('Failed to access dataset infos for ego2global poses.')

    history_cache = deque(maxlen=args.history_frames + 1)

    processed = 0
    camera_mask_warned = False
    missing_pose_warned = False
    records = []

    stop_early = False
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = move_to_device(data['inputs'], device)
            inputs = normalize_inference_inputs(inputs)
            data_samples = move_to_device(data['data_samples'], device)
            results = model(inputs=inputs, data_samples=data_samples, mode='predict')

            for i, result in enumerate(results):
                sample = data_samples[i]
                sample_idx = int(sample.metainfo.get('sample_idx', batch_idx))
                token = str(sample.metainfo.get('sample_token', sample_idx))
                scene_name = sample.metainfo.get('scene_name', None)

                labels = np.asarray(result['sem_pred'], dtype=np.int64)
                occ_loc = np.asarray(result['occ_loc'], dtype=np.int64)

                mask_camera, _ = get_mask_camera(sample, base_dataset)
                if mask_camera is None:
                    if not camera_mask_warned:
                        print('[MultiFrameInference] mask_camera not found, keep unmasked predictions.')
                        camera_mask_warned = True
                else:
                    occ_loc, labels = apply_dense_mask(occ_loc, labels, mask_camera)

                if scene_name is None:
                    scene_name = '__unknown_scene__'

                if history_cache and history_cache[-1]['scene_name'] != scene_name:
                    history_cache.clear()

                if 0 <= sample_idx < len(data_infos):
                    info = data_infos[sample_idx]
                else:
                    info = None

                if info is None and not missing_pose_warned:
                    print('[MultiFrameInference] Missing sample info, fallback to identity ego2global.')
                    missing_pose_warned = True

                current_e2g = build_ego2global(info)
                current_e2o = ensure_ego2occ(sample.metainfo)

                current_inside = _inside_voxel_bounds(occ_loc, voxel_shape) if occ_loc.size > 0 else np.zeros((0,), dtype=np.bool_)
                current_valid = current_inside & (labels >= 0) & (labels < num_classes)
                current_coords = occ_loc[current_valid]
                current_labels = labels[current_valid]

                history_cache.append(dict(
                    sample_idx=sample_idx,
                    token=token,
                    scene_name=scene_name,
                    occ_loc=occ_loc,
                    sem_pred=labels,
                    T_e2g=current_e2g,
                    T_e2o=current_e2o,
                ))

                mapped_coords_list = []
                mapped_labels_list = []

                inv_current_e2g = np.linalg.inv(current_e2g)
                for hist in history_cache:
                    hist_occ = hist['occ_loc']
                    hist_labels = hist['sem_pred']
                    if hist_occ.size == 0 or hist_labels.size == 0:
                        continue

                    transform_src_to_tgt = (
                        current_e2o
                        @ inv_current_e2g
                        @ hist['T_e2g']
                        @ np.linalg.inv(hist['T_e2o'])
                    )

                    mapped_coords, keep_mask = map_occ_between_frames(
                        hist_occ,
                        transform_src_to_tgt,
                        pc_range,
                        voxel_size,
                        voxel_shape,
                    )
                    if mapped_coords.size == 0:
                        continue

                    mapped_labels = hist_labels[keep_mask]
                    valid_labels = (mapped_labels >= 0) & (mapped_labels < num_classes)
                    if not np.any(valid_labels):
                        continue

                    mapped_coords_list.append(mapped_coords[valid_labels])
                    mapped_labels_list.append(mapped_labels[valid_labels].astype(np.int64, copy=False))

                winner_coords, winner_labels = vote_voxel_labels(
                    mapped_coords_list,
                    mapped_labels_list,
                    current_coords,
                    current_labels,
                    num_classes=num_classes,
                    voxel_shape=voxel_shape,
                )

                output_file, non_empty_voxels, saved_voxels = save_prediction(
                    winner_coords=winner_coords,
                    winner_labels=winner_labels,
                    sample_idx=sample_idx,
                    token=token,
                    args=args,
                    work_dir=work_dir,
                    pc_range=pc_range,
                    voxel_size=voxel_size,
                    voxel_shape=voxel_shape,
                    empty_label=empty_label,
                    palette=palette,
                    dense_dtype=dense_dtype,
                )

                mapped_input_points = int(sum(coords.shape[0] for coords in mapped_coords_list))
                records.append(dict(
                    sample_idx=sample_idx,
                    token=token,
                    scene_name=scene_name,
                    history_window=int(len(history_cache)),
                    mapped_input_points=mapped_input_points,
                    non_empty_voxels=non_empty_voxels,
                    saved_voxels=saved_voxels,
                    output_file=output_file,
                ))

                processed += 1
                if args.max_samples > 0 and processed >= args.max_samples:
                    stop_early = True
                    break

            if stop_early:
                break

    manifest = dict(
        config=args.config,
        weights=args.weights,
        split=args.split,
        history_frames=int(args.history_frames),
        output_format=args.output_format,
        max_voxels=int(args.max_voxels),
        num_classes=num_classes,
        empty_label=empty_label,
        dense_dtype=np.dtype(dense_dtype).name if dense_dtype is not None else None,
        voxel_shape=list(voxel_shape),
        voxel_size=voxel_size.tolist(),
        pc_range=pc_range.tolist(),
        num_shards=int(args.num_shards),
        shard_id=int(args.shard_id),
        processed=int(processed),
        records=records,
    )
    manifest_path = osp.join(work_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'[MultiFrameInference] Saved {processed} samples to: {work_dir}')
    print(f'[MultiFrameInference] Format: {args.output_format}')
    print(f'[MultiFrameInference] Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
