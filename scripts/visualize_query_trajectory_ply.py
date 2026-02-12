import os
import os.path as osp
import sys
import argparse
import importlib
import copy
import shutil
from datetime import datetime

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from mmengine.config import Config, DictAction
from mmengine.dataset import DefaultSampler, pseudo_collate
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.registry import DATASETS, MODELS
from torch.utils.data import DataLoader, Subset

from models.bbox.utils import decode_points


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize query initialization and refined query distribution as PLY on test samples.')
    parser.add_argument('--config', required=True, help='Path to config file.')
    parser.add_argument('--weights', required=True,
                        help='Checkpoint file path, or path to last_checkpoint, or run directory containing last_checkpoint.')
    parser.add_argument('--save-dir', default='demo_outputs/query_vis',
                        help='Output root directory.')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test',
                        help='Dataset split.')
    parser.add_argument('--max-samples', type=int, default=3,
                        help='Number of samples to export. <=0 means all.')
    parser.add_argument('--sample-indices', type=int, nargs='*', default=None,
                        help='Explicit dataset indices to visualize. If set, max-samples is ignored.')
    parser.add_argument('--random-sample', action='store_true',
                        help='Randomly choose max-samples from the split.')
    parser.add_argument('--batch-size', type=int, default=1, help='Inference batch size (recommended 1).')
    parser.add_argument('--num-workers', type=int, default=2, help='Dataloader workers.')
    parser.add_argument('--num-views', type=int, default=6,
                        help='Number of camera views per frame.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic behavior.')

    parser.add_argument('--refine-layer', type=int, default=-1,
                        help='Which refine layer to export. -1 means final layer.')
    parser.add_argument('--max-query-points', type=int, default=600000,
                        help='Cap exported refined query points per sample. <=0 means all.')
    parser.add_argument('--export-query-pointset', action='store_true',
                        help='Also export per-query point set from all_refine_pts (not OPUSSampling sample points).')

    parser.add_argument('--show-query-range', action='store_true',
                        help='Export an extra PLY showing per-query spatial range from query point set (AABB wireframe).')
    parser.add_argument('--range-samples-per-edge', type=int, default=4,
                        help='Number of sampled points per AABB edge when --show-query-range is on.')
    parser.add_argument('--max-range-queries', type=int, default=-1,
                        help='Limit queries used for range visualization. <=0 means all.')

    parser.add_argument('--override', nargs='+', action=DictAction,
                        help='Optional config overrides, e.g. key=value.')
    return parser.parse_args()


def sanitize_name(value):
    return str(value).replace(' ', '_').replace('/', '_').replace('\\', '_')


def resolve_checkpoint_path(path):
    path = osp.abspath(osp.expanduser(path))

    if osp.isdir(path):
        last_ckpt = osp.join(path, 'last_checkpoint')
        if osp.isfile(last_ckpt):
            path = last_ckpt
        else:
            raise FileNotFoundError(f'No checkpoint file found in directory: {path}')

    if osp.basename(path) == 'last_checkpoint':
        with open(path, 'r') as f:
            line = f.readline().strip()
        if not line:
            raise ValueError(f'Empty last_checkpoint file: {path}')
        ckpt_path = line if osp.isabs(line) else osp.join(osp.dirname(path), line)
        ckpt_path = osp.abspath(ckpt_path)
        if not osp.isfile(ckpt_path):
            raise FileNotFoundError(f'Checkpoint listed in last_checkpoint does not exist: {ckpt_path}')
        return ckpt_path

    if not osp.isfile(path):
        raise FileNotFoundError(f'Checkpoint file not found: {path}')
    return path


def build_palette(num_colors, seed=0):
    rng = np.random.RandomState(seed)
    palette = rng.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)
    if num_colors > 0:
        palette[0] = np.array([255, 64, 64], dtype=np.uint8)
    return palette


def write_ply(path, xyz, rgb=None, labels=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    n = int(xyz.shape[0])

    cols = [xyz]
    fmt = ['%.6f', '%.6f', '%.6f']
    props = ['property float x', 'property float y', 'property float z']

    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(n, 3)
        cols.append(rgb)
        fmt += ['%d', '%d', '%d']
        props += ['property uchar red', 'property uchar green', 'property uchar blue']

    if labels is not None:
        labels = np.asarray(labels, dtype=np.int32).reshape(n, 1)
        cols.append(labels)
        fmt += ['%d']
        props += ['property int label']

    data = np.concatenate(cols, axis=1) if n > 0 else np.zeros((0, len(fmt)), dtype=np.float32)

    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        for p in props:
            f.write(f'{p}\n')
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


def resolve_image_path(path_str, data_root=None):
    if path_str is None:
        return None
    path_str = str(path_str)
    candidates = []
    if osp.isabs(path_str):
        candidates.append(path_str)
    else:
        candidates.append(osp.abspath(path_str))
        if data_root:
            candidates.append(osp.abspath(osp.join(str(data_root), path_str)))

    for c in candidates:
        if osp.isfile(c):
            return c
    return None


def pick_current_frame_indices(meta, num_views):
    filenames = meta.get('filename', None)
    if not isinstance(filenames, list) or len(filenames) == 0:
        return []

    total = len(filenames)
    if total <= num_views:
        return list(range(total))

    timestamps = meta.get('img_timestamp', None)
    if not isinstance(timestamps, list) or len(timestamps) != total or total % num_views != 0:
        return list(range(min(num_views, total)))

    num_frames = total // num_views
    block_means = []
    for i in range(num_frames):
        block = timestamps[i * num_views:(i + 1) * num_views]
        block_means.append(float(np.mean(block)))

    curr_block = int(np.argmax(np.asarray(block_means)))
    start = curr_block * num_views
    return list(range(start, start + num_views))


def copy_current_frame_images(meta, sample_dir, data_root, num_views):
    filenames = meta.get('filename', None)
    if not isinstance(filenames, list) or len(filenames) == 0:
        return 0

    keep_indices = pick_current_frame_indices(meta, num_views=num_views)
    if not keep_indices:
        return 0

    out_dir = osp.join(sample_dir, 'current_frame_images')
    os.makedirs(out_dir, exist_ok=True)

    copied = 0
    records = []
    for local_i, src_i in enumerate(keep_indices):
        src_rel = filenames[src_i]
        src_abs = resolve_image_path(src_rel, data_root=data_root)
        if src_abs is None:
            records.append(f'[{local_i}] MISSING {src_rel}')
            continue

        base = osp.basename(src_abs)
        dst_name = f'view{local_i:02d}_{base}'
        dst_path = osp.join(out_dir, dst_name)
        shutil.copy2(src_abs, dst_path)
        copied += 1
        records.append(f'[{local_i}] {src_abs} -> {dst_name}')

    with open(osp.join(out_dir, 'mapping.txt'), 'w') as f:
        f.write('\n'.join(records) + '\n')

    return copied


def normalize_inputs(inputs):
    if not isinstance(inputs, dict):
        return inputs
    img = inputs.get('img', None)
    if isinstance(img, list) and img and torch.is_tensor(img[0]):
        inputs['img'] = torch.stack(img, dim=0)
    return inputs


def select_dataset_cfg(cfg, split):
    if split == 'test':
        return copy.deepcopy(cfg.test_dataloader.dataset)
    if split == 'val':
        return copy.deepcopy(cfg.val_dataloader.dataset)
    return copy.deepcopy(cfg.train_dataloader.dataset)


def maybe_force_offline_sweeps(dataset_cfg):
    pipeline = dataset_cfg.get('pipeline', [])
    for p in pipeline:
        if p.get('type') == 'LoadMultiViewImageFromMultiSweeps':
            p['force_offline'] = True


def sample_points(xyz, rgb=None, labels=None, max_points=-1, seed=0):
    n = int(xyz.shape[0])
    if max_points <= 0 or n <= max_points:
        return xyz, rgb, labels

    rng = np.random.RandomState(seed)
    choice = rng.choice(n, size=max_points, replace=False)
    xyz = xyz[choice]
    rgb = rgb[choice] if rgb is not None else None
    labels = labels[choice] if labels is not None else None
    return xyz, rgb, labels


def build_query_range_wireframe(query_pts, samples_per_edge=4):
    """query_pts: [Q, P, 3] in world coords. Returns [M, 3], [M] query_ids."""
    q, p, _ = query_pts.shape
    if q == 0 or p == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    mins = query_pts.min(axis=1)  # [Q, 3]
    maxs = query_pts.max(axis=1)  # [Q, 3]

    corners = np.zeros((q, 8, 3), dtype=np.float32)
    corners[:, 0] = np.stack([mins[:, 0], mins[:, 1], mins[:, 2]], axis=1)
    corners[:, 1] = np.stack([maxs[:, 0], mins[:, 1], mins[:, 2]], axis=1)
    corners[:, 2] = np.stack([mins[:, 0], maxs[:, 1], mins[:, 2]], axis=1)
    corners[:, 3] = np.stack([maxs[:, 0], maxs[:, 1], mins[:, 2]], axis=1)
    corners[:, 4] = np.stack([mins[:, 0], mins[:, 1], maxs[:, 2]], axis=1)
    corners[:, 5] = np.stack([maxs[:, 0], mins[:, 1], maxs[:, 2]], axis=1)
    corners[:, 6] = np.stack([mins[:, 0], maxs[:, 1], maxs[:, 2]], axis=1)
    corners[:, 7] = np.stack([maxs[:, 0], maxs[:, 1], maxs[:, 2]], axis=1)

    edges = np.asarray([
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int64)

    s = max(int(samples_per_edge), 2)
    t = np.linspace(0.0, 1.0, s, dtype=np.float32)

    p0 = corners[:, edges[:, 0], :]  # [Q, 12, 3]
    p1 = corners[:, edges[:, 1], :]  # [Q, 12, 3]
    wire = (1.0 - t[None, None, :, None]) * p0[:, :, None, :] + t[None, None, :, None] * p1[:, :, None, :]
    wire = wire.reshape(q, -1, 3)

    query_ids = np.arange(q, dtype=np.int32)[:, None]
    query_ids = np.repeat(query_ids, wire.shape[1], axis=1)

    return wire.reshape(-1, 3), query_ids.reshape(-1)


def extract_img_metas(data_samples):
    if data_samples is None:
        return []
    if isinstance(data_samples, list):
        return [sample.metainfo for sample in data_samples]
    return [data_samples.metainfo]


def forward_query_outputs(model, inputs, data_samples):
    img = inputs.get('img') if isinstance(inputs, dict) else inputs
    points = inputs.get('points') if isinstance(inputs, dict) else None
    img_metas = extract_img_metas(data_samples)

    img_feats = None if not getattr(model, 'with_img_backbone', False) else model.extract_img_feat(img, img_metas)
    pts_feats = None
    if getattr(model, 'with_pts_backbone', False):
        pts_feats = model.extract_pts_feat(points)
        if isinstance(pts_feats, (list, tuple)):
            if hasattr(model, 'final_conv'):
                pts_feats = model.final_conv(pts_feats[0])
            elif len(pts_feats) == 1:
                pts_feats = pts_feats[0]

    try:
        outs = model.pts_bbox_head(mlvl_feats=img_feats, pts_feats=pts_feats, points=points, img_metas=img_metas)
    except TypeError:
        try:
            outs = model.pts_bbox_head(mlvl_feats=img_feats, points=points, img_metas=img_metas)
        except TypeError:
            outs = model.pts_bbox_head(mlvl_feats=img_feats, img_metas=img_metas)

    return outs, img_metas


def main():
    args = parse_args()

    from mmdet3d.utils import register_all_modules
    from mmengine.registry import init_default_scope

    try:
        register_all_modules(init_default_scope=True)
    except KeyError as exc:
        if 'LoadMultiViewImageFromFiles' not in str(exc):
            raise
        init_default_scope('mmdet3d')

    cfg = Config.fromfile(args.config)
    if args.override is not None:
        cfg.merge_from_dict(args.override)

    importlib.import_module('models')
    importlib.import_module('loaders')

    ckpt_path = resolve_checkpoint_path(args.weights)

    set_random_seed(args.seed, deterministic=args.deterministic)
    cudnn.benchmark = not args.deterministic
    np.random.seed(args.seed)

    dataset_cfg = select_dataset_cfg(cfg, args.split)
    maybe_force_offline_sweeps(dataset_cfg)
    dataset = DATASETS.build(dataset_cfg)
    dataset_data_root = dataset_cfg.get('data_root', None)

    total_count = len(dataset)
    if args.sample_indices is not None and len(args.sample_indices) > 0:
        indices = [i for i in args.sample_indices if 0 <= i < total_count]
        if len(indices) == 0:
            raise ValueError('No valid --sample-indices after bounds check.')
        dataset = Subset(dataset, indices)
        print(f'[query_vis] Using explicit indices: {indices}')
    elif args.random_sample and args.max_samples > 0:
        count = min(args.max_samples, total_count)
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(total_count, size=count, replace=False).tolist()
        dataset = Subset(dataset, indices)
        print(f'[query_vis] Randomly sampled indices: {indices}')

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
    load_checkpoint(model, ckpt_path, map_location=device, strict=False)
    model.to(device)
    model.eval()

    pc_range = torch.tensor(cfg.point_cloud_range, dtype=torch.float32, device=device)

    run_name = osp.splitext(osp.basename(args.config))[0]
    run_name += '_' + osp.splitext(osp.basename(ckpt_path))[0]
    run_name += '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_root = osp.join(args.save_dir, f'{args.split}_{run_name}')
    os.makedirs(out_root, exist_ok=True)

    print(f'[query_vis] config: {args.config}')
    print(f'[query_vis] checkpoint: {ckpt_path}')
    print(f'[query_vis] split: {args.split}, dataset size: {len(dataset)}')
    print(f'[query_vis] output dir: {out_root}')
    print('[query_vis] note: exported query coordinates come from init_points/all_refine_pts, '
          'not OPUSSampling feature-sample points.')

    exported = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = normalize_inputs(move_to_device(data['inputs'], device))
            data_samples = move_to_device(data['data_samples'], device)

            outs, _ = forward_query_outputs(model, inputs, data_samples)

            init_points = outs.get('init_points', None)
            all_refine_pts = outs.get('all_refine_pts', None)
            if init_points is None or all_refine_pts is None:
                raise RuntimeError('Model head output does not contain init_points/all_refine_pts. '
                                   'This script targets OPUS-style heads.')

            num_layers = len(all_refine_pts)
            layer_idx = args.refine_layer
            if layer_idx < 0:
                layer_idx = num_layers + layer_idx
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f'Invalid --refine-layer={args.refine_layer}, available [0, {num_layers - 1}]')

            init_world = decode_points(init_points, pc_range).detach().cpu().numpy()  # [B, Q, 1, 3]
            refine_world = decode_points(all_refine_pts[layer_idx], pc_range).detach().cpu().numpy()  # [B, Q, P, 3]

            batch_size = init_world.shape[0]

            query_palette = build_palette(init_world.shape[1], seed=args.seed)

            for i in range(batch_size):
                data_sample = data_samples[i]
                meta = getattr(data_sample, 'metainfo', {})
                token = sanitize_name(meta.get('sample_token', f'b{batch_idx:06d}_{i:02d}'))
                scene = sanitize_name(meta.get('scene_name', 'unknown_scene'))
                sample_dir = osp.join(out_root, scene, token)
                os.makedirs(sample_dir, exist_ok=True)

                copied_views = copy_current_frame_images(
                    meta, sample_dir, data_root=dataset_data_root, num_views=args.num_views)

                # init query points [Q, 3]
                init_xyz = init_world[i, :, 0, :].astype(np.float32)
                q_ids = np.arange(init_xyz.shape[0], dtype=np.int32)
                init_rgb = query_palette[q_ids]
                write_ply(osp.join(sample_dir, 'query_init_points.ply'), init_xyz, rgb=init_rgb, labels=q_ids)

                # refined query points [Q, P, 3]
                refine_qp = refine_world[i].astype(np.float32)
                q, p, _ = refine_qp.shape
                refine_center = refine_qp.mean(axis=1)
                center_ids = np.arange(q, dtype=np.int32)
                center_rgb = query_palette[center_ids]
                write_ply(osp.join(sample_dir, f'query_refine_layer{layer_idx}_centers.ply'),
                          refine_center, rgb=center_rgb, labels=center_ids)

                if args.export_query_pointset:
                    refine_all = refine_qp.reshape(-1, 3)
                    refine_ids = np.repeat(np.arange(q, dtype=np.int32), p)
                    refine_rgb = query_palette[refine_ids]
                    refine_all, refine_rgb, refine_ids = sample_points(
                        refine_all, refine_rgb, refine_ids,
                        max_points=args.max_query_points,
                        seed=args.seed + exported + 101)
                    write_ply(osp.join(sample_dir, f'query_refine_layer{layer_idx}_querypointset.ply'),
                              refine_all, rgb=refine_rgb, labels=refine_ids)

                # Optional query range visualization
                if args.show_query_range:
                    range_qp = refine_qp
                    if args.max_range_queries > 0 and range_qp.shape[0] > args.max_range_queries:
                        keep = np.linspace(0, range_qp.shape[0] - 1, args.max_range_queries).round().astype(np.int64)
                        range_qp = range_qp[keep]
                        range_ids = keep.astype(np.int32)
                        range_palette = query_palette[range_ids]
                    else:
                        range_ids = np.arange(range_qp.shape[0], dtype=np.int32)
                        range_palette = query_palette

                    range_xyz, range_labels_local = build_query_range_wireframe(
                        range_qp, samples_per_edge=args.range_samples_per_edge)
                    range_labels = range_ids[range_labels_local]
                    range_rgb = range_palette[range_labels_local]
                    write_ply(osp.join(sample_dir, f'query_refine_layer{layer_idx}_range_wireframe.ply'),
                              range_xyz, rgb=range_rgb, labels=range_labels)

                # Combined ply for quick check
                combined_xyz = np.concatenate([init_xyz, refine_center], axis=0)
                combined_rgb = np.concatenate([
                    np.full_like(init_rgb, [255, 40, 40]),
                    np.full_like(center_rgb, [40, 255, 40]),
                ], axis=0)
                combined_label = np.concatenate([
                    np.full((init_xyz.shape[0],), -2, dtype=np.int32),
                    np.full((refine_center.shape[0],), -3, dtype=np.int32),
                ], axis=0)
                write_ply(osp.join(sample_dir, f'combined_init_refine_layer{layer_idx}.ply'),
                          combined_xyz, rgb=combined_rgb, labels=combined_label)

                exported += 1
                print(f'[query_vis] exported sample {exported}: {scene}/{token} '
                      f'(copied current images: {copied_views})')

                if args.max_samples > 0 and exported >= args.max_samples and args.sample_indices is None and not args.random_sample:
                    print(f'[query_vis] reached max-samples={args.max_samples}')
                    print(f'[query_vis] done. Outputs in: {out_root}')
                    return

    print(f'[query_vis] done. Exported {exported} samples to: {out_root}')


if __name__ == '__main__':
    main()
