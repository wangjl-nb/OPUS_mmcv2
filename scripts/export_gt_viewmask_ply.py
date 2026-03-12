#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_PC_RANGE = [-20.0, -20.0, -3.0, 20.0, 20.0, 5.0]
DEFAULT_VOXEL_SIZE = [0.1, 0.1, 0.1]
FRONT_PRIORITY = ['CAM_FRONT']
THREE_VIEW_PRIORITY = ['CAM_LEFT', 'CAM_FRONT', 'CAM_RIGHT']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export current-GT visible voxels to PLY for 1/3/6-view subsets.')
    parser.add_argument(
        '--ann-file',
        default='/root/wjl/OPUS_mmcv2/data/TartanGround_Indoor/train.pkl',
        help='Annotation pkl containing infos.')
    parser.add_argument(
        '--gt-root',
        required=True,
        help='GT root containing labels.npz with mask_camera_bits.')
    parser.add_argument(
        '--sample-index',
        type=int,
        default=0,
        help='Sample index inside ann_file.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory. Defaults under demos/.')
    parser.add_argument(
        '--pc-range',
        nargs=6,
        type=float,
        default=DEFAULT_PC_RANGE,
        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
        help='Point cloud range used to decode voxels.')
    parser.add_argument(
        '--voxel-size',
        nargs=3,
        type=float,
        default=DEFAULT_VOXEL_SIZE,
        metavar=('VX', 'VY', 'VZ'),
        help='Voxel size used to decode voxels.')
    parser.add_argument(
        '--max-points',
        type=int,
        default=0,
        help='Randomly keep at most this many visible voxels per export; <=0 keeps all.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed for optional voxel subsampling.')
    return parser.parse_args()


def load_infos(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, list):
        return data
    raise TypeError(f'Unsupported ann format: {type(data)}')


def resolve_gt_path(gt_root, scene_name, token):
    root = Path(gt_root)
    direct = root / scene_name / token / 'labels.npz'
    if direct.exists():
        return direct
    matches = list(root.rglob(f'{token}/labels.npz'))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f'No labels.npz found for token={token} under {gt_root}')
    raise RuntimeError(f'Multiple labels.npz matches for token={token}: {matches}')


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
    with open(path, 'w') as handle:
        handle.write('ply\n')
        handle.write('format ascii 1.0\n')
        handle.write(f'element vertex {n}\n')
        for prop in header_props:
            handle.write(f'{prop}\n')
        handle.write('end_header\n')
        np.savetxt(handle, data, fmt=' '.join(fmt))


def build_palette(num_classes):
    rng = np.random.default_rng(seed=20260312)
    palette = rng.integers(0, 256, size=(max(1, num_classes), 3), dtype=np.uint8)
    palette[0] = np.array([180, 180, 180], dtype=np.uint8)
    return palette


def voxel_indices_to_xyz(indices, pc_range, voxel_size):
    indices = indices.astype(np.float32)
    x = (indices[:, 0] + 0.5) * voxel_size[0] + pc_range[0]
    y = (indices[:, 1] + 0.5) * voxel_size[1] + pc_range[1]
    z = (indices[:, 2] + 0.5) * voxel_size[2] + pc_range[2]
    return np.stack([x, y, z], axis=1)


def choose_camera_subset(camera_names, count):
    available = list(camera_names)
    if count >= len(available):
        return available
    if count == 1:
        for name in FRONT_PRIORITY:
            if name in available:
                return [name]
        return [available[0]]
    if count == 3:
        if all(name in available for name in THREE_VIEW_PRIORITY):
            return list(THREE_VIEW_PRIORITY)
        return available[:3]
    return available[:count]


def subset_mask_from_bits(mask_camera_bits, camera_names, selected_names):
    selected = set(selected_names)
    mask = np.zeros_like(mask_camera_bits, dtype=np.bool_)
    for cam_idx, cam_name in enumerate(camera_names):
        if cam_name in selected:
            mask |= (mask_camera_bits & (1 << cam_idx)) != 0
    return mask


def maybe_subsample(indices, max_points, seed):
    if max_points is None or max_points <= 0 or indices.shape[0] <= max_points:
        return indices
    rng = np.random.default_rng(seed)
    choice = rng.choice(indices.shape[0], size=max_points, replace=False)
    return indices[choice]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_images(info, selected_names, dst_dir):
    ensure_dir(dst_dir)
    copied = []
    for cam_name in selected_names:
        cam_info = info.get('cams', {}).get(cam_name)
        if cam_info is None:
            continue
        src = cam_info.get('data_path')
        if not isinstance(src, str) or not osp.exists(src):
            continue
        stem = osp.basename(src)
        dst = osp.join(dst_dir, stem)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def export_subset(out_dir, name, info, semantics, mask_camera_bits, camera_names,
                  pc_range, voxel_size, max_points, seed):
    selected_names = choose_camera_subset(camera_names, name)
    subset_name = f'{len(selected_names)}view'
    subset_dir = osp.join(out_dir, subset_name)
    ensure_dir(subset_dir)

    visible_mask = subset_mask_from_bits(mask_camera_bits, camera_names, selected_names)
    visible_indices = np.argwhere(visible_mask)
    visible_indices = maybe_subsample(visible_indices, max_points=max_points, seed=seed + len(selected_names))

    xyz = voxel_indices_to_xyz(visible_indices, pc_range=pc_range, voxel_size=voxel_size)
    labels = semantics[visible_indices[:, 0], visible_indices[:, 1], visible_indices[:, 2]]

    palette = build_palette(int(semantics.max()) + 1)
    rgb = palette[labels.clip(min=0, max=palette.shape[0] - 1)]

    ply_path = osp.join(subset_dir, f'gt_visible_{subset_name}.ply')
    write_ply(ply_path, xyz, rgb=rgb, labels=labels)

    images_dir = osp.join(subset_dir, 'images')
    copied_images = copy_images(info, selected_names, images_dir)

    meta = dict(
        scene_name=info.get('scene_name'),
        token=info.get('token'),
        selected_cameras=selected_names,
        num_visible_voxels=int(visible_indices.shape[0]),
        ply_path=ply_path,
        copied_images=copied_images,
    )
    with open(osp.join(subset_dir, 'meta.json'), 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return meta


def main():
    args = parse_args()
    infos = load_infos(args.ann_file)
    if args.sample_index < 0 or args.sample_index >= len(infos):
        raise IndexError(f'sample_index out of range: {args.sample_index}, dataset len={len(infos)}')

    info = infos[args.sample_index]
    scene_name = info['scene_name']
    token = info['token']
    gt_path = resolve_gt_path(args.gt_root, scene_name=scene_name, token=token)

    occ = np.load(gt_path, allow_pickle=False)
    if 'mask_camera_bits' not in occ.files:
        raise KeyError(f'GT file does not contain mask_camera_bits: {gt_path}')
    if 'camera_names' not in occ.files:
        raise KeyError(f'GT file does not contain camera_names: {gt_path}')

    semantics = np.asarray(occ['semantics'])
    mask_camera_bits = np.asarray(occ['mask_camera_bits'], dtype=np.uint8)
    camera_names = [str(x) for x in occ['camera_names'].tolist()]

    if args.out_dir is None:
        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.out_dir = osp.join(
            '/root/wjl/OPUS_mmcv2/demos',
            f'gt_viewmask_{Path(args.ann_file).stem}_{args.sample_index:06d}_{token}_{stamp}',
        )
    ensure_dir(args.out_dir)

    results = []
    for count in [1, 3, 6]:
        results.append(
            export_subset(
                out_dir=args.out_dir,
                name=count,
                info=info,
                semantics=semantics,
                mask_camera_bits=mask_camera_bits,
                camera_names=camera_names,
                pc_range=args.pc_range,
                voxel_size=args.voxel_size,
                max_points=args.max_points,
                seed=args.seed,
            )
        )

    summary = dict(
        ann_file=args.ann_file,
        gt_root=args.gt_root,
        gt_path=str(gt_path),
        sample_index=args.sample_index,
        scene_name=scene_name,
        token=token,
        camera_names=camera_names,
        outputs=results,
    )
    with open(osp.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
