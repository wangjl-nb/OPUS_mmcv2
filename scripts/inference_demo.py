import os
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

import argparse
import importlib
import os.path as osp
from datetime import datetime

import cv2
import matplotlib as mpl
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from mmengine.config import Config, DictAction
from mmengine.dataset import DefaultSampler, pseudo_collate
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.registry import DATASETS, MODELS
from torch.utils.data import DataLoader, Subset


def build_palette(num_classes):
    cmap = mpl.colormaps['turbo']
    palette = (cmap(np.linspace(0.0, 1.0, num_classes))[:, :3] * 255).astype(np.uint8)
    return palette


def visualize_occ(x, y, z, labels, palette, voxel_size, vmin=0, vmax=None):
    if vmax is None:
        vmax = palette.shape[0] - 1

    try:
        import mayavi.mlab as mlab

        mlab.options.offscreen = True
        if palette.shape[1] == 3:
            palette = np.concatenate([palette, np.ones((palette.shape[0], 1)) * 255], axis=1)
        fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
        plot = mlab.points3d(
            x, y, z,
            labels,
            scale_factor=voxel_size,
            mode='cube',
            scale_mode='vector',
            opacity=1.0,
            vmin=vmin,
            vmax=vmax)
        plot.module_manager.scalar_lut_manager.lut.table = palette

        f = mlab.gcf()
        f.scene._lift()
        save_fig = mlab.screenshot()
        mlab.close(fig)
        return save_fig
    except Exception:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        max_draw = 120000
        if labels.shape[0] > max_draw:
            choose = np.random.choice(labels.shape[0], max_draw, replace=False)
            x, y, z, labels = x[choose], y[choose], z[choose], labels[choose]

        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        clamped = np.clip(labels.astype(np.int64), vmin, vmax)
        colors = palette[clamped] / 255.0
        ax.scatter(x, y, z, c=colors, s=0.35, marker='s', linewidths=0)
        ax.view_init(elev=28, azim=132)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = buffer.reshape(height, width, 3)
        plt.close(fig)
        return image


def write_ply(path, xyz, rgb=None, labels=None):
    xyz = np.asarray(xyz, dtype=np.float32)
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
        cols.append(rgb)
        fmt += ['%d', '%d', '%d']
        header_props += [
            'property uchar red',
            'property uchar green',
            'property uchar blue',
        ]

    if labels is not None:
        labels = np.asarray(labels, dtype=np.int32).reshape(-1, 1)
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


def occ_loc_to_xyz(occ_loc, pc_range, voxel_size):
    occ_loc = occ_loc.astype(np.float32)
    x = (occ_loc[:, 0] + 0.5) * voxel_size[0] + pc_range[0]
    y = (occ_loc[:, 1] + 0.5) * voxel_size[1] + pc_range[1]
    z = (occ_loc[:, 2] + 0.5) * voxel_size[2] + pc_range[2]
    return x, y, z


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


def main():
    parser = argparse.ArgumentParser(description='Inference demo with visualization')
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
    parser.add_argument('--max-points', type=int, default=2000000000, help='Max points to draw')
    parser.add_argument('--save-pred-ply', action='store_true', help='Save predicted points as PLY')
    parser.add_argument('--save-gt-ply', action='store_true', help='Save GT points as PLY')
    parser.add_argument('--max-gt-points', type=int, default=20000000000, help='Max GT points to save')
    parser.add_argument('--gt-mask', choices=['none', 'camera', 'lidar'], default='camera',
                        help='Mask for GT voxels')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--random-train-sample', action='store_true',
                        help='Run inference on one random sample from train split')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic algorithms')
    parser.add_argument('--override', nargs='+', action=DictAction, help='Override config')
    args = parser.parse_args()

    from mmdet3d.utils import register_all_modules
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

    if args.random_train_sample:
        if args.split != 'train':
            raise ValueError('--random-train-sample can only be used with --split train')
        rng = np.random.default_rng(args.seed)
        rand_index = int(rng.integers(0, len(dataset)))
        print(f'[InferenceDemo] Selected random train index: {rand_index}/{len(dataset)}')
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
        print(f'[InferenceDemo] Using shard {args.shard_id}/{args.num_shards}, ' 
              f'samples: {len(shard_indices)}/{len(dataset)}')
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

    pc_range = cfg.point_cloud_range
    voxel_size = cfg.voxel_size
    num_classes = cfg.model['pts_bbox_head']['num_classes']
    empty_label = cfg.model['pts_bbox_head'].get('empty_label', num_classes)
    palette = build_palette(num_classes)

    processed = 0
    camera_mask_warned = False
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = move_to_device(data['inputs'], device)
            inputs = normalize_inference_inputs(inputs)
            data_samples = move_to_device(data['data_samples'], device)
            results = model(inputs=inputs, data_samples=data_samples, mode='predict')

            for i, result in enumerate(results):
                sample = data_samples[i]
                sample_idx = sample.metainfo.get('sample_idx', batch_idx)
                token = sample.metainfo.get('sample_token', sample_idx)
                scene_name = sample.metainfo.get('scene_name', None)

                labels = np.asarray(result['sem_pred'])
                occ_loc = np.asarray(result['occ_loc'])

                mask_camera, occ_infos = get_mask_camera(sample, base_dataset)
                mask_camera = None
                if mask_camera is None:
                    if not camera_mask_warned:
                        print('[InferenceDemo] mask_camera not found, keep unmasked predictions.')
                        camera_mask_warned = True
                else:
                    occ_loc, labels = apply_dense_mask(occ_loc, labels, mask_camera)

                if labels.size == 0:
                    continue

                x, y, z = occ_loc_to_xyz(occ_loc, pc_range, voxel_size)

                if args.max_points > 0 and labels.shape[0] > args.max_points:
                    choice = np.random.choice(labels.shape[0], args.max_points, replace=False)
                    x, y, z, labels = x[choice], y[choice], z[choice], labels[choice]

                labels = labels.astype(np.int64, copy=False)
                img = visualize_occ(
                    x, y, z,
                    labels,
                    palette,
                    voxel_size[0],
                    vmin=0,
                    vmax=num_classes - 1)
                out_path = osp.join(work_dir, f'{sample_idx:0>6}_{token}_pred.jpg')
                cv2.imwrite(out_path, img[..., ::-1])

                if args.save_pred_ply:
                    pred_xyz = np.stack([x, y, z], axis=1)
                    pred_rgb = palette[labels]
                    ply_path = osp.join(work_dir, f'{sample_idx:0>6}_{token}_pred.ply')
                    write_ply(ply_path, pred_xyz, rgb=pred_rgb, labels=labels)

                if args.save_gt_ply:
                    if scene_name is None:
                        continue
                    if occ_infos is None:
                        occ_infos = load_occ_infos(base_dataset, scene_name, token)
                    if occ_infos is None or 'semantics' not in occ_infos:
                        continue

                    occ_labels = occ_infos['semantics']
                    mask = occ_labels != empty_label
                    if args.gt_mask == 'camera' and 'mask_camera' in occ_infos:
                        mask = mask & occ_infos['mask_camera'].astype(np.bool_)
                    elif args.gt_mask == 'lidar' and 'mask_lidar' in occ_infos:
                        mask = mask & occ_infos['mask_lidar'].astype(np.bool_)

                    gt_pos = np.argwhere(mask)
                    gt_labels = occ_labels[mask]
                    if args.max_gt_points > 0 and gt_labels.shape[0] > args.max_gt_points:
                        choice = np.random.choice(gt_labels.shape[0], args.max_gt_points, replace=False)
                        gt_pos = gt_pos[choice]
                        gt_labels = gt_labels[choice]

                    gt_x = (gt_pos[:, 0] + 0.5) * voxel_size[0] + pc_range[0]
                    gt_y = (gt_pos[:, 1] + 0.5) * voxel_size[1] + pc_range[1]
                    gt_z = (gt_pos[:, 2] + 0.5) * voxel_size[2] + pc_range[2]
                    gt_xyz = np.stack([gt_x, gt_y, gt_z], axis=1)
                    gt_rgb = palette[gt_labels.astype(np.int64)]
                    gt_ply = osp.join(work_dir, f'{sample_idx:0>6}_{token}_gt.ply')
                    write_ply(gt_ply, gt_xyz, rgb=gt_rgb, labels=gt_labels)

                processed += 1
                if args.max_samples > 0 and processed >= args.max_samples:
                    return


if __name__ == '__main__':
    main()
