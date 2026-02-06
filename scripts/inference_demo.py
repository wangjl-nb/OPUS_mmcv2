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
from torch.utils.data import DataLoader


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


def main():
    parser = argparse.ArgumentParser(description='Inference demo with visualization')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--weights', required=True, help='Path to checkpoint')
    parser.add_argument('--save-dir', type=str, default='demo_outputs', help='Output directory')
    parser.add_argument('--split', choices=['val', 'test'], default='test', help='Dataset split')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--max-samples', type=int, default=-1, help='Max samples to run')
    parser.add_argument('--max-points', type=int, default=200000, help='Max points to draw')
    parser.add_argument('--save-pred-ply', action='store_true', help='Save predicted points as PLY')
    parser.add_argument('--save-gt-ply', action='store_true', help='Save GT points as PLY')
    parser.add_argument('--max-gt-points', type=int, default=200000, help='Max GT points to save')
    parser.add_argument('--gt-mask', choices=['none', 'camera', 'lidar'], default='camera',
                        help='Mask for GT voxels')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
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
    else:
        dataset_cfg = cfg.val_dataloader.dataset

    for p in dataset_cfg.pipeline:
        if p['type'] == 'LoadMultiViewImageFromMultiSweeps':
            p['force_offline'] = True

    dataset = DATASETS.build(dataset_cfg)
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
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = move_to_device(data['inputs'], device)
            inputs = normalize_inference_inputs(inputs)
            data_samples = move_to_device(data['data_samples'], device)
            results = model(inputs=inputs, data_samples=data_samples, mode='predict')

            for i, result in enumerate(results):
                sample_idx = data_samples[i].metainfo.get('sample_idx', batch_idx)
                token = data_samples[i].metainfo.get('sample_token', sample_idx)

                labels = result['sem_pred']
                occ_loc = result['occ_loc']

                if labels.size == 0:
                    continue

                x, y, z = occ_loc_to_xyz(occ_loc, pc_range, voxel_size)

                if args.max_points > 0 and labels.shape[0] > args.max_points:
                    choice = np.random.choice(labels.shape[0], args.max_points, replace=False)
                    x, y, z, labels = x[choice], y[choice], z[choice], labels[choice]

                img = visualize_occ(
                    x, y, z,
                    labels.astype(np.int64),
                    palette,
                    voxel_size[0],
                    vmin=0,
                    vmax=num_classes - 1)
                out_path = osp.join(work_dir, f'{sample_idx:0>6}_{token}_pred.jpg')
                cv2.imwrite(out_path, img[..., ::-1])

                if args.save_pred_ply:
                    pred_xyz = np.stack([x, y, z], axis=1)
                    pred_rgb = palette[labels.astype(np.int64)]
                    ply_path = osp.join(work_dir, f'{sample_idx:0>6}_{token}_pred.ply')
                    write_ply(ply_path, pred_xyz, rgb=pred_rgb, labels=labels)

                if args.save_gt_ply:
                    scene_name = data_samples[i].metainfo.get('scene_name', None)
                    if scene_name is None:
                        continue
                    occ_root = dataset.occ_root or osp.join(dataset.data_root, 'gts')
                    occ_file = osp.join(occ_root, scene_name, str(token), 'labels.npz')
                    if not osp.exists(occ_file):
                        continue
                    occ_infos = np.load(occ_file)
                    occ_labels = occ_infos['semantics']
                    mask = occ_labels != empty_label
                    if args.gt_mask == 'camera':
                        mask = mask & occ_infos['mask_camera'].astype(np.bool_)
                    elif args.gt_mask == 'lidar':
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
