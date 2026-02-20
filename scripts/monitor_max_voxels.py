#!/usr/bin/env python3
"""Quick monitor for voxel-cap saturation (max_voxels) on a config dataloader."""

import argparse
import importlib
from typing import List

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Monitor max_voxels saturation.')
    parser.add_argument('--config', required=True, help='Path to config file.')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--num-batches', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def resolve_cap(cfg: Config, split: str) -> int:
    cap_cfg = cfg.model.pts_voxel_layer.get('max_voxels', None)
    if isinstance(cap_cfg, (list, tuple)):
        if split == 'train':
            return int(cap_cfg[0])
        return int(cap_cfg[1])
    return int(cap_cfg)


def to_points_list(points_data, device: torch.device) -> List[torch.Tensor]:
    if isinstance(points_data, list):
        pts_list = points_data
    else:
        pts_list = [points_data]

    out = []
    for pts in pts_list:
        if hasattr(pts, 'tensor'):
            pts = pts.tensor
        if not isinstance(pts, torch.Tensor):
            pts = torch.as_tensor(pts)
        out.append(pts.to(device=device, non_blocking=True))
    return out


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    importlib.import_module('models')
    importlib.import_module('loaders')

    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'

    dataloader_key = f'{args.split}_dataloader'
    dataloader_cfg = cfg.get(dataloader_key)
    if dataloader_cfg is None:
        raise ValueError(f'Config has no {dataloader_key}.')

    if args.batch_size is not None:
        dataloader_cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        dataloader_cfg.num_workers = args.num_workers
        if args.num_workers == 0 and dataloader_cfg.get('persistent_workers', False):
            dataloader_cfg.persistent_workers = False

    cap = resolve_cap(cfg, args.split)

    dataloader = Runner.build_dataloader(dataloader_cfg)
    model = MODELS.build(cfg.model)

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but CUDA is not available.')
    model.to(device)
    if args.split == 'train':
        model.train()
    else:
        model.eval()

    sample_voxel_counts = []
    hit_flags = []
    processed_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        points_data = batch['inputs']['points']
        points_list = to_points_list(points_data, device)

        with torch.no_grad():
            _, _, coors = model.voxelize(points_list)

        # coors[:, 0] is the sample index inside this mini-batch.
        sample_counts = torch.bincount(
            coors[:, 0].long(), minlength=len(points_list)
        ).cpu().numpy().tolist()

        batch_hits = [int(c >= cap) for c in sample_counts]
        sample_voxel_counts.extend(sample_counts)
        hit_flags.extend(batch_hits)
        processed_batches += 1

        batch_hit_num = int(np.sum(batch_hits))
        print(
            f'batch={batch_idx:04d} samples={len(sample_counts)} '
            f'max={max(sample_counts):6d} mean={np.mean(sample_counts):8.1f} '
            f'hit={batch_hit_num}/{len(sample_counts)} cap={cap}'
        )

    if not sample_voxel_counts:
        raise RuntimeError('No batch was processed. Check dataloader settings.')

    counts = np.array(sample_voxel_counts, dtype=np.float64)
    hits = np.array(hit_flags, dtype=np.int64)
    hit_ratio = float(hits.mean())

    print('\n=== max_voxels report ===')
    print(f'split: {args.split}')
    print(f'processed batches: {processed_batches}')
    print(f'processed samples: {counts.size}')
    print(f'cap: {cap}')
    print(
        f'voxel_count mean/p95/max: '
        f'{counts.mean():.1f} / {np.percentile(counts, 95):.1f} / {counts.max():.0f}'
    )
    print(f'cap-hit samples: {hits.sum()} ({hit_ratio * 100:.2f}%)')

    if hit_ratio > 0.20:
        print('suggestion: hit ratio is high, consider increasing max_voxels by 25%-50%.')
    elif hit_ratio > 0.05:
        print('suggestion: moderate cap pressure, try +10%-25% max_voxels and re-check speed/memory.')
    else:
        print('suggestion: cap pressure is low, keeping current max_voxels is usually fine.')


if __name__ == '__main__':
    main()
