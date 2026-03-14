#!/usr/bin/env python
import argparse
import ast
import importlib
import os.path as osp
import subprocess
import sys

from vis_utils import (
    TEXTSIM_VIEW_CASES,
    maybe_force_offline_sweeps,
    resolve_sample_indices,
    select_dataset_cfg,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run multi-frame comparison and query trajectory visualization in one command.'
    )
    parser.add_argument('--config', required=True, help='Config path')
    parser.add_argument('--weights', required=True, help='Checkpoint path')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--max-samples', type=int, default=5,
                        help='Number of samples to run (random or first N depending on flags)')
    parser.add_argument('--sample-indices', type=int, nargs='*', default=None,
                        help='Explicit dataset indices to run. If set, max-samples is ignored.')
    parser.add_argument('--random-sample', action='store_true',
                        help='Randomly choose max-samples from the split.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--override', nargs='+', default=None,
                        help='Optional config overrides forwarded to both scripts.')
    parser.add_argument('--prediction-adapter',
                        choices=['standard', 'talk2dino_textsim'],
                        default='standard',
                        help='Prediction export adapter.')
    parser.add_argument('--text-query', default=None,
                        help='Free-form text query used by talk2dino_textsim mode.')
    parser.add_argument('--textsim-save-dir', default=None,
                        help='Output root for export_text_query_similarity_pointcloud.py')
    parser.add_argument('--talk2dino-env', default='talk2dino',
                        help='Conda env used by Talk2DINO helper.')
    parser.add_argument('--talk2dino-config',
                        default='/root/wjl/Talk2DINO/tartanground_label_ae/configs/pca256_talk2dino_reg.json',
                        help='Talk2DINO latent config json.')
    parser.add_argument('--helper-script',
                        default=osp.join(osp.dirname(__file__), 'encode_text_query_talk2dino.py'),
                        help='Path to Talk2DINO helper script.')

    parser.add_argument('--compare-save-dir', default='outputs',
                        help='Output root for compare_multiframe_frames.py')
    parser.add_argument('--history-a', type=int, default=0)
    parser.add_argument('--history-b', type=int, default=5)
    parser.add_argument('--compare-batch-size', type=int, default=1)
    parser.add_argument('--compare-num-workers', type=int, default=4)
    parser.add_argument('--output-format', choices=['ply', 'npz'], default='ply')
    parser.add_argument('--max-voxels', type=int, default=300000000)
    parser.add_argument('--disable-camera-mask', action='store_true',
                        help='Forward to compare script to export no-mask sparse prediction PLYs')
    parser.add_argument('--eval-metrics', action='store_true')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-id', type=int, default=0)

    parser.add_argument('--vis-save-dir', default='outputs',
                        help='Output root for visualize_query_trajectory_ply.py')
    parser.add_argument('--vis-batch-size', type=int, default=1)
    parser.add_argument('--vis-num-workers', type=int, default=2)
    parser.add_argument('--num-views', type=int, default=6)
    parser.add_argument('--refine-layer', type=int, default=-1)
    parser.add_argument('--max-query-points', type=int, default=6000000)
    parser.add_argument('--export-query-pointset',action='store_true')
    parser.add_argument('--show-query-range',default=True )
    parser.add_argument('--range-samples-per-edge', type=int, default=4)
    parser.add_argument('--max-range-queries', type=int, default=-1)
    return parser.parse_args()


def _extend_with_indices(cmd, indices):
    if indices:
        cmd += ['--sample-indices'] + [str(i) for i in indices]


def _parse_override_list(items):
    parsed = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f'Invalid override item: {item}')
        key, value = item.split('=', 1)
        try:
            parsed[key] = ast.literal_eval(value)
        except Exception:
            parsed[key] = value
    return parsed


def _resolve_textsim_indices(args):
    from mmengine.config import Config
    from mmdet3d.registry import DATASETS
    from mmdet3d.utils import register_all_modules
    from mmengine.registry import init_default_scope

    try:
        register_all_modules(init_default_scope=True)
    except KeyError as exc:
        if 'LoadMultiViewImageFromFiles' not in str(exc):
            raise
        init_default_scope('mmdet3d')

    cfg = Config.fromfile(args.config)
    if args.override:
        cfg.merge_from_dict(_parse_override_list(args.override))

    importlib.import_module('loaders')
    dataset_cfg = select_dataset_cfg(cfg, args.split)
    maybe_force_offline_sweeps(dataset_cfg)
    dataset = DATASETS.build(dataset_cfg)
    if hasattr(dataset, 'full_init'):
        dataset.full_init()

    return resolve_sample_indices(
        total_count=len(dataset),
        max_samples=args.max_samples,
        sample_indices=args.sample_indices,
        random_sample=args.random_sample,
        seed=args.seed,
    )


def _run_textsim_cases(args, script_dir):
    if args.eval_metrics:
        raise ValueError('talk2dino_textsim mode does not support --eval-metrics')
    if not args.text_query:
        raise ValueError('--text-query is required when --prediction-adapter=talk2dino_textsim')

    target_indices = _resolve_textsim_indices(args)
    if not target_indices:
        raise ValueError('No target sample indices resolved for talk2dino_textsim mode')

    textsim_save_dir = args.textsim_save_dir or args.compare_save_dir
    textsim_script = osp.join(script_dir, 'export_text_query_similarity_pointcloud.py')

    print('[Combo] Running talk2dino_textsim mode only; compare/query-vis steps are skipped.')
    print(f'[Combo] Target sample indices: {target_indices}')
    for view_case in TEXTSIM_VIEW_CASES:
        case_dir = osp.join(textsim_save_dir, view_case['name'])
        cmd = [
            sys.executable,
            textsim_script,
            '--config', args.config,
            '--weights', args.weights,
            '--text-query', args.text_query,
            '--save-dir', case_dir,
            '--split', args.split,
            '--batch-size', str(args.vis_batch_size),
            '--num-workers', str(args.vis_num_workers),
            '--max-voxels', str(args.max_voxels),
            '--seed', str(args.seed),
            '--view-case-name', view_case['name'],
            '--num-views', str(len(view_case['cameras'])),
            '--talk2dino-env', args.talk2dino_env,
            '--talk2dino-config', args.talk2dino_config,
            '--helper-script', args.helper_script,
            '--active-cameras',
        ] + list(view_case['cameras'])
        _extend_with_indices(cmd, target_indices)
        if args.deterministic:
            cmd.append('--deterministic')
        if args.disable_camera_mask:
            cmd.append('--disable-camera-mask')
        if args.override:
            cmd += ['--override'] + args.override

        print(f'[Combo] Running text-sim case {view_case["name"]}')
        print('[Combo] Command:', ' '.join(cmd))
        subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    if args.sample_indices and args.random_sample:
        raise ValueError('--sample-indices cannot be combined with --random-sample')
    if args.random_sample and args.max_samples <= 0:
        raise ValueError('--random-sample requires --max-samples > 0')
    if args.num_shards < 1:
        raise ValueError('--num-shards must be >= 1')
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError('--shard-id must satisfy 0 <= shard_id < num_shards')

    script_dir = osp.dirname(__file__)

    if args.prediction_adapter == 'talk2dino_textsim':
        _run_textsim_cases(args, script_dir)
        return

    compare_script = osp.join(script_dir, 'compare_multiframe_frames.py')
    compare_cmd = [
        sys.executable,
        compare_script,
        '--config', args.config,
        '--weights', args.weights,
        '--split', args.split,
        '--history-a', str(args.history_a),
        '--history-b', str(args.history_b),
        '--batch-size', str(args.compare_batch_size),
        '--num-workers', str(args.compare_num_workers),
        '--max-samples', str(args.max_samples),
        '--output-format', args.output_format,
        '--max-voxels', str(args.max_voxels),
        '--save-dir', args.compare_save_dir,
        '--num-shards', str(args.num_shards),
        '--shard-id', str(args.shard_id),
        '--seed', str(args.seed),
    ]
    if args.deterministic:
        compare_cmd.append('--deterministic')
    if args.random_sample:
        compare_cmd.append('--random-sample')
    if args.disable_camera_mask:
        compare_cmd.append('--disable-camera-mask')
    _extend_with_indices(compare_cmd, args.sample_indices)
    if args.eval_metrics:
        compare_cmd.append('--eval-metrics')
    if args.override:
        compare_cmd += ['--override'] + args.override

    print('[Combo] Running compare_multiframe_frames.py')
    print('[Combo] Command:', ' '.join(compare_cmd))
    subprocess.run(compare_cmd, check=True)

    vis_script = osp.join(script_dir, 'visualize_query_trajectory_ply.py')
    vis_cmd = [
        sys.executable,
        vis_script,
        '--config', args.config,
        '--weights', args.weights,
        '--save-dir', args.vis_save_dir,
        '--split', args.split,
        '--max-samples', str(args.max_samples),
        '--batch-size', str(args.vis_batch_size),
        '--num-workers', str(args.vis_num_workers),
        '--num-views', str(args.num_views),
        '--refine-layer', str(args.refine_layer),
        '--max-query-points', str(args.max_query_points),
        '--seed', str(args.seed),
    ]
    if args.deterministic:
        vis_cmd.append('--deterministic')
    if args.random_sample:
        vis_cmd.append('--random-sample')
    _extend_with_indices(vis_cmd, args.sample_indices)
    if args.export_query_pointset:
        vis_cmd.append('--export-query-pointset')
    if args.show_query_range:
        vis_cmd.append('--show-query-range')
    vis_cmd += ['--range-samples-per-edge', str(args.range_samples_per_edge)]
    if args.max_range_queries is not None:
        vis_cmd += ['--max-range-queries', str(args.max_range_queries)]
    if args.override:
        vis_cmd += ['--override'] + args.override

    print('[Combo] Running visualize_query_trajectory_ply.py')
    print('[Combo] Command:', ' '.join(vis_cmd))
    subprocess.run(vis_cmd, check=True)


if __name__ == '__main__':
    main()
