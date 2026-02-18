#!/usr/bin/env python
import argparse
import os.path as osp
import subprocess
import sys


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

    parser.add_argument('--compare-save-dir', default='outputs',
                        help='Output root for compare_multiframe_frames.py')
    parser.add_argument('--history-a', type=int, default=0)
    parser.add_argument('--history-b', type=int, default=5)
    parser.add_argument('--compare-batch-size', type=int, default=1)
    parser.add_argument('--compare-num-workers', type=int, default=4)
    parser.add_argument('--output-format', choices=['ply', 'npz'], default='ply')
    parser.add_argument('--max-voxels', type=int, default=300000000)
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


def main():
    args = parse_args()

    if args.sample_indices and args.random_sample:
        raise ValueError('--sample-indices cannot be combined with --random-sample')
    if args.random_sample and args.max_samples <= 0:
        raise ValueError('--random-sample requires --max-samples > 0')

    script_dir = osp.dirname(__file__)

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
