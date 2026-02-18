import argparse
import json
import os
import os.path as osp
import pickle
import runpy
import subprocess
import sys
from typing import Dict, List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run and compare multi-frame inference with two history lengths.')
    parser.add_argument('--config', required=True, help='Config path')
    parser.add_argument('--weights', required=True, help='Checkpoint path')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--save-dir', default='compare_multiframe_outputs',
                        help='Output root for both runs and summary')
    parser.add_argument('--history-a', type=int, default=0,
                        help='First history setting (e.g., 0)')
    parser.add_argument('--history-b', type=int, default=5,
                        help='Second history setting (e.g., 5)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=5,
                        help='Run only a few samples by default for quick comparison')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--random-train-sample', action='store_true')
    parser.add_argument('--sample-indices', type=int, nargs='*', default=None,
                        help='Explicit dataset indices to run. If set, max-samples is ignored.')
    parser.add_argument('--random-sample', action='store_true',
                        help='Randomly choose max-samples from the split.')
    parser.add_argument('--output-format', choices=['ply', 'npz'], default='ply',
                        help='Forwarded to inference script')
    parser.add_argument('--max-voxels', type=int, default=300000000,
                        help='Max voxels saved in PLY mode, forwarded to inference script')
    parser.add_argument('--eval-metrics', default=False, action='store_true',
                        help='Compute mIoU/IoU after inference; requires NPZ outputs')
    parser.add_argument('--skip-infer', action='store_true',
                        help='Skip inference and only process existing run dirs')
    parser.add_argument('--run-dir-a', type=str, default=None,
                        help='Existing run dir for history-a (used with --skip-infer)')
    parser.add_argument('--run-dir-b', type=str, default=None,
                        help='Existing run dir for history-b (used with --skip-infer)')
    parser.add_argument('--override', nargs='+', default=None,
                        help='Optional config override forwarded to inference script')
    return parser.parse_args()


def _load_data_infos(ann_file: str) -> List[Dict]:
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'infos' in data:
        infos = data['infos']
    elif isinstance(data, dict) and 'data_list' in data:
        infos = data['data_list']
    elif isinstance(data, list):
        infos = data
    else:
        raise TypeError(f'Unsupported annotation format: {type(data)}')
    return infos


def _info_with_aliases(info: Dict) -> Dict:
    info = dict(info)
    if 'token' in info:
        info.setdefault('sample_token', info['token'])
    if 'sample_token' in info:
        info.setdefault('token', info['sample_token'])
    return info


def _build_occ_path(occ_root: str, path_template: str, info: Dict) -> str:
    fmt = _info_with_aliases(info)
    return osp.join(occ_root, path_template.format(**fmt))


def _compute_hist(gt: np.ndarray, pred: np.ndarray, num_classes: int, mask: np.ndarray = None):
    gt = np.asarray(gt, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)

    if mask is not None:
        mask = np.asarray(mask, dtype=np.bool_)
        gt = gt[mask]
        pred = pred[mask]

    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.float64)

    return np.bincount(
        num_classes * gt[valid] + pred[valid], minlength=num_classes**2
    ).reshape(num_classes, num_classes).astype(np.float64)


def _compute_per_class_iou(hist: np.ndarray):
    hist = np.asarray(hist, dtype=np.float64)
    denom = hist.sum(1) + hist.sum(0) - np.diag(hist)
    return np.divide(
        np.diag(hist),
        denom,
        out=np.full(hist.shape[0], np.nan, dtype=np.float64),
        where=denom > 0,
    )


def _compute_miou(hist: np.ndarray, empty_label: int):
    iou = _compute_per_class_iou(hist)
    if empty_label is not None and 0 <= empty_label < hist.shape[0]:
        valid_indices = [i for i in range(hist.shape[0]) if i != empty_label]
        mean_iou = np.nanmean(iou[valid_indices])
    else:
        mean_iou = np.nanmean(iou)
    return float(mean_iou) * 100.0


def _compute_occupied_iou(hist: np.ndarray, empty_label: int):
    num_classes = hist.shape[0]
    if empty_label is None or empty_label < 0 or empty_label >= num_classes:
        return float('nan')
    occ_mask = np.ones(num_classes, dtype=np.bool_)
    occ_mask[empty_label] = False
    tp = hist[np.ix_(occ_mask, occ_mask)].sum()
    fp = hist[empty_label, occ_mask].sum()
    fn = hist[occ_mask, empty_label].sum()
    denom = tp + fp + fn
    if denom <= 0:
        return float('nan')
    return float(tp) / float(denom) * 100.0


def _latest_subdir(path: str) -> str:
    subdirs = [
        osp.join(path, d) for d in os.listdir(path)
        if osp.isdir(osp.join(path, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f'No run directories found under: {path}')
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


def _resolve_eval_spec(config_path: str, split: str):
    cfg = runpy.run_path(config_path)

    dataloader_key = f'{split}_dataloader'
    if dataloader_key not in cfg:
        raise KeyError(f'Missing `{dataloader_key}` in config: {config_path}')

    dataset_cfg = cfg[dataloader_key]['dataset']
    data_root = dataset_cfg.get('data_root', None)
    ann_file = dataset_cfg['ann_file']
    if not osp.isabs(ann_file) and data_root is not None:
        ann_file = osp.join(data_root, ann_file)

    occ_root = dataset_cfg.get('occ_root', None)
    if occ_root is None:
        if data_root is None:
            raise KeyError('Cannot resolve occ_root: both occ_root and data_root are missing')
        occ_root = osp.join(data_root, 'gts')

    dataset_meta = dataset_cfg.get('dataset_cfg', {}) or {}
    occ_io_cfg = dataset_meta.get('occ_io', {}) or {}

    path_template = occ_io_cfg.get('path_template', '{scene_name}/{token}/labels.npz')
    semantics_key = occ_io_cfg.get('semantics_key', 'semantics')
    mask_camera_key = occ_io_cfg.get('mask_camera_key', 'mask_camera')

    model_cfg = cfg.get('model', {})
    head_cfg = model_cfg.get('pts_bbox_head', {}) if isinstance(model_cfg, dict) else {}
    num_classes = int(head_cfg.get('num_classes', 0))
    empty_label = int(dataset_meta.get('empty_label', head_cfg.get('empty_label', max(num_classes, 1) - 1)))

    class_names = dataset_meta.get('class_names', None)
    if class_names is not None:
        class_names = list(class_names)

    return dict(
        ann_file=ann_file,
        occ_root=occ_root,
        path_template=path_template,
        semantics_key=semantics_key,
        mask_camera_key=mask_camera_key,
        empty_label=empty_label,
        num_classes=num_classes,
        class_names=class_names,
    )


def run_inference(history_frames: int, args) -> str:
    out_root = osp.join(args.save_dir, f'history_{history_frames}')
    os.makedirs(out_root, exist_ok=True)

    script_path = osp.join(osp.dirname(__file__), 'inference_demo_multiframe.py')
    cmd = [
        sys.executable,
        script_path,
        '--config', args.config,
        '--weights', args.weights,
        '--save-dir', out_root,
        '--split', args.split,
        '--batch-size', str(args.batch_size),
        '--num-workers', str(args.num_workers),
        '--max-samples', str(args.max_samples),
        '--num-shards', str(args.num_shards),
        '--shard-id', str(args.shard_id),
        '--history-frames', str(history_frames),
        '--output-format', args.output_format,
        '--max-voxels', str(args.max_voxels),
        '--seed', str(args.seed),
    ]
    if args.deterministic:
        cmd.append('--deterministic')
    if args.random_train_sample:
        cmd.append('--random-train-sample')
    if args.random_sample:
        cmd.append('--random-sample')
    if args.sample_indices:
        cmd += ['--sample-indices'] + [str(i) for i in args.sample_indices]
    if args.override:
        cmd += ['--override'] + args.override

    print(f'[Compare] Running history={history_frames}')
    print('[Compare] Command:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

    return _latest_subdir(out_root)


def evaluate_run(run_dir: str, eval_spec: Dict):
    manifest_path = osp.join(run_dir, 'manifest.json')
    if not osp.isfile(manifest_path):
        raise FileNotFoundError(f'Missing manifest: {manifest_path}')

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    records = manifest.get('records', [])
    if not records:
        raise RuntimeError(f'No sample records found in manifest: {manifest_path}')

    infos = _load_data_infos(eval_spec['ann_file'])

    class_cap = [
        int(eval_spec['empty_label']) + 1,
        int(eval_spec['num_classes']),
    ]
    if eval_spec['class_names'] is not None:
        class_cap.append(len(eval_spec['class_names']))
    num_classes = max(class_cap)

    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    used_samples = 0

    for rec in records:
        sample_idx = int(rec['sample_idx'])
        if sample_idx < 0 or sample_idx >= len(infos):
            continue

        pred_file = osp.join(run_dir, rec['output_file'])
        if not osp.isfile(pred_file):
            continue

        with np.load(pred_file) as pred_raw:
            if 'semantics' not in pred_raw:
                continue
            pred_sem = np.asarray(pred_raw['semantics'], dtype=np.int64)

        info = infos[sample_idx]
        gt_file = _build_occ_path(eval_spec['occ_root'], eval_spec['path_template'], info)
        if not osp.isfile(gt_file):
            continue

        with np.load(gt_file) as gt_raw:
            if eval_spec['semantics_key'] not in gt_raw:
                continue
            gt_sem = np.asarray(gt_raw[eval_spec['semantics_key']], dtype=np.int64)
            if eval_spec['mask_camera_key'] in gt_raw:
                mask = np.asarray(gt_raw[eval_spec['mask_camera_key']], dtype=np.bool_)
            else:
                mask = np.ones_like(gt_sem, dtype=np.bool_)

        if pred_sem.shape != gt_sem.shape:
            raise ValueError(
                f'Shape mismatch @ sample_idx={sample_idx}: '
                f'pred {pred_sem.shape} vs gt {gt_sem.shape}')

        local_max = max(
            int(np.max(gt_sem)) if gt_sem.size else -1,
            int(np.max(pred_sem)) if pred_sem.size else -1,
            eval_spec['empty_label'],
        )
        needed_classes = local_max + 1
        if needed_classes > hist.shape[0]:
            new_hist = np.zeros((needed_classes, needed_classes), dtype=np.float64)
            new_hist[:hist.shape[0], :hist.shape[1]] = hist
            hist = new_hist

        hist += _compute_hist(gt_sem, pred_sem, hist.shape[0], mask=mask)
        used_samples += 1

    per_class_iou = _compute_per_class_iou(hist)
    metrics = dict(
        run_dir=run_dir,
        processed=int(manifest.get('processed', 0)),
        evaluated_samples=int(used_samples),
        mIoU=round(_compute_miou(hist, eval_spec['empty_label']), 2),
        IoU=round(_compute_occupied_iou(hist, eval_spec['empty_label']), 2),
        non_empty_avg=round(float(np.mean([r.get('non_empty_voxels', 0) for r in records])) if records else 0.0, 2),
    )

    if eval_spec['class_names'] is not None:
        class_names = eval_spec['class_names']
    else:
        class_names = [f'class_{i}' for i in range(hist.shape[0])]

    classwise = {}
    for idx, iou in enumerate(per_class_iou):
        name = class_names[idx] if idx < len(class_names) else f'class_{idx}'
        key = f'IoU_{idx:02d}_{name}'
        classwise[key] = None if np.isnan(iou) else round(float(iou) * 100.0, 2)

    metrics['classwise_iou'] = classwise
    return metrics


def _collect_preview(run_dir: str, max_items: int = 5):
    manifest_path = osp.join(run_dir, 'manifest.json')
    if not osp.isfile(manifest_path):
        return dict(run_dir=run_dir, processed=0, files=[])

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    records = manifest.get('records', [])
    files = [r.get('output_file') for r in records[:max_items]]
    return dict(
        run_dir=run_dir,
        processed=int(manifest.get('processed', 0)),
        output_format=manifest.get('output_format', None),
        files=files,
    )


def main():
    args = parse_args()

    if not args.skip_infer:
        if args.sample_indices and (args.random_sample or args.random_train_sample):
            raise ValueError('--sample-indices cannot be combined with random sampling flags')
        if args.random_sample and args.random_train_sample:
            raise ValueError('--random-sample cannot be combined with --random-train-sample')
        if args.random_sample and args.max_samples <= 0:
            raise ValueError('--random-sample requires --max-samples > 0')

    if args.skip_infer:
        if not args.run_dir_a or not args.run_dir_b:
            raise ValueError('--skip-infer requires both --run-dir-a and --run-dir-b')
        run_dir_a = args.run_dir_a
        run_dir_b = args.run_dir_b
    else:
        if args.eval_metrics and args.output_format != 'npz':
            raise ValueError('--eval-metrics requires --output-format npz when running inference')
        run_dir_a = run_inference(args.history_a, args)
        run_dir_b = run_inference(args.history_b, args)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.eval_metrics:
        eval_spec = _resolve_eval_spec(args.config, args.split)
        metrics_a = evaluate_run(run_dir_a, eval_spec)
        metrics_b = evaluate_run(run_dir_b, eval_spec)

        delta_miou = round(metrics_b['mIoU'] - metrics_a['mIoU'], 2)
        delta_iou = round(metrics_b['IoU'] - metrics_a['IoU'], 2)

        summary = dict(
            config=args.config,
            weights=args.weights,
            split=args.split,
            history_a=args.history_a,
            history_b=args.history_b,
            result_a=metrics_a,
            result_b=metrics_b,
            delta=dict(
                mIoU=delta_miou,
                IoU=delta_iou,
            ),
        )

        summary_path = osp.join(
            args.save_dir,
            f'compare_h{args.history_a}_h{args.history_b}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print('\n[Compare] Result')
        print(f"  history={args.history_a}: mIoU={metrics_a['mIoU']:.2f}, IoU={metrics_a['IoU']:.2f}, eval_samples={metrics_a['evaluated_samples']}")
        print(f"  history={args.history_b}: mIoU={metrics_b['mIoU']:.2f}, IoU={metrics_b['IoU']:.2f}, eval_samples={metrics_b['evaluated_samples']}")
        print(f'  delta (h{args.history_b} - h{args.history_a}): mIoU={delta_miou:+.2f}, IoU={delta_iou:+.2f}')
        print(f'[Compare] Summary saved to: {summary_path}')
        return

    preview_a = _collect_preview(run_dir_a)
    preview_b = _collect_preview(run_dir_b)

    summary = dict(
        config=args.config,
        weights=args.weights,
        split=args.split,
        history_a=args.history_a,
        history_b=args.history_b,
        max_samples=args.max_samples,
        output_format=args.output_format,
        run_a=preview_a,
        run_b=preview_b,
    )

    summary_path = osp.join(
        args.save_dir,
        f'compare_h{args.history_a}_h{args.history_b}_preview.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n[Compare] Preview mode (no metric evaluation)')
    print(f'  history={args.history_a} run: {preview_a["run_dir"]}')
    print(f'  history={args.history_b} run: {preview_b["run_dir"]}')
    print(f'  summary: {summary_path}')
    print('  You can open generated PLY files with Open3D.')


if __name__ == '__main__':
    main()
