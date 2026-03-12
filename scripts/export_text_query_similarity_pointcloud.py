#!/usr/bin/env python3
import argparse
import copy
import importlib
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from mmengine.config import Config, DictAction
from mmengine.dataset import DefaultSampler, pseudo_collate
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.registry import DATASETS, MODELS
from torch.utils.data import DataLoader, Subset

from models.bbox.utils import decode_points


DEFAULT_SAVE_DIR = '/root/wjl/OPUS_mmcv2/demos/text_query_similarity'
DEFAULT_TALK2DINO_ENV = 'talk2dino'
DEFAULT_TALK2DINO_CONFIG = (
    '/root/wjl/Talk2DINO/tartanground_label_ae/configs/pca256_talk2dino_reg.json'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export sparse OPUS voxels colored by text-query feature similarity.')
    parser.add_argument('--config', required=True, help='Path to frozen or source config file.')
    parser.add_argument('--weights', required=True,
                        help='Checkpoint file, last_checkpoint file, or run directory containing last_checkpoint.')
    parser.add_argument('--text-query', required=True, help='Free-form text query.')
    parser.add_argument('--save-dir', default=DEFAULT_SAVE_DIR, help='Output root directory.')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val',
                        help='Dataset split.')
    parser.add_argument('--max-samples', type=int, default=1,
                        help='Number of samples to export when --sample-indices is not set.')
    parser.add_argument('--sample-indices', type=int, nargs='*', default=None,
                        help='Explicit dataset indices to export.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Inference batch size. Recommended 1.')
    parser.add_argument('--num-workers', type=int, default=2, help='Dataloader workers.')
    parser.add_argument('--max-voxels', type=int, default=300000,
                        help='Max voxels saved to PLY per sample; <=0 keeps all.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic behavior.')
    parser.add_argument('--disable-camera-mask', action='store_true',
                        help='Do not crop sparse predictions by mask_camera before saving outputs.')
    parser.add_argument('--talk2dino-env', default=DEFAULT_TALK2DINO_ENV,
                        help='Conda environment used by the Talk2DINO text helper.')
    parser.add_argument('--talk2dino-config', default=DEFAULT_TALK2DINO_CONFIG,
                        help='Talk2DINO text latent config json.')
    parser.add_argument('--helper-script', default=osp.join(
        ROOT, 'scripts', 'encode_text_query_talk2dino.py'),
        help='Path to Talk2DINO text helper script.')
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
        with open(path, 'r') as handle:
            line = handle.readline().strip()
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
    for transform in pipeline:
        if transform.get('type') == 'LoadMultiViewImageFromMultiSweeps':
            transform['force_offline'] = True


def extract_img_metas(data_samples):
    if data_samples is None:
        return []
    if isinstance(data_samples, list):
        return [sample.metainfo for sample in data_samples]
    return [data_samples.metainfo]


def forward_query_outputs(model, inputs, data_samples):
    img = inputs.get('img') if isinstance(inputs, dict) else inputs
    points = inputs.get('points') if isinstance(inputs, dict) else None
    mapanything_extra = inputs.get('mapanything_extra') if isinstance(inputs, dict) else None
    img_metas = extract_img_metas(data_samples)

    if isinstance(img, torch.Tensor) and img.dim() >= 1 and hasattr(model, '_normalize_mapanything_extra'):
        mapanything_extra = model._normalize_mapanything_extra(
            mapanything_extra, batch_size=int(img.shape[0]))

    need_img_branch = getattr(model, '_need_img_branch', None)
    if callable(need_img_branch):
        img_feats = model.extract_img_feat(
            img,
            img_metas,
            points=points,
            mapanything_extra=mapanything_extra) if need_img_branch() else None
    else:
        img_feats = None if not getattr(model, 'with_img_backbone', False) else model.extract_img_feat(img, img_metas)

    pts_feats = None
    if hasattr(model, '_extract_pts_feat_for_head'):
        pts_feats = model._extract_pts_feat_for_head(points)
        if pts_feats is not None and hasattr(model, '_maybe_drop_lidar_feat'):
            pts_feats = model._maybe_drop_lidar_feat(pts_feats)
    elif getattr(model, 'with_pts_backbone', False):
        pts_feats = model.extract_pts_feat(points)
        if isinstance(pts_feats, (list, tuple)):
            if hasattr(model, 'final_conv'):
                pts_feats = model.final_conv(pts_feats[0])
            elif len(pts_feats) == 1:
                pts_feats = pts_feats[0]

    tpv_feats = model._extract_tpv_feat_for_head(points) if hasattr(model, '_extract_tpv_feat_for_head') else None

    try:
        outs = model.pts_bbox_head(
            mlvl_feats=img_feats,
            pts_feats=pts_feats,
            tpv_feats=tpv_feats,
            points=points,
            img_metas=img_metas)
    except TypeError:
        try:
            outs = model.pts_bbox_head(mlvl_feats=img_feats, points=points, img_metas=img_metas)
        except TypeError:
            outs = model.pts_bbox_head(mlvl_feats=img_feats, img_metas=img_metas)
    return outs, img_metas


def occ_idx_to_xyz(occ_loc, pc_range, voxel_size):
    occ_loc = np.asarray(occ_loc, dtype=np.float64)
    return (occ_loc + 0.5) * voxel_size[None, :] + pc_range[None, :3]


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


def apply_sparse_mask(occ_loc, values, dense_mask):
    if dense_mask is None or occ_loc.size == 0:
        return occ_loc, values

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
    return occ_loc[keep], values[keep]


def build_red_blue_rgb(similarity):
    activation = np.clip(np.asarray(similarity, dtype=np.float32), 0.0, 1.0)
    rgb = np.empty((activation.shape[0], 3), dtype=np.uint8)
    rgb[:, 0] = np.clip(np.round(255.0 * activation), 0, 255).astype(np.uint8)
    rgb[:, 1] = 0
    rgb[:, 2] = np.clip(np.round(255.0 * (1.0 - activation)), 0, 255).astype(np.uint8)
    return rgb, activation


def write_similarity_ply(path, xyz, rgb, similarity):
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    similarity = np.asarray(similarity, dtype=np.float32).reshape(-1, 1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'xyz must have shape [N, 3], got {xyz.shape}')
    if rgb.shape != (xyz.shape[0], 3):
        raise ValueError(f'rgb must have shape {(xyz.shape[0], 3)}, got {rgb.shape}')
    if similarity.shape[0] != xyz.shape[0]:
        raise ValueError(f'similarity length mismatch: {similarity.shape[0]} vs {xyz.shape[0]}')

    os.makedirs(osp.dirname(path), exist_ok=True)
    data = np.concatenate([xyz, rgb, similarity], axis=1)
    with open(path, 'w') as handle:
        handle.write('ply\n')
        handle.write('format ascii 1.0\n')
        handle.write(f'element vertex {xyz.shape[0]}\n')
        handle.write('property float x\n')
        handle.write('property float y\n')
        handle.write('property float z\n')
        handle.write('property uchar red\n')
        handle.write('property uchar green\n')
        handle.write('property uchar blue\n')
        handle.write('property float similarity\n')
        handle.write('end_header\n')
        if data.shape[0] > 0:
            np.savetxt(handle, data, fmt='%.4f %.4f %.4f %d %d %d %.6f')


def resolve_conda_executable():
    conda_bin = shutil.which('conda')
    if conda_bin is not None:
        return conda_bin
    fallback = '/root/miniconda3/bin/conda'
    if osp.isfile(fallback):
        return fallback
    raise FileNotFoundError('Failed to locate conda executable.')


def resolve_env_python(env_name):
    if osp.isabs(env_name):
        if osp.isfile(env_name):
            return env_name
        raise FileNotFoundError(f'Python executable not found: {env_name}')

    base_root = '/root/miniconda3'
    if env_name in ('base', 'root'):
        candidate = osp.join(base_root, 'bin', 'python')
    else:
        candidate = osp.join(base_root, 'envs', env_name, 'bin', 'python')
    if osp.isfile(candidate):
        return candidate
    return None


def run_text_helper(args, work_dir):
    helper_script = osp.abspath(args.helper_script)
    if not osp.isfile(helper_script):
        raise FileNotFoundError(f'Talk2DINO helper script not found: {helper_script}')
    if not osp.isfile(args.talk2dino_config):
        raise FileNotFoundError(f'Talk2DINO config not found: {args.talk2dino_config}')

    latent_path = osp.join(work_dir, 'query_latent.npy')
    meta_path = osp.join(work_dir, 'query_latent_meta.json')
    env_python = resolve_env_python(args.talk2dino_env)
    if env_python is not None:
        cmd = [
            env_python,
            helper_script,
            '--text-query',
            args.text_query,
            '--config-json',
            osp.abspath(args.talk2dino_config),
            '--out-npy',
            latent_path,
            '--out-json',
            meta_path,
        ]
    else:
        conda_bin = resolve_conda_executable()
        cmd = [
            conda_bin,
            'run',
            '-n',
            args.talk2dino_env,
            'python',
            helper_script,
            '--text-query',
            args.text_query,
            '--config-json',
            osp.abspath(args.talk2dino_config),
            '--out-npy',
            latent_path,
            '--out-json',
            meta_path,
        ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            'Talk2DINO text helper failed.\n'
            f'Command: {" ".join(cmd)}\n'
            f'STDOUT:\n{result.stdout}\n'
            f'STDERR:\n{result.stderr}')

    latent = np.load(latent_path).astype(np.float32)
    if latent.shape != (latent.shape[0],):
        latent = latent.reshape(-1)
    if latent.ndim != 1:
        raise ValueError(f'Unexpected text latent shape: {latent.shape}')
    if not np.all(np.isfinite(latent)):
        raise ValueError('Text latent contains non-finite values.')
    norm = float(np.linalg.norm(latent))
    if norm <= 1e-8:
        raise ValueError('Text latent has near-zero norm.')
    latent = latent / norm
    np.save(latent_path, latent.astype(np.float32))

    with open(meta_path, 'r', encoding='utf-8') as handle:
        meta = json.load(handle)
    with open(meta_path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    return latent.astype(np.float32), meta, latent_path, meta_path


def extract_similarity_results(head, pred_dicts, text_latent):
    if head.test_cfg.get('padding', True):
        raise NotImplementedError(
            'Text-query voxel feature export requires test_cfg.padding=False.')

    if head.score_mode == 'binary_occ':
        all_score_preds = pred_dicts['all_occ_scores']
        score_channels = head.occ_out_channels
    else:
        all_score_preds = pred_dicts['all_cls_scores']
        score_channels = head.num_classes
    all_refine_pts = pred_dicts['all_refine_pts']
    all_feat_scores = pred_dicts.get('all_feat_scores', None)
    if all_feat_scores is None or all_feat_scores[-1] is None:
        raise RuntimeError('Model output does not contain all_feat_scores; feature similarity is unavailable.')

    score_preds = all_score_preds[-1].sigmoid()
    refine_pts = all_refine_pts[-1]
    feat_scores = all_feat_scores[-1]
    text_latent = torch.as_tensor(text_latent, device=refine_pts.device, dtype=feat_scores.dtype)
    text_latent = F.normalize(text_latent, dim=0)

    batch_size = refine_pts.shape[0]
    ctr_dist_thr = head.test_cfg.get('ctr_dist_thr', 3.0)
    score_thr = head.test_cfg.get('score_thr', 0.0)

    result_list = []
    for batch_idx in range(batch_size):
        refine_pts_i = decode_points(refine_pts[batch_idx], head.pc_range)
        score_preds_i = score_preds[batch_idx]
        feat_scores_i = feat_scores[batch_idx]

        centers = refine_pts_i.mean(dim=1, keepdim=True)
        ctr_dists = torch.norm(refine_pts_i - centers, dim=-1)
        mask_dist = ctr_dists < ctr_dist_thr
        mask_score = (score_preds_i > score_thr).any(dim=-1)
        mask = mask_dist & mask_score
        refine_pts_i = refine_pts_i[mask]
        score_preds_i = score_preds_i[mask]
        feat_scores_i = feat_scores_i[mask]

        if refine_pts_i.numel() == 0:
            result_list.append(dict(
                occ_loc=np.zeros((0, 3), dtype=np.int64),
                similarity=np.zeros((0,), dtype=np.float32)))
            continue

        pts = torch.cat([refine_pts_i, score_preds_i, feat_scores_i], dim=-1)
        pts_infos, voxels, num_pts = head.voxel_generator(pts)
        voxels = torch.flip(voxels, [1]).long()
        score_start = 3
        score_end = score_start + score_channels
        scores = pts_infos[..., score_start:score_end]
        scores = scores.sum(dim=1) / num_pts[..., None]
        voxel_feats = pts_infos[..., score_end:]
        voxel_feats = voxel_feats.sum(dim=1) / num_pts[..., None]

        if head.score_mode == 'binary_occ':
            voxel_keep_mask = scores.squeeze(-1) > score_thr
            voxels = voxels[voxel_keep_mask]
            voxel_feats = voxel_feats[voxel_keep_mask]

        if voxels.numel() == 0:
            result_list.append(dict(
                occ_loc=np.zeros((0, 3), dtype=np.int64),
                similarity=np.zeros((0,), dtype=np.float32)))
            continue

        voxel_feats = F.normalize(voxel_feats, dim=-1)
        similarity = voxel_feats @ text_latent
        result_list.append(dict(
            occ_loc=voxels.detach().cpu().numpy(),
            similarity=similarity.detach().cpu().numpy().astype(np.float32),
            feature_dims=int(voxel_feats.shape[-1]),
        ))

    return result_list


def save_sample_outputs(sample_dir, sample_idx, token, occ_loc, similarity, pc_range, voxel_size, max_voxels, seed):
    os.makedirs(sample_dir, exist_ok=True)
    xyz = occ_idx_to_xyz(occ_loc, pc_range, voxel_size).astype(np.float32)
    similarity = np.asarray(similarity, dtype=np.float32)
    npz_path = osp.join(sample_dir, f'{sample_idx:0>6}_{token}_textsim_voxel.npz')
    np.savez_compressed(npz_path, occ_loc=occ_loc.astype(np.int64), xyz=xyz, similarity=similarity)

    save_xyz = xyz
    save_sim = similarity
    if max_voxels > 0 and occ_loc.shape[0] > max_voxels:
        rng = np.random.default_rng(seed + sample_idx)
        choice = rng.choice(occ_loc.shape[0], size=max_voxels, replace=False)
        save_xyz = xyz[choice]
        save_sim = similarity[choice]

    save_rgb, alpha = build_red_blue_rgb(save_sim)
    ply_path = osp.join(sample_dir, f'{sample_idx:0>6}_{token}_textsim_voxel.ply')
    write_similarity_ply(ply_path, save_xyz, save_rgb, save_sim)
    return dict(
        npz_path=npz_path,
        ply_path=ply_path,
        num_voxels_full=int(occ_loc.shape[0]),
        num_voxels_ply=int(save_xyz.shape[0]),
        similarity_min=float(similarity.min()) if similarity.size > 0 else None,
        similarity_max=float(similarity.max()) if similarity.size > 0 else None,
        similarity_mean=float(similarity.mean()) if similarity.size > 0 else None,
        alpha_min=float(alpha.min()) if alpha.size > 0 else None,
        alpha_max=float(alpha.max()) if alpha.size > 0 else None,
        alpha_mean=float(alpha.mean()) if alpha.size > 0 else None,
    )


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

    run_name = osp.splitext(osp.basename(args.config))[0]
    run_name += '_' + osp.splitext(osp.basename(ckpt_path))[0]
    run_name += '_' + sanitize_name(args.text_query)
    run_name += '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_root = osp.join(args.save_dir, f'{args.split}_{run_name}')
    os.makedirs(out_root, exist_ok=True)

    text_latent, text_meta, latent_path, latent_meta_path = run_text_helper(args, out_root)

    dataset_cfg = select_dataset_cfg(cfg, args.split)
    maybe_force_offline_sweeps(dataset_cfg)
    dataset = DATASETS.build(dataset_cfg)
    base_dataset = dataset
    if hasattr(base_dataset, 'full_init'):
        base_dataset.full_init()

    total_count = len(dataset)
    if args.sample_indices is not None and len(args.sample_indices) > 0:
        indices = [index for index in args.sample_indices if 0 <= index < total_count]
        if len(indices) == 0:
            raise ValueError('No valid --sample-indices after bounds check.')
        dataset = Subset(dataset, indices)
        print(f'[textsim] using explicit indices: {indices}')
    elif args.max_samples > 0:
        count = min(args.max_samples, total_count)
        dataset = Subset(dataset, list(range(count)))

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

    pc_range = np.asarray(cfg.point_cloud_range, dtype=np.float64)
    voxel_size = np.asarray(cfg.voxel_size, dtype=np.float64)

    print(f'[textsim] config: {args.config}')
    print(f'[textsim] checkpoint: {ckpt_path}')
    print(f'[textsim] split: {args.split}, dataset size: {len(dataset)}')
    print(f'[textsim] text query: {args.text_query}')
    print(f'[textsim] output dir: {out_root}')

    records = []
    processed = 0
    camera_mask_warned = False

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = normalize_inputs(move_to_device(data['inputs'], device))
            data_samples = move_to_device(data['data_samples'], device)

            outs, _ = forward_query_outputs(model, inputs, data_samples)
            similarity_results = extract_similarity_results(model.pts_bbox_head, outs, text_latent)

            for local_idx, data_sample in enumerate(data_samples):
                meta = getattr(data_sample, 'metainfo', {})
                sample_idx = int(meta.get('sample_idx', batch_idx))
                token = sanitize_name(meta.get('sample_token', f'{sample_idx:06d}'))
                scene = sanitize_name(meta.get('scene_name', 'unknown_scene'))
                sample_dir = osp.join(out_root, scene, token)

                occ_loc = similarity_results[local_idx]['occ_loc']
                similarity = similarity_results[local_idx]['similarity']

                if not args.disable_camera_mask:
                    mask_camera, _ = get_mask_camera(data_sample, base_dataset)
                    if mask_camera is None:
                        if not camera_mask_warned:
                            print('[textsim] mask_camera not found, keep unmasked predictions.')
                            camera_mask_warned = True
                    else:
                        occ_loc, similarity = apply_sparse_mask(occ_loc, similarity, mask_camera)

                save_info = save_sample_outputs(
                    sample_dir=sample_dir,
                    sample_idx=sample_idx,
                    token=token,
                    occ_loc=occ_loc,
                    similarity=similarity,
                    pc_range=pc_range,
                    voxel_size=voxel_size,
                    max_voxels=args.max_voxels,
                    seed=args.seed)

                record = dict(
                    sample_idx=sample_idx,
                    scene_name=scene,
                    token=token,
                    feature_dims=similarity_results[local_idx].get('feature_dims', None),
                    disable_camera_mask=bool(args.disable_camera_mask),
                    **save_info,
                )
                records.append(record)
                processed += 1
                print(f'[textsim] exported sample {processed}: {scene}/{token} '
                      f'voxels={record["num_voxels_full"]}')

    manifest = dict(
        config=osp.abspath(args.config),
        weights=ckpt_path,
        split=args.split,
        text_query=args.text_query,
        disable_camera_mask=bool(args.disable_camera_mask),
        helper_script=osp.abspath(args.helper_script),
        talk2dino_env=args.talk2dino_env,
        talk2dino_config=osp.abspath(args.talk2dino_config),
        text_latent_npy=latent_path,
        text_latent_meta=latent_meta_path,
        text_meta=text_meta,
        max_voxels=int(args.max_voxels),
        processed=int(processed),
        records=records,
    )
    manifest_path = osp.join(out_root, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f'[textsim] manifest: {manifest_path}')
    print(f'[textsim] done. Exported {processed} samples to: {out_root}')


if __name__ == '__main__':
    main()
