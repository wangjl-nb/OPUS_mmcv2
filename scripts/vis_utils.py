import copy
import os
import os.path as osp
import shutil

import numpy as np


TEXTSIM_VIEW_CASES = (
    dict(name='view1_front', cameras=['CAM_FRONT']),
    dict(name='view2_front_right', cameras=['CAM_FRONT', 'CAM_RIGHT']),
    dict(name='view4_side', cameras=['CAM_LEFT', 'CAM_BACK', 'CAM_FRONT', 'CAM_RIGHT']),
)


def sanitize_name(value):
    return str(value).replace(' ', '_').replace('/', '_').replace('\\', '_')


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


def resolve_sample_indices(total_count,
                           max_samples,
                           sample_indices=None,
                           random_sample=False,
                           seed=0,
                           candidate_indices=None):
    if sample_indices is not None and len(sample_indices) > 0:
        indices = [int(i) for i in sample_indices if 0 <= int(i) < total_count]
        if not indices:
            raise ValueError('No valid --sample-indices after bounds check.')
        return indices

    pool = list(range(total_count)) if candidate_indices is None else [int(i) for i in candidate_indices]
    pool = [i for i in pool if 0 <= i < total_count]
    if not pool:
        return []

    if random_sample:
        if max_samples <= 0:
            raise ValueError('--random-sample requires --max-samples > 0')
        count = min(int(max_samples), len(pool))
        rng = np.random.RandomState(seed)
        return [int(i) for i in rng.choice(pool, size=count, replace=False).tolist()]

    if max_samples is not None and int(max_samples) > 0:
        return pool[:min(int(max_samples), len(pool))]

    return pool


def build_same_scene_history_indices(data_infos, target_indices, history_frames):
    target_indices = sorted({int(i) for i in target_indices})
    if history_frames <= 0:
        return target_indices, {int(i): [int(i)] for i in target_indices}

    context_indices = set()
    history_by_target = {}
    for target_idx in target_indices:
        if target_idx < 0 or target_idx >= len(data_infos):
            continue
        scene_name = data_infos[target_idx].get('scene_name', None)
        history = []
        cursor = target_idx
        while cursor >= 0 and len(history) < int(history_frames) + 1:
            info = data_infos[cursor]
            if info.get('scene_name', None) != scene_name:
                break
            history.append(int(cursor))
            cursor -= 1
        history.reverse()
        history_by_target[int(target_idx)] = history
        context_indices.update(history)

    return sorted(context_indices), history_by_target


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

    for candidate in candidates:
        if osp.isfile(candidate):
            return candidate
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
    for frame_idx in range(num_frames):
        block = timestamps[frame_idx * num_views:(frame_idx + 1) * num_views]
        block_means.append(float(np.mean(block)))

    current_block = int(np.argmax(np.asarray(block_means)))
    start = current_block * num_views
    return list(range(start, start + num_views))


def copy_current_frame_images(meta,
                              sample_dir,
                              data_root,
                              num_views,
                              out_dir_name='current_frame_images'):
    filenames = meta.get('filename', None)
    if not isinstance(filenames, list) or len(filenames) == 0:
        return 0

    keep_indices = pick_current_frame_indices(meta, num_views=int(num_views))
    if not keep_indices:
        return 0

    out_dir = osp.join(sample_dir, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    copied = 0
    records = []
    for local_idx, src_idx in enumerate(keep_indices):
        src_rel = filenames[src_idx]
        src_abs = resolve_image_path(src_rel, data_root=data_root)
        if src_abs is None:
            records.append(f'[{local_idx}] MISSING {src_rel}')
            continue

        base = osp.basename(src_abs)
        dst_name = f'view{local_idx:02d}_{base}'
        dst_path = osp.join(out_dir, dst_name)
        shutil.copy2(src_abs, dst_path)
        copied += 1
        records.append(f'[{local_idx}] {src_abs} -> {dst_name}')

    with open(osp.join(out_dir, 'mapping.txt'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(records) + '\n')
    return copied


def mask_from_bits(mask_camera_bits, camera_names, selected_names):
    mask = np.zeros_like(mask_camera_bits, dtype=np.bool_)
    selected = {str(name) for name in selected_names}
    for cam_idx, cam_name in enumerate(camera_names):
        if str(cam_name) in selected:
            mask |= (mask_camera_bits & (1 << cam_idx)) != 0
    return mask


def _iter_dataset_cfgs(cfg):
    for split_key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        split_cfg = cfg.get(split_key, None)
        if split_cfg is None or 'dataset' not in split_cfg:
            continue
        yield split_cfg['dataset']


def apply_camera_subset_to_cfg(cfg, active_camera_names):
    active_camera_names = [str(name) for name in active_camera_names]
    active_num_views = len(active_camera_names)
    if active_num_views <= 0:
        raise ValueError('active_camera_names must not be empty')

    for dataset_cfg in _iter_dataset_cfgs(cfg):
        inner_cfg = dataset_cfg.get('dataset_cfg', None)
        if isinstance(inner_cfg, dict):
            inner_cfg['cam_types'] = list(active_camera_names)
            inner_cfg['num_views'] = int(active_num_views)
            occ_io = inner_cfg.setdefault('occ_io', {})
            occ_io.setdefault('mask_camera_bits_key', 'mask_camera_bits')
            occ_io.setdefault('camera_names_key', 'camera_names')
            occ_io['mask_camera_select_names'] = list(active_camera_names)

        pipeline = dataset_cfg.get('pipeline', [])
        for transform in pipeline:
            if 'cam_types' in transform:
                transform['cam_types'] = list(active_camera_names)
            if transform.get('type') == 'LoadMultiViewImageFromFiles':
                transform['num_views'] = int(active_num_views)
            if transform.get('type') == 'LoadOcc3DFromFile':
                transform.setdefault('mask_camera_bits_key', 'mask_camera_bits')
                transform.setdefault('camera_names_key', 'camera_names')
                transform['mask_camera_select_names'] = list(active_camera_names)

    model_cfg = cfg.get('model', {})
    img_encoder_cfg = model_cfg.get('img_encoder', None)
    if isinstance(img_encoder_cfg, dict):
        img_encoder_cfg['num_views'] = int(active_num_views)
        anyup_cfg = img_encoder_cfg.get('anyup_cfg', None)
        if isinstance(anyup_cfg, dict) and 'view_batch_size' in anyup_cfg:
            anyup_cfg['view_batch_size'] = min(int(anyup_cfg['view_batch_size']), int(active_num_views))

    pts_bbox_head_cfg = model_cfg.get('pts_bbox_head', None)
    if isinstance(pts_bbox_head_cfg, dict):
        transformer_cfg = pts_bbox_head_cfg.get('transformer', None)
        if isinstance(transformer_cfg, dict):
            transformer_cfg['num_views'] = int(active_num_views)

    if 'num_views' in cfg:
        cfg['num_views'] = int(active_num_views)

    return cfg
