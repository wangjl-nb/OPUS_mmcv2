#!/usr/bin/env python3
import argparse
import copy
import json
import os
import os.path as osp
import pickle
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

ROOT = '/root/wjl/OPUS_mmcv2'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loaders.pipelines.loading import LoadMapAnythingExtraFromDepth
from models.mapanything.opus_mapanything_wrapper import MapAnythingOPUSEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize MapAnything multimodal encoder features on one sample.')
    parser.add_argument(
        '--config',
        type=str,
        default='/root/wjl/OPUS_mmcv2/configs/opusv1-fusion_nusc-occ3d/Tartanground_office_res_map_sum_0.5_gts0.1.py',
        help='Config path used to build MapAnything encoder settings.')
    parser.add_argument(
        '--pkl',
        type=str,
        default=None,
        help='Optional pkl path. If omitted, derive from config dataset root test.pkl')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index in pkl infos.')
    parser.add_argument('--view-idx', type=int, default=0, help='View index inside sample cams order.')
    parser.add_argument('--cam-name', type=str, default=None, help='Optional camera name (overrides view-idx).')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='/root/wjl/OPUS_mmcv2/outputs/mapanything_feature_vis',
        help='Output directory for visualization images.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device, e.g. cuda:0 or cpu.')
    parser.add_argument(
        '--disable-anyup',
        action='store_true',
        default=False,
        help='Disable AnyUp branch. By default, AnyUp visualization is generated when cfg enables AnyUp.')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def get_default_pkl_from_cfg(cfg):
    dataset_root = cfg.get('dataset_root', None)
    if isinstance(dataset_root, str):
        return osp.join(dataset_root, 'test.pkl')
    test_loader = cfg.get('test_dataloader', None)
    if isinstance(test_loader, dict):
        ds = test_loader.get('dataset', None)
        if isinstance(ds, dict) and isinstance(ds.get('ann_file', None), str):
            return ds['ann_file']
    raise ValueError('Cannot infer pkl path from config, please provide --pkl')


def load_sample_info(pkl_path, sample_idx):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data.get('infos', None)
    if not isinstance(infos, list) or len(infos) == 0:
        raise ValueError(f'Invalid pkl: no infos in {pkl_path}')
    if sample_idx < 0 or sample_idx >= len(infos):
        raise IndexError(f'sample-idx={sample_idx} out of range [0,{len(infos)-1}]')
    return infos[sample_idx]


def resolve_view(cam_names, view_idx, cam_name):
    if cam_name is not None:
        if cam_name not in cam_names:
            raise KeyError(f'cam-name={cam_name} not found in {cam_names}')
        return cam_names.index(cam_name), cam_name
    if view_idx < 0 or view_idx >= len(cam_names):
        raise IndexError(f'view-idx={view_idx} out of range [0,{len(cam_names)-1}]')
    return view_idx, cam_names[view_idx]


def find_mapextra_cfg_from_pipeline(pipeline):
    if not isinstance(pipeline, list):
        return dict(strict=True)
    for step in pipeline:
        if isinstance(step, dict) and step.get('type', None) == 'LoadMapAnythingExtraFromDepth':
            cfg = copy.deepcopy(step)
            cfg.pop('type', None)
            return cfg
    return dict(strict=True)


def build_results_for_mapextra(info):
    cams = info['cams']
    cam_names = list(cams.keys())
    filenames = []
    imgs = []
    for cam_name in cam_names:
        cam_info = cams[cam_name]
        image_path = cam_info['data_path']
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        filenames.append(image_path)
        imgs.append(image_rgb)

    results = dict(
        filename=filenames,
        img=imgs,
        cams=info.get('cams', {}),
        cam_sweeps=info.get('cam_sweeps', []),
    )
    return results, cam_names


def pca_rgb(feature_chw):
    # feature_chw: [C,H,W]
    feat = np.asarray(feature_chw, dtype=np.float32)
    c, h, w = feat.shape
    x = feat.reshape(c, -1).T  # [HW, C]
    if x.shape[0] == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    x = x - np.mean(x, axis=0, keepdims=True)

    # SVD-based PCA (stable and dependency-free)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(3, vt.shape[0])
    y = x @ vt[:k].T
    if k < 3:
        pad = np.zeros((y.shape[0], 3 - k), dtype=np.float32)
        y = np.concatenate([y, pad], axis=1)

    y_img = y.reshape(h, w, 3)
    out = np.zeros_like(y_img, dtype=np.float32)
    for ch in range(3):
        comp = y_img[..., ch]
        lo = np.percentile(comp, 1.0)
        hi = np.percentile(comp, 99.0)
        if hi <= lo:
            out[..., ch] = 0.0
        else:
            out[..., ch] = np.clip((comp - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def save_image(path, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def cosine_similarity_map(feat_a_chw, feat_b_chw, target_hw=None):
    a = torch.from_numpy(np.asarray(feat_a_chw, dtype=np.float32)).unsqueeze(0)  # [1,C,H,W]
    b = torch.from_numpy(np.asarray(feat_b_chw, dtype=np.float32)).unsqueeze(0)  # [1,C,H,W]
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError(
            f'Expected CHW features for similarity, got a={a.shape}, b={b.shape}')
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f'Feature channel mismatch for similarity: {a.shape[1]} vs {b.shape[1]}')

    if target_hw is None:
        target_hw = (int(a.shape[-2]), int(a.shape[-1]))
    target_hw = (int(target_hw[0]), int(target_hw[1]))

    if a.shape[-2:] != target_hw:
        a = F.interpolate(a, size=target_hw, mode='bilinear', align_corners=False)
    if b.shape[-2:] != target_hw:
        b = F.interpolate(b, size=target_hw, mode='bilinear', align_corners=False)

    a = F.normalize(a, p=2, dim=1, eps=1e-6)
    b = F.normalize(b, p=2, dim=1, eps=1e-6)
    sim = torch.sum(a * b, dim=1).squeeze(0)  # [H,W], range about [-1,1]
    return sim.detach().cpu().numpy().astype(np.float32)


def similarity_to_heatmap(sim_map):
    sim = np.asarray(sim_map, dtype=np.float32)
    sim = np.clip(sim, -1.0, 1.0)
    sim_u8 = ((sim + 1.0) * 0.5 * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(sim_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def summarize_similarity(sim_map):
    s = np.asarray(sim_map, dtype=np.float32).reshape(-1)
    if s.size == 0:
        return dict(mean=None, min=None, max=None, p10=None, p90=None)
    return dict(
        mean=float(np.mean(s)),
        min=float(np.min(s)),
        max=float(np.max(s)),
        p10=float(np.percentile(s, 10)),
        p90=float(np.percentile(s, 90)),
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    pkl_path = args.pkl if args.pkl is not None else get_default_pkl_from_cfg(cfg)
    info = load_sample_info(pkl_path, args.sample_idx)

    results, cam_names = build_results_for_mapextra(info)
    selected_view_idx, selected_cam_name = resolve_view(cam_names, args.view_idx, args.cam_name)

    mapextra_cfg = find_mapextra_cfg_from_pipeline(cfg.get('train_pipeline', []))
    mapextra_loader = LoadMapAnythingExtraFromDepth(**mapextra_cfg)
    results = mapextra_loader(results)
    mapanything_extra = results['mapanything_extra']

    imgs_np = results['img']
    img_raw = imgs_np[selected_view_idx]

    img_tensor = torch.from_numpy(np.stack(imgs_np, axis=0)).permute(0, 3, 1, 2).contiguous()
    img_tensor = img_tensor.unsqueeze(0)  # [1, TN, 3, H, W]

    enc_cfg = copy.deepcopy(cfg.model['img_encoder'])
    if enc_cfg.get('type', None) == 'MapAnythingOccEncoder':
        wrapper_cfg = dict(
            repo_root=enc_cfg['repo_root'],
            mapanything_model_cfg=enc_cfg['mapanything_model_cfg'],
            mapanything_preprocess_cfg=enc_cfg.get('mapanything_preprocess_cfg', {}),
            anyup_cfg=enc_cfg.get('anyup_cfg', {}),
            freeze=bool(enc_cfg.get('freeze', True)),
            enable_random_mask=False,
            random_mask_cfg=None,
            strip_to_feature_mode=True,
            strict_shapes=True,
            expect_contiguous=True,
            tn_align_mode='strict',
            lidar_injection='shared',
            preserve_meta_keys=None,
        )
    else:
        raise ValueError('Expected cfg.model.img_encoder.type == MapAnythingOccEncoder')

    if args.disable_anyup:
        wrapper_cfg['anyup_cfg'] = dict(enabled=False)

    wrapper = MapAnythingOPUSEncoder(**wrapper_cfg)
    device = torch.device(args.device if ('cuda' in args.device and torch.cuda.is_available()) else 'cpu')
    wrapper = wrapper.to(device)
    wrapper.eval()

    img_tensor = img_tensor.to(device)

    # Build processed views exactly like wrapper.forward
    batch_views = wrapper.input_adapter(
        img=img_tensor,
        points=None,
        img_metas=[{}],
        mapanything_extra=mapanything_extra,
    )
    sample_views = batch_views[0]
    processed_views = wrapper._preprocess_inputs(sample_views, **wrapper.mapanything_preprocess_cfg)
    processed_views_before = [dict(v) for v in processed_views]
    processed_views = wrapper._preprocess_input_views_for_inference(processed_views)
    wrapper._normalize_is_metric_scale(processed_views, sample_idx=0)
    wrapper._validate_preprocessed_views_for_geometry(processed_views_before, processed_views, sample_idx=0)
    processed_views = wrapper._move_processed_views_to_model_device(processed_views)

    # Force deterministic full-geometry usage during this visualization
    if hasattr(wrapper.map_model, '_configure_geometric_input_config'):
        wrapper.map_model._configure_geometric_input_config(
            use_calibration=True,
            use_depth=True,
            use_pose=True,
            use_depth_scale=True,
            use_pose_scale=True,
        )

    with torch.no_grad():
        # DINOv2 image-only encoder feature (before geometry fusion)
        dino_feats_per_view, _ = wrapper.map_model._encode_n_views(processed_views)
        # Final MapAnything fused feature (after geometry fusion)
        fused_feats_per_view = wrapper.map_model.forward(processed_views)
        anyup_feat_vis = None
        if wrapper.anyup_enabled:
            # Visualize AnyUp output right after upsampling fused MapAnything feature
            # (before pyramid downsampling in training path).
            hr_image = wrapper._view_image_from_processed_view(processed_views[selected_view_idx])
            upsample_hw = wrapper._compute_anyup_upsample_size(
                int(hr_image.shape[-2]), int(hr_image.shape[-1]))
            anyup_feat_vis = wrapper._run_anyup_model(
                hr_image,
                fused_feats_per_view[selected_view_idx],
                output_size=upsample_hw)
            if not isinstance(anyup_feat_vis, torch.Tensor) or anyup_feat_vis.dim() != 4:
                raise ValueError(
                    f'Unexpected AnyUp output shape for visualization: '
                    f'{type(anyup_feat_vis)} / {getattr(anyup_feat_vis, "shape", None)}')

    dino_feat = dino_feats_per_view[selected_view_idx][0].detach().float().cpu().numpy()
    fused_feat = fused_feats_per_view[selected_view_idx][0].detach().float().cpu().numpy()
    anyup_feat = None
    if anyup_feat_vis is not None:
        anyup_feat = anyup_feat_vis[0].detach().float().cpu().numpy()

    dino_vis = pca_rgb(dino_feat)
    fused_vis = pca_rgb(fused_feat)
    anyup_vis = pca_rgb(anyup_feat) if anyup_feat is not None else None

    # Pixel-wise cosine similarity maps
    sim_map_fused_vs_dino = cosine_similarity_map(
        fused_feat, dino_feat, target_hw=fused_feat.shape[1:])
    sim_vis_fused_vs_dino = similarity_to_heatmap(sim_map_fused_vs_dino)
    sim_stats_fused_vs_dino = summarize_similarity(sim_map_fused_vs_dino)

    sim_map_fused_vs_anyup = None
    sim_vis_fused_vs_anyup = None
    sim_stats_fused_vs_anyup = None
    if anyup_feat is not None:
        sim_map_fused_vs_anyup = cosine_similarity_map(
            fused_feat, anyup_feat, target_hw=fused_feat.shape[1:])
        sim_vis_fused_vs_anyup = similarity_to_heatmap(sim_map_fused_vs_anyup)
        sim_stats_fused_vs_anyup = summarize_similarity(sim_map_fused_vs_anyup)

    # Resize feature visualizations to raw image size for easy side-by-side comparison
    h_raw, w_raw = img_raw.shape[:2]
    dino_vis = cv2.resize(dino_vis, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR)
    fused_vis = cv2.resize(fused_vis, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR)
    sim_vis_fused_vs_dino = cv2.resize(
        sim_vis_fused_vs_dino, (w_raw, h_raw), interpolation=cv2.INTER_NEAREST)
    if anyup_vis is not None:
        anyup_vis = cv2.resize(anyup_vis, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR)
    if sim_vis_fused_vs_anyup is not None:
        sim_vis_fused_vs_anyup = cv2.resize(
            sim_vis_fused_vs_anyup, (w_raw, h_raw), interpolation=cv2.INTER_NEAREST)

    stem = f"sample{args.sample_idx:06d}_{selected_cam_name}"
    path_raw = osp.join(args.out_dir, f'{stem}_original.png')
    path_dino = osp.join(args.out_dir, f'{stem}_dinov2_pca.png')
    path_fused = osp.join(args.out_dir, f'{stem}_mapanything_fused_pca.png')
    path_anyup = osp.join(args.out_dir, f'{stem}_anyup_upsampled_pca.png')
    path_sim_dino = osp.join(args.out_dir, f'{stem}_sim_mapanything_vs_dinov2.png')
    path_sim_anyup = osp.join(args.out_dir, f'{stem}_sim_mapanything_vs_anyup.png')
    path_panel = osp.join(args.out_dir, f'{stem}_panel.png')

    save_image(path_raw, img_raw)
    save_image(path_dino, dino_vis)
    save_image(path_fused, fused_vis)
    save_image(path_sim_dino, sim_vis_fused_vs_dino)
    if anyup_vis is not None:
        save_image(path_anyup, anyup_vis)
    if sim_vis_fused_vs_anyup is not None:
        save_image(path_sim_anyup, sim_vis_fused_vs_anyup)

    panel_images = [img_raw, dino_vis, fused_vis]
    if anyup_vis is not None:
        panel_images.append(anyup_vis)
    panel_images.append(sim_vis_fused_vs_dino)
    if sim_vis_fused_vs_anyup is not None:
        panel_images.append(sim_vis_fused_vs_anyup)
    panel = np.concatenate(panel_images, axis=1)
    save_image(path_panel, panel)

    summary = {
        'config': args.config,
        'pkl': pkl_path,
        'sample_idx': int(args.sample_idx),
        'cam_name': selected_cam_name,
        'view_idx': int(selected_view_idx),
        'raw_image_shape': list(img_raw.shape),
        'dino_feat_shape': list(dino_feat.shape),
        'fused_feat_shape': list(fused_feat.shape),
        'anyup_feat_shape': list(anyup_feat.shape) if anyup_feat is not None else None,
        'similarity_stats': {
            'mapanything_vs_dinov2': sim_stats_fused_vs_dino,
            'mapanything_vs_anyup': sim_stats_fused_vs_anyup,
        },
        'files': {
            'original': path_raw,
            'dinov2_pca': path_dino,
            'mapanything_fused_pca': path_fused,
            'anyup_upsampled_pca': path_anyup if anyup_feat is not None else None,
            'sim_mapanything_vs_dinov2': path_sim_dino,
            'sim_mapanything_vs_anyup': path_sim_anyup if anyup_feat is not None else None,
            'panel': path_panel,
        },
    }
    summary_path = osp.join(args.out_dir, f'{stem}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('[Done] Saved visualization files:')
    print(f'  original: {path_raw}')
    print(f'  dinov2_pca: {path_dino}')
    print(f'  mapanything_fused_pca: {path_fused}')
    if anyup_feat is not None:
        print(f'  anyup_upsampled_pca: {path_anyup}')
    print(f'  sim_mapanything_vs_dinov2: {path_sim_dino}')
    if sim_vis_fused_vs_anyup is not None:
        print(f'  sim_mapanything_vs_anyup: {path_sim_anyup}')
    print(f'  sim_stats.mapanything_vs_dinov2: {sim_stats_fused_vs_dino}')
    if sim_stats_fused_vs_anyup is not None:
        print(f'  sim_stats.mapanything_vs_anyup: {sim_stats_fused_vs_anyup}')
    print(f'  panel: {path_panel}')
    print(f'  summary: {summary_path}')


if __name__ == '__main__':
    main()
