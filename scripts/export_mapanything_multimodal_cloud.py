#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import pickle
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export one pkl sample to fused 3D cloud using MapAnything modalities (depth+intrinsics+pose).')
    parser.add_argument(
        '--pkl',
        type=str,
        default='/root/wjl/OPUS_mmcv2/data/tartanground_demo/test.pkl',
        help='Path to generated pkl file.')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index inside pkl infos.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='/root/wjl/OPUS_mmcv2/outputs/mapanything_multimodal_cloud',
        help='Output directory for ply/bin/json.')
    parser.add_argument('--min-depth', type=float, default=1e-6, help='Minimum valid depth.')
    parser.add_argument('--max-depth', type=float, default=80.0, help='Maximum valid depth.')
    parser.add_argument('--stride', type=int, default=1, help='Pixel sampling stride for unprojection.')
    parser.add_argument(
        '--cams',
        type=str,
        nargs='*',
        default=None,
        help='Optional camera names to use, e.g. CAM_FRONT CAM_LEFT.')
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=0.0,
        help='Optional voxel downsample size in meters. 0 disables downsample.')
    parser.add_argument(
        '--save-bin',
        action='store_true',
        help='Also save float32 .bin as [x,y,z,r,g,b] rows.')
    parser.add_argument(
        '--no-rgb',
        action='store_true',
        help='Do not load rgb, export geometry only (gray color).')
    return parser.parse_args()


def load_pkl_info(pkl_path: str, sample_idx: int) -> Dict:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data.get('infos', None)
    if not isinstance(infos, list) or len(infos) == 0:
        raise ValueError(f'Invalid pkl: infos is empty in {pkl_path}')
    if sample_idx < 0 or sample_idx >= len(infos):
        raise IndexError(f'sample-idx={sample_idx} out of range [0, {len(infos)-1}]')
    return infos[sample_idx]


def load_depth(depth_path: str) -> np.ndarray:
    if depth_path.endswith('.npy'):
        depth = np.load(depth_path)
    else:
        depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None:
            raise FileNotFoundError(f'Failed to decode depth: {depth_path}')
        depth = depth_rgba.view('<f4').squeeze()
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim > 2:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f'Depth map must be HxW, got {depth.shape} ({depth_path})')
    return depth


def load_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {image_path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def as_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotation)
    if rot.shape == (3, 3):
        return rot.astype(np.float64)

    quat = np.asarray(rotation, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f'Unsupported rotation shape: {rot.shape}')

    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise ValueError('Quaternion has zero norm')

    w, x, y, z = quat / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def build_cam2world(cam_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    rot = as_rotation_matrix(cam_info['sensor2global_rotation'])
    trans = np.asarray(cam_info['sensor2global_translation'], dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(rot)) or not np.all(np.isfinite(trans)):
        raise ValueError('Non-finite camera pose values found')
    return rot, trans


def unproject_depth_to_world(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    rot: np.ndarray,
    trans: np.ndarray,
    min_depth: float,
    max_depth: float,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    step = max(1, int(stride))

    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[0:h:step, 0:w:step]

    valid = np.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid].astype(np.float64)

    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f'cam_intrinsic must be 3x3, got {k.shape}')

    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        raise ValueError(f'Invalid intrinsics fx/fy: fx={fx}, fy={fy}')

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)

    # Column-vector convention: Xw = R @ Xc + t
    pts_world = (rot @ pts_cam.T).T + trans[None, :]

    return pts_world.astype(np.float32), u.astype(np.int32), v.astype(np.int32)


def voxel_downsample_numpy(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0 or points.shape[0] == 0:
        return points, colors

    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            f'voxel_size>0 requires open3d, import failed: {e}. '
            'Use --voxel-size 0 to disable downsample.')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float64) / 255.0))
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    ds_points = np.asarray(pcd.points, dtype=np.float32)
    ds_colors = np.clip(np.asarray(pcd.colors, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return ds_points, ds_colors


def write_ply_ascii(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError('xyz/rgb length mismatch')

    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {xyz.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        out = np.concatenate([xyz, rgb], axis=1)
        np.savetxt(f, out, fmt='%.5f %.5f %.5f %d %d %d')


def write_bin_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    arr = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    arr.tofile(path)


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    info = load_pkl_info(args.pkl, args.sample_idx)

    cams = info.get('cams', {})
    if not isinstance(cams, dict) or len(cams) == 0:
        raise ValueError('Sample has empty cams dict')

    cam_names: Iterable[str]
    if args.cams:
        cam_names = args.cams
    else:
        cam_names = cams.keys()

    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    per_cam_stats = {}
    use_rgb = not args.no_rgb

    for cam_name in cam_names:
        if cam_name not in cams:
            raise KeyError(f'Camera {cam_name} not found in sample cams')
        cam_info = cams[cam_name]

        image_path = cam_info.get('data_path', None)
        depth_path = cam_info.get('depth_path', None)
        if not isinstance(depth_path, str) or not depth_path:
            raise KeyError(f'{cam_name} missing depth_path')

        depth = load_depth(depth_path)
        rot, trans = build_cam2world(cam_info)
        intrinsics = np.asarray(cam_info.get('cam_intrinsic', None), dtype=np.float32)

        points_world, u_idx, v_idx = unproject_depth_to_world(
            depth=depth,
            intrinsics=intrinsics,
            rot=rot,
            trans=trans,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            stride=args.stride,
        )

        if use_rgb:
            if not isinstance(image_path, str) or not image_path:
                raise KeyError(f'{cam_name} missing data_path for rgb loading')
            rgb = load_rgb(image_path)
            if rgb.shape[:2] != depth.shape:
                raise ValueError(
                    f'{cam_name} image/depth shape mismatch: image={rgb.shape[:2]} depth={depth.shape}')
            colors = rgb[v_idx, u_idx, :].astype(np.uint8)
        else:
            colors = np.full((points_world.shape[0], 3), 180, dtype=np.uint8)

        all_points.append(points_world)
        all_colors.append(colors)
        per_cam_stats[cam_name] = int(points_world.shape[0])

    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=np.float32)
    colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.uint8)

    points, colors = voxel_downsample_numpy(points, colors, voxel_size=float(args.voxel_size))

    sample_token = str(info.get('token', f'sample_{args.sample_idx}'))
    stem = f'mapanything_multimodal_{args.sample_idx:06d}_{sample_token}'

    ply_path = osp.join(args.out_dir, f'{stem}.ply')
    write_ply_ascii(ply_path, points, colors)

    bin_path = None
    if args.save_bin:
        bin_path = osp.join(args.out_dir, f'{stem}.bin')
        write_bin_xyzrgb(bin_path, points, colors)

    summary = {
        'pkl': args.pkl,
        'sample_idx': int(args.sample_idx),
        'sample_token': sample_token,
        'scene_name': str(info.get('scene_name', 'unknown')),
        'num_cams_used': int(len(list(cam_names))),
        'per_cam_point_count': per_cam_stats,
        'total_points_saved': int(points.shape[0]),
        'min_depth': float(args.min_depth),
        'max_depth': float(args.max_depth),
        'stride': int(args.stride),
        'voxel_size': float(args.voxel_size),
        'ply_path': ply_path,
        'bin_path': bin_path,
    }

    summary_path = osp.join(args.out_dir, f'{stem}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('[Done] Export finished')
    print(f'  ply: {ply_path}')
    if bin_path:
        print(f'  bin: {bin_path}')
    print(f'  summary: {summary_path}')
    print(f'  total_points: {points.shape[0]}')


if __name__ == '__main__':
    main()
