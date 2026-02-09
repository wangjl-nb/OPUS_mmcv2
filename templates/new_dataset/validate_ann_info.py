#!/usr/bin/env python3
"""Validate ann_file schema for Occ3D dataset adapter templates."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def _load_infos(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, list):
        return data
    raise TypeError(f'Unsupported ann format: {type(data)}')


def _shape_ok(value, expected):
    arr = np.asarray(value)
    return tuple(arr.shape) == tuple(expected)


def _check_cam(cam_name, cam_info, prefix):
    errors = []
    required = [
        'data_path', 'timestamp',
        'sensor2global_translation', 'sensor2global_rotation',
        'cam_intrinsic',
    ]
    for key in required:
        if key not in cam_info:
            errors.append(f'{prefix}.{cam_name}: missing `{key}`')

    if 'sensor2global_translation' in cam_info and not _shape_ok(cam_info['sensor2global_translation'], (3,)):
        errors.append(f'{prefix}.{cam_name}.sensor2global_translation: expect shape (3,)')
    if 'sensor2global_rotation' in cam_info and not _shape_ok(cam_info['sensor2global_rotation'], (3, 3)):
        errors.append(f'{prefix}.{cam_name}.sensor2global_rotation: expect shape (3,3)')
    if 'cam_intrinsic' in cam_info and not _shape_ok(cam_info['cam_intrinsic'], (3, 3)):
        errors.append(f'{prefix}.{cam_name}.cam_intrinsic: expect shape (3,3)')

    return errors


def _check_info(info, idx):
    errors = []
    prefix = f'info[{idx}]'

    required_top = [
        'token', 'scene_name', 'timestamp',
        'ego2global_translation', 'ego2global_rotation',
        'lidar2ego_translation', 'lidar2ego_rotation',
        'cams', 'cam_sweeps',
        'lidar_path', 'lidar_points', 'lidar_sweeps',
    ]
    for key in required_top:
        if key not in info:
            errors.append(f'{prefix}: missing `{key}`')

    if 'ego2global_translation' in info and not _shape_ok(info['ego2global_translation'], (3,)):
        errors.append(f'{prefix}.ego2global_translation: expect shape (3,)')
    if 'ego2global_rotation' in info and not _shape_ok(info['ego2global_rotation'], (4,)):
        errors.append(f'{prefix}.ego2global_rotation: expect shape (4,), quaternion [w,x,y,z]')
    if 'lidar2ego_translation' in info and not _shape_ok(info['lidar2ego_translation'], (3,)):
        errors.append(f'{prefix}.lidar2ego_translation: expect shape (3,)')
    if 'lidar2ego_rotation' in info and not _shape_ok(info['lidar2ego_rotation'], (4,)):
        errors.append(f'{prefix}.lidar2ego_rotation: expect shape (4,), quaternion [w,x,y,z]')

    cams = info.get('cams', None)
    if not isinstance(cams, dict) or len(cams) == 0:
        errors.append(f'{prefix}.cams: expect non-empty dict')
    else:
        for cam_name, cam_info in cams.items():
            if not isinstance(cam_info, dict):
                errors.append(f'{prefix}.cams.{cam_name}: expect dict')
                continue
            errors.extend(_check_cam(cam_name, cam_info, f'{prefix}.cams'))

    cam_sweeps = info.get('cam_sweeps', None)
    if not isinstance(cam_sweeps, list):
        errors.append(f'{prefix}.cam_sweeps: expect list')
    elif cam_sweeps:
        first = cam_sweeps[0]
        if not isinstance(first, dict):
            errors.append(f'{prefix}.cam_sweeps[0]: expect dict[cam_name->cam_info]')
        else:
            for cam_name, cam_info in first.items():
                if isinstance(cam_info, dict):
                    errors.extend(_check_cam(cam_name, cam_info, f'{prefix}.cam_sweeps[0]'))

    lidar_points = info.get('lidar_points', None)
    if not isinstance(lidar_points, dict) or 'lidar_path' not in lidar_points:
        errors.append(f'{prefix}.lidar_points: expect dict with key `lidar_path`')

    lidar_sweeps = info.get('lidar_sweeps', None)
    if not isinstance(lidar_sweeps, list):
        errors.append(f'{prefix}.lidar_sweeps: expect list')
    elif lidar_sweeps:
        first = lidar_sweeps[0]
        if not isinstance(first, dict):
            errors.append(f'{prefix}.lidar_sweeps[0]: expect dict')
        else:
            for key in ['data_path', 'timestamp', 'sensor2lidar_rotation', 'sensor2lidar_translation']:
                if key not in first:
                    errors.append(f'{prefix}.lidar_sweeps[0]: missing `{key}`')
            if 'sensor2lidar_rotation' in first and not _shape_ok(first['sensor2lidar_rotation'], (3, 3)):
                errors.append(f'{prefix}.lidar_sweeps[0].sensor2lidar_rotation: expect shape (3,3)')
            if 'sensor2lidar_translation' in first and not _shape_ok(first['sensor2lidar_translation'], (3,)):
                errors.append(f'{prefix}.lidar_sweeps[0].sensor2lidar_translation: expect shape (3,)')

    return errors


def main():
    parser = argparse.ArgumentParser(description='Validate ann_file schema for Occ3D templates.')
    parser.add_argument('--ann-file', required=True, help='Path to train/val/test pkl')
    parser.add_argument('--num-samples', type=int, default=3, help='How many samples to check from head')
    args = parser.parse_args()

    ann_path = Path(args.ann_file)
    if not ann_path.exists():
        print(f'[ERROR] ann file not found: {ann_path}')
        return 2

    infos = _load_infos(ann_path)
    if not infos:
        print('[ERROR] ann file is empty')
        return 2

    n = min(len(infos), max(1, args.num_samples))
    all_errors = []
    for i in range(n):
        all_errors.extend(_check_info(infos[i], i))

    print(f'Checked {n} sample(s) from {ann_path}')
    if all_errors:
        print(f'Found {len(all_errors)} issue(s):')
        for err in all_errors:
            print(f'  - {err}')
        return 1

    print('Schema check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
