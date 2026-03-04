_base_ = ['./tartanground_demo_r50_640x640_9f_100e.py']

# =========================
# Core experiment knobs
# =========================
# Point input switch:
# - 'lidar': original LiDAR file + sweeps pipeline
# - 'depth': multi-view depth -> pseudo-LiDAR points
point_input_source = 'depth'
assert point_input_source in ['lidar', 'depth']
eval_score_thr = 0.3

depth_points_cfg = dict(
    type='LoadPointsFromMultiViewDepth',
    # Keep current-frame points exact time=0 for query init with init_pos_lidar='curr'.
    sample_stride=4,
    max_points_total=560000,
    depth_min=0.01,
    depth_max=30.0,
    # Tartanground camera extrinsics are consistent with OpenCV camera frame:
    # x right, y down, z forward. This matches LiDAR frame mapping better.
    coord_convention='opencv',
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    time_dim=4,
    strict_depth_exist=True,
    fallback_depth_from_image_path=True,
)

if point_input_source == 'depth':
    point_load_transforms = [depth_points_cfg]
else:
    point_load_transforms = [
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
        dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, use_dim=[0, 1, 2, 3, 4],
             pad_empty_sweeps=True, remove_close=True),
    ]

# Unified split pkl naming by a single suffix variable.
ann_pkl_suffix = '_with_mapanything_depth' if point_input_source == 'depth' else '' # depth gt
train_ann_file = f'{_base_.dataset_root}train{ann_pkl_suffix}.pkl'
val_ann_file = f'{_base_.dataset_root}val{ann_pkl_suffix}.pkl'
test_ann_file = f'{_base_.dataset_root}test{ann_pkl_suffix}.pkl'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color', imdecode_backend=_base_.imdecode_backend),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_base_.num_frames - 1,
         imdecode_backend=_base_.imdecode_backend, cam_types=_base_.dataset_cfg['cam_types']),
    *point_load_transforms,
    dict(type='LiDARToOccSpace'),
    dict(type='LoadOcc3DFromFile', occ_root=_base_.occ_root,
         path_template=_base_.dataset_cfg['occ_io']['path_template'],
         semantics_key=_base_.dataset_cfg['occ_io']['semantics_key'],
         mask_camera_key=_base_.dataset_cfg['occ_io']['mask_camera_key'],
         mask_lidar_key=_base_.dataset_cfg['occ_io']['mask_lidar_key'],
         class_names=_base_.dataset_cfg['class_names'],
         empty_label=_base_.dataset_cfg['empty_label']),
    dict(type='RandomTransformImage', ida_aug_conf=_base_.ida_aug_conf, training=True),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='PackOcc3DInputs', meta_keys=(
        'sample_idx', 'sample_token', 'scene_name',
        'filename', 'ori_shape', 'img_shape', 'pad_shape',
        'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color', imdecode_backend=_base_.imdecode_backend),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_base_.num_frames - 1,
         test_mode=True, imdecode_backend=_base_.imdecode_backend, cam_types=_base_.dataset_cfg['cam_types']),
    *point_load_transforms,
    dict(type='LiDARToOccSpace'),
    dict(type='RandomTransformImage', ida_aug_conf=_base_.ida_aug_conf, training=False),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='PackOcc3DInputs', meta_keys=(
        'sample_idx', 'sample_token', 'scene_name',
        'filename', 'ori_shape', 'img_shape', 'pad_shape',
        'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp'))
]

train_dataloader = dict(dataset=dict(ann_file=train_ann_file, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(ann_file=val_ann_file, pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(ann_file=test_ann_file, pipeline=test_pipeline))

# Depth pseudo-points have lower early-stage confidence; 0.3 can over-prune all
# predictions and make val mIoU collapse to 0.0 in first epochs.
model = dict(
    test_cfg=dict(
        pts=dict(
            score_thr=eval_score_thr,
        ),
    ),
)
