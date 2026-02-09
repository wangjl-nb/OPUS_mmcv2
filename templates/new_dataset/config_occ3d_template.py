"""Config template for a new Occ3D-like dataset.

Copy to configs/... and fill placeholders.
"""

default_scope = 'mmdet3d'
custom_imports = dict(
    imports=['models', 'loaders', 'loaders.my_occ3d_dataset'],
    allow_failed_imports=False)

# ------------------------
# Dataset
# ------------------------
dataset_type = 'MyOcc3DDataset'
dataset_root = '/path/to/dataset_root/'
occ_root = '/path/to/occ_root/'

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

occ_names = ['class_a', 'class_b']
point_cloud_range = [-20.0, -20.0, -3.0, 20.0, 20.0, 5.0]
voxel_size = [0.1, 0.1, 0.1]

# Centralized dataset params.
dataset_cfg = dict(
    cam_types=['CAM_FRONT', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BACK'],
    num_views=4,
    occ_io=dict(
        path_template='{scene_name}/{token}/labels.npz',
        semantics_key='semantics',
        mask_camera_key='mask_camera',
        mask_lidar_key='mask_lidar',
    ),
    class_names=occ_names + ['free'],
    empty_label=len(occ_names),
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    ray=dict(
        num_workers=8,
        max_origins=8,
        origin_xy_bound=39.0,
        lidar=dict(
            mode='nuscenes_default',
            azimuth_step_deg=1.0,
            pitch_angles=None,
        ),
    ),
)

# ------------------------
# Pipeline
# ------------------------
num_frames = 8

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(
        type='LoadMultiViewImageFromMultiSweeps',
        sweeps_num=num_frames - 1,
        cam_types=dataset_cfg['cam_types']),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LiDARToOccSpace'),
    dict(
        type='LoadOcc3DFromFile',
        occ_root=occ_root,
        path_template=dataset_cfg['occ_io']['path_template'],
        semantics_key=dataset_cfg['occ_io']['semantics_key'],
        mask_camera_key=dataset_cfg['occ_io']['mask_camera_key'],
        mask_lidar_key=dataset_cfg['occ_io']['mask_lidar_key'],
        class_names=dataset_cfg['class_names'],
        empty_label=dataset_cfg['empty_label']),
    dict(type='PackOcc3DInputs', meta_keys=(
        'sample_idx', 'sample_token', 'scene_name',
        'filename', 'ori_shape', 'img_shape', 'pad_shape',
        'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp')),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(
        type='LoadMultiViewImageFromMultiSweeps',
        sweeps_num=num_frames - 1,
        test_mode=True,
        cam_types=dataset_cfg['cam_types']),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LiDARToOccSpace'),
    dict(type='PackOcc3DInputs', meta_keys=(
        'sample_idx', 'sample_token', 'scene_name',
        'filename', 'ori_shape', 'img_shape', 'pad_shape',
        'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=False),
    collate_fn=dict(type='pseudo_collate'),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True),
    collate_fn=dict(type='pseudo_collate'),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='Occ3DMetric',
    ann_file=dataset_root + 'val.pkl',
    occ_root=occ_root,
    occ_path_template=dataset_cfg['occ_io']['path_template'],
    semantics_key=dataset_cfg['occ_io']['semantics_key'],
    mask_camera_key=dataset_cfg['occ_io']['mask_camera_key'],
    mask_lidar_key=dataset_cfg['occ_io']['mask_lidar_key'],
    ray_num_workers=dataset_cfg['ray']['num_workers'],
    ray_cfg=dataset_cfg['ray'],
    empty_label=dataset_cfg['empty_label'],
    use_camera_mask=True,
    compute_rayiou=False,
    pc_range=dataset_cfg['pc_range'],
    voxel_size=dataset_cfg['voxel_size'],
    class_names=dataset_cfg['class_names'],
)

test_evaluator = val_evaluator
