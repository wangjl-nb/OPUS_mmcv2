default_scope = 'mmdet3d'
custom_imports = dict(imports=['models', 'loaders'], allow_failed_imports=False)

dataset_type = 'NuScenesOcc3DDataset'
dataset_root = 'data/nuscenes/'
occ_root = 'data/nuscenes/gts/'

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

# For nuScenes we usually do 10-class detection
object_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
pc_voxel_size = [0.05, 0.05, 0.16]
voxel_size = [0.4, 0.4, 0.4]

dataset_cfg = dict(
    cam_types=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    num_views=input_modality.get('num_cams', 6),
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


# arch config
embed_dims = 256
num_layers = 6
num_query = 600
num_frames = 8
num_levels = 4
num_points = 4
num_refines = [1, 4, 16, 32, 64, 128]

img_backbone = dict(
    type='ResNet',
    _scope_='mmdet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)
img_neck = dict(
    type='FPN',
    _scope_='mmdet',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

pts_voxel_layer=dict(max_num_points=10, voxel_size=pc_voxel_size, deterministic=False,
                     max_voxels=(90000, 120000), point_cloud_range=point_cloud_range)
pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5)
pts_middle_encoder=dict(
    type='SparseEncoder',
    in_channels=5,
    sparse_shape=[41, 1600, 1600],
    output_channels=128,
    order=('conv', 'norm', 'act'),
    encoder_channels=((16, 16, 32), 
                      (32, 32, 64), 
                      (64, 64, 128), 
                      (128,128)),
    encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
    block_type='basicblock')
pts_backbone=dict(
    type='SECOND',
    in_channels=256,
    out_channels=[128, 256],
    layer_nums=[5, 5],
    layer_strides=[1, 2],
    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    conv_cfg=dict(type='Conv2d', bias=False))
pts_neck=dict(
    type='SECONDFPN',
    in_channels=[128, 256],
    out_channels=[256, 256],
    upsample_strides=[1, 2],
    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    upsample_cfg=dict(type='deconv', bias=False),
    use_conv_for_no_stride=True)

model = dict(
    type='OPUSV1Fusion',
    data_preprocessor=dict(type='BaseDataPreprocessor'),
    use_grid_mask=False,
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_voxel_layer=pts_voxel_layer,
    pts_voxel_encoder=pts_voxel_encoder,
    pts_middle_encoder=pts_middle_encoder,
    pts_backbone=pts_backbone,
    pts_neck=pts_neck,
    pts_bbox_head=dict(
        type='OPUSV1FusionHead',
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        init_pos_lidar='curr',
        transformer=dict(
            type='OPUSV1FusionTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_views=dataset_cfg['num_views'],
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            scales=[0.5],
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            _scope_='mmdet',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5)),
    train_cfg=dict(
        pts=dict(
            cls_weights=[
               
                ],
            )
        ),
    test_cfg=dict(
        pts=dict(
            score_thr=0.5,
            padding=True)
    )
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, cam_types=dataset_cfg['cam_types']),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, use_dim=[0, 1, 2, 3, 4],
         pad_empty_sweeps=True, remove_close=True),
    dict(type='LiDARToOccSpace'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadOcc3DFromFile', occ_root=occ_root, path_template=dataset_cfg['occ_io']['path_template'], semantics_key=dataset_cfg['occ_io']['semantics_key'], mask_camera_key=dataset_cfg['occ_io']['mask_camera_key'], mask_lidar_key=dataset_cfg['occ_io']['mask_lidar_key'], class_names=dataset_cfg['class_names'], empty_label=dataset_cfg['empty_label']),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=object_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PackOcc3DInputs', meta_keys=(
        'sample_idx', 'sample_token', 'scene_name',
        'filename', 'ori_shape', 'img_shape', 'pad_shape',
        'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True, cam_types=dataset_cfg['cam_types']),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, use_dim=[0, 1, 2, 3, 4],
         pad_empty_sweeps=True, remove_close=True),
    dict(type='LiDARToOccSpace'),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PackOcc3DInputs', meta_keys=(
        'sample_idx', 'sample_token', 'scene_name',
        'filename', 'ori_shape', 'img_shape', 'pad_shape',
        'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp'))
]

batch_size = 64

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    collate_fn=dict(type='pseudo_collate'),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
        box_type_3d='LiDAR'),
    collate_fn=dict(type='pseudo_collate'),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
        box_type_3d='LiDAR'),
    collate_fn=dict(type='pseudo_collate'),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    loss_scale=512.0,
    clip_grad=dict(max_norm=35, norm_type=2),
)

# learning policy
total_epochs = 100
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=total_epochs,
        by_epoch=True,
        begin=0,
        end=total_epochs,
        eta_min=2e-4 * 1e-3),
]

# load pretrained weights
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=total_epochs)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='Occ3DMetric',
    ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
    occ_root=occ_root,
    class_names=dataset_cfg['class_names'],
    pc_range=dataset_cfg['pc_range'],
    voxel_size=dataset_cfg['voxel_size'],
    occ_path_template=dataset_cfg['occ_io']['path_template'],
    semantics_key=dataset_cfg['occ_io']['semantics_key'],
    mask_camera_key=dataset_cfg['occ_io']['mask_camera_key'],
    mask_lidar_key=dataset_cfg['occ_io']['mask_lidar_key'],
    ray_num_workers=dataset_cfg['ray']['num_workers'],
    ray_cfg=dataset_cfg['ray'],
    empty_label=dataset_cfg['empty_label'],
    use_camera_mask=True,
    compute_rayiou=True,
)
test_evaluator = dict(
    type='Occ3DMetric',
    ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
    occ_root=occ_root,
    class_names=dataset_cfg['class_names'],
    pc_range=dataset_cfg['pc_range'],
    voxel_size=dataset_cfg['voxel_size'],
    occ_path_template=dataset_cfg['occ_io']['path_template'],
    semantics_key=dataset_cfg['occ_io']['semantics_key'],
    mask_camera_key=dataset_cfg['occ_io']['mask_camera_key'],
    mask_lidar_key=dataset_cfg['occ_io']['mask_lidar_key'],
    ray_num_workers=dataset_cfg['ray']['num_workers'],
    ray_cfg=dataset_cfg['ray'],
    empty_label=dataset_cfg['empty_label'],
    use_camera_mask=True,
    compute_rayiou=True,
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(type='TensorboardVisBackend')],
)

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)

randomness = dict(seed=0, deterministic=True)

# load pretrained weights
load_from = 'pretrain/fusion_pretrain_model.pth'
revise_keys = []

# resume the last training
resume_from = None

# other flags
debug = False
