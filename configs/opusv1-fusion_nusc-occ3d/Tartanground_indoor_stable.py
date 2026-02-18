default_scope = 'mmdet3d'
custom_imports = dict(imports=['models', 'loaders'], allow_failed_imports=False)

dataset_type = 'TartangroundOcc3DDataset'
dataset_root = '/root/wjl/OPUS_mmcv2/data/TartanGround_Indoor/'
occ_root = '/root/wjl/OPUS_mmcv2/data/TartanGround_Indoor/gts/'

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
    'others', 'acousticceiling', 'airconditiongvent', 'atm', 'awning', 'bakerycounter',
    'bannersign', 'barcounter', 'barrel', 'barshelf', 'beam', 'bed',
    'bench', 'bleach', 'blinds', 'boatcovered', 'book', 'boxedfood',
    'boxedlaundrydetergent', 'bp', 'building', 'buildingtunnel', 'cabinet', 'cabinets',
    'cable', 'cabledrum', 'cameraactor', 'car', 'carpet', 'cashregister',
    'ceiling', 'cementcolumn', 'chain', 'chair', 'chairs', 'chasis',
    'childactor', 'chipscylinder', 'cieling', 'cleaningpowder', 'column', 'concretebarricade',
    'couch', 'counch', 'cranerail', 'cucumbers', 'cupboard', 'curtain',
    'decorativecookiebox', 'desk', 'donuts', 'door', 'doorway', 'dumpster',
    'elevator', 'elevatordoor', 'fabricsoftener', 'fence', 'fencecolumn', 'floor',
    'flowerpot', 'fluorescentlight', 'foldingscreen', 'frozenfoodbagged', 'fryingoil', 'garagedoor',
    'garbagecan', 'gate', 'grass', 'ground', 'handtruck', 'heater',
    'instancedfoliageactor', 'ivy', 'laboratorytable', 'lamp', 'largeconduit', 'light',
    'lightbulbstring', 'meatchub', 'metalcieling', 'metalfloor', 'metalhandrail', 'metalpanel',
    'metalpillar', 'metalplatform', 'metalpole', 'metalramp', 'metalstair', 'metalsupport',
    'metaltop', 'mirror', 'monitor', 'oranges', 'overheadcrane', 'painting',
    'papertowels', 'petfoodtub', 'piano', 'picture', 'pictureframe', 'pillar',
    'pipe', 'plant', 'plantholder', 'platform', 'post', 'railgatebase',
    'railing', 'railtrack', 'railtrackend', 'receptiondesk', 'refrigerator', 'road',
    'robotarm', 'rock', 'rollingcabinet', 'roof', 'rug', 'saucebottle',
    'screendivider', 'seat', 'sewergrate', 'shampoo', 'shelf', 'shoppingbags',
    'sidewalk', 'sign', 'sky', 'sodabottle', 'sofa', 'stair',
    'stairs', 'staticmeshactor', 'storagerack', 'streetlight', 'supportarch', 'supportbeam',
    'supportcolumn', 'table', 'tireassembly', 'toolbox', 'trashcan', 'tree',
    'tunnel', 'tv', 'vendingmachine', 'vent', 'ventilationgrill', 'ventpipe',
    'vine', 'walkway', 'wall', 'wallarch', 'wallarchway', 'walllight',
    'walls', 'window', 'windowframe', 'woodenbox', 'woodenpallet', 'woodenplank',
]
occ_eval_names = occ_names + ['free']

# Precomputed from semantic_unify_report.json.
# cls_weights in [1, 30] by inverse-frequency sqrt mapping:
#   w_i = round(1 + 29 * norm((max_count / count_i) ** 0.5))
# others(id=0) is manually fixed to 10.
rare_classes = [
    13, 26, 50, 65, 76, 140, 134, 112, 56, 39, 32, 5,
    29, 55, 19, 96, 37, 66, 106, 129, 53, 61, 23, 92,
]
cls_weights = [
    10, 2, 15, 23, 7, 28, 26, 14, 18, 12, 7, 22, 20, 30, 10, 13,
    17, 14, 26, 27, 1, 3, 5, 26, 7, 18, 29, 16, 9, 28, 2, 9,
    28, 7, 20, 12, 12, 27, 22, 28, 14, 18, 15, 26, 16, 10, 14, 25,
    21, 13, 29, 4, 8, 27, 5, 28, 28, 7, 15, 1, 26, 26, 26, 18,
    21, 29, 27, 25, 4, 3, 21, 20, 4, 20, 24, 14, 29, 7, 11, 12,
    2, 5, 9, 6, 16, 11, 8, 19, 9, 12, 24, 23, 26, 25, 17, 23,
    27, 21, 25, 22, 16, 5, 2, 17, 22, 18, 27, 18, 8, 12, 26, 26,
    28, 9, 13, 14, 23, 7, 6, 24, 22, 12, 23, 5, 16, 25, 5, 7,
    2, 27, 8, 15, 4, 16, 28, 13, 5, 4, 19, 9, 29, 15, 23, 24,
    4, 16, 15, 8, 21, 11, 16, 4, 1, 6, 20, 9, 5, 15, 7, 17,
    23, 26,
]
# Soften static class weighting for multi-scene stability.
cls_weights = [max(1, int(round(w * 0.6))) for w in cls_weights]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-20.0, -20.0, -3.0, 20.0, 20.0, 5.0]
pc_voxel_size = [0.05, 0.05, 0.05]
voxel_size = [0.05, 0.05, 0.05]

dataset_cfg = dict(
    cam_types=['CAM_LEFT', 'CAM_BACK', 'CAM_FRONT', 'CAM_BOTTOM', 'CAM_TOP', 'CAM_RIGHT'],
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


# architecture config (capacity/performance trade-offs)
# - num_query: number of occupancy queries (higher -> better recall, more memory/latency).
# - num_frames: temporal frames consumed by camera branch.
# - num_refines: points-per-query schedule across decoder layers.
embed_dims = 256
num_layers = 6
num_query = 6400  # Main occupancy query budget.
num_frames = 9  # 1 current + 8 history sweeps.
num_levels = 4
num_points = 4
num_refines = [
                1,
                4,
                4,
                16,
                32,
                64,
            ]  # Decoder point expansion per layer.

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
    sparse_shape=[161, 800, 800],
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
    in_channels=1152,
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
        empty_label=dataset_cfg['empty_label'],  # Index of free/unoccupied voxel label.
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
            # Phase-2 (default OFF): re-allocate query budget once during decoding.
            query_allocator=dict(
                enabled=True,  # Keep disabled for safe rollout; enable for gray testing.
                switch_layer=2,  # Apply once after decoder layer index 1.
                context_ratio=0.6,  # Ratio of high-confidence context queries to keep.
                detail_jitter_std=0.01,  # Jitter for detail queries (decoded XYZ space).
                score_weights=dict(nonempty=0.5, uncertainty=0.5),  # Weights for detail ranking.
            ),
            pc_range=point_cloud_range),
        # Classification and point-regression losses.
        loss_cls=dict(
            type='FocalLoss',
            _scope_='mmdet',
            gamma=2.0,  # Focus on hard classification samples.
            alpha=0.25,  # Class-balance factor in focal loss.
            loss_weight=1.3),  # Global weight of classification branch.
        loss_pts=dict(
            type='SmoothL1Loss',
            _scope_='mmdet',
            beta=0.2,  # Transition point between L1 and L2 region.
            loss_weight=0.8)),  # Global weight of point-regression branch.
    train_cfg=dict(
        pts=dict(
            # Base long-tail priors.
            cls_weights=cls_weights,  # Per-class static re-weight for classification.
            rare_classes=rare_classes,  # Fallback tail class ids.
            rare_weights=2,  # Legacy rare-class minimum regression weight.
            hard_camera_mask=True,  # Only keep camera-visible voxels as GT anchors.

            # Phase-1: tail-aware dynamic weighting (enabled by default).
            tail_focus=dict(
                enabled=True,
                policy='ema_freq',  # Select tail classes from EMA class frequency.
                ema_momentum=0.9,  # Higher -> smoother but slower adaptation.
                freq_thr=0.01,  # Class is tail when EMA frequency <= threshold.
                min_tail_classes=4,  # Force at least this many tail classes.
                max_tail_classes=16,  # Cap tail set size.
                sync_stats=True,  # All-reduce class stats in DDP.
                fallback='rare_classes',  # Use rare_classes when EMA result is unstable/empty.
                tail_weight=2,  # Extra weight multiplier/min-clamp for tail samples.
            ),

            # Phase-1: pred->gt regression hard mining.
            hard_mining=dict(
                enabled=True,
                pred_topk_ratio=0.75,  # Keep top-75% hardest pred->gt pairs by distance.
                min_keep=2048,  # Floor for kept pairs to avoid over-pruning.
            ),

            # Phase-1: class-balanced GT sampling before matching.
            gt_balance=dict(
                enabled=True,
                per_class_cap=8192,  # Max GT points per class.
                tail_min_keep=64,  # Tail classes are repeated to reach this minimum.
                sample_mode='deterministic',  # Deterministic sampling/repeat order.
            ),

            # Phase-1: query init mix (LiDAR FPS + random) with strict count conservation.
            query_init_mix=dict(
                enabled=True,
                lidar_ratio=0.7,  # Ratio of FPS queries from LiDAR points.
                random_ratio=0.3,  # Ratio of random queries.
                random_mode='uniform_pc_range',  # Back-fill random points from pc_range.
            ),
        )
    ),
    test_cfg=dict(
        pts=dict(
            score_thr=0.3,
            padding=False)
    )
)

# Image-domain augmentation (currently locked to fixed resize for small-object study).
ida_aug_conf = {
    'resize_lim': (1.0, 1.0),  # Locked: no random resize jitter.
    'final_dim': (640, 640),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 640, 'W': 640,
    'rand_flip': True,
}

imdecode_backend = 'pillow'
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color',imdecode_backend=imdecode_backend),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1,imdecode_backend=imdecode_backend, cam_types=dataset_cfg['cam_types']),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, use_dim=[0, 1, 2, 3, 4],
         pad_empty_sweeps=True, remove_close=True),
    dict(type='LiDARToOccSpace'),
    # dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=True, with_attr_label=False),
    dict(type='LoadOcc3DFromFile', occ_root=occ_root, path_template=dataset_cfg['occ_io']['path_template'], semantics_key=dataset_cfg['occ_io']['semantics_key'], mask_camera_key=dataset_cfg['occ_io']['mask_camera_key'], mask_lidar_key=dataset_cfg['occ_io']['mask_lidar_key'], class_names=dataset_cfg['class_names'], empty_label=dataset_cfg['empty_label']),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=object_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PackOcc3DInputs', meta_keys=(
        'sample_idx', 'sample_token', 'scene_name',
        'filename', 'ori_shape', 'img_shape', 'pad_shape',
        'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color',imdecode_backend=imdecode_backend),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True,imdecode_backend=imdecode_backend, cam_types=dataset_cfg['cam_types']),
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

batch_size = 16

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'train.pkl',
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=False),
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
        ann_file=dataset_root + 'val.pkl',
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True),
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
        ann_file=dataset_root + 'test.pkl',
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True),
    collate_fn=dict(type='pseudo_collate'),
)

optimizer = dict(
    type='AdamW',
    lr=1.5e-4,
    weight_decay=0.01
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    loss_scale=512.0,
    clip_grad=dict(max_norm=35, norm_type=2),
)

# learning policy
total_epochs = 100
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=1500),
    dict(
        type='CosineAnnealingLR',
        T_max=total_epochs,
        by_epoch=True,
        begin=0,
        end=total_epochs,
        eta_min=2e-4 * 1e-3),
]

# load pretrained weights
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='Occ3DMetric',
    ann_file=dataset_root + 'val.pkl',
    occ_root=occ_root,
    occ_path_template=dataset_cfg['occ_io']['path_template'],
    semantics_key=dataset_cfg['occ_io']['semantics_key'],
    mask_camera_key=dataset_cfg['occ_io']['mask_camera_key'],
    mask_lidar_key=dataset_cfg['occ_io']['mask_lidar_key'],
    empty_label=dataset_cfg['empty_label'],
    use_camera_mask=True,
    pc_range=dataset_cfg['pc_range'],
    voxel_size=dataset_cfg['voxel_size'],
    class_names=dataset_cfg['class_names'],
    miou_num_workers=32,
    # Extra reporting focused on long-tail classes.
    focus_eval=dict(
        enabled=True,
        policy='ema_freq',  # Auto-pick small/tail classes by frequency.
        ema_momentum=0.9,
        freq_thr=0.02,
        min_classes=8,
        max_classes=24,
        fallback='rare_classes',
        rare_classes=rare_classes,
    ),
)
test_evaluator = dict(
    type='Occ3DMetric',
    ann_file=dataset_root + 'test.pkl',
    occ_root=occ_root,
    occ_path_template=dataset_cfg['occ_io']['path_template'],
    semantics_key=dataset_cfg['occ_io']['semantics_key'],
    mask_camera_key=dataset_cfg['occ_io']['mask_camera_key'],
    mask_lidar_key=dataset_cfg['occ_io']['mask_lidar_key'],
    empty_label=dataset_cfg['empty_label'],
    use_camera_mask=True,
    pc_range=dataset_cfg['pc_range'],
    voxel_size=dataset_cfg['voxel_size'],
    class_names=dataset_cfg['class_names'],
    miou_num_workers=32,
    # Extra reporting focused on long-tail classes.
    focus_eval=dict(
        enabled=True,
        policy='ema_freq',  # Auto-pick small/tail classes by frequency.
        ema_momentum=0.9,
        freq_thr=0.02,
        min_classes=8,
        max_classes=24,
        fallback='rare_classes',
        rare_classes=rare_classes,
    ),
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=5, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)

visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(type='TensorboardVisBackend')],
)

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)

randomness = dict(seed=0, deterministic=False)

# load pretrained weights
load_from = 'pretrain/fusion_pretrain_model.pth'

# resume the last training
resume_from = None
