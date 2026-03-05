# -------------------- Runtime Scope --------------------
default_scope = 'mmdet3d'  # Registry scope used by MMEngine.
custom_imports = dict(
    imports=['models', 'loaders'],  # Project-local modules required by this config.
    allow_failed_imports=False,  # Fail fast if custom modules are missing.
)


# -------------------- Dataset Paths --------------------
dataset_type = 'TartangroundOcc3DDataset'  # Dataset class for Tartanground occupancy data.
dataset_root = '/root/wjl/OPUS_mmcv2/data/tartanground_demo/'  # Root for split pkl files and metadata.
occ_root = '/root/wjl/OPUS_mmcv2/data/tartanground_demo/gts/'  # Root for occupancy label npz files.


# -------------------- Experiment Knobs --------------------
point_input_source = 'lidar'  # Use raw LiDAR points for the point branch.
eval_score_thr = 0.3  # Inference score threshold for occupancy query filtering.
val_interval = 10  # Run validation every N epochs.
total_epochs = 150  # Train for 150 epochs.

# Use GT-depth split files (depth_gt setting).
ann_pkl_suffix = '_with_depth'  # train_with_depth.pkl / val_with_depth.pkl / test_with_depth.pkl
train_ann_file = f'{dataset_root}train{ann_pkl_suffix}.pkl'  # Training annotation pkl.
val_ann_file = f'{dataset_root}val{ann_pkl_suffix}.pkl'  # Validation annotation pkl.
test_ann_file = f'{dataset_root}test{ann_pkl_suffix}.pkl'  # Test annotation pkl.

input_modality = dict(
    use_lidar=True,  # Point branch consumes pseudo-LiDAR/depth points.
    use_camera=True,  # Multi-view images are enabled.
    use_radar=False,  # Radar is disabled.
    use_map=False,  # HD map input is disabled.
    use_external=True,  # External signals (for this project) are allowed.
)


# -------------------- Class Definitions --------------------
object_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]  # Detection names kept for dataset compatibility.

occ_names = [
    'others', 'bathroompartition', 'binder', 'blinds', 'book', 'bottle',
    'briefcase', 'building', 'cardboardbox', 'carpet', 'ceiling', 'ceilingvent',
    'chair', 'clipboard', 'clock', 'coffeecup', 'computerbaseunit', 'computermouse',
    'computermousepad', 'copier', 'cup', 'cupboard', 'desk', 'donut', 'door',
    'eraser', 'filecabinet', 'firealarm', 'firesprinkler', 'floor', 'goblet',
    'handdryer', 'headsculpture', 'jug', 'keyboard', 'lamp', 'laptop', 'light',
    'marker', 'microphone', 'mirror', 'monitor', 'paperstack', 'paperstand',
    'papertray', 'pen', 'pencil', 'pencilcontainer', 'phone', 'picture', 'pictureframe',
    'plant', 'printer', 'rack', 'receptiondesk', 'rollingcabinet', 'securitycamera',
    'sign', 'sink', 'skysphere', 'smartphone', 'smokedetector', 'soapdispenser', 'sofa',
    'sticker', 'table', 'tabletpc', 'toilet', 'trashcan', 'tray', 'tv', 'urinal', 'vase',
    'wall', 'watercooler', 'whiteboardmagnet', 'window', 'writingmat', 'z'
]  # Occupancy semantic classes (without free).
occ_eval_names = occ_names + ['free']  # Evaluation classes with free-space appended.

rare_classes = [13, 17, 23, 25, 28, 32, 39, 45, 60, 64, 66, 75, 77, 78]  # Tail class ids.
cls_weights = [
    1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
    1, 1, 1, 5, 1, 3, 1, 3, 2, 1,
    2, 1, 1, 5, 1, 10, 1, 1, 3, 1,
    1, 1, 3, 1, 2, 2, 2, 1, 1, 6,
    1, 1, 1, 3, 2, 5, 2, 2, 2, 1,
    1, 1, 1, 2, 1, 1, 2, 2, 1, 1,
    5, 2, 1, 1, 4, 1, 7, 1, 1, 1,
    1, 1, 1, 1, 1, 10, 1, 8, 7,
]  # Static per-class reweighting for long-tail occupancy.


# -------------------- Geometry --------------------
point_cloud_range = [-20.0, -20.0, -3.0, 20.0, 20.0, 5.0]  # [x_min, y_min, z_min, x_max, y_max, z_max]
pc_voxel_size = [0.05, 0.05, 0.05]  # Point-branch voxelization size.
voxel_size = [0.05, 0.05, 0.05]  # Occupancy voxel size for head/evaluator.

dataset_cfg = dict(
    cam_types=['CAM_LEFT', 'CAM_BACK', 'CAM_FRONT', 'CAM_BOTTOM', 'CAM_TOP', 'CAM_RIGHT'],  # Camera order.
    num_views=input_modality.get('num_cams', 6),  # Number of camera views.
    occ_io=dict(
        path_template='{scene_name}/{token}/labels.npz',  # Relative path format under occ_root.
        semantics_key='semantics',  # Semantic volume key in npz.
        mask_camera_key='mask_camera',  # Camera visibility mask key.
        mask_lidar_key='mask_lidar',  # LiDAR visibility mask key.
    ),
    class_names=occ_names + ['free'],  # Full occupancy class names.
    empty_label=len(occ_names),  # Index of free-space class.
    pc_range=point_cloud_range,  # Keep geometry shared across dataset/model/eval.
    voxel_size=voxel_size,  # Voxel size used for metrics and decoding.
    ray=dict(
        num_workers=8,  # Workers for ray-related preprocessing/metrics.
        max_origins=8,  # Max origins sampled in ray utilities.
        origin_xy_bound=39.0,  # XY bound for origin sampling.
        lidar=dict(
            mode='nuscenes_default',  # Ray casting preset.
            azimuth_step_deg=1.0,  # Horizontal angle step.
            pitch_angles=None,  # Use preset pitch layout.
        ),
    ),
)


# -------------------- Model Capacity --------------------
embed_dims = 256  # Shared embedding channels.
num_layers = 6  # Transformer decoder layers.
num_query = 9600  # Occupancy query count.
num_frames = 9  # 1 current + 8 sweeps.
num_levels = 4  # FPN feature levels.
num_points = 4  # Sampling points per query/level.
num_refines = [1, 4, 16, 32, 64, 128]  # Progressive point refinements per decoder layer.

query_init_mix_cfg = dict(
    enabled=True,  # Enable mixed query initialization.
    lidar_ratio=0.7,  # Fraction from point-based FPS seeds.
    random_ratio=0.3,  # Fraction from random seeds.
    random_mode='uniform_pc_range',  # Uniform random in point cloud range.
)


# -------------------- Image Branch --------------------
img_backbone = dict(
    type='ResNet',  # 2D backbone.
    _scope_='mmdet',  # Use mmdet registry for ResNet.
    depth=50,  # ResNet-50.
    num_stages=4,  # Four stages.
    out_indices=(0, 1, 2, 3),  # Export all stages.
    frozen_stages=1,  # Freeze stem + stage1 for stability.
    norm_cfg=dict(type='BN2d', requires_grad=True),  # BatchNorm2d with trainable params.
    norm_eval=True,  # Freeze BN running stats in train.
    style='pytorch',  # PyTorch ResNet style.
    with_cp=True,  # Enable checkpointing to save memory.
)

img_neck = dict(
    type='FPN',  # Feature pyramid network.
    _scope_='mmdet',  # Use mmdet registry.
    in_channels=[256, 512, 1024, 2048],  # ResNet stage channels.
    out_channels=embed_dims,  # Unified FPN channels.
    num_outs=num_levels,  # Number of output scales.
)

mapanything_model_cfg = dict(
    load_type='from_pretrained',  # Load MapAnything from pretrained assets.
    model_id_or_path='facebook/map-anything-apache-v1',  # HF model id.
    from_pretrained_kwargs=dict(local_files_only=True),  # Do not download during model load.
)

mapanything_preprocess_cfg = dict(
    resize_mode='fixed_size',  # Deterministic resize for consistent features.
    size=(616, 616),  # Input spatial size for MapAnything.
    patch_size=14,  # Patch size used by the encoder.
    norm_type='dinov2',  # Normalization convention.
)

img_encoder = dict(
    type='MapAnythingOccEncoder',  # Main image encoder.
    repo_root='/root/wjl/OPUS_mmcv2/third_party/map-anything',  # Local MapAnything source root.
    freeze=True,  # Keep pretrained encoder frozen.
    freeze_via_wrapper=False,  # Do not freeze via wrapper; direct module freeze is used.
    num_views=dataset_cfg['num_views'],  # Camera count.
    num_frames=num_frames,  # Temporal frame count.
    chunk_by_frame=True,  # Split processing by frame to control memory.
    mapanything_model_cfg=mapanything_model_cfg,  # MapAnything model loading config.
    mapanything_preprocess_cfg=mapanything_preprocess_cfg,  # MapAnything preprocess config.
    anyup_cfg=dict(
        enabled=False,  # AnyUp is disabled in this experiment.
        repo_root='/root/wjl/OPUS_mmcv2/third_party/anyup',  # AnyUp local source root.
        variant='anyup_multi_backbone',  # AnyUp model variant.
        checkpoint_path='/root/wjl/OPUS_mmcv2/third_party/anyup/checkpoints/anyup_multi_backbone.pth',  # AnyUp checkpoint path.
        allow_online_download_if_missing=False,  # Disallow online fallback download by default.
        q_chunk_size=64,  # Query chunk size for AnyUp runtime.
        view_batch_size=6,  # Per-step view batch size.
        output_in_channels=1024,  # Input channels expected by AnyUp output head.
        output_channels=256,  # Output channels aligned to fusion dims.
        upsample_output_divisor=4,  # Upsampling target divisor.
        freeze=True,  # Freeze AnyUp when enabled.
        mode='bilinear',  # Bilinear mode requirement.
        pyramid=dict(
            output_divisors=[4, 8, 16, 32],  # Pyramid scales.
            downsample_mode='bilinear',  # Bilinear downsample.
            num_levels=4,  # Number of pyramid outputs.
            align_corners=False,  # Standard bilinear alignment behavior.
        ),
    ),
)

img_feature_fusion = dict(
    alpha=[0.5, 0.5, 0.5, 0.5],  # Per-level weighting factor alpha.
    beta=[0.5, 0.5, 0.5, 0.5],  # Per-level weighting factor beta.
    interp_mode='bilinear',  # Bilinear fusion interpolation.
    align_corners=False,  # Keep bilinear align_corners disabled.
)

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],  # ImageNet-like mean.
    std=[58.395, 57.120, 57.375],  # ImageNet-like std.
    to_rgb=True,  # Convert BGR->RGB if needed.
)


# -------------------- Point Branch --------------------
pts_voxel_layer = dict(
    max_num_points=10,  # Max points per voxel.
    voxel_size=pc_voxel_size,  # Voxel resolution for sparse conv input.
    deterministic=False,  # Allow non-deterministic speed path.
    max_voxels=(90000, 120000),  # Train/test max voxel count.
    point_cloud_range=point_cloud_range,  # Spatial clipping range.
)

pts_voxel_encoder = dict(
    type='HardSimpleVFE',  # Simple hard voxel feature encoder.
    num_features=5,  # x/y/z/intensity/time.
)

pts_middle_encoder = dict(
    type='SparseEncoder',  # Sparse 3D encoder.
    in_channels=5,  # Input point feature dim.
    sparse_shape=[161, 800, 800],  # Sparse volume dimensions.
    output_channels=128,  # Output BEV channels.
    order=('conv', 'norm', 'act'),  # Module order.
    encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),  # Block channels.
    encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),  # Block paddings.
    block_type='basicblock',  # Sparse block type.
    return_middle_feats=False,  # Use standard BEV path; TPV intermediate features not needed.
)

pts_backbone = dict(
    type='SECOND',  # 2D backbone on BEV features.
    in_channels=1152,  # Input channels from sparse encoder collapse.
    out_channels=[128, 256],  # Per-stage output channels.
    layer_nums=[5, 5],  # Blocks per stage.
    layer_strides=[1, 2],  # Strides per stage.
    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),  # BN settings.
    conv_cfg=dict(type='Conv2d', bias=False),  # Conv settings.
)

pts_neck = dict(
    type='SECONDFPN',  # Neck for BEV features.
    in_channels=[128, 256],  # Input stage channels.
    out_channels=[256, 256],  # Unified output channels.
    upsample_strides=[1, 2],  # Upsampling strides.
    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),  # BN settings.
    upsample_cfg=dict(type='deconv', bias=False),  # Deconvolution upsample.
    use_conv_for_no_stride=True,  # Use conv branch for stride=1 stage.
)

# TPV encoder is enabled in this run.
tpv_encoder = dict(
    type='TPVLiteEncoder',  # Lightweight TPV encoder.
    in_channels=128,  # Input from point/image branch fusion source.
    skip_in_channels=64,  # Skip connection channels.
    fpn_channels=64,  # Internal TPV FPN channels.
    out_channels=256,  # Output channels for TPV features.
    use_skip=True,  # Enable skip fusion.
)


# -------------------- Full Model --------------------
model = dict(
    type='OPUSV1Fusion',  # Main occupancy model.
    data_preprocessor=dict(type='BaseDataPreprocessor'),  # Default preprocessor.
    use_grid_mask=False,  # Disable grid mask augmentation.
    data_aug=dict(
        img_color_aug=True,  # Keep image color aug on GPU path.
        img_norm_cfg=img_norm_cfg,  # Image normalization config.
        img_pad_cfg=dict(size_divisor=14),  # Pad to transformer patch divisor.
    ),
    stop_prev_grad=0,  # Temporal gradient stop strategy.

    # Branch switches (TPV on, pure point-only branch off).
    enable_tpv_feature_branch=False,
    enable_pts_feature_branch=True,

    # Image stack.
    img_backbone=img_backbone,
    img_neck=img_neck,
    img_encoder=img_encoder,
    img_feature_fusion=img_feature_fusion,

    # Point stack.
    pts_voxel_layer=pts_voxel_layer,
    pts_voxel_encoder=pts_voxel_encoder,
    pts_middle_encoder=pts_middle_encoder,
    pts_backbone=pts_backbone,
    pts_neck=pts_neck,

    # TPV stack.
    tpv_encoder=tpv_encoder,

    # Occupancy head.
    pts_bbox_head=dict(
        type='OPUSV1FusionHead',  # Dense query-based occupancy head.
        num_classes=len(occ_names),  # Exclude free class from prediction targets.
        in_channels=embed_dims,  # Input feature channels.
        num_query=num_query,  # Query count.
        pc_range=point_cloud_range,  # Scene range.
        empty_label=dataset_cfg['empty_label'],  # Index of free label.
        voxel_size=voxel_size,  # Voxel decoding granularity.
        init_pos_lidar='curr',  # Initialize from current-frame coordinates.
        transformer=dict(
            type='OPUSV1FusionTransformer',  # Fusion transformer.
            embed_dims=embed_dims,  # Hidden channels.
            num_frames=num_frames,  # Temporal frames.
            num_views=dataset_cfg['num_views'],  # Camera views.
            num_points=num_points,  # Sample points per query.
            num_layers=num_layers,  # Decoder depth.
            num_levels=num_levels,  # Feature pyramid levels.
            num_classes=len(occ_names),  # Semantic class count.
            num_refines=num_refines,  # Progressive refinement schedule.
            scales=[0.5],  # Coordinate scaling factors.

            # TPV sampling setup from TPV-lite depth-gt config.
            use_pts_sampling=True,  # Enable point-feature sampling path.
            use_tpv_sampling=False,  # Disable TPV sampling path.
            tpv_fusion_mode='query_attn',  # Use query attention for TPV fusion.

            # Query allocator configuration.
            query_allocator=dict(
                enabled=True,  # Enable one-shot query reallocation.
                switch_layer=2,  # Trigger after decoder layer index 1.
                context_ratio=0.6,  # Ratio kept as context queries.
                detail_jitter_std=0.01,  # Jitter for detail query generation.
                score_weights=dict(
                    nonempty=0.5,  # Non-empty prior weight.
                    uncertainty=0.5,  # Uncertainty prior weight.
                ),
            ),
            pc_range=point_cloud_range,  # Geometry range for decoding/sampling.
        ),

        # Classification and regression losses.
        loss_cls=dict(
            type='FocalLoss',
            _scope_='mmdet',
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_pts=dict(
            type='SmoothL1Loss',
            _scope_='mmdet',
            beta=0.2,
            loss_weight=0.5,
        ),
    ),

    # Training-time matching and weighting settings.
    train_cfg=dict(
        pts=dict(
            cls_weights=cls_weights,  # Static class weights.
            rare_classes=rare_classes,  # Tail class ids.
            rare_weights=12,  # Legacy tail floor.
            hard_camera_mask=True,  # Match only camera-visible voxels.

            tail_focus=dict(
                enabled=True,
                policy='ema_freq',
                ema_momentum=0.9,
                freq_thr=0.02,
                min_tail_classes=8,
                max_tail_classes=24,
                sync_stats=True,
                fallback='rare_classes',
                tail_weight=12,
            ),

            hard_mining=dict(
                enabled=True,
                pred_topk_ratio=0.5,
                min_keep=1024,
            ),

            gt_balance=dict(
                enabled=True,
                per_class_cap=8192,
                tail_min_keep=128,
                sample_mode='random',
            ),

            query_init_mix=query_init_mix_cfg,  # Mixed query initialization.
        )
    ),

    # Inference-time settings.
    test_cfg=dict(
        pts=dict(
            score_thr=eval_score_thr,  # Keep low threshold for early depth-point stability.
            padding=False,  # Do not pad query outputs at test time.
        )
    ),
)


# -------------------- Data Augmentation --------------------
ida_aug_conf = {
    'resize_lim': (1.0, 1.0),  # Locked scale (no random resize).
    'final_dim': (616, 616),  # Final training/eval image shape.
    'bot_pct_lim': (0.0, 0.0),  # No random bottom crop.
    'rot_lim': (0.0, 0.0),  # No random rotation.
    'H': 616,  # Original height prior for augmentation.
    'W': 616,  # Original width prior for augmentation.
    'rand_flip': True,  # Random horizontal flip enabled.
}

imdecode_backend = 'pillow'  # Decoder backend for image loading.


# -------------------- Point/Extra Loaders --------------------
depth_points_cfg = dict(
    type='LoadPointsFromMultiViewDepth',  # Build pseudo points from multi-view depth.
    sample_stride=4,  # Keep temporal stride for current-frame alignment.
    max_points_total=560000,  # Upper bound to control memory.
    depth_min=0.01,  # Minimum valid depth.
    depth_max=30.0,  # Maximum valid depth.
    coord_convention='opencv',  # Camera coordinate convention.
    load_dim=5,  # x/y/z/intensity/time.
    use_dim=[0, 1, 2, 3, 4],  # Use all loaded dimensions.
    time_dim=4,  # Time index in point feature.
    strict_depth_exist=True,  # Require depth source presence.
    fallback_depth_from_image_path=True,  # Allow derived depth path fallback.
)

point_load_transforms = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, use_dim=[0, 1, 2, 3, 4],
         pad_empty_sweeps=True, remove_close=True),
]  # Use raw LiDAR points + sweeps for point branch.

mapanything_extra_loader = dict(
    type='LoadMapAnythingExtraFromDepth',  # Build mapanything_extra from depth/intrinsics/pose.
    strict=True,  # Enforce strict metadata checks.
    filter_depth_by_pcrange=True,  # Clip by point cloud range.
    point_cloud_range=point_cloud_range,  # Shared spatial bounds.
    min_valid_depth_ratio=1e-6,  # Keep near-empty frames valid with tiny threshold.
)

pack_meta_keys = (
    'sample_idx', 'sample_token', 'scene_name',
    'filename', 'ori_shape', 'img_shape', 'pad_shape',
    'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp',
)  # Metadata consumed by model/evaluator hooks.


# -------------------- Pipelines --------------------
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='color',
        imdecode_backend=imdecode_backend,
    ),  # Load current multi-view images.
    dict(
        type='LoadMultiViewImageFromMultiSweeps',
        sweeps_num=num_frames - 1,
        imdecode_backend=imdecode_backend,
        cam_types=dataset_cfg['cam_types'],
    ),  # Load history sweeps.
    *point_load_transforms,
    dict(type='LiDARToOccSpace'),  # Convert points into occupancy coordinate system.
    dict(
        type='LoadOcc3DFromFile',
        occ_root=occ_root,
        path_template=dataset_cfg['occ_io']['path_template'],
        semantics_key=dataset_cfg['occ_io']['semantics_key'],
        mask_camera_key=dataset_cfg['occ_io']['mask_camera_key'],
        mask_lidar_key=dataset_cfg['occ_io']['mask_lidar_key'],
        class_names=dataset_cfg['class_names'],
        empty_label=dataset_cfg['empty_label'],
    ),  # Load occupancy GT for training.
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),  # Image augmentation.
    mapanything_extra_loader,  # Inject mapanything_extra before range filtering.
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),  # Clip points to scene range.
    dict(
        type='PackOcc3DInputs',
        meta_keys=pack_meta_keys,
        extra_input_keys=('mapanything_extra',),
    ),  # Pack regular inputs + mapanything_extra.
]

test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='color',
        imdecode_backend=imdecode_backend,
    ),  # Load current multi-view images.
    dict(
        type='LoadMultiViewImageFromMultiSweeps',
        sweeps_num=num_frames - 1,
        test_mode=True,
        imdecode_backend=imdecode_backend,
        cam_types=dataset_cfg['cam_types'],
    ),  # Load history sweeps in test mode.
    *point_load_transforms,
    dict(type='LiDARToOccSpace'),  # Convert points into occupancy coordinate system.
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),  # Deterministic eval transform.
    mapanything_extra_loader,  # Inject mapanything_extra for val/test too.
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),  # Clip points to scene range.
    dict(
        type='PackOcc3DInputs',
        meta_keys=pack_meta_keys,
        extra_input_keys=('mapanything_extra',),
    ),  # Pack regular inputs + mapanything_extra.
]


# -------------------- Dataloaders --------------------
batch_size = 12  # Per-GPU train batch size.

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=False,
    ),
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
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
    ),
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
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
    ),
    collate_fn=dict(type='pseudo_collate'),
)


# -------------------- Optimization --------------------
optimizer = dict(
    type='AdamW',  # Optimizer type.
    lr=2e-4,  # Base learning rate.
    weight_decay=0.01,  # Weight decay.
)

optim_wrapper = dict(
    type='AmpOptimWrapper',  # Mixed precision wrapper.
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # Lower LR for pretrained image backbone.
            'sampling_offset': dict(lr_mult=0.1),  # Lower LR for offset params.
        }
    ),
    loss_scale=512.0,  # Static loss scale for AMP.
    clip_grad=dict(max_norm=35, norm_type=2),  # Gradient clipping.
)


# -------------------- Schedule --------------------
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500,
    ),  # Warm-up for first 500 iterations.
    dict(
        type='CosineAnnealingLR',
        T_max=total_epochs,
        by_epoch=True,
        begin=0,
        end=total_epochs,
        eta_min=2e-4 * 1e-3,
    ),  # Cosine decay over full 150 epochs.
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=total_epochs,
    val_interval=val_interval,
)
val_cfg = dict(type='ValLoop')  # Standard validation loop.
test_cfg = dict(type='TestLoop')  # Standard test loop.


# -------------------- Evaluation --------------------
val_evaluator = dict(
    type='Occ3DMetric',
    ann_file=val_ann_file,
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
    focus_eval=dict(
        enabled=True,
        policy='ema_freq',
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
    ann_file=test_ann_file,
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
    focus_eval=dict(
        enabled=True,
        policy='ema_freq',
        ema_momentum=0.9,
        freq_thr=0.02,
        min_classes=8,
        max_classes=24,
        fallback='rare_classes',
        rare_classes=rare_classes,
    ),
)


# -------------------- Hooks And Runtime --------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=5, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)  # Log smoothing setup.

visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(type='TensorboardVisBackend')],
)

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)

randomness = dict(seed=0, deterministic=False)  # Reproducibility settings.

load_from = 'pretrain/fusion_pretrain_model.pth'  # Pretrained initialization checkpoint.
resume_from = None  # Do not auto-resume.
