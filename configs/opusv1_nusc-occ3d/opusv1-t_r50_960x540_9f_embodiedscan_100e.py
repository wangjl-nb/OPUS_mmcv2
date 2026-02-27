default_scope = "mmdet3d"
custom_imports = dict(imports=["models", "loaders"], allow_failed_imports=False)

dataset_type = "TartangroundOcc3DDataset"
dataset_root = "/mnt/cpfs-E/data/Embodiedscan/EmbodiedScan_OPUS/"
occ_root = "/mnt/cpfs-E/data/Embodiedscan/EmbodiedScan_OPUS/gts/"

# Optional prefix/suffix for ann files (empty -> default train/val).
# Examples:
#   ann_prefix = "mini_"  -> mini_train.pkl / mini_val.pkl
#   ann_suffix = "_good"  -> train_good.pkl / val_good.pkl
ann_prefix = ""
ann_suffix = "_good"
train_ann = f"{ann_prefix}train{ann_suffix}.pkl"
val_ann = f"{ann_prefix}val{ann_suffix}.pkl"

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

# For nuScenes we usually do 10-class detection
object_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

occ_names = [
    "others",
    "floor",
    "wall",
    "chair",
    "cabinet",
    "door",
    "table",
    "couch",
    "shelf",
    "window",
    "bed",
    "curtain",
    "desk",
    "doorframe",
    "plant",
    "stairs",
    "pillow",
    "wardrobe",
    "picture",
    "bathtub",
    "box",
    "counter",
    "bench",
    "stand",
    "rail",
    "sink",
    "clothes",
    "mirror",
    "toilet",
    "refrigerator",
    "lamp",
    "book",
    "dresser",
    "stool",
    "fireplace",
    "tv",
    "blanket",
    "commode",
    "washing machine",
    "monitor",
    "window frame",
    "radiator",
    "mat",
    "shower",
    "rack",
    "towel",
    "ottoman",
    "column",
    "blinds",
    "stove",
    "bar",
    "pillar",
    "bin",
    "heater",
    "clothes dryer",
    "backpack",
    "blackboard",
    "decoration",
    "roof",
    "bag",
    "steps",
    "windowsill",
    "cushion",
    "carpet",
    "copier",
    "board",
    "countertop",
    "basket",
    "mailbox",
    "kitchen island",
    "washbasin",
    "bicycle",
    "drawer",
    "oven",
    "piano",
    "excercise equipment",
    "beam",
    "partition",
    "printer",
    "microwave",
    "frame",
]
occ_eval_names = occ_names + ["free"]

# Computed from EmbodiedScan_OPUS gts (mask_camera, exclude free).
# Weights in [1, 10] using quadratic decay on normalized proportions:
#   w_i = round(1 + 9 * (1 - (p_i / p_max))^2)
rare_classes = [
    58, 76, 75, 80, 70, 77, 73, 60, 66, 72, 71, 79,
    63, 68, 78, 57, 50, 48, 67, 74, 69, 40, 61, 44,
]
cls_weights = [
    5, 1, 1, 6, 8, 8, 7, 7, 8, 9, 8, 9, 9, 9, 10, 10,
    9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10,
]

# Ensure class weights length matches occ_names.
cls_weights = cls_weights[: len(occ_names)]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78]
voxel_size = [0.16, 0.16, 0.16]

dataset_cfg = dict(
    cam_types=["CAM_FRONT"],
    num_views=1,
    occ_io=dict(
        path_template="{scene_name}/{token}/labels.npz",
        semantics_key="semantics",
        mask_camera_key="mask_camera",
        mask_lidar_key="mask_lidar",
    ),
    class_names=occ_names + ["free"],
    empty_label=len(occ_names),
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
)

# arch config
embed_dims = 256
num_layers = 6
num_query = 2400
num_frames = 9
num_levels = 4
num_points = 4
num_refines = [1, 2, 4, 8, 16, 32]

img_backbone = dict(
    type="ResNet",
    _scope_="mmdet",
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type="BN2d", requires_grad=True),
    norm_eval=True,
    style="pytorch",
    with_cp=True,
)
img_neck = dict(
    type="FPN",
    _scope_="mmdet",
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels,
)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True,
)

model = dict(
    type="OPUSV1",
    data_preprocessor=dict(type="BaseDataPreprocessor"),
    use_grid_mask=False,
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32),
    ),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type="OPUSV1Head",
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        pc_range=point_cloud_range,
        empty_label=dataset_cfg["empty_label"],
        voxel_size=voxel_size,
        transformer=dict(
            type="OPUSV1Transformer",
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_views=dataset_cfg["num_views"],
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            scales=[0.5],
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss",
            _scope_="mmdet",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_pts=dict(_scope_="mmdet", type="SmoothL1Loss", beta=0.2, loss_weight=0.5),
    ),
    train_cfg=dict(
        pts=dict(
            cls_weights=cls_weights,
            rare_classes=rare_classes,
            rare_weights=2,
            hard_camera_mask=True,
        )
    ),
    test_cfg=dict(
        pts=dict(
            score_thr=0.2,
            padding=False,
        )
    ),
)

# Image-domain augmentation
ida_aug_conf = {
    "resize_lim": (1.0, 1.2),
    "final_dim": (540, 960),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 540,
    "W": 960,
    "rand_flip": True,
}

imdecode_backend = "pillow"
train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=False,
        color_type="color",
        imdecode_backend=imdecode_backend,
    ),
    dict(
        type="LoadMultiViewImageFromMultiSweeps",
        sweeps_num=num_frames - 1,
        imdecode_backend=imdecode_backend,
        cam_types=dataset_cfg["cam_types"],
    ),
    dict(
        type="LoadOcc3DFromFile",
        occ_root=occ_root,
        path_template=dataset_cfg["occ_io"]["path_template"],
        semantics_key=dataset_cfg["occ_io"]["semantics_key"],
        mask_camera_key=dataset_cfg["occ_io"]["mask_camera_key"],
        mask_lidar_key=dataset_cfg["occ_io"]["mask_lidar_key"],
        class_names=dataset_cfg["class_names"],
        empty_label=dataset_cfg["empty_label"],
    ),
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=True),
    dict(
        type="PackOcc3DInputs",
        meta_keys=(
            "sample_idx",
            "sample_token",
            "scene_name",
            "filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "ego2occ",
            "ego2img",
            "ego2lidar",
            "img_timestamp",
        ),
    ),
]

test_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=False,
        color_type="color",
        imdecode_backend=imdecode_backend,
    ),
    dict(
        type="LoadMultiViewImageFromMultiSweeps",
        sweeps_num=num_frames - 1,
        test_mode=True,
        imdecode_backend=imdecode_backend,
        cam_types=dataset_cfg["cam_types"],
    ),
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type="PackOcc3DInputs",
        meta_keys=(
            "sample_idx",
            "sample_token",
            "scene_name",
            "filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "ego2occ",
            "ego2img",
            "ego2lidar",
            "img_timestamp",
        ),
    ),
]

batch_size = 32

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + train_ann,
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=False,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + val_ann,
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + val_ann,
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=0.01,
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            "sampling_offset": dict(lr_mult=0.1),
        }
    ),
    loss_scale=512.0,
    clip_grad=dict(max_norm=35, norm_type=2),
)

# learning policy
total_epochs = 100
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        T_max=total_epochs,
        by_epoch=True,
        begin=0,
        end=total_epochs,
        eta_min=2e-4 * 1e-3,
    ),
]

# load pretrained weights
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

val_evaluator = dict(
    type="Occ3DMetric",
    ann_file=dataset_root + val_ann,
    occ_root=occ_root,
    occ_path_template=dataset_cfg["occ_io"]["path_template"],
    semantics_key=dataset_cfg["occ_io"]["semantics_key"],
    mask_camera_key=dataset_cfg["occ_io"]["mask_camera_key"],
    mask_lidar_key=dataset_cfg["occ_io"]["mask_lidar_key"],
    empty_label=dataset_cfg["empty_label"],
    use_camera_mask=True,
    pc_range=dataset_cfg["pc_range"],
    voxel_size=dataset_cfg["voxel_size"],
    class_names=dataset_cfg["class_names"],
    miou_num_workers=32,
    focus_eval=dict(
        enabled=True,
        policy="ema_freq",
        ema_momentum=0.9,
        freq_thr=0.02,
        min_classes=8,
        max_classes=24,
        fallback="rare_classes",
        rare_classes=rare_classes,
    ),
)

test_evaluator = dict(
    type="Occ3DMetric",
    ann_file=dataset_root + val_ann,
    occ_root=occ_root,
    occ_path_template=dataset_cfg["occ_io"]["path_template"],
    semantics_key=dataset_cfg["occ_io"]["semantics_key"],
    mask_camera_key=dataset_cfg["occ_io"]["mask_camera_key"],
    mask_lidar_key=dataset_cfg["occ_io"]["mask_lidar_key"],
    empty_label=dataset_cfg["empty_label"],
    use_camera_mask=True,
    pc_range=dataset_cfg["pc_range"],
    voxel_size=dataset_cfg["voxel_size"],
    class_names=dataset_cfg["class_names"],
    miou_num_workers=32,
    focus_eval=dict(
        enabled=True,
        policy="ema_freq",
        ema_momentum=0.9,
        freq_thr=0.02,
        min_classes=8,
        max_classes=24,
        fallback="rare_classes",
        rare_classes=rare_classes,
    ),
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=1),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=2, max_keep_ckpts=5, save_last=True),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

log_processor = dict(type="LogProcessor", window_size=1, by_epoch=True)

visualizer = dict(
    type="Visualizer",
    vis_backends=[dict(type="TensorboardVisBackend")],
)

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

randomness = dict(seed=0, deterministic=False)

# load pretrained weights
load_from = "pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth"
revise_keys = [("backbone", "img_backbone")]

# resume the last training
resume_from = None
default_scope = "mmdet3d"
custom_imports = dict(imports=["models", "loaders"], allow_failed_imports=False)

dataset_type = "TartangroundOcc3DDataset"
dataset_root = "/mnt/cpfs-E/data/Embodiedscan/EmbodiedScan_OPUS/"
occ_root = "/mnt/cpfs-E/data/Embodiedscan/EmbodiedScan_OPUS/gts/"

# Optional prefix/suffix for ann files (empty -> default train/val).
# Examples:
#   ann_prefix = "mini_"  -> mini_train.pkl / mini_val.pkl
#   ann_suffix = "_good"  -> train_good.pkl / val_good.pkl
ann_prefix = ""
ann_suffix = "_good"
train_ann = f"{ann_prefix}train{ann_suffix}.pkl"
val_ann = f"{ann_prefix}val{ann_suffix}.pkl"

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

# For nuScenes we usually do 10-class detection
object_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

occ_names = [
    "others",
    "floor",
    "wall",
    "chair",
    "cabinet",
    "door",
    "table",
    "couch",
    "shelf",
    "window",
    "bed",
    "curtain",
    "desk",
    "doorframe",
    "plant",
    "stairs",
    "pillow",
    "wardrobe",
    "picture",
    "bathtub",
    "box",
    "counter",
    "bench",
    "stand",
    "rail",
    "sink",
    "clothes",
    "mirror",
    "toilet",
    "refrigerator",
    "lamp",
    "book",
    "dresser",
    "stool",
    "fireplace",
    "tv",
    "blanket",
    "commode",
    "washing machine",
    "monitor",
    "window frame",
    "radiator",
    "mat",
    "shower",
    "rack",
    "towel",
    "ottoman",
    "column",
    "blinds",
    "stove",
    "bar",
    "pillar",
    "bin",
    "heater",
    "clothes dryer",
    "backpack",
    "blackboard",
    "decoration",
    "roof",
    "bag",
    "steps",
    "windowsill",
    "cushion",
    "carpet",
    "copier",
    "board",
    "countertop",
    "basket",
    "mailbox",
    "kitchen island",
    "washbasin",
    "bicycle",
    "drawer",
    "oven",
    "piano",
    "excercise equipment",
    "beam",
    "partition",
    "printer",
    "microwave",
    "frame",
]
occ_eval_names = occ_names + ["free"]

# Computed from EmbodiedScan_OPUS gts (mask_camera, exclude free).
# Weights in [1, 10] using quadratic decay on normalized proportions:
#   w_i = round(1 + 9 * (1 - (p_i / p_max))^2)
rare_classes = [
    58, 76, 75, 80, 70, 77, 73, 60, 66, 72, 71, 79,
    63, 68, 78, 57, 50, 48, 67, 74, 69, 40, 61, 44,
]
cls_weights = [
    5, 1, 1, 6, 8, 8, 7, 7, 8, 9, 8, 9, 9, 9, 10, 10,
    9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10,
]

# Ensure class weights length matches occ_names.
cls_weights = cls_weights[: len(occ_names)]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78]
voxel_size = [0.16, 0.16, 0.16]

dataset_cfg = dict(
    cam_types=["CAM_FRONT"],
    num_views=1,
    occ_io=dict(
        path_template="{scene_name}/{token}/labels.npz",
        semantics_key="semantics",
        mask_camera_key="mask_camera",
        mask_lidar_key="mask_lidar",
    ),
    class_names=occ_names + ["free"],
    empty_label=len(occ_names),
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
)

# arch config
embed_dims = 256
num_layers = 6
num_query = 2400
num_frames = 9
num_levels = 4
num_points = 4
num_refines = [1, 2, 4, 8, 16, 32]

img_backbone = dict(
    type="ResNet",
    _scope_="mmdet",
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type="BN2d", requires_grad=True),
    norm_eval=True,
    style="pytorch",
    with_cp=True,
)
img_neck = dict(
    type="FPN",
    _scope_="mmdet",
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels,
)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True,
)

model = dict(
    type="OPUSV1",
    data_preprocessor=dict(type="BaseDataPreprocessor"),
    use_grid_mask=False,
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32),
    ),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type="OPUSV1Head",
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        pc_range=point_cloud_range,
        empty_label=dataset_cfg["empty_label"],
        voxel_size=voxel_size,
        transformer=dict(
            type="OPUSV1Transformer",
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_views=dataset_cfg["num_views"],
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            scales=[0.5],
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss",
            _scope_="mmdet",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_pts=dict(_scope_="mmdet", type="SmoothL1Loss", beta=0.2, loss_weight=0.5),
    ),
    train_cfg=dict(
        pts=dict(
            cls_weights=cls_weights,
            rare_classes=rare_classes,
            rare_weights=2,
            hard_camera_mask=True,
        )
    ),
    test_cfg=dict(
        pts=dict(
            score_thr=0.2,
            padding=False,
        )
    ),
)

# Image-domain augmentation
ida_aug_conf = {
    "resize_lim": (1.0, 1.2),
    "final_dim": (540, 960),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 540,
    "W": 960,
    "rand_flip": True,
}

imdecode_backend = "pillow"
train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=False,
        color_type="color",
        imdecode_backend=imdecode_backend,
    ),
    dict(
        type="LoadMultiViewImageFromMultiSweeps",
        sweeps_num=num_frames - 1,
        imdecode_backend=imdecode_backend,
        cam_types=dataset_cfg["cam_types"],
    ),
    dict(
        type="LoadOcc3DFromFile",
        occ_root=occ_root,
        path_template=dataset_cfg["occ_io"]["path_template"],
        semantics_key=dataset_cfg["occ_io"]["semantics_key"],
        mask_camera_key=dataset_cfg["occ_io"]["mask_camera_key"],
        mask_lidar_key=dataset_cfg["occ_io"]["mask_lidar_key"],
        class_names=dataset_cfg["class_names"],
        empty_label=dataset_cfg["empty_label"],
    ),
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=True),
    dict(
        type="PackOcc3DInputs",
        meta_keys=(
            "sample_idx",
            "sample_token",
            "scene_name",
            "filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "ego2occ",
            "ego2img",
            "ego2lidar",
            "img_timestamp",
        ),
    ),
]

test_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=False,
        color_type="color",
        imdecode_backend=imdecode_backend,
    ),
    dict(
        type="LoadMultiViewImageFromMultiSweeps",
        sweeps_num=num_frames - 1,
        test_mode=True,
        imdecode_backend=imdecode_backend,
        cam_types=dataset_cfg["cam_types"],
    ),
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type="PackOcc3DInputs",
        meta_keys=(
            "sample_idx",
            "sample_token",
            "scene_name",
            "filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "ego2occ",
            "ego2img",
            "ego2lidar",
            "img_timestamp",
        ),
    ),
]

batch_size = 32

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + train_ann,
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=False,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + val_ann,
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + val_ann,
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=0.01,
)

optim_wrapper = dict(
    type="SafeAmpOptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            "sampling_offset": dict(lr_mult=0.1),
        }
    ),
    loss_scale=256.0,
    clip_grad=dict(max_norm=35, norm_type=2),
    sanitize_nonfinite_grads=True,
    log_nonfinite_stats=True,
)

# learning policy
total_epochs = 100
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        T_max=total_epochs,
        by_epoch=True,
        begin=0,
        end=total_epochs,
        eta_min=2e-4 * 1e-3,
    ),
]

# load pretrained weights
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

val_evaluator = dict(
    type="Occ3DMetric",
    ann_file=dataset_root + val_ann,
    occ_root=occ_root,
    occ_path_template=dataset_cfg["occ_io"]["path_template"],
    semantics_key=dataset_cfg["occ_io"]["semantics_key"],
    mask_camera_key=dataset_cfg["occ_io"]["mask_camera_key"],
    mask_lidar_key=dataset_cfg["occ_io"]["mask_lidar_key"],
    empty_label=dataset_cfg["empty_label"],
    use_camera_mask=True,
    pc_range=dataset_cfg["pc_range"],
    voxel_size=dataset_cfg["voxel_size"],
    class_names=dataset_cfg["class_names"],
    miou_num_workers=32,
    focus_eval=dict(
        enabled=True,
        policy="ema_freq",
        ema_momentum=0.9,
        freq_thr=0.02,
        min_classes=8,
        max_classes=24,
        fallback="rare_classes",
        rare_classes=rare_classes,
    ),
)

test_evaluator = dict(
    type="Occ3DMetric",
    ann_file=dataset_root + val_ann,
    occ_root=occ_root,
    occ_path_template=dataset_cfg["occ_io"]["path_template"],
    semantics_key=dataset_cfg["occ_io"]["semantics_key"],
    mask_camera_key=dataset_cfg["occ_io"]["mask_camera_key"],
    mask_lidar_key=dataset_cfg["occ_io"]["mask_lidar_key"],
    empty_label=dataset_cfg["empty_label"],
    use_camera_mask=True,
    pc_range=dataset_cfg["pc_range"],
    voxel_size=dataset_cfg["voxel_size"],
    class_names=dataset_cfg["class_names"],
    miou_num_workers=32,
    focus_eval=dict(
        enabled=True,
        policy="ema_freq",
        ema_momentum=0.9,
        freq_thr=0.02,
        min_classes=8,
        max_classes=24,
        fallback="rare_classes",
        rare_classes=rare_classes,
    ),
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=1),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=5, max_keep_ckpts=5, save_last=True),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

log_processor = dict(type="LogProcessor", window_size=1, by_epoch=True)

visualizer = dict(
    type="Visualizer",
    vis_backends=[dict(type="TensorboardVisBackend")],
)

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

randomness = dict(seed=0, deterministic=False)

# load pretrained weights
load_from = "pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth"
revise_keys = [("backbone", "img_backbone")]

# resume the last training
resume_from = None
