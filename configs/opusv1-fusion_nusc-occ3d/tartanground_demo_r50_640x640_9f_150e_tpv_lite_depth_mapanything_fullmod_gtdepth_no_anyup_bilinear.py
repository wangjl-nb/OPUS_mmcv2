from copy import deepcopy

_base_ = ['./tartanground_demo_r50_640x640_9f_100e_tpv_lite_depth_gt.py']

# Combined experiment:
# - TPV feature branch + TPV sampling
# - GT-depth pseudo-points input (train/val/test *_with_depth.pkl)
# - MapAnything full-modality extra inputs from depth/intrinsics/pose
# - AnyUp disabled + bilinear interpolation
mapanything_extra_loader = dict(
    type='LoadMapAnythingExtraFromDepth',
    strict=True,
    filter_depth_by_pcrange=True,
    point_cloud_range=_base_.point_cloud_range,
    min_valid_depth_ratio=1e-6,
)

train_pipeline = []
for transform in _base_.train_pipeline:
    t = deepcopy(transform)
    if t.get('type') == 'PointsRangeFilter':
        train_pipeline.append(deepcopy(mapanything_extra_loader))
    if t.get('type') == 'PackOcc3DInputs':
        t['extra_input_keys'] = ('mapanything_extra', )
    train_pipeline.append(t)

test_pipeline = []
for transform in _base_.test_pipeline:
    t = deepcopy(transform)
    if t.get('type') == 'PointsRangeFilter':
        test_pipeline.append(deepcopy(mapanything_extra_loader))
    if t.get('type') == 'PackOcc3DInputs':
        t['extra_input_keys'] = ('mapanything_extra', )
    test_pipeline.append(t)

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

total_epochs = 150
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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=5)

model = dict(
    img_encoder=dict(
        freeze_via_wrapper=False,
        anyup_cfg=dict(
            enabled=False,
            repo_root='/root/wjl/OPUS_mmcv2/third_party/anyup',
            variant='anyup_multi_backbone',
            checkpoint_path='/root/wjl/OPUS_mmcv2/third_party/anyup/checkpoints/anyup_multi_backbone.pth',
            allow_online_download_if_missing=True,
            q_chunk_size=64,
            view_batch_size=6,
            output_in_channels=1024,
            output_channels=256,
            upsample_output_divisor=4,
            freeze=True,
            mode='bilinear',
            pyramid=dict(
                output_divisors=[4, 8, 16, 32],
                downsample_mode='bilinear',
                num_levels=4,
                align_corners=False,
            ),
        ),
    ),
    img_feature_fusion=dict(
        interp_mode='bilinear',
        align_corners=False,
    ),
)
