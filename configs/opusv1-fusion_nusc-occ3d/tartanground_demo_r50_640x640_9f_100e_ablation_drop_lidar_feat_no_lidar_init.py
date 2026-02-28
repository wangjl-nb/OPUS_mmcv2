_base_ = ['./tartanground_demo_r50_640x640_9f_100e.py']

# Ablation target:
# - drop LiDAR feature contribution in fusion branch
# - disable LiDAR-based query initialization (fall back to learned init points)
model = dict(
    drop_lidar_feat=True,
    pts_bbox_head=dict(
        init_pos_lidar=None,
    ),
    train_cfg=dict(
        pts=dict(
            # Explicitly disable mix policy since LiDAR init path is disabled.
            query_init_mix=dict(enabled=False),
        ),
    ),
)
