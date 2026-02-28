_base_ = ['./tartanground_demo_r50_640x640_9f_100e.py']

# Ablation target:
# - keep LiDAR points for query initialization (`init_pos_lidar` from base config)
# - remove LiDAR feature contribution in fusion branch
model = dict(
    drop_lidar_feat=True,
)
