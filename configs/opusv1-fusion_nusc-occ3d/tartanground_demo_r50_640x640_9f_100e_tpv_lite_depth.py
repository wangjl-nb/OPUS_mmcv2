_base_ = ['./tartanground_demo_r50_640x640_9f_100e_depth_points_switch.py']

model = dict(
    enable_tpv_feature_branch=True,
    enable_pts_feature_branch=False,
    pts_middle_encoder=dict(
        return_middle_feats=True,
    ),
    tpv_encoder=dict(
        type='TPVLiteEncoder',
        in_channels=128,
        skip_in_channels=64,
        fpn_channels=64,
        out_channels=256,
        use_skip=True,
    ),
    pts_bbox_head=dict(
        transformer=dict(
            use_pts_sampling=False,
            use_tpv_sampling=True,
            tpv_fusion_mode='query_attn',
        ),
    ),
)
