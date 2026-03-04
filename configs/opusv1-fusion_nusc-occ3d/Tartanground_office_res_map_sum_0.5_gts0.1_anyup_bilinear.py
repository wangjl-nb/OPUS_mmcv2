_base_ = ['./Tartanground_office_res_map_sum_0.5_gts0.1.py']

# Ablation: keep the same external MapAnything pipeline but replace AnyUp
# super-resolution with bilinear interpolation for upsampling.
model = dict(
    img_encoder=dict(
        anyup_cfg=dict(
            enabled=True,
            mode='bilinear',
            pyramid=dict(
                downsample_mode='bilinear',
                align_corners=False,
            ),
        ),
    )
)
