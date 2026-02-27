import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SingleLevelToFPN(BaseModule):
    """Lightweight pyramid adapter for single-level image features.

    Input:
        Tensor[B, TN, C_in, H, W] or [Tensor[B, TN, C_in, H, W]]
    Output:
        list of Tensor[B, TN, C_out, H_l, W_l]
    """

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_outs=4,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if num_outs < 1 or num_outs > 4:
            raise ValueError(f'num_outs must be in [1, 4], got {num_outs}')

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_outs = int(num_outs)

        self.proj0 = ConvModule(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.down1 = ConvModule(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.down2 = ConvModule(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.down3 = ConvModule(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def _unpack_input(self, x):
        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError(
                    f'SingleLevelToFPN expects a single feature level, got {len(x)} levels')
            x = x[0]

        if not isinstance(x, torch.Tensor) or x.dim() != 5:
            raise ValueError(
                'SingleLevelToFPN expects Tensor[B, TN, C, H, W], '
                f'got type={type(x)} shape={getattr(x, "shape", None)}')
        return x

    def forward(self, x):
        x = self._unpack_input(x)
        bsz, tnv, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f'Input channel mismatch: expected {self.in_channels}, got {channels}')

        x = x.reshape(bsz * tnv, channels, height, width)

        p0 = self.proj0(x)
        p1 = self.down1(p0)
        p2 = self.down2(p1)
        p3 = self.down3(p2)

        outs = [p0, p1, p2, p3][:self.num_outs]
        reshaped_outs = []
        for feat in outs:
            _, c_out, h_out, w_out = feat.shape
            reshaped_outs.append(feat.reshape(bsz, tnv, c_out, h_out, w_out))
        return reshaped_outs
