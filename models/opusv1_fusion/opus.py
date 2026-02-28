import time
import queue
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.dist import get_dist_info
from mmcv.cnn import ConvModule
from mmcv.ops import Voxelization
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from ..compat import auto_fp16, cast_tensor_type
from ..utils import (GridMask, pad_multiple, GpuPhotoMetricDistortion,
                     disable_all_fp16_function, debug_is_finite)


@MODELS.register_module()
class OPUSV1Fusion(MVXTwoStageDetector):
    '''
        specifically for 2D SECOND

        Adding:
            2D feature

    '''
    def __init__(self,
                 use_grid_mask=True,
                 data_aug=None,
                 stop_prev_grad=0,
                 drop_lidar_feat=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 img_encoder=None,
                 img_feature_fusion=None,
                 use_external_img_encoder=False,
                 enable_pts_feature_branch=True,
                 external_img_cache=False,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 second_out_dim=512,
                 pts_feat_dim=256,
                 data_preprocessor=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs,
        )
        if pretrained is not None and init_cfg is None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.use_external_img_encoder = bool(use_external_img_encoder)
        self.enable_pts_feature_branch = bool(enable_pts_feature_branch)
        self.external_img_cache = bool(external_img_cache)
        self.img_encoder = MODELS.build(img_encoder) if img_encoder is not None else None
        if self.use_external_img_encoder and self.img_encoder is None:
            raise ValueError('img_encoder must be provided when use_external_img_encoder=True')
        self.img_feature_fusion_cfg = copy.deepcopy(img_feature_fusion) if img_feature_fusion else None
        self.use_img_feature_fusion = bool(
            self.img_feature_fusion_cfg is not None
            and self.img_encoder is not None
            and not self.use_external_img_encoder)
        img_encoder_cfg_freeze = False
        if isinstance(img_encoder, dict):
            img_encoder_cfg_freeze = bool(img_encoder.get('freeze', False))
        self._freeze_img_encoder = bool(
            img_encoder_cfg_freeze or
            (self.img_feature_fusion_cfg is not None
             and self.img_feature_fusion_cfg.get('freeze_img_encoder', True)))

        self.img_fusion_proj = None
        self.img_fusion_interp_mode = 'bilinear'
        self.img_fusion_align_corners = False
        if self.use_img_feature_fusion:
            if not self.with_img_backbone:
                raise ValueError('img_feature_fusion requires img_backbone to be enabled')
            fusion_out_channels = self._infer_img_fusion_out_channels(
                img_neck_cfg=img_neck,
                pts_bbox_head_cfg=pts_bbox_head)
            map_in_channels = int(self.img_feature_fusion_cfg.get('map_in_channels', 1024))
            self.img_fusion_proj = nn.Conv2d(
                map_in_channels,
                fusion_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True)
            self.img_fusion_interp_mode = self.img_feature_fusion_cfg.get('interp_mode', 'bilinear')
            self.img_fusion_align_corners = bool(
                self.img_feature_fusion_cfg.get('align_corners', False))
        if self._freeze_img_encoder and self.img_encoder is not None:
            self._freeze_module(self.img_encoder)

        self.pts_voxel_layer = None
        if pts_voxel_layer is not None:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = use_grid_mask

        self.memory = {}
        self.queue = queue.Queue()

        self.final_conv = None
        if self.enable_pts_feature_branch:
            self.final_conv = ConvModule(
                second_out_dim,
                pts_feat_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d')
            )

        self.pts_feat_dim=pts_feat_dim
        if not self.enable_pts_feature_branch:
            # These modules are intentionally disabled in this training setup.
            # Mark them frozen to avoid DDP unused-parameter reduction errors.
            self._freeze_module(self.pts_voxel_encoder)
            self._freeze_module(self.pts_middle_encoder)
            self._freeze_module(self.pts_backbone)
            self._freeze_module(self.pts_neck)

    @staticmethod
    def _freeze_module(module):
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval()

    @staticmethod
    def _infer_img_fusion_out_channels(img_neck_cfg, pts_bbox_head_cfg):
        if isinstance(img_neck_cfg, dict):
            out_channels = img_neck_cfg.get('out_channels', None)
            if isinstance(out_channels, (list, tuple)) and len(out_channels) > 0:
                return int(out_channels[0])
            if out_channels is not None:
                return int(out_channels)
        if isinstance(pts_bbox_head_cfg, dict):
            in_channels = pts_bbox_head_cfg.get('in_channels', None)
            if in_channels is not None:
                return int(in_channels)
        return 256

    def train(self, mode=True):
        super().train(mode)
        if self._freeze_img_encoder and self.img_encoder is not None:
            self.img_encoder.eval()
        return self

    def _need_img_branch(self):
        return self.use_external_img_encoder or self.with_img_backbone

    @staticmethod
    def _get_level_weights(weight_cfg, num_levels, name):
        if isinstance(weight_cfg, (float, int)):
            return [float(weight_cfg) for _ in range(num_levels)]
        if isinstance(weight_cfg, tuple):
            weight_cfg = list(weight_cfg)
        if not isinstance(weight_cfg, list):
            raise TypeError(
                f'{name} must be float/int or list/tuple of floats, got {type(weight_cfg)}')
        if len(weight_cfg) == 1:
            return [float(weight_cfg[0]) for _ in range(num_levels)]
        if len(weight_cfg) != num_levels:
            raise ValueError(
                f'{name} length mismatch: expected 1 or {num_levels}, got {len(weight_cfg)}')
        return [float(v) for v in weight_cfg]

    def _normalize_points_list(self, points):
        if points is None:
            return None
        if isinstance(points, torch.Tensor):
            if points.dim() == 2:
                return [points]
            if points.dim() != 3:
                raise TypeError(
                    'points Tensor input must have shape [N, C] or [B, N, C] when passed to external img encoder, '
                    f'but got {tuple(points.shape)}')
            return [points[i] for i in range(points.shape[0])]
        if isinstance(points, tuple):
            points = list(points)
        if not isinstance(points, list):
            raise TypeError(
                'points for external img encoder must be None, Tensor[B,N,C], or list of tensors, '
                f'but got {type(points)}')
        normalized = []
        for pts in points:
            if hasattr(pts, 'tensor'):
                pts = pts.tensor
            normalized.append(pts)
        return normalized

    def _run_external_img_encoder(self, img, points, img_metas, mapanything_extra=None):
        points = self._normalize_points_list(points)
        if self._freeze_img_encoder:
            with torch.no_grad():
                return self.img_encoder(
                    img,
                    points=points,
                    img_metas=img_metas,
                    mapanything_extra=mapanything_extra)
        return self.img_encoder(
            img,
            points=points,
            img_metas=img_metas,
            mapanything_extra=mapanything_extra)

    def _extract_external_img_feat_tensor(self, img, points, img_metas, mapanything_extra=None):
        img_feats = self._run_external_img_encoder(
            img,
            points=points,
            img_metas=img_metas,
            mapanything_extra=mapanything_extra)
        if isinstance(img_feats, (list, tuple)):
            if len(img_feats) == 1 and isinstance(img_feats[0], torch.Tensor):
                img_feats = img_feats[0]
            else:
                raise ValueError(
                    'External img encoder must output a single Tensor[B, TN, C, H, W] '
                    'when img_feature_fusion is enabled.')
        if not isinstance(img_feats, torch.Tensor) or img_feats.dim() != 5:
            raise ValueError(
                'External img encoder output must be Tensor[B, TN, C, H, W], '
                f"but got type={type(img_feats)} shape={getattr(img_feats, 'shape', None)}")
        return img_feats

    def _extract_external_img_feat(self, img, points, img_metas, mapanything_extra=None):
        img_feats = self._run_external_img_encoder(
            img,
            points=points,
            img_metas=img_metas,
            mapanything_extra=mapanything_extra)

        if isinstance(img_feats, (list, tuple)):
            if len(img_feats) == 1 and isinstance(img_feats[0], torch.Tensor):
                img_feats = img_feats[0]
            else:
                return list(img_feats)

        if not isinstance(img_feats, torch.Tensor) or img_feats.dim() != 5:
            raise ValueError(
                'External img encoder output must be Tensor[B, TN, C, H, W] or single-element list, '
                f"but got type={type(img_feats)} shape={getattr(img_feats, 'shape', None)}")

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        else:
            img_feats = [img_feats]

        if not isinstance(img_feats, (list, tuple)):
            raise TypeError(f'img_neck output must be list/tuple, got {type(img_feats)}')
        for lvl, feat in enumerate(img_feats):
            if not isinstance(feat, torch.Tensor) or feat.dim() != 5:
                raise ValueError(
                    f'Image feature at level {lvl} must be Tensor[B, TN, C, H, W], '
                    f"got type={type(feat)} shape={getattr(feat, 'shape', None)}")
        return list(img_feats)

    def _interpolate_external_feat(self, feat_2d, target_hw):
        if feat_2d.shape[-2:] == target_hw:
            return feat_2d
        if self.img_fusion_interp_mode in ('nearest', 'area', 'nearest-exact'):
            return F.interpolate(
                feat_2d,
                size=target_hw,
                mode=self.img_fusion_interp_mode)
        return F.interpolate(
            feat_2d,
            size=target_hw,
            mode=self.img_fusion_interp_mode,
            align_corners=self.img_fusion_align_corners)

    def _fuse_img_features(self, fpn_feats, external_feat):
        if not isinstance(fpn_feats, (list, tuple)) or len(fpn_feats) == 0:
            raise ValueError('fpn_feats must be a non-empty list of image features')
        if self.img_fusion_proj is None:
            return list(fpn_feats)
        if not isinstance(external_feat, torch.Tensor) or external_feat.dim() != 5:
            raise ValueError(
                'external_feat must be Tensor[B, TN, C, H, W], '
                f'but got type={type(external_feat)} shape={getattr(external_feat, "shape", None)}')

        B_ext, TN_ext, C_ext, H_ext, W_ext = external_feat.shape
        ext_feat_2d = external_feat.reshape(B_ext * TN_ext, C_ext, H_ext, W_ext)
        ext_feat_2d = ext_feat_2d.to(dtype=self.img_fusion_proj.weight.dtype)
        ext_feat_proj_2d = self.img_fusion_proj(ext_feat_2d)
        proj_channels = ext_feat_proj_2d.shape[1]

        num_levels = len(fpn_feats)
        alpha = self._get_level_weights(
            self.img_feature_fusion_cfg.get('alpha', 1.0),
            num_levels=num_levels,
            name='img_feature_fusion.alpha')
        beta = self._get_level_weights(
            self.img_feature_fusion_cfg.get('beta', 0.0),
            num_levels=num_levels,
            name='img_feature_fusion.beta')

        fused_feats = []
        for lvl, fpn_feat in enumerate(fpn_feats):
            if not isinstance(fpn_feat, torch.Tensor) or fpn_feat.dim() != 5:
                raise ValueError(
                    f'FPN feature level {lvl} must be Tensor[B, TN, C, H, W], '
                    f'got type={type(fpn_feat)} shape={getattr(fpn_feat, "shape", None)}')
            B_fpn, TN_fpn, C_fpn, H_fpn, W_fpn = fpn_feat.shape
            if B_fpn != B_ext or TN_fpn != TN_ext:
                raise ValueError(
                    f'Feature batch/TN mismatch at level {lvl}: '
                    f'fpn={tuple(fpn_feat.shape[:2])}, external={(B_ext, TN_ext)}')
            if C_fpn != proj_channels:
                raise ValueError(
                    f'Feature channel mismatch at level {lvl}: '
                    f'fpn C={C_fpn}, projected external C={proj_channels}. '
                    'Please check img_neck out_channels and img_feature_fusion.map_in_channels.')
            ext_lvl_2d = self._interpolate_external_feat(
                ext_feat_proj_2d,
                target_hw=(H_fpn, W_fpn))
            ext_lvl = ext_lvl_2d.view(B_fpn, TN_fpn, C_fpn, H_fpn, W_fpn)
            ext_lvl = ext_lvl.to(dtype=fpn_feat.dtype)
            fused = alpha[lvl] * fpn_feat + beta[lvl] * ext_lvl
            fused_feats.append(fused)
        return fused_feats

    def _extract_pts_feat_for_head(self, points):
        if (not self.enable_pts_feature_branch) or (not self.with_pts_backbone):
            return None
        pts_feats = self.extract_pts_feat(points)
        return self.final_conv(pts_feats[0])

    @torch.no_grad()
    def voxelize(self, points):
        """Apply hard voxelization on a batch of points."""
        assert self.pts_voxel_layer is not None, 'pts_voxel_layer is not set'
        voxels, num_points, coors = [], [], []
        for i, res in enumerate(points):
            if hasattr(res, 'tensor'):
                res = res.tensor
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            num_points.append(res_num_points)
            coors.append(res_coors)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors = torch.cat(coors, dim=0)
        return voxels, num_points, coors

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat_(self, img):
        # import pdb; pdb.set_trace()
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_img_feat(self, img, img_metas, points=None, mapanything_extra=None):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5
        img_for_external = img

        if self.use_external_img_encoder:
            return self._extract_external_img_feat(
                img_for_external,
                points=points,
                img_metas=img_metas,
                mapanything_extra=mapanything_extra)

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            num_views = self._get_num_views()
            img = img.reshape(B, -1, num_views, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            all_img_feats = [self.extract_img_feat_(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat_(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, num_views, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat_(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        if self.use_img_feature_fusion:
            external_feat = self._extract_external_img_feat_tensor(
                img_for_external,
                points=points,
                img_metas=img_metas,
                mapanything_extra=mapanything_extra)
            img_feats_reshaped = self._fuse_img_features(img_feats_reshaped, external_feat)

        return img_feats_reshaped
    
    @auto_fp16(apply_to=('pts'), out_fp32=True)
    def extract_pts_feat(self, pts):
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        return self._forward(inputs, data_samples)

    def _forward(self, inputs, data_samples=None):
        img = inputs.get('img') if isinstance(inputs, dict) else inputs
        img_metas = self._collect_img_metas(data_samples)
        return self.extract_feat(img, img_metas)

    def loss(self, inputs, data_samples):
        img = inputs.get('img') if isinstance(inputs, dict) else inputs
        points = inputs.get('points') if isinstance(inputs, dict) else None
        mapanything_extra = inputs.get('mapanything_extra') if isinstance(inputs, dict) else None
        img_metas = self._collect_img_metas(data_samples)
        voxel_semantics = self._stack_data_samples(data_samples, 'voxel_semantics')
        mask_camera = self._stack_data_samples(data_samples, 'mask_camera')
        debug_is_finite('input.img', img)
        debug_is_finite('input.points', points)
        debug_is_finite('input.voxel_semantics', voxel_semantics)
        debug_is_finite('input.mask_camera', mask_camera)
        if img_metas:
            for i, meta in enumerate(img_metas):
                if 'ego2img' in meta:
                    debug_is_finite(f'img_metas[{i}].ego2img', np.asarray(meta['ego2img']))
                if 'ego2occ' in meta:
                    debug_is_finite(f'img_metas[{i}].ego2occ', np.asarray(meta['ego2occ']))
                if 'ego2lidar' in meta:
                    debug_is_finite(f'img_metas[{i}].ego2lidar', np.asarray(meta['ego2lidar']))
        return self.forward_train(
            img=img,
            points=points,
            mapanything_extra=mapanything_extra,
            img_metas=img_metas,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera,
        )

    def predict(self, inputs, data_samples, rescale=False):
        img = inputs.get('img') if isinstance(inputs, dict) else inputs
        points = inputs.get('points') if isinstance(inputs, dict) else None
        mapanything_extra = inputs.get('mapanything_extra') if isinstance(inputs, dict) else None
        img_metas = self._collect_img_metas(data_samples)
        return self.simple_test(img_metas, img, points, rescale=rescale,
                                mapanything_extra=mapanything_extra)

    def forward_train(self,
                      points=None,
                      mapanything_extra=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      voxel_semantics=None,
                      mask_camera=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img_feats = self.extract_img_feat(
            img,
            img_metas,
            points=points,
            mapanything_extra=mapanything_extra) \
            if self._need_img_branch() else None
        if pts_feats is not None:
            pts_feats = self._extract_pts_feat_for_head(points)
            pts_feats = self._maybe_drop_lidar_feat(pts_feats)
        debug_is_finite('img_feats', img_feats)
        debug_is_finite('pts_feats', pts_feats)

        # forward occ head
        outs = self.pts_bbox_head(mlvl_feats=img_feats, pts_feats=pts_feats,
                                  img_metas=img_metas, points=points)
        if mask_camera is None:
            mask_camera = torch.ones_like(voxel_semantics)
        loss_inputs = [voxel_semantics, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    @disable_all_fp16_function
    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        return self.simple_test(img_metas[0], img[0], points[0], **kwargs)

    def _get_num_views(self):
        transformer = getattr(self.pts_bbox_head, 'transformer', None)
        if transformer is not None and hasattr(transformer, 'num_views'):
            return int(transformer.num_views)
        return 6

    def _can_use_online_test(self, img_metas, img):
        if get_dist_info()[1] != 1:
            return False
        if not isinstance(img_metas, list) or len(img_metas) != 1:
            return False
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            return False
        num_views = self._get_num_views()
        return num_views > 0 and img.shape[1] % num_views == 0

    def simple_test(self,
                    img_metas,
                    img=None,
                    points=None,
                    rescale=False,
                    mapanything_extra=None):
        if self._can_use_online_test(img_metas, img):
            return self.simple_test_online(
                img_metas,
                img,
                points,
                rescale,
                mapanything_extra=mapanything_extra)
        return self.simple_test_offline(
            img_metas,
            img,
            points,
            rescale,
            mapanything_extra=mapanything_extra)

    def _collect_img_metas(self, data_samples):
        if not data_samples:
            return []
        return [sample.metainfo for sample in data_samples]

    def _stack_data_samples(self, data_samples, key):
        if not data_samples:
            return None
        values = [getattr(sample, key) for sample in data_samples if hasattr(sample, key)]
        if not values:
            return None
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values, dim=0)
        return torch.tensor(values)

    def _normalize_mapanything_extra(self, mapanything_extra, batch_size):
        if mapanything_extra is None:
            return None
        if isinstance(mapanything_extra, dict):
            return [copy.deepcopy(mapanything_extra) for _ in range(batch_size)]
        if isinstance(mapanything_extra, tuple):
            mapanything_extra = list(mapanything_extra)
        if not isinstance(mapanything_extra, list):
            raise TypeError(
                'mapanything_extra must be None, dict, or list of dict, '
                f'but got {type(mapanything_extra)}')
        if len(mapanything_extra) != batch_size:
            raise ValueError(
                f'mapanything_extra batch size mismatch: got {len(mapanything_extra)}, '
                f'expected {batch_size}')
        normalized = []
        for item in mapanything_extra:
            if item is None:
                normalized.append(None)
            elif isinstance(item, dict):
                normalized.append(copy.deepcopy(item))
            else:
                raise TypeError(
                    f'Each mapanything_extra item must be dict/None, got {type(item)}')
        return normalized

    def _slice_mapanything_extra(self, mapanything_extra, img_indices, total_views):
        if mapanything_extra is None:
            return None

        sliced = []
        for sample_extra in mapanything_extra:
            if sample_extra is None:
                sliced.append(None)
                continue
            if not isinstance(sample_extra, dict):
                raise TypeError(
                    f'Each mapanything_extra item must be dict/None, got {type(sample_extra)}')
            sample_out = copy.deepcopy(sample_extra)
            views = sample_out.get('views', None)
            if isinstance(views, (list, tuple)) and len(views) == total_views:
                sample_out['views'] = [views[j] for j in img_indices]
            sliced.append(sample_out)
        return sliced

    def simple_test_offline(self,
                            img_metas,
                            img=None,
                            points=None,
                            rescale=False,
                            mapanything_extra=None):
        img_feats = self.extract_img_feat(
            img,
            img_metas,
            points=points,
            mapanything_extra=mapanything_extra) \
            if self._need_img_branch() else None
        pts_feats = self._extract_pts_feat_for_head(points)

        outs = self.pts_bbox_head(mlvl_feats=img_feats, pts_feats=pts_feats,
                                  img_metas=img_metas, points=points)
        return self.pts_bbox_head.get_occ(outs, img_metas[0], rescale=rescale)

    def simple_test_online(self,
                           img_metas,
                           img=None,
                           points=None,
                           rescale=False,
                           mapanything_extra=None):
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape
        mapanything_extra = self._normalize_mapanything_extra(mapanything_extra, B)
        num_views = self._get_num_views()
        img = img.reshape(B, N // num_views, num_views, C, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // num_views
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        uses_external_encoder = self.use_external_img_encoder or self.use_img_feature_fusion
        allow_cache = not (uses_external_encoder and not self.external_img_cache)
        if mapanything_extra is not None:
            # Optional extra modalities can change feature semantics even for same filename.
            allow_cache = False
        for i in range(num_frames):
            img_indices = list(np.arange(i * num_views, (i + 1) * num_views))
            extra_curr = self._slice_mapanything_extra(
                mapanything_extra,
                img_indices=img_indices,
                total_views=num_views * num_frames)

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                item = img_metas[0][k]
                if isinstance(item, list) and (len(item) == num_views * num_frames):
                    img_metas_curr[0][k] = [item[j] for j in img_indices]
                else:
                    img_metas_curr[0][k] = item

            if allow_cache and img_filenames[img_indices[0]] in self.memory:
                # found in memory
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                # extract feature and put into memory
                img_feats_curr = self.extract_img_feat(
                    img[:, i],
                    img_metas_curr,
                    points=points,
                    mapanything_extra=extra_curr)
                if allow_cache:
                    self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                    self.queue.put(img_filenames[img_indices[0]])
                    while self.queue.qsize() >= 16:  # avoid OOM
                        pop_key = self.queue.get()
                        self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # extract points features
        pts_feats = self._extract_pts_feat_for_head(points)
        if pts_feats is not None:
                pts_feats = self._maybe_drop_lidar_feat(pts_feats)

        # run occupancy predictor
        outs = self.pts_bbox_head(mlvl_feats=img_feats, pts_feats=pts_feats,
                                  img_metas=img_metas, points=points)
        return self.pts_bbox_head.get_occ(outs, img_metas[0], rescale=rescale)
