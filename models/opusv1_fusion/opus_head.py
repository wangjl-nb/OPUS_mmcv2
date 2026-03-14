import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import furthest_point_sample, gather_points, knn, Voxelization
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from ..bbox.utils import decode_points, encode_points
from ..compat import build_loss, build_transformer, force_fp32, multi_apply
from ..utils import debug_is_finite


@MODELS.register_module()
class OPUSV1FusionHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 init_pos_lidar=None,
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 feature_supervision=None,
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_occ=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                 loss_pts=dict(type='L1Loss'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        loss_cls_cfg = dict(loss_cls or {})
        if 'type' in loss_cls_cfg and '_scope_' not in loss_cls_cfg:
            loss_cls_cfg['_scope_'] = 'mmdet'
        self.loss_cls = build_loss(loss_cls_cfg)
        loss_pts_cfg = dict(loss_pts or {})
        if 'type' in loss_pts_cfg and '_scope_' not in loss_pts_cfg:
            loss_pts_cfg['_scope_'] = 'mmdet'
        self.loss_pts = build_loss(loss_pts_cfg)
        self.loss_occ_cfg = dict(loss_occ or {})
        self.feature_supervision = dict(feature_supervision or {})
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.feature_dims = int(getattr(self.transformer, 'feature_dims', 0) or 0)
        self.score_mode = str(getattr(self.transformer, 'score_mode', 'semantic')).lower()
        self.occ_out_channels = int(getattr(self.transformer, 'occ_out_channels', 1) or 1)
        self.voxel_generator = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=10,
            max_voxels=self.num_query * self.num_refines[-1],
            deterministic=False)

        # position initialization
        assert init_pos_lidar in [None, 'all', 'curr'], \
            'init_pos_lidar should be one of [None, "all", "curr"], ' \
            f'but got {init_pos_lidar}'
        self.init_pos_lidar = init_pos_lidar

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()

        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)
        # Keep EMA stats ephemeral so older checkpoints can load without strict mismatch.
        self.register_buffer(
            'tail_class_freq_ema',
            torch.zeros(self.num_classes, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            'tail_ema_updates',
            torch.zeros((), dtype=torch.long),
            persistent=False)
        occ_target_cfg = dict((self.train_cfg or {}).get('occ_target_cfg', {}))
        self.occ_pos_dist_thr = float(occ_target_cfg.get('pos_dist_thr', 0.10))
        self.occ_neg_dist_thr = float(occ_target_cfg.get('neg_dist_thr', 0.20))
        if self.occ_pos_dist_thr < 0 or self.occ_neg_dist_thr < self.occ_pos_dist_thr:
            raise ValueError(
                'occ_target_cfg must satisfy 0 <= pos_dist_thr <= neg_dist_thr, '
                f'got pos={self.occ_pos_dist_thr}, neg={self.occ_neg_dist_thr}')
        self._init_feature_supervision()

        self._init_layers()

    def _init_layers(self):
        if not self.init_pos_lidar:
            self.init_points = nn.Embedding(self.num_query, 3)
            nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()

    def _init_feature_supervision(self):
        cfg = self.feature_supervision
        self.feature_supervision_enabled = bool(cfg.get('enabled', False))
        self.prototype_decode_enabled = bool(cfg.get('prototype_decode', self.feature_supervision_enabled))
        self.prototype_metric = str(cfg.get('similarity', 'cosine')).lower()
        self.loss_feat_cosine_weight = float(cfg.get('loss_weights', {}).get('cosine', 1.0))
        self.loss_feat_mse_weight = float(cfg.get('loss_weights', {}).get('mse', 0.1))
        self.loss_feat_ce_weight = float(cfg.get('loss_weights', {}).get('ce', 0.0))
        self.loss_feat_margin_weight = float(cfg.get('loss_weights', {}).get('margin', 0.0))
        self.loss_feat_temperature = float(cfg.get('temperature', 0.07))
        self.loss_feat_margin = float(cfg.get('margin', 0.10))
        self.prototype_field = str(cfg.get('prototype_field', 'latent_128'))
        self.prototype_label_names = list(cfg.get('class_names', []))
        self.feature_only_positive = bool(cfg.get('only_positive', False))
        self.feature_strong_pos_dist_thr = float(
            cfg.get('strong_pos_dist_thr', self.occ_pos_dist_thr))
        self.feature_weak_pos_dist_thr = float(
            cfg.get('weak_pos_dist_thr', self.feature_strong_pos_dist_thr))
        self.feature_weak_pos_weight = float(cfg.get('weak_pos_weight', 1.0))
        self.tail_feature_weight = float(cfg.get('tail_feature_weight', 1.0))
        self.register_buffer('prototype_bank', torch.empty(0), persistent=False)

        if self.loss_feat_temperature <= 0:
            raise ValueError(
                f'feature_supervision.temperature must be positive, got {self.loss_feat_temperature}')
        if self.feature_strong_pos_dist_thr < 0 or \
           self.feature_weak_pos_dist_thr < self.feature_strong_pos_dist_thr:
            raise ValueError(
                'feature_supervision strong/weak positive thresholds must satisfy '
                f'0 <= strong <= weak, got strong={self.feature_strong_pos_dist_thr}, '
                f'weak={self.feature_weak_pos_dist_thr}')
        if self.feature_weak_pos_weight < 0:
            raise ValueError(
                f'feature_supervision.weak_pos_weight must be non-negative, got {self.feature_weak_pos_weight}')

        if not self.feature_supervision_enabled:
            return
        if self.feature_dims <= 0:
            raise ValueError('feature_supervision.enabled=True requires transformer.feature_dims > 0')
        if not self.prototype_label_names:
            raise ValueError('feature_supervision.enabled=True requires class_names')

        prototype_npz_path = cfg.get('prototype_npz_path', None)
        if not prototype_npz_path:
            raise ValueError('feature_supervision.enabled=True requires prototype_npz_path')
        prototype_npz_path = Path(prototype_npz_path)
        if not prototype_npz_path.exists():
            raise FileNotFoundError(f'Prototype npz does not exist: {prototype_npz_path}')

        bridge_entries = None
        prototype_bridge_path = cfg.get('prototype_bridge_path', None)
        if prototype_bridge_path:
            prototype_bridge_path = Path(prototype_bridge_path)
            if not prototype_bridge_path.exists():
                raise FileNotFoundError(f'Prototype bridge json does not exist: {prototype_bridge_path}')
            with prototype_bridge_path.open('r', encoding='utf-8') as handle:
                bridge_entries = json.load(handle).get('entries', [])

        with np.load(prototype_npz_path, allow_pickle=False) as data:
            if self.prototype_field not in data.files:
                raise KeyError(
                    f'Prototype npz missing field "{self.prototype_field}": {prototype_npz_path}')
            vectors = np.asarray(data[self.prototype_field], dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(f'Prototype bank must be 2D, got {vectors.shape}')
        if vectors.shape[1] != self.feature_dims:
            raise ValueError(
                f'Prototype dim mismatch: expected {self.feature_dims}, got {vectors.shape[1]}')

        if bridge_entries is not None:
            if len(bridge_entries) != vectors.shape[0]:
                raise ValueError(
                    f'Prototype bridge length mismatch: entries={len(bridge_entries)} '
                    f'vs vectors={vectors.shape[0]}')
            raw_to_vec = {
                str(entry['raw_name']): vectors[idx]
                for idx, entry in enumerate(bridge_entries)
            }
        else:
            with np.load(prototype_npz_path, allow_pickle=True) as data:
                if 'raw_names' not in data.files:
                    raise KeyError(
                        'Prototype npz missing raw_names and no prototype_bridge_path was provided: '
                        f'{prototype_npz_path}')
                raw_names = [str(x) for x in data['raw_names'].tolist()]
            raw_to_vec = {raw_name: vectors[idx] for idx, raw_name in enumerate(raw_names)}
        bank = []
        missing = []
        for label_name in self.prototype_label_names:
            vec = raw_to_vec.get(str(label_name), None)
            if vec is None:
                missing.append(str(label_name))
                vec = np.zeros((self.feature_dims,), dtype=np.float32)
            bank.append(vec)
        if missing:
            raise KeyError(f'Prototype bank missing labels: {missing}')

        bank = torch.from_numpy(np.stack(bank, axis=0))
        bank = F.normalize(bank, dim=-1)
        self.prototype_bank = bank

    @staticmethod
    def _weighted_mean(values, weights):
        if values.numel() == 0:
            return values.sum() * 0.0
        weights = weights.to(values.dtype)
        denom = weights.sum().clamp(min=1e-6)
        return (values * weights).sum() / denom

    def _build_binary_occ_targets(self, pred_pts, pred_paired_pts):
        pred_dist = torch.norm(pred_pts - pred_paired_pts, dim=-1)
        positive_mask = pred_dist <= self.occ_pos_dist_thr
        negative_mask = pred_dist >= self.occ_neg_dist_thr
        valid_mask = positive_mask | negative_mask
        occ_targets = positive_mask.to(dtype=pred_pts.dtype)
        return occ_targets, valid_mask, positive_mask

    def _compute_binary_occ_loss(self, score_preds, occ_targets, valid_mask):
        if score_preds is None:
            return None
        if score_preds.shape[-1] != 1:
            raise ValueError(
                f'binary occ mode expects score dim=1, got {score_preds.shape[-1]}')
        if valid_mask.sum() <= 0:
            return score_preds.sum() * 0.0

        score_preds = score_preds.squeeze(-1)
        score_preds = score_preds[valid_mask]
        occ_targets = occ_targets[valid_mask]

        gamma = float(self.loss_occ_cfg.get('gamma', 2.0))
        alpha = float(self.loss_occ_cfg.get('alpha', 0.25))
        loss_weight = float(self.loss_occ_cfg.get('loss_weight', 1.0))

        prob = score_preds.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            score_preds, occ_targets, reduction='none')
        p_t = prob * occ_targets + (1.0 - prob) * (1.0 - occ_targets)
        focal_weight = (1.0 - p_t).pow(gamma)
        alpha_factor = alpha * occ_targets + (1.0 - alpha) * (1.0 - occ_targets)
        loss = ce_loss * focal_weight * alpha_factor
        return loss.sum() / max(int(valid_mask.sum().item()), 1) * loss_weight

    def _build_feature_supervision_mask_weights(
            self, pred_pts, pred_paired_pts, labels, cls_weights, positive_mask=None):
        if pred_pts.numel() == 0:
            empty_mask = pred_pts.new_zeros((0,), dtype=torch.bool)
            empty_weights = pred_pts.new_zeros((0,), dtype=pred_pts.dtype)
            return empty_mask, empty_weights

        pred_dist = torch.norm(pred_pts - pred_paired_pts, dim=-1)
        strong_mask = pred_dist <= self.feature_strong_pos_dist_thr
        weak_mask = (pred_dist > self.feature_strong_pos_dist_thr) & \
                    (pred_dist <= self.feature_weak_pos_dist_thr)

        if self.feature_only_positive:
            feature_mask = strong_mask | weak_mask
        else:
            feature_mask = torch.ones_like(strong_mask, dtype=torch.bool)

        sample_weights = pred_pts.new_ones(pred_dist.shape[0])
        sample_weights[weak_mask] = self.feature_weak_pos_weight

        row_idx = torch.arange(labels.shape[0], device=labels.device)
        scalar_cls_weights = cls_weights[row_idx, labels.long()].to(sample_weights.dtype)
        sample_weights = sample_weights * scalar_cls_weights

        if self.tail_feature_weight > 1.0:
            rare_classes = [int(x) for x in (self.train_cfg or {}).get('rare_classes', [])]
            if rare_classes:
                tail_mask = torch.zeros_like(feature_mask)
                for cls_idx in rare_classes:
                    tail_mask |= (labels == cls_idx)
                sample_weights[tail_mask] = sample_weights[tail_mask] * self.tail_feature_weight

        if positive_mask is not None and self.feature_strong_pos_dist_thr <= self.occ_pos_dist_thr \
           and self.feature_weak_pos_dist_thr <= self.occ_pos_dist_thr:
            # Backward-compatible path when semantic supervision only uses strict occupied queries.
            feature_mask = positive_mask

        return feature_mask, sample_weights

    def _compute_feature_loss(
            self,
            feat_scores,
            labels,
            cls_weights,
            pred_pts=None,
            pred_paired_pts=None,
            positive_mask=None):
        if (not self.feature_supervision_enabled) or feat_scores is None:
            return None

        if pred_pts is None or pred_paired_pts is None:
            feature_mask = positive_mask
            if feature_mask is None:
                feature_mask = torch.ones(
                    feat_scores.shape[0], device=feat_scores.device, dtype=torch.bool)
            row_idx = torch.arange(labels.shape[0], device=labels.device)
            sample_weights = cls_weights[row_idx, labels.long()].to(feat_scores.dtype)
        else:
            feature_mask, sample_weights = self._build_feature_supervision_mask_weights(
                pred_pts,
                pred_paired_pts,
                labels,
                cls_weights,
                positive_mask=positive_mask)

        if feature_mask is not None:
            if feature_mask.sum() <= 0:
                return feat_scores.sum() * 0.0
            feat_scores = feat_scores[feature_mask]
            labels = labels[feature_mask]
            sample_weights = sample_weights[feature_mask]

        feat_scores = F.normalize(feat_scores, dim=-1)
        target_feats = self.prototype_bank.index_select(0, labels.long())
        target_feats = F.normalize(target_feats, dim=-1)

        loss_feat = feat_scores.sum() * 0.0
        if self.loss_feat_cosine_weight > 0:
            cosine_loss = 1.0 - F.cosine_similarity(feat_scores, target_feats, dim=-1)
            loss_feat = loss_feat + self.loss_feat_cosine_weight * self._weighted_mean(
                cosine_loss, sample_weights)

        if self.loss_feat_mse_weight > 0:
            mse_loss = F.mse_loss(feat_scores, target_feats, reduction='none').mean(dim=-1)
            loss_feat = loss_feat + self.loss_feat_mse_weight * self._weighted_mean(
                mse_loss, sample_weights)

        if self.loss_feat_ce_weight > 0:
            logits = torch.matmul(feat_scores, self.prototype_bank.t()) / self.loss_feat_temperature
            ce_loss = F.cross_entropy(logits, labels.long(), reduction='none')
            loss_feat = loss_feat + self.loss_feat_ce_weight * self._weighted_mean(
                ce_loss, sample_weights)

        if self.loss_feat_margin_weight > 0:
            similarity = torch.matmul(feat_scores, self.prototype_bank.t())
            row_idx = torch.arange(labels.shape[0], device=labels.device)
            sim_pos = similarity[row_idx, labels.long()]
            similarity_neg = similarity.clone()
            similarity_neg[row_idx, labels.long()] = torch.finfo(similarity_neg.dtype).min
            sim_neg_max = similarity_neg.max(dim=-1).values
            margin_loss = F.relu(sim_neg_max - sim_pos + self.loss_feat_margin)
            loss_feat = loss_feat + self.loss_feat_margin_weight * self._weighted_mean(
                margin_loss, sample_weights)

        return loss_feat

    def _uniform_sample_pc_range(self, num_points, device, dtype):
        if num_points <= 0:
            return torch.zeros((0, 3), device=device, dtype=dtype)
        low = self.pc_range[:3].to(device=device, dtype=dtype)
        high = self.pc_range[3:].to(device=device, dtype=dtype)
        rand = torch.rand((num_points, 3), device=device, dtype=dtype)
        return rand * (high - low) + low

    def _sample_lidar_queries(self, pts_xyz, num_lidar):
        if num_lidar <= 0:
            return pts_xyz.new_zeros((0, 3))
        if pts_xyz.numel() == 0:
            return self._uniform_sample_pc_range(num_lidar, pts_xyz.device, pts_xyz.dtype)

        pts_xyz = pts_xyz.contiguous()
        sample_num = min(num_lidar, pts_xyz.shape[0])
        sample_idx = furthest_point_sample(pts_xyz.unsqueeze(0), sample_num)
        sampled = gather_points(
            pts_xyz.unsqueeze(0).transpose(1, 2).contiguous(),
            sample_idx).transpose(1, 2).squeeze(0)

        if sampled.shape[0] < num_lidar:
            repeat_idx = torch.arange(
                num_lidar - sampled.shape[0], device=pts_xyz.device) % max(sampled.shape[0], 1)
            if sampled.shape[0] > 0:
                sampled = torch.cat([sampled, sampled[repeat_idx]], dim=0)
            else:
                sampled = self._uniform_sample_pc_range(num_lidar, pts_xyz.device, pts_xyz.dtype)
        return sampled[:num_lidar]

    def _sample_random_queries(self, pts_xyz, num_random, random_mode='uniform_pc_range'):
        if num_random <= 0:
            return pts_xyz.new_zeros((0, 3))

        sampled = []
        # Keep compatibility with small-object focus default: random queries prefer real lidar points,
        # then back-fill from the valid pc range.
        use_point_first = random_mode in ['uniform_pc_range', 'point_then_uniform', 'point_random']
        if use_point_first and pts_xyz.numel() > 0:
            sample_num = min(num_random, pts_xyz.shape[0])
            rand_idx = torch.randperm(pts_xyz.shape[0], device=pts_xyz.device)[:sample_num]
            sampled.append(pts_xyz[rand_idx])
            num_random -= sample_num

        if num_random > 0:
            sampled.append(self._uniform_sample_pc_range(num_random, pts_xyz.device, pts_xyz.dtype))

        if not sampled:
            return pts_xyz.new_zeros((0, 3))
        return torch.cat(sampled, dim=0)

    def _resolve_query_init_mix(self):
        mix_cfg = (self.train_cfg or {}).get('query_init_mix', {})
        num_query = int(getattr(self, 'num_query', 0))
        if num_query <= 0:
            return 0, 0
        if not mix_cfg.get('enabled', False):
            return num_query, 0

        lidar_ratio = float(mix_cfg.get('lidar_ratio', 1.0))
        random_ratio = float(mix_cfg.get('random_ratio', 1.0 - lidar_ratio))
        lidar_ratio = max(lidar_ratio, 0.0)
        random_ratio = max(random_ratio, 0.0)
        ratio_sum = lidar_ratio + random_ratio
        if ratio_sum <= 0:
            return num_query, 0

        lidar_ratio = lidar_ratio / ratio_sum
        num_lidar = int(round(num_query * lidar_ratio))
        num_lidar = max(0, min(num_query, num_lidar))
        num_random = num_query - num_lidar
        return num_lidar, num_random

    def get_init_position(self, points, mlvl_feats, pts_feats, img_metas):
        B, Q = mlvl_feats[0].shape[0], self.num_query

        if not self.init_pos_lidar:
            init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
            query_feat = init_points.new_zeros(B, Q, self.embed_dims)
            return init_points, query_feat

        with torch.no_grad():
            assert points is not None
            mix_cfg = (self.train_cfg or {}).get('query_init_mix', {})
            random_mode = mix_cfg.get('random_mode', 'uniform_pc_range')
            num_lidar, num_random = self._resolve_query_init_mix()

            init_points = []
            for pts in points:
                if hasattr(pts, 'tensor'):
                    pts = pts.tensor
                if self.init_pos_lidar == 'curr' and pts.shape[-1] > 3:
                    # Only use the current lidar points.
                    pts = pts[pts[:, -1] == 0]

                pts_xyz = pts[..., :3].contiguous()
                lidar_points = self._sample_lidar_queries(pts_xyz, num_lidar)
                random_points = self._sample_random_queries(
                    pts_xyz,
                    num_random,
                    random_mode=random_mode)

                mixed_points = torch.cat([lidar_points, random_points], dim=0)
                if mixed_points.shape[0] < self.num_query:
                    extra = self._uniform_sample_pc_range(
                        self.num_query - mixed_points.shape[0],
                        mixed_points.device,
                        mixed_points.dtype)
                    mixed_points = torch.cat([mixed_points, extra], dim=0)
                elif mixed_points.shape[0] > self.num_query:
                    mixed_points = mixed_points[:self.num_query]
                init_points.append(mixed_points.unsqueeze(0))

            init_points = torch.cat(init_points, dim=0).unsqueeze(2)
            init_points = encode_points(init_points, self.pc_range)

        query_feat = init_points.new_zeros(B, Q, self.embed_dims)
        return init_points, query_feat

    def forward(self,
                mlvl_feats=None,
                pts_feats=None,
                tpv_feats=None,
                points=None,
                img_metas=None):
        init_points, query_feat = self.get_init_position(points, mlvl_feats,
                                                         pts_feats, img_metas)
        score_preds, refine_pts, feat_scores = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            pts_feats,
            img_metas=img_metas,
            tpv_feats=tpv_feats,
        )

        outputs = dict(
            init_points=init_points,
            all_refine_pts=refine_pts,
            all_feat_scores=feat_scores)
        if self.score_mode == 'binary_occ':
            outputs['all_occ_scores'] = score_preds
        else:
            outputs['all_cls_scores'] = score_preds
        return outputs

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist_xy = (pts - centers[None, ...])[..., :2]
        dist_xy = torch.norm(dist_xy, dim=-1)
        return dist_xy / max_dist + 1

    def discretize(self, pts, clip=True, decode=False):
        loc = torch.floor((pts - self.pc_range[:3]) / self.voxel_size)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)

        return loc.long() if not decode else \
            (loc + 0.5) * self.voxel_size + self.pc_range[:3]

    def _build_class_mask(self, class_ids, device):
        mask = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        empty_label = int(getattr(self, 'empty_label', -1))
        for cls_idx in class_ids:
            cls_idx = int(cls_idx)
            if 0 <= cls_idx < self.num_classes and cls_idx != empty_label:
                mask[cls_idx] = True
        return mask

    def _get_fallback_tail_mask(self, device):
        rare_classes = (self.train_cfg or {}).get('rare_classes', [])
        return self._build_class_mask(rare_classes, device)

    def _sync_class_count(self, class_count, sync_stats=True):
        if not sync_stats:
            return class_count
        if not dist.is_available() or not dist.is_initialized():
            return class_count
        synced = class_count.clone()
        dist.all_reduce(synced, op=dist.ReduceOp.SUM)
        return synced

    def _update_tail_freq_ema(self, class_count, tail_cfg):
        class_count = self._sync_class_count(
            class_count,
            sync_stats=tail_cfg.get('sync_stats', True))
        total_count = class_count.sum()
        if total_count <= 0:
            return None

        batch_freq = class_count / total_count
        momentum = float(tail_cfg.get('ema_momentum', 0.9))
        momentum = min(max(momentum, 0.0), 1.0)

        if (not hasattr(self, 'tail_class_freq_ema')) or \
           self.tail_class_freq_ema.numel() != batch_freq.numel():
            return batch_freq

        if int(self.tail_ema_updates.item()) <= 0:
            self.tail_class_freq_ema.copy_(batch_freq)
        else:
            self.tail_class_freq_ema.mul_(momentum)
            self.tail_class_freq_ema.add_(batch_freq, alpha=1.0 - momentum)
        self.tail_ema_updates.add_(1)
        return self.tail_class_freq_ema

    def _estimate_class_count(self, gt_labels_list, device):
        class_count = torch.zeros(self.num_classes, device=device, dtype=torch.float32)
        for labels in gt_labels_list:
            if labels.numel() == 0:
                continue
            bincount = torch.bincount(labels.long(), minlength=self.num_classes).to(torch.float32)
            class_count += bincount[:self.num_classes]

        if 0 <= self.empty_label < self.num_classes:
            class_count[self.empty_label] = 0
        return class_count

    def _select_tail_classes(self, class_freq, tail_cfg):
        class_freq = torch.as_tensor(class_freq, dtype=torch.float32)
        num_classes = int(getattr(self, 'num_classes', class_freq.numel()))
        if class_freq.numel() < num_classes:
            pad = class_freq.new_full((num_classes - class_freq.numel(),), float('inf'))
            class_freq = torch.cat([class_freq, pad], dim=0)
        elif class_freq.numel() > num_classes:
            class_freq = class_freq[:num_classes]

        tail_cfg = tail_cfg or {}
        freq_thr = float(tail_cfg.get('freq_thr', 0.02))
        min_tail = int(tail_cfg.get('min_tail_classes', tail_cfg.get('min_classes', 0)))
        max_tail = int(tail_cfg.get('max_tail_classes', tail_cfg.get('max_classes', num_classes)))

        empty_label = int(getattr(self, 'empty_label', -1))
        valid_mask = torch.ones(num_classes, dtype=torch.bool, device=class_freq.device)
        if 0 <= empty_label < num_classes:
            valid_mask[empty_label] = False

        safe_freq = torch.where(
            torch.isfinite(class_freq),
            class_freq,
            class_freq.new_full(class_freq.shape, float('inf')))

        tail_mask = valid_mask & (safe_freq <= freq_thr)
        num_valid = int(valid_mask.sum().item())
        max_tail = max(0, min(max_tail, num_valid))
        if max_tail > 0 and min_tail > max_tail:
            min_tail = max_tail

        current = int(tail_mask.sum().item())
        if current < min_tail:
            candidate = torch.nonzero(valid_mask & (~tail_mask), as_tuple=False).flatten()
            if candidate.numel() > 0:
                _, order = torch.sort(safe_freq[candidate], descending=False)
                add_num = min(min_tail - current, candidate.numel())
                tail_mask[candidate[order[:add_num]]] = True

        current = int(tail_mask.sum().item())
        if max_tail == 0:
            tail_mask.zero_()
        elif current > max_tail:
            selected = torch.nonzero(tail_mask, as_tuple=False).flatten()
            _, order = torch.sort(safe_freq[selected], descending=False)
            keep = selected[order[:max_tail]]
            new_mask = torch.zeros_like(tail_mask)
            new_mask[keep] = True
            tail_mask = new_mask

        if 0 <= empty_label < num_classes:
            tail_mask[empty_label] = False
        return tail_mask

    def _resolve_tail_class_mask(self, gt_labels_list):
        if gt_labels_list:
            device = gt_labels_list[0].device
        else:
            device = self.pc_range.device

        tail_cfg = (self.train_cfg or {}).get('tail_focus', {})
        if not tail_cfg.get('enabled', False):
            return self._get_fallback_tail_mask(device)

        class_count = self._estimate_class_count(gt_labels_list, device)
        if class_count.sum() <= 0:
            return self._get_fallback_tail_mask(device)

        policy = tail_cfg.get('policy', 'ema_freq')
        if policy == 'ema_freq':
            class_freq = self._update_tail_freq_ema(class_count, tail_cfg)
            if class_freq is None:
                return self._get_fallback_tail_mask(device)
        else:
            class_freq = class_count / class_count.sum().clamp(min=1.0)

        tail_mask = self._select_tail_classes(class_freq, tail_cfg)
        if tail_mask.sum() <= 0 and tail_cfg.get('fallback', 'rare_classes') == 'rare_classes':
            tail_mask = self._get_fallback_tail_mask(device)
        return tail_mask.to(device=device)

    def _balance_gt_indices(self, labels, tail_class_mask=None):
        labels = labels.long()
        if labels.numel() == 0:
            return labels.new_zeros((0,), dtype=torch.long)

        gt_balance_cfg = (self.train_cfg or {}).get('gt_balance', {})
        if not gt_balance_cfg.get('enabled', False):
            return torch.arange(labels.shape[0], device=labels.device, dtype=torch.long)

        per_class_cap = int(gt_balance_cfg.get('per_class_cap', labels.shape[0]))
        per_class_cap = max(per_class_cap, 1)
        tail_min_keep = int(gt_balance_cfg.get('tail_min_keep', per_class_cap))
        tail_min_keep = max(tail_min_keep, 1)
        sample_mode = gt_balance_cfg.get('sample_mode', 'deterministic')

        if tail_class_mask is None:
            tail_class_mask = torch.zeros(
                int(getattr(self, 'num_classes', labels.max().item() + 1)),
                dtype=torch.bool,
                device=labels.device)
        else:
            tail_class_mask = tail_class_mask.to(device=labels.device, dtype=torch.bool)

        keep_indices = []
        unique_labels = labels.unique(sorted=True)
        for cls_idx in unique_labels.tolist():
            cls_idx = int(cls_idx)
            if cls_idx == int(getattr(self, 'empty_label', -1)):
                continue

            cls_indices = torch.nonzero(labels == cls_idx, as_tuple=False).flatten()
            if cls_indices.numel() == 0:
                continue

            if sample_mode == 'deterministic':
                selected = cls_indices[:per_class_cap]
            else:
                perm = torch.randperm(cls_indices.numel(), device=labels.device)
                selected = cls_indices[perm[:per_class_cap]]

            is_tail = cls_idx < tail_class_mask.numel() and bool(tail_class_mask[cls_idx])
            if is_tail and selected.numel() < tail_min_keep:
                need = tail_min_keep - selected.numel()
                if sample_mode == 'deterministic':
                    repeat = cls_indices[torch.arange(need, device=labels.device) % cls_indices.numel()]
                else:
                    repeat = cls_indices[torch.randint(0, cls_indices.numel(), (need,), device=labels.device)]
                selected = torch.cat([selected, repeat], dim=0)

            keep_indices.append(selected)

        if not keep_indices:
            return torch.arange(labels.shape[0], device=labels.device, dtype=torch.long)
        return torch.cat(keep_indices, dim=0)

    def _apply_gt_balance(self, gt_points_list, gt_masks_list, gt_labels_list, tail_class_mask):
        gt_balance_cfg = (self.train_cfg or {}).get('gt_balance', {})
        if not gt_balance_cfg.get('enabled', False):
            return gt_points_list, gt_masks_list, gt_labels_list

        balanced_points, balanced_masks, balanced_labels = [], [], []
        for points, masks, labels in zip(gt_points_list, gt_masks_list, gt_labels_list):
            keep_indices = self._balance_gt_indices(labels, tail_class_mask=tail_class_mask)
            if keep_indices.numel() == 0:
                keep_indices = torch.arange(labels.shape[0], device=labels.device, dtype=torch.long)
            balanced_points.append(points[keep_indices])
            balanced_masks.append(masks[keep_indices])
            balanced_labels.append(labels[keep_indices])
        return balanced_points, balanced_masks, balanced_labels

    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_masks, gt_labels, tail_class_mask=None):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).reshape(-1).long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).reshape(-1).long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        pred_paired_pts = gt_points[pred_paired_idx]

        # cls assignment
        refine_pts_labels = gt_labels[pred_paired_idx]
        cls_weights = self.train_cfg.get('cls_weights', [1] * self.num_classes)
        cls_weights = refine_pts.new_tensor(cls_weights)
        if cls_weights.numel() < self.num_classes:
            pad = cls_weights.new_ones(self.num_classes - cls_weights.numel())
            cls_weights = torch.cat([cls_weights, pad], dim=0)
        elif cls_weights.numel() > self.num_classes:
            cls_weights = cls_weights[:self.num_classes]
        label_weights = cls_weights * self.get_dis_weight(pred_paired_pts)[..., None]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist_gt = torch.norm(gt_points - gt_paired_pts, dim=-1)
        gt_masks = gt_masks.bool()
        empty_mask = (dist_gt > empty_dist_thr) & gt_masks
        gt_pts_weights[empty_mask] = empty_weights

        tail_focus_cfg = (self.train_cfg or {}).get('tail_focus', {})
        if tail_focus_cfg.get('enabled', False) and tail_class_mask is not None:
            tail_class_mask = tail_class_mask.to(device=gt_labels.device, dtype=torch.bool)
            tail_weight = float(tail_focus_cfg.get(
                'tail_weight',
                self.train_cfg.get('rare_weights', 10)))
            tail_weight = max(tail_weight, 1.0)

            pred_tail_mask = tail_class_mask[refine_pts_labels]
            if pred_tail_mask.any():
                label_weights[pred_tail_mask] = label_weights[pred_tail_mask] * tail_weight

            gt_tail_mask = tail_class_mask[gt_labels] & gt_masks
            if gt_tail_mask.any():
                gt_pts_weights[gt_tail_mask] = gt_pts_weights[gt_tail_mask].clamp(min=tail_weight)
        else:
            # Legacy fallback: static rare class emphasis on gt->pred regression.
            rare_classes = self.train_cfg.get('rare_classes', [0, 2, 5, 8])
            rare_weights = self.train_cfg.get('rare_weights', 10)
            for cls_idx in rare_classes:
                rare_mask = (gt_labels == int(cls_idx)) & gt_masks
                gt_pts_weights[rare_mask] = gt_pts_weights[rare_mask].clamp(min=rare_weights)

        return (refine_pts_labels, gt_paired_idx, pred_paired_idx, label_weights,
                gt_pts_weights)

    def get_targets(self):
        # To instantiate the abstract method.
        pass

    def _apply_pred_hard_mining(self, pred_pts, pred_paired_pts):
        mining_cfg = (self.train_cfg or {}).get('hard_mining', {})
        if (not mining_cfg.get('enabled', False)) or pred_pts.numel() == 0:
            return pred_pts, pred_paired_pts

        pred_topk_ratio = float(mining_cfg.get('pred_topk_ratio', 1.0))
        pred_topk_ratio = min(max(pred_topk_ratio, 0.0), 1.0)
        min_keep = int(mining_cfg.get('min_keep', 0))

        dist_pred = torch.norm(pred_pts - pred_paired_pts, dim=-1)
        keep_k = int(round(dist_pred.shape[0] * pred_topk_ratio))
        keep_k = max(keep_k, min_keep)
        if keep_k <= 0 or keep_k >= dist_pred.shape[0]:
            return pred_pts, pred_paired_pts

        keep_idx = torch.topk(dist_pred, k=keep_k, largest=True).indices
        return pred_pts[keep_idx], pred_paired_pts[keep_idx]

    def loss_single(self,
                    score_preds,
                    refine_pts,
                    feat_scores,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list,
                    tail_class_mask=None):
        num_imgs = score_preds.size(0)  # B
        score_dim = self.num_classes if self.score_mode == 'semantic' else self.occ_out_channels
        score_preds = score_preds.reshape(num_imgs, -1, score_dim)
        if feat_scores is not None:
            feat_scores = feat_scores.reshape(num_imgs, -1, self.feature_dims)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        valid_sample_ids = [
            i for i in range(num_imgs)
            if gt_points_list[i] is not None and gt_points_list[i].numel() > 0
        ]
        if not valid_sample_ids:
            zero_score = score_preds.sum() * 0.0
            zero_pts = refine_pts.sum() * 0.0
            zero_feat = feat_scores.sum() * 0.0 if feat_scores is not None else None
            return zero_score, zero_pts, zero_feat

        score_preds_list = [score_preds[i] for i in valid_sample_ids]
        feat_scores_list = [feat_scores[i] for i in valid_sample_ids] if feat_scores is not None else None
        refine_pts_list = [refine_pts[i] for i in valid_sample_ids]
        gt_points_list = [gt_points_list[i] for i in valid_sample_ids]
        gt_masks_list = [gt_masks_list[i] for i in valid_sample_ids]
        gt_labels_list = [gt_labels_list[i] for i in valid_sample_ids]

        tail_mask_list = [tail_class_mask for _ in range(len(valid_sample_ids))]
        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
             self._get_target_single, refine_pts_list, gt_points_list,
             gt_masks_list, gt_labels_list, tail_mask_list)

        gt_paired_pts, pred_paired_pts = [], []
        for i in range(len(valid_sample_ids)):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        score_preds = torch.cat(score_preds_list)
        feat_scores = torch.cat(feat_scores_list) if feat_scores_list is not None else None
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        positive_mask = None
        if self.score_mode == 'binary_occ':
            occ_targets, occ_valid_mask, positive_mask = self._build_binary_occ_targets(
                pred_pts, pred_paired_pts)
            loss_score = self._compute_binary_occ_loss(
                score_preds, occ_targets, occ_valid_mask)
        else:
            loss_score = self.loss_cls(
                score_preds,
                labels,
                weight=cls_weights,
                avg_factor=score_preds.shape[0])

        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(
            gt_pts,
            gt_paired_pts,
            weight=gt_pts_weights[..., None],
            avg_factor=max(gt_pts.shape[0], 1))

        mined_pred_pts, mined_pred_paired_pts = self._apply_pred_hard_mining(pred_pts, pred_paired_pts)
        loss_pts += self.loss_pts(
            mined_pred_pts,
            mined_pred_paired_pts,
            avg_factor=max(mined_pred_pts.shape[0], 1))

        loss_feat = self._compute_feature_loss(
            feat_scores,
            labels,
            cls_weights,
            pred_pts=pred_pts,
            pred_paired_pts=pred_paired_pts,
            positive_mask=positive_mask)
        return loss_score, loss_pts, loss_feat

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        if self.score_mode == 'binary_occ':
            all_score_preds = preds_dicts['all_occ_scores']
        else:
            all_score_preds = preds_dicts['all_cls_scores']  # [L, B, Q, P, C]
        all_refine_pts = preds_dicts['all_refine_pts']
        all_feat_scores = preds_dicts.get('all_feat_scores', None)
        debug_is_finite('head.init_points', init_points)
        debug_is_finite('head.all_score_preds', all_score_preds)
        debug_is_finite('head.all_refine_pts', all_refine_pts)
        debug_is_finite('head.all_feat_scores', all_feat_scores)

        num_dec_layers = len(all_score_preds)
        gt_points_list, gt_masks_list, gt_labels_list = self.get_sparse_voxels(
            voxel_semantics, mask_camera)
        tail_class_mask = self._resolve_tail_class_mask(gt_labels_list)
        gt_points_list, gt_masks_list, gt_labels_list = self._apply_gt_balance(
            gt_points_list,
            gt_masks_list,
            gt_labels_list,
            tail_class_mask=tail_class_mask)

        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_tail_masks = [tail_class_mask for _ in range(num_dec_layers)]

        if all_feat_scores is None:
            all_feat_scores = [None for _ in range(num_dec_layers)]
        losses_score, losses_pts, losses_feat = multi_apply(
            self.loss_single,
            all_score_preds,
            all_refine_pts,
            all_feat_scores,
            all_gt_points_list,
            all_gt_masks_list,
            all_gt_labels_list,
            all_tail_masks)

        loss_dict = dict()
        # loss of init_points
        if init_points is not None and not self.init_pos_lidar:
            score_dim = self.num_classes if self.score_mode == 'semantic' else self.occ_out_channels
            pseudo_scores = init_points.new_zeros(*init_points.shape[:-1], score_dim)
            _, init_loss_pts, _ = self.loss_single(
                pseudo_scores,
                init_points,
                None,
                gt_points_list,
                gt_masks_list,
                gt_labels_list,
                tail_class_mask=tail_class_mask)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer
        score_loss_name = 'loss_occ' if self.score_mode == 'binary_occ' else 'loss_cls'
        loss_dict[score_loss_name] = losses_score[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        if self.feature_supervision_enabled and losses_feat[-1] is not None:
            loss_dict['loss_feat'] = losses_feat[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_score_i, loss_pts_i, loss_feat_i in zip(
                losses_score[:-1], losses_pts[:-1], losses_feat[:-1]):
            loss_dict[f'd{num_dec_layer}.{score_loss_name}'] = loss_score_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            if self.feature_supervision_enabled and loss_feat_i is not None:
                loss_dict[f'd{num_dec_layer}.loss_feat'] = loss_feat_i
            num_dec_layer += 1
        return loss_dict

    def get_occ(self, pred_dicts, img_metas, rescale=False):
        if self.score_mode == 'binary_occ':
            all_score_preds = pred_dicts['all_occ_scores']
            score_channels = self.occ_out_channels
        else:
            all_score_preds = pred_dicts['all_cls_scores']
            score_channels = self.num_classes
        all_refine_pts = pred_dicts['all_refine_pts']
        all_feat_scores = pred_dicts.get('all_feat_scores', None)
        score_preds = all_score_preds[-1].sigmoid()
        refine_pts = all_refine_pts[-1]
        feat_scores = all_feat_scores[-1] if all_feat_scores is not None else None

        batch_size = refine_pts.shape[0]
        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr', 3.)
        score_thr = self.test_cfg.get('score_thr', 0.)

        result_list = []
        for i in range(batch_size):
            refine_pts_i, score_preds_i = refine_pts[i], score_preds[i]
            feat_scores_i = feat_scores[i] if feat_scores is not None else None
            refine_pts_i = decode_points(refine_pts_i, self.pc_range)

            # filter weak points by distance and score
            centers = refine_pts_i.mean(dim=1, keepdim=True)
            ctr_dists = torch.norm(refine_pts_i - centers, dim=-1)
            mask_dist = ctr_dists < ctr_dist_thr
            mask_score = (score_preds_i > score_thr).any(dim=-1)
            mask = mask_dist & mask_score
            refine_pts_i = refine_pts_i[mask]
            score_preds_i = score_preds_i[mask]
            if feat_scores_i is not None:
                feat_scores_i = feat_scores_i[mask]

            if refine_pts_i.numel() == 0:
                result_list.append(dict(
                    sem_pred=np.zeros((0,), dtype=np.int64),
                    occ_loc=np.zeros((0, 3), dtype=np.int64)))
                continue

            if feat_scores_i is not None:
                pts = torch.cat([refine_pts_i, score_preds_i, feat_scores_i], dim=-1)
            else:
                pts = torch.cat([refine_pts_i, score_preds_i], dim=-1)
            pts_infos, voxels, num_pts = self.voxel_generator(pts)
            voxels = torch.flip(voxels, [1]).long()
            score_start = 3
            score_end = score_start + score_channels
            scores = pts_infos[..., score_start:score_end]
            scores = scores.sum(dim=1) / num_pts[..., None]
            voxel_feats = None
            if feat_scores_i is not None:
                voxel_feats = pts_infos[..., score_end:]
                voxel_feats = voxel_feats.sum(dim=1) / num_pts[..., None]

            if self.test_cfg.get('padding', True):
                if voxel_feats is not None and self.prototype_decode_enabled:
                    raise NotImplementedError(
                        'feature-supervised prototype decoding requires test_cfg.padding=False')
                occ = scores.new_zeros((self.voxel_num[0], self.voxel_num[1],
                                        self.voxel_num[2], score_channels))
                occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
                occ = occ.permute(3, 0, 1, 2).unsqueeze(0)
                # padding
                dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
                eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)
                # repalce with original occ prediction
                original_mask = (occ > score_thr).any(dim=1, keepdim=True)
                original_mask = original_mask.expand_as(eroded_occ)
                eroded_occ[original_mask] = occ[original_mask]
                # sparse dense occ
                eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
                voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
                scores = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :]

            if self.score_mode == 'binary_occ':
                voxel_keep_mask = scores.squeeze(-1) > score_thr
                voxels = voxels[voxel_keep_mask]
                scores = scores[voxel_keep_mask]
                if voxel_feats is not None:
                    voxel_feats = voxel_feats[voxel_keep_mask]

            if voxels.numel() == 0:
                result_list.append(dict(
                    sem_pred=np.zeros((0,), dtype=np.int64),
                    occ_loc=np.zeros((0, 3), dtype=np.int64)))
                continue

            if voxel_feats is not None and self.prototype_decode_enabled:
                if self.prototype_metric != 'cosine':
                    raise ValueError(
                        f'Unsupported prototype similarity metric: {self.prototype_metric}')
                voxel_feats = F.normalize(voxel_feats, dim=-1)
                similarity = voxel_feats @ self.prototype_bank.t()
                labels = similarity.argmax(dim=-1)
            else:
                labels = scores.argmax(dim=-1)
            result_list.append(dict(
                sem_pred=labels.detach().cpu().numpy(),
                occ_loc=voxels.detach().cpu().numpy()))

        return result_list

    def get_sparse_voxels(self, voxel_semantics, mask_camera):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()
        hard_camera_mask = self.train_cfg.get('hard_camera_mask', False)

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)  # actual space

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            non_empty_mask = voxel_semantics[i] != self.empty_label
            if hard_camera_mask:
                visible_mask = mask_camera[i].bool()
                final_mask = non_empty_mask & visible_mask
            else:
                final_mask = non_empty_mask

            gt_points.append(coors[final_mask])
            gt_labels.append(voxel_semantics[i][final_mask])
            if hard_camera_mask:
                gt_masks.append(torch.ones_like(gt_labels[-1], dtype=torch.bool))
            else:
                # Legacy behavior: use camera visibility to reweight non-empty GT.
                gt_masks.append(mask_camera[i][non_empty_mask])

        return gt_points, gt_masks, gt_labels
