import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.ops import furthest_point_sample, gather_points, knn
from mmengine.model import BaseModule, bias_init_with_prob
from mmdet3d.registry import MODELS
from spconv.pytorch import SparseConv3d, SparseConvTensor, SparseSequential, SubMConv3d

from ..bbox.utils import decode_points, encode_points
from ..compat import build_loss, build_transformer, force_fp32, multi_apply


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        return torch.cat([x, x_max[unq_inv, :]], dim=1)


class Densifier(BaseModule):
    def __init__(self, in_channels, pfn_channels, num_classes, pc_range, voxel_num):
        super().__init__()
        self.pfn_layers = nn.ModuleList()
        for i, pfn_channel in enumerate(pfn_channels):
            last_layer = (i == len(pfn_channels) - 1)
            self.pfn_layers.append(
                PFNLayer(
                    in_channels=in_channels + 6 if i == 0 else pfn_channel,
                    out_channels=pfn_channel,
                    use_norm=True,
                    last_layer=last_layer,
                ))
        self.cls_branch = SparseSequential(
            SparseConv3d(
                pfn_channels[-1], pfn_channels[-1],
                kernel_size=3, stride=1, padding=1, indice_key='densifier1'),
            nn.ReLU(inplace=True),
            SubMConv3d(
                pfn_channels[-1], num_classes,
                kernel_size=3, stride=1, padding=1, indice_key='densifier2'))
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('voxel_num', voxel_num)
        self.scale_xyz = self.voxel_num[0] * self.voxel_num[1] * self.voxel_num[2]
        self.scale_yz = self.voxel_num[1] * self.voxel_num[2]
        self.scale_z = self.voxel_num[2]

    def init_weights(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def discretize(self, pts, clip=True):
        encoded_pts = encode_points(pts, self.pc_range)
        voxel_num = self.voxel_num.to(encoded_pts.dtype)
        loc = torch.floor(encoded_pts * voxel_num)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)

        centers = decode_points((loc + 0.5) / voxel_num, self.pc_range)
        return loc.long(), centers

    def forward(self, pts_feats, refine_pts):
        if pts_feats is None:
            return None, None

        B, Q, P, _ = refine_pts.shape
        refine_pts = decode_points(refine_pts, self.pc_range)

        batch_index = torch.arange(B, device=pts_feats.device).view(B, 1, 1, 1).expand(B, Q, P, 1)
        coors, coor_centers = self.discretize(refine_pts.detach())
        coors = torch.cat([batch_index, coors], dim=-1)
        coors = coors[..., 0] * self.scale_xyz + coors[..., 1] * self.scale_yz + \
            coors[..., 2] * self.scale_z + coors[..., 3]

        centers = refine_pts.mean(dim=2, keepdim=True)
        pts_feats = torch.cat([refine_pts - coor_centers, refine_pts - centers, pts_feats], dim=-1)
        pts_feats = pts_feats.reshape(B * Q * P, -1)
        coors = coors.reshape(B * Q * P)

        unq_coors, unq_inv = coors.unique(return_inverse=True)
        for pfn_layer in self.pfn_layers:
            pts_feats = pfn_layer(pts_feats, unq_inv)
        voxel_coors = torch.stack(
            [unq_coors // self.scale_xyz,
             (unq_coors % self.scale_xyz) // self.scale_yz,
             (unq_coors % self.scale_yz) // self.scale_z,
             (unq_coors % self.scale_z)], dim=1)

        sparse_feats = SparseConvTensor(
            features=pts_feats,
            indices=voxel_coors.int(),
            spatial_shape=self.voxel_num.tolist(),
            batch_size=B,
        )
        res = self.cls_branch(sparse_feats)
        densified_cls_scores = res.features
        densified_voxel_coors = res.indices.long()

        return densified_cls_scores, densified_voxel_coors


@MODELS.register_module()
class OPUSV2FusionHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=(),
                 empty_label=17,
                 pfn_channels=(64, 64),
                 voxel_size=(),
                 init_pos_lidar=None,
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
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
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims

        assert init_pos_lidar in [None, 'all', 'curr'], \
            'init_pos_lidar should be one of [None, "all", "curr"]'
        self.init_pos_lidar = init_pos_lidar

        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)
        self.register_buffer(
            'tail_class_freq_ema',
            torch.zeros(self.num_classes, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            'tail_ema_updates',
            torch.zeros((), dtype=torch.long),
            persistent=False)

        self.densifiers = nn.ModuleList()
        for _ in range(self.transformer.num_layers):
            self.densifiers.append(
                Densifier(
                    in_channels=self.transformer.num_pt_channels,
                    pfn_channels=pfn_channels,
                    num_classes=num_classes,
                    pc_range=self.pc_range,
                    voxel_num=self.voxel_num,
                ))

        self._init_layers()

    def _init_layers(self):
        if not self.init_pos_lidar:
            self.init_points = nn.Embedding(self.num_query, 3)
            nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()
        for densifier in self.densifiers:
            densifier.init_weights()

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
            repeat_idx = torch.arange(num_lidar - sampled.shape[0], device=pts_xyz.device) % max(sampled.shape[0], 1)
            if sampled.shape[0] > 0:
                sampled = torch.cat([sampled, sampled[repeat_idx]], dim=0)
            else:
                sampled = self._uniform_sample_pc_range(num_lidar, pts_xyz.device, pts_xyz.dtype)
        return sampled[:num_lidar]

    def _sample_random_queries(self, pts_xyz, num_random, random_mode='uniform_pc_range'):
        if num_random <= 0:
            return pts_xyz.new_zeros((0, 3))

        sampled = []
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
        default_cfg = dict(
            enabled=True,
            lidar_ratio=0.7,
            random_ratio=0.3,
        )
        mix_cfg = dict(default_cfg)
        mix_cfg.update((self.train_cfg or {}).get('query_init_mix', {}))
        num_query = int(getattr(self, 'num_query', 0))
        if num_query <= 0:
            return 0, 0
        if not mix_cfg.get('enabled', True):
            return num_query, 0

        lidar_ratio = float(mix_cfg.get('lidar_ratio', 0.7))
        random_ratio = float(mix_cfg.get('random_ratio', 0.3))
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
                points=None,
                img_metas=None):
        init_points, query_feat = self.get_init_position(points, mlvl_feats, pts_feats, img_metas)

        pt_feats, refine_pts = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            pts_feats,
            img_metas=img_metas,
        )

        cls_scores, voxel_coors = [], []
        for i, densifier in enumerate(self.densifiers):
            cls_score, voxel_coor = densifier(pt_feats[i], refine_pts[i])
            cls_scores.append(cls_score)
            voxel_coors.append(voxel_coor)

        return dict(
            init_points=init_points,
            all_refine_pts=refine_pts,
            all_cls_scores=cls_scores,
            all_voxel_coors=voxel_coors)

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
    def _get_regression_target_single(self, refine_pts, gt_points, gt_masks, gt_labels, tail_class_mask=None):
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_pts = refine_pts[gt_paired_idx]

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
            tail_weight = float(tail_focus_cfg.get('tail_weight', self.train_cfg.get('rare_weights', 10)))
            tail_weight = max(tail_weight, 1.0)
            gt_tail_mask = tail_class_mask[gt_labels] & gt_masks
            if gt_tail_mask.any():
                gt_pts_weights[gt_tail_mask] = gt_pts_weights[gt_tail_mask].clamp(min=tail_weight)
        else:
            rare_classes = self.train_cfg.get('rare_classes', [0, 2, 5, 8])
            rare_weights = self.train_cfg.get('rare_weights', 10)
            for cls_idx in rare_classes:
                rare_mask = (gt_labels == int(cls_idx)) & gt_masks
                gt_pts_weights[rare_mask] = gt_pts_weights[rare_mask].clamp(min=rare_weights)

        return gt_paired_idx, pred_paired_idx, gt_pts_weights

    def get_targets(self):
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
                    refine_pts,
                    cls_scores,
                    voxel_coors,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list,
                    gt_dense_occ,
                    tail_class_mask=None):
        num_imgs = refine_pts.size(0)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        tail_mask_list = [tail_class_mask for _ in range(num_imgs)]
        gt_paired_idx_list, pred_paired_idx_list, gt_pts_weights = multi_apply(
            self._get_regression_target_single,
            refine_pts_list,
            gt_points_list,
            gt_masks_list,
            gt_labels_list,
            tail_mask_list)

        gt_paired_pts, pred_paired_pts = [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

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

        loss_cls = pred_pts.new_tensor(0)
        if (cls_scores is not None) and (voxel_coors is not None):
            b, x, y, z = voxel_coors.unbind(dim=-1)
            gt_labels = gt_dense_occ[b, x, y, z]

            cls_weights = self.train_cfg.get('cls_weights', [1] * self.num_classes)
            cls_weights = cls_scores.new_tensor(cls_weights)
            if cls_weights.numel() < self.num_classes:
                pad = cls_weights.new_ones(self.num_classes - cls_weights.numel())
                cls_weights = torch.cat([cls_weights, pad], dim=0)
            elif cls_weights.numel() > self.num_classes:
                cls_weights = cls_weights[:self.num_classes]
            cls_weights = cls_weights[None, :].expand(cls_scores.shape[0], -1).clone()

            tail_focus_cfg = (self.train_cfg or {}).get('tail_focus', {})
            if tail_focus_cfg.get('enabled', False) and tail_class_mask is not None:
                tail_class_mask = tail_class_mask.to(device=gt_labels.device, dtype=torch.bool)
                tail_weight = float(tail_focus_cfg.get('tail_weight', self.train_cfg.get('rare_weights', 10)))
                tail_weight = max(tail_weight, 1.0)
                sample_tail = tail_class_mask[gt_labels]
                if sample_tail.any():
                    cls_weights[sample_tail] = cls_weights[sample_tail] * tail_weight

            avg_factor = (gt_labels != self.empty_label).sum().clamp(min=1)
            loss_cls += self.loss_cls(
                cls_scores,
                gt_labels,
                weight=cls_weights,
                avg_factor=avg_factor)

        return loss_cls, loss_pts

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        init_points = preds_dicts['init_points']
        all_refine_pts = preds_dicts['all_refine_pts']
        all_cls_scores = preds_dicts['all_cls_scores']
        all_voxel_coors = preds_dicts['all_voxel_coors']
        voxel_semantics = voxel_semantics.long()
        mask_camera = mask_camera.bool()

        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_masks_list, gt_labels_list = self.get_sparse_voxels(voxel_semantics, mask_camera)
        tail_class_mask = self._resolve_tail_class_mask(gt_labels_list)
        gt_points_list, gt_masks_list, gt_labels_list = self._apply_gt_balance(
            gt_points_list, gt_masks_list, gt_labels_list, tail_class_mask=tail_class_mask)

        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_dense_occ = [voxel_semantics for _ in range(num_dec_layers)]
        all_tail_masks = [tail_class_mask for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single,
            all_refine_pts,
            all_cls_scores,
            all_voxel_coors,
            all_gt_points_list,
            all_gt_masks_list,
            all_gt_labels_list,
            all_gt_dense_occ,
            all_tail_masks)

        loss_dict = dict()
        if init_points is not None and not self.init_pos_lidar:
            _, init_loss_pts = self.loss_single(
                init_points,
                None,
                None,
                gt_points_list,
                gt_masks_list,
                gt_labels_list,
                voxel_semantics,
                tail_class_mask=tail_class_mask)
            loss_dict['init_loss_pts'] = init_loss_pts

        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict

    def get_occ(self, pred_dicts, img_metas, rescale=False):
        voxels = pred_dicts['all_voxel_coors'][-1]
        cls_scores = pred_dicts['all_cls_scores'][-1].sigmoid()
        batch_size = voxels[-1, 0].item() + 1
        scores, labels = cls_scores.max(dim=-1)

        mask = scores > self.test_cfg.get('score_thr', 0.0)
        labels = labels[mask]
        voxels = voxels[mask]

        result_list = []
        for i in range(batch_size):
            _labels = labels[voxels[:, 0] == i]
            _voxels = voxels[voxels[:, 0] == i, 1:]
            result_list.append(dict(
                sem_pred=_labels.detach().cpu().numpy(),
                occ_loc=_voxels.detach().cpu().numpy()))

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
        coors = torch.stack([xx, yy, zz], dim=-1)

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            non_empty_mask = voxel_semantics[i] != self.empty_label
            if hard_camera_mask:
                visible_mask = mask_camera[i].bool()
                final_mask = non_empty_mask & visible_mask
                gt_points.append(coors[final_mask])
                gt_labels.append(voxel_semantics[i][final_mask])
                gt_masks.append(torch.ones_like(gt_labels[-1], dtype=torch.bool))
            else:
                gt_points.append(coors[non_empty_mask])
                gt_labels.append(voxel_semantics[i][non_empty_mask])
                gt_masks.append(mask_camera[i][non_empty_mask].bool())

        return gt_points, gt_masks, gt_labels
