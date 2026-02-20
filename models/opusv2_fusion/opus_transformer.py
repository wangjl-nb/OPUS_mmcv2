import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from ..bbox.utils import decode_points, encode_points
from ..checkpoint import checkpoint as cp
from ..csrc.wrapper import MSMV_CUDA
from ..utils import DUMP
from .opus_sampling import sampling_4d, sampling_pts_feats


@MODELS.register_module()
class OPUSV2FusionTransformer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_groups=4,
                 num_refines=(1, 2, 4, 8, 16, 32),
                 num_pt_channels=32,
                 scales=(1.0,),
                 query_allocator=None,
                 pc_range=(),
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed.'
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines
        self.num_pt_channels = num_pt_channels
        self.num_layers = num_layers

        self.decoder = OPUSTransformerDecoder(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_views=num_views,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_groups=num_groups,
            num_refines=num_refines,
            num_pt_channels=num_pt_channels,
            scales=scales,
            query_allocator=query_allocator,
            pc_range=pc_range,
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, pts_feats, img_metas):
        pt_feats, refine_pts = self.decoder(
            query_points, query_feat, mlvl_feats, pts_feats, img_metas)

        pt_feats = [None if f is None else torch.nan_to_num(f) for f in pt_feats]
        refine_pts = [None if p is None else torch.nan_to_num(p) for p in refine_pts]

        return pt_feats, refine_pts


class OPUSTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_groups=4,
                 num_refines=16,
                 num_pt_channels=32,
                 scales=(1.0,),
                 query_allocator=None,
                 pc_range=(),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_groups = num_groups
        self.query_allocator = query_allocator or {}

        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        before_refines = [1] + num_refines

        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                OPUSTransformerDecoderLayer(
                    embed_dims=embed_dims,
                    num_frames=num_frames,
                    num_views=num_views,
                    num_points=num_points,
                    num_levels=num_levels,
                    num_groups=num_groups,
                    num_pt_channels=num_pt_channels,
                    num_refines=num_refines[i],
                    last_refines=before_refines[i],
                    last_layer=i == num_layers - 1,
                    scale=scales[i],
                    pc_range=pc_range,
                ))

    @torch.no_grad()
    def init_weights(self):
        for layer in self.decoder_layers:
            if hasattr(layer, 'init_weights'):
                layer.init_weights()

    def _apply_query_allocator(self, query_points, query_feat, pt_feat, layer_idx):
        allocator_cfg = self.query_allocator or {}
        if not allocator_cfg.get('enabled', False):
            return query_points, query_feat
        if pt_feat is None:
            return query_points, query_feat

        switch_layer = int(allocator_cfg.get('switch_layer', 2))
        if layer_idx != switch_layer - 1:
            return query_points, query_feat

        B, Q, P, Cxyz = query_points.shape
        if Q <= 1:
            return query_points, query_feat

        feat_prob = pt_feat.sigmoid()
        nonempty_score = feat_prob.max(dim=-1).values.mean(dim=-1)

        num_channels = max(feat_prob.shape[-1], 1)
        entropy = -(feat_prob.clamp(min=1e-6, max=1 - 1e-6) *
                    torch.log(feat_prob.clamp(min=1e-6, max=1 - 1e-6))).sum(dim=-1)
        entropy = entropy / max(np.log(float(num_channels)), 1.0)
        uncertainty_score = entropy.mean(dim=-1)

        score_weights = allocator_cfg.get('score_weights', {})
        nonempty_w = float(score_weights.get('nonempty', 0.5))
        uncertainty_w = float(score_weights.get('uncertainty', 0.5))

        context_ratio = float(allocator_cfg.get('context_ratio', 0.6))
        context_ratio = min(max(context_ratio, 0.0), 1.0)
        num_context = int(round(Q * context_ratio))
        num_context = max(0, min(Q, num_context))
        num_detail = Q - num_context

        detail_score = nonempty_w * (1.0 - nonempty_score) + uncertainty_w * uncertainty_score
        context_rank = torch.argsort(nonempty_score, dim=-1, descending=True)
        detail_rank = torch.argsort(detail_score, dim=-1, descending=True)

        if num_context > 0:
            context_idx = context_rank[:, :num_context]
        else:
            context_idx = context_rank[:, :0]

        detail_indices = []
        for batch_idx in range(B):
            detail_idx = detail_rank[batch_idx]
            if num_context > 0:
                used = torch.zeros(Q, dtype=torch.bool, device=detail_idx.device)
                used[context_idx[batch_idx]] = True
                detail_idx = detail_idx[~used[detail_idx]]
            if detail_idx.numel() < num_detail:
                short = num_detail - detail_idx.numel()
                detail_idx = torch.cat([detail_idx, detail_rank[batch_idx, :short]], dim=0)
            detail_indices.append(detail_idx[:num_detail])

        if num_detail > 0:
            detail_idx = torch.stack(detail_indices, dim=0)
            gather_idx = torch.cat([context_idx, detail_idx], dim=1)
        else:
            gather_idx = context_idx

        gather_points_idx = gather_idx[..., None, None].expand(B, Q, P, Cxyz)
        gather_feat_idx = gather_idx[..., None].expand(B, Q, query_feat.shape[-1])
        new_query_points = torch.gather(query_points, dim=1, index=gather_points_idx)
        new_query_feat = torch.gather(query_feat, dim=1, index=gather_feat_idx)

        detail_jitter_std = float(allocator_cfg.get('detail_jitter_std', 0.0))
        if detail_jitter_std > 0 and num_detail > 0:
            detail_slice = slice(num_context, Q)
            detail_points = decode_points(new_query_points[:, detail_slice], self.pc_range)
            detail_points = detail_points + torch.randn_like(detail_points) * detail_jitter_std

            pc_range = detail_points.new_tensor(self.pc_range)
            lower = pc_range[:3]
            upper = pc_range[3:]
            detail_points = detail_points.clamp(min=lower, max=upper)

            new_query_points = new_query_points.clone()
            new_query_points[:, detail_slice] = encode_points(detail_points, self.pc_range)

        return new_query_points, new_query_feat

    def forward(self, query_points, query_feat, mlvl_feats, pts_feats, img_metas):
        pt_feats, refine_pts = [], []

        ego2img = np.asarray([m['ego2img'] for m in img_metas]).astype(np.float32)
        ego2img = query_feat.new_tensor(ego2img)
        ego2occ = np.asarray([m['ego2occ'] for m in img_metas]).astype(np.float32)
        occ2ego = torch.inverse(query_feat.new_tensor(ego2occ))
        occ2ego = occ2ego[:, None].expand_as(ego2img)
        occ2img = ego2img @ occ2ego

        ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
        ego2lidar = query_feat.new_tensor(ego2lidar)
        occ2lidar = ego2lidar @ occ2ego[:, 0]

        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape
            N, T, G, C = self.num_views, self.num_frames, self.num_groups, GC // self.num_groups
            assert T * N == TN
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)
                feat = feat.reshape(B * T * G, N, H, W, C)
            else:
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)
                feat = feat.reshape(B * T * G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        B, GC, H, W = pts_feats.shape
        G, C = self.num_groups, GC // self.num_groups
        pts_feats = pts_feats.reshape(B, G, C, H, W).reshape(B * G, C, H, W)

        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i

            query_points = query_points.detach()
            query_feat, pt_feat, query_points = decoder_layer(
                query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar, img_metas)

            pt_feats.append(pt_feat)
            refine_pts.append(query_points)
            query_points, query_feat = self._apply_query_allocator(
                query_points=query_points,
                query_feat=query_feat,
                pt_feat=pt_feat,
                layer_idx=i,
            )

        return pt_feats, refine_pts


class OPUSTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_groups=4,
                 num_pt_channels=32,
                 num_refines=16,
                 last_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 last_layer=False,
                 scale=1.0,
                 pc_range=(),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_points = num_points
        self.num_refines = num_refines
        self.last_refines = last_refines
        self.last_layer = last_layer
        self.scale = scale

        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.last_refines, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = OPUSSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = OPUSSampling(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_views=num_views,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range,
        )
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * (num_frames + 1),
            n_groups=num_groups,
            out_points=32,
        )
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, num_pt_channels * self.num_refines))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 3 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def forward(self, query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar, img_metas):
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        query_feat = query_feat + query_pos

        sampled_img_feat, sampled_pts_feat = self.sampling(
            query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar, img_metas)
        sampled_feat = torch.cat([sampled_img_feat, sampled_pts_feat], dim=-2)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        B, Q = query_points.shape[:2]
        reg_offset = self.scale * self.reg_branch(query_feat)
        refine_pt = self.refine_points(query_points, reg_offset)

        pt_feat = None
        if self.training or self.last_layer:
            pt_feat = self.cls_branch(query_feat)
            pt_feat = pt_feat.reshape(B, Q, self.num_refines, -1)

        return query_feat, pt_feat, refine_pt


class OPUSSelfAttention(BaseModule):
    """Scale-adaptive self attention."""

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=(),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_points, query_feat):
        dist = self.calc_points_dists(query_points)
        tau = self.gen_tau(query_feat)

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)
        attn_mask = dist[:, None, :, :] * tau[..., None]
        attn_mask = attn_mask.flatten(0, 1)
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_points, query_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat, use_reentrant=False)
        return self.inner_forward(query_points, query_feat)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist


class OPUSSampling(BaseModule):
    """Adaptive spatio-temporal sampling for image + LiDAR features."""

    def __init__(self,
                 embed_dims=256,
                 num_frames=4,
                 num_views=6,
                 num_groups=4,
                 num_points=8,
                 num_levels=4,
                 pc_range=(),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_views = num_views
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_prototype = nn.Embedding(num_groups * num_points, 3)
        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.normal_(self.sampling_prototype.weight, mean=0, std=1)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar, img_metas):
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.ones_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, -1, 3)

        prototype = self.sampling_prototype.weight[None, None, ...].repeat(B, Q, 1, 1)
        sampling_points = query_center + prototype * query_scale + sampling_offset
        sampling_points = sampling_points.view(B, Q, self.num_groups, self.num_points, 3)

        img_sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        img_sampling_points = img_sampling_points.expand(
            B, Q, self.num_frames, self.num_groups, self.num_points, 3)
        pts_sampling_points = sampling_points.clone()

        scale_weights = self.scale_weights(query_feat).view(
            B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(
            B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        sampled_img_feats = sampling_4d(
            img_sampling_points,
            mlvl_feats,
            scale_weights,
            occ2img,
            image_h,
            image_w,
            self.num_views,
        )
        sampled_pts_feats = sampling_pts_feats(
            pts_sampling_points,
            pts_feats,
            occ2lidar,
            self.pc_range,
        )

        return sampled_img_feats, sampled_pts_feats

    def forward(self, query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(
                self.inner_forward,
                query_points,
                query_feat,
                mlvl_feats,
                pts_feats,
                occ2img,
                occ2lidar,
                img_metas,
                use_reentrant=False,
            )
        return self.inner_forward(query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar, img_metas)


class AdaptiveMixing(nn.Module):
    """Adaptive mixing."""

    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        params = self.parameter_generator(query)
        params = params.reshape(B * Q, G, -1)
        out = x.reshape(B * Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B * Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B * Q, G, self.out_points, self.in_points)

        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        out = torch.matmul(S, out)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out
        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        return self.inner_forward(x, query)
