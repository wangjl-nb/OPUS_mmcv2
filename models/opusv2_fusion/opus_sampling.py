import torch
import torch.nn.functional as F

from ..bbox.utils import decode_points, encode_points
from ..utils import DUMP
from ..csrc.wrapper import msmv_sampling


def make_sample_points(query_points, offset, pc_range):
    """
    query_points: [B, Q, P, 3] (x, y, z)
    offset: [B, Q, G, P, 3]
    """
    xyz = decode_points(query_points, pc_range)
    xyz = xyz[..., None, None, :]
    sample_xyz = xyz + offset
    return sample_xyz


def sampling_4d(sample_points,
                mlvl_feats,
                scale_weights,
                occ2img,
                image_h,
                image_w,
                num_views=6,
                eps=1e-5):
    """Multi-frame multi-view image feature sampling.

    Args:
        sample_points: [B, Q, T, G, P, 3]
        mlvl_feats: list of [B*T*G, C, N, H, W] or channel-last CUDA format
        scale_weights: [B, Q, G, T, P, L]
        occ2img: [B, T*N, 4, 4]
    """
    B, Q, T, G, P, _ = sample_points.shape
    N = num_views

    sample_points = sample_points.reshape(B, Q, T, G * P, 3)

    occ2img = occ2img[:, :, None, None, :, :]
    occ2img = occ2img.expand(B, T * N, Q, G * P, 4, 4)
    occ2img = occ2img.reshape(B, T, N, Q, G * P, 4, 4)

    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)
    sample_points = sample_points[:, :, None, ..., None]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 3)

    with torch.cuda.amp.autocast(enabled=False):
        sample_points_cam = torch.matmul(occ2img.float(), sample_points.float()).squeeze(-1)
        homo = sample_points_cam[..., 2:3]
        homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
        sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero
        sample_points_cam = torch.nan_to_num(sample_points_cam, nan=0.0, posinf=2.0, neginf=-2.0)

    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h
    sample_points_cam = sample_points_cam.clamp(-2.0, 2.0)

    valid_mask = ((homo > eps)
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()

    if DUMP.enabled:
        torch.save(torch.cat([sample_points_cam, homo_nonzero], dim=-1).cpu(),
                   '{}/sample_points_cam_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
        torch.save(valid_mask.cpu(),
                   '{}/sample_points_cam_valid_mask_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)

    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)

    i_view = torch.argmax(valid_mask, dim=-1)[..., None]

    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]

    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (N - 1)], dim=-1)

    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)
    sample_points_cam = sample_points_cam.reshape(B * T * G, Q, P, 3)
    sample_points_cam = sample_points_cam.contiguous().to(mlvl_feats[0].dtype)

    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B * G * T, Q, P, -1)
    scale_weights = scale_weights.contiguous()

    final = msmv_sampling(mlvl_feats, sample_points_cam, scale_weights)

    C = final.shape[2]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4)
    final = final.flatten(3, 4)

    return final


def sampling_pts_feats(sample_points, pts_feats, occ2lidar, pc_range):
    """Sample LiDAR BEV features at query-centered 3D points.

    Args:
        sample_points: [B, Q, G, P, 3]
        pts_feats: [B*G, C, H, W]
        occ2lidar: [B, 4, 4]
    Returns:
        sampled features in shape [B, Q, G, P, C]
    """
    C = pts_feats.shape[1]
    B, Q, G, P, _ = sample_points.shape
    sample_points = sample_points.permute(0, 2, 1, 3, 4)
    sample_points = sample_points.reshape(B * G, Q, P, 3)

    occ2lidar = occ2lidar[:, None].expand(B, G, 4, 4).reshape(B * G, 4, 4)
    occ2lidar = occ2lidar[:, None, None].expand(B * G, Q, P, 4, 4)

    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)[..., None]
    with torch.cuda.amp.autocast(enabled=False):
        sample_points = torch.matmul(occ2lidar.float(), sample_points.float()).squeeze(-1)

    norm_sample_points = encode_points(sample_points[..., :3], pc_range)
    norm_sample_points = norm_sample_points[..., :2] * 2 - 1
    norm_sample_points = torch.nan_to_num(norm_sample_points, nan=0.0, posinf=2.0, neginf=-2.0)
    norm_sample_points = norm_sample_points.clamp(-2.0, 2.0)
    norm_sample_points = norm_sample_points.to(pts_feats.dtype)

    feat = F.grid_sample(pts_feats, norm_sample_points, padding_mode='zeros', align_corners=True)
    feat = feat.permute(0, 2, 3, 1)
    feat = feat.reshape(B, G, Q, P, C)
    feat = feat.permute(0, 2, 1, 3, 4)

    return feat
