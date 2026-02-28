import copy
import warnings

import torch


class OPUSToMapAnythingInputAdapter:
    """Adapt OPUS batch tensors to MapAnything-style list[view-dict] inputs."""

    _DEFAULT_META_KEYS = (
        'filename',
        'img_shape',
        'ori_shape',
        'pad_shape',
        'img_timestamp',
        'ego2img',
        'ego2occ',
        'ego2lidar',
        'sample_token',
        'scene_name',
        'sample_idx',
    )

    def __init__(self,
                 lidar_injection='shared',
                 tn_align_mode='strict',
                 preserve_meta_keys=None):
        valid_lidar_modes = ('none', 'shared', 'per_view', 'both')
        if lidar_injection not in valid_lidar_modes:
            raise ValueError(
                f'lidar_injection must be one of {valid_lidar_modes}, got {lidar_injection}')
        valid_align_modes = ('strict', 'pad_last', 'truncate_tail')
        if tn_align_mode not in valid_align_modes:
            raise ValueError(
                f'tn_align_mode must be one of {valid_align_modes}, got {tn_align_mode}')

        self.lidar_injection = lidar_injection
        self.tn_align_mode = tn_align_mode
        self.preserve_meta_keys = tuple(preserve_meta_keys or self._DEFAULT_META_KEYS)

    def _normalize_points(self, points, batch_size):
        if points is None:
            return None

        if isinstance(points, torch.Tensor):
            if points.dim() == 3 and points.shape[0] == batch_size:
                return [points[i] for i in range(batch_size)]
            if points.dim() == 2 and batch_size == 1:
                return [points]
            raise TypeError(
                'points tensor input must have shape [B, N, C] when passed as Tensor, '
                f'but got {tuple(points.shape)}')

        if not isinstance(points, (list, tuple)):
            raise TypeError(
                'points must be None, Tensor[B,N,C], or list/tuple of per-sample tensors, '
                f'but got {type(points)}')

        points_list = list(points)
        if len(points_list) != batch_size:
            raise ValueError(
                f'points batch size mismatch: got {len(points_list)} samples but expected {batch_size}')

        normalized = []
        for pts in points_list:
            if hasattr(pts, 'tensor'):
                pts = pts.tensor
            normalized.append(pts)
        return normalized

    def _normalize_mapanything_extra(self, mapanything_extra, batch_size):
        if mapanything_extra is None:
            return [dict() for _ in range(batch_size)]

        if isinstance(mapanything_extra, dict):
            return [copy.deepcopy(mapanything_extra) for _ in range(batch_size)]

        if not isinstance(mapanything_extra, (list, tuple)):
            raise TypeError(
                'mapanything_extra must be None, dict, or list/tuple of dict, '
                f'but got {type(mapanything_extra)}')

        mapanything_extra = list(mapanything_extra)
        if len(mapanything_extra) != batch_size:
            raise ValueError(
                f'mapanything_extra batch size mismatch: got {len(mapanything_extra)} samples, '
                f'expected {batch_size}')

        normalized = []
        for item in mapanything_extra:
            if item is None:
                normalized.append(dict())
            elif isinstance(item, dict):
                normalized.append(copy.deepcopy(item))
            else:
                raise TypeError(
                    'Each mapanything_extra item must be dict/None, '
                    f'but got {type(item)}')
        return normalized

    def _align_extra_views(self, views_extra, target_tn):
        if views_extra is None:
            return [dict() for _ in range(target_tn)]
        if not isinstance(views_extra, (list, tuple)):
            raise TypeError(
                f'mapanything_extra["views"] must be list/tuple, got {type(views_extra)}')

        views_extra = list(views_extra)
        cur_tn = len(views_extra)
        if cur_tn == target_tn:
            return views_extra

        if self.tn_align_mode == 'strict':
            raise ValueError(
                f'mapanything_extra views length mismatch: expected TN={target_tn}, got {cur_tn}')

        if self.tn_align_mode == 'truncate_tail':
            if cur_tn < target_tn:
                raise ValueError(
                    f'truncate_tail mode requires extra views >= TN={target_tn}, got {cur_tn}')
            warnings.warn(
                f'Truncating mapanything_extra views from {cur_tn} to {target_tn}.',
                stacklevel=2)
            return views_extra[:target_tn]

        # pad_last mode
        if cur_tn > target_tn:
            warnings.warn(
                f'Truncating mapanything_extra views from {cur_tn} to {target_tn}.',
                stacklevel=2)
            return views_extra[:target_tn]

        if cur_tn == 0:
            pad_item = dict()
        else:
            pad_item = views_extra[-1]
        out = list(views_extra)
        while len(out) < target_tn:
            out.append(copy.deepcopy(pad_item))
        warnings.warn(
            f'Padding mapanything_extra views from {cur_tn} to {target_tn} using pad_last.',
            stacklevel=2)
        return out

    def _slice_meta_value(self, value, idx, total_views):
        if isinstance(value, list) and len(value) == total_views:
            return value[idx]
        if isinstance(value, tuple) and len(value) == total_views:
            return value[idx]
        if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == total_views:
            return value[idx]
        return value

    def _merge_view_extra(self, base_view, view_extra):
        if view_extra is None:
            return base_view
        if not isinstance(view_extra, dict):
            raise TypeError(
                f'Each view-level extra must be dict/None, got {type(view_extra)}')
        if 'img' in view_extra:
            raise KeyError('mapanything_extra.views[*] must not override the "img" field')
        merged = dict(base_view)
        merged.update(view_extra)
        return merged

    def __call__(self, img, points, img_metas, mapanything_extra=None):
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            raise ValueError(
                f'img must be Tensor[B, TN, C, H, W], got type={type(img)} '
                f'shape={getattr(img, "shape", None)}')

        batch_size, total_views, channels, _, _ = img.shape
        if channels != 3:
            raise ValueError(f'Expected 3-channel image tensor, got C={channels}')
        if not isinstance(img_metas, list) or len(img_metas) != batch_size:
            raise ValueError(
                f'img_metas must be list of length B={batch_size}, got {type(img_metas)}')

        points_list = self._normalize_points(points, batch_size)
        extra_list = self._normalize_mapanything_extra(mapanything_extra, batch_size)

        batch_views = []
        for batch_idx in range(batch_size):
            meta = img_metas[batch_idx] if isinstance(img_metas[batch_idx], dict) else {}
            extra = extra_list[batch_idx]

            shared_extra = dict(extra.get('shared', {}))
            if self.lidar_injection in ('shared', 'both') and points_list is not None:
                shared_extra.setdefault('lidar_points', points_list[batch_idx])

            views_extra = self._align_extra_views(extra.get('views', None), total_views)

            sample_views = []
            for view_idx in range(total_views):
                view = dict(img=img[batch_idx, view_idx].permute(1, 2, 0).contiguous())
                for key in self.preserve_meta_keys:
                    if key in meta:
                        view[key] = self._slice_meta_value(meta[key], view_idx, total_views)

                if self.lidar_injection in ('per_view', 'both') and points_list is not None:
                    view['lidar_points'] = points_list[batch_idx]
                if self.lidar_injection in ('shared', 'both'):
                    view['shared'] = shared_extra

                view = self._merge_view_extra(view, views_extra[view_idx])
                sample_views.append(view)

            batch_views.append(sample_views)
        return batch_views
