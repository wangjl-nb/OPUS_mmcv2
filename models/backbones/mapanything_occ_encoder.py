import copy
import warnings

<<<<<<< HEAD
import numpy as np
import torch
from mmengine.model import BaseModule

=======
import torch
from mmengine.model import BaseModule
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
from mmdet3d.registry import MODELS

from ..mapanything.opus_mapanything_wrapper import MapAnythingOPUSEncoder


@MODELS.register_module()
class MapAnythingOccEncoder(BaseModule):
    """Backbone bridge that runs internal OPUS MapAnything wrapper.

    Input:
        img: Tensor[B, TN, 3, H, W]
        points: list[Tensor] or Tensor[B, N, C]
        img_metas: list[dict]
        mapanything_extra: optional list[dict] for modality extension
    Output:
        Tensor[B, TN, C, Hf, Wf]
    """

    def __init__(self,
                 repo_root,
<<<<<<< HEAD
                 mapanything_model_cfg,
                 mapanything_preprocess_cfg=None,
                 num_views=6,
                 num_frames=None,
                 chunk_by_frame=True,
                 freeze=True,
=======
                 mapanything_model_cfg=None,
                 mapanything_preprocess_cfg=None,
                 freeze=True,
                 batch_forward_size=None,
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
                 enable_random_mask=False,
                 random_mask_cfg=None,
                 strip_to_feature_mode=True,
                 strict_shapes=True,
                 expect_contiguous=True,
<<<<<<< HEAD
                 tn_align_mode='strict',
                 lidar_injection='shared',
                 preserve_meta_keys=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.num_views = int(num_views)
        self.num_frames = None if num_frames is None else int(num_frames)
=======
                 num_views=6,
                 num_frames=1,
                 chunk_by_frame=True,
                 tn_align_mode='strict',
                 lidar_injection='shared',
                 preserve_meta_keys=('img_shape', 'pad_shape', 'ori_shape'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_views = int(num_views)
        self.num_frames = int(num_frames)
        self.freeze = bool(freeze)
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
        self.chunk_by_frame = bool(chunk_by_frame)
        self.strict_shapes = bool(strict_shapes)
        self.expect_contiguous = bool(expect_contiguous)
        self.tn_align_mode = tn_align_mode

        valid_align_modes = ('strict', 'pad_last', 'truncate_tail')
<<<<<<< HEAD
        if tn_align_mode not in valid_align_modes:
            raise ValueError(
                f'tn_align_mode must be one of {valid_align_modes}, got {tn_align_mode}')
=======
        if self.tn_align_mode not in valid_align_modes:
            raise ValueError(
                f'tn_align_mode must be one of {valid_align_modes}, got {self.tn_align_mode}')
        if self.num_views <= 0:
            raise ValueError(f'num_views must be positive, got {self.num_views}')
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)

        self.wrapper = MapAnythingOPUSEncoder(
            repo_root=repo_root,
            mapanything_model_cfg=mapanything_model_cfg,
            mapanything_preprocess_cfg=mapanything_preprocess_cfg,
            freeze=freeze,
<<<<<<< HEAD
=======
            batch_forward_size=batch_forward_size,
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
            enable_random_mask=enable_random_mask,
            random_mask_cfg=random_mask_cfg,
            strip_to_feature_mode=strip_to_feature_mode,
            strict_shapes=strict_shapes,
            expect_contiguous=expect_contiguous,
            tn_align_mode=tn_align_mode,
            lidar_injection=lidar_injection,
<<<<<<< HEAD
            preserve_meta_keys=preserve_meta_keys,
        )

    def train(self, mode=True):
        super().train(mode)
        self.wrapper.train(mode)
        return self

    def _expected_tn(self):
        if self.num_frames is None:
=======
            preserve_meta_keys=preserve_meta_keys)
        self.out_channels = int(getattr(self.wrapper, 'out_channels', 1024))
        self._freeze_wrapper_if_needed()

    def _freeze_wrapper_if_needed(self):
        if not self.freeze:
            return
        for parameter in self.wrapper.parameters():
            parameter.requires_grad = False
        self.wrapper.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.wrapper.eval()
        return self

    def _expected_tn(self):
        if self.num_views <= 0 or self.num_frames <= 0:
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
            return None
        return self.num_views * self.num_frames

    def _align_tn_img(self, img, target_tn):
<<<<<<< HEAD
        cur_tn = img.shape[1]
        if cur_tn == target_tn:
            return img

        if self.tn_align_mode == 'strict':
            raise ValueError(
                f'Input TN mismatch: expected {target_tn}, got {cur_tn}')

        if self.tn_align_mode == 'truncate_tail':
            if cur_tn < target_tn:
                raise ValueError(
                    f'truncate_tail mode requires TN >= {target_tn}, got {cur_tn}')
            warnings.warn(
                f'Truncating image TN from {cur_tn} to {target_tn}.',
                stacklevel=2)
            return img[:, :target_tn]

        # pad_last mode
        if cur_tn > target_tn:
            warnings.warn(
                f'Truncating image TN from {cur_tn} to {target_tn}.',
                stacklevel=2)
            return img[:, :target_tn]

        pad_count = target_tn - cur_tn
        pad_feat = img[:, -1:].expand(img.shape[0], pad_count, *img.shape[2:])
        warnings.warn(
            f'Padding image TN from {cur_tn} to {target_tn} using pad_last.',
            stacklevel=2)
=======
        cur_tn = int(img.shape[1])
        if cur_tn == target_tn:
            return img
        if self.tn_align_mode == 'strict':
            raise ValueError(f'Input TN mismatch: expected {target_tn}, got {cur_tn}')
        if self.tn_align_mode == 'truncate_tail':
            if cur_tn < target_tn:
                raise ValueError(f'truncate_tail mode requires TN >= {target_tn}, got {cur_tn}')
            warnings.warn(f'Truncating image TN from {cur_tn} to {target_tn}.', stacklevel=2)
            return img[:, :target_tn]

        # pad_last
        warnings.warn(
            f'Padding image TN from {cur_tn} to {target_tn} using pad_last.',
            stacklevel=2)
        if cur_tn <= 0:
            raise ValueError('Cannot pad empty TN dimension')
        pad_count = target_tn - cur_tn
        pad_feat = img[:, -1:].expand(-1, pad_count, -1, -1, -1)
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
        return torch.cat([img, pad_feat], dim=1)

    def _align_meta_value(self, value, target_tn):
        if isinstance(value, list):
<<<<<<< HEAD
            cur_tn = len(value)
            if cur_tn == target_tn:
                return value
            if self.tn_align_mode == 'strict':
                return value
            if self.tn_align_mode == 'truncate_tail':
                if cur_tn >= target_tn:
                    return value[:target_tn]
                return value
            # pad_last mode
            if cur_tn >= target_tn:
                return value[:target_tn]
            if cur_tn == 0:
                return [None for _ in range(target_tn)]
            return value + [copy.deepcopy(value[-1]) for _ in range(target_tn - cur_tn)]

        if isinstance(value, tuple):
            aligned = self._align_meta_value(list(value), target_tn)
            return tuple(aligned)

        if isinstance(value, torch.Tensor) and value.dim() > 0:
            cur_tn = int(value.shape[0])
            if cur_tn == target_tn:
                return value
            if self.tn_align_mode == 'strict':
                return value
            if self.tn_align_mode == 'truncate_tail':
                if cur_tn >= target_tn:
                    return value[:target_tn]
                return value
            # pad_last mode
            if cur_tn >= target_tn:
                return value[:target_tn]
            if cur_tn == 0:
                return value
            last = value[-1:]
            pad = last.repeat(target_tn - cur_tn, *([1] * (last.dim() - 1)))
            return torch.cat([value, pad], dim=0)

        if isinstance(value, np.ndarray) and value.ndim > 0:
            cur_tn = int(value.shape[0])
            if cur_tn == target_tn:
                return value
            if self.tn_align_mode == 'strict':
                return value
            if self.tn_align_mode == 'truncate_tail':
                if cur_tn >= target_tn:
                    return value[:target_tn]
                return value
            # pad_last mode
            if cur_tn >= target_tn:
                return value[:target_tn]
            if cur_tn == 0:
                return value
            pad_count = target_tn - cur_tn
            last = value[-1:]
            pad = np.repeat(last, repeats=pad_count, axis=0)
            return np.concatenate([value, pad], axis=0)
=======
            if len(value) == target_tn:
                return value
            if self.tn_align_mode == 'strict':
                return value
            if len(value) > target_tn:
                return value[:target_tn]
            if len(value) == 0:
                return value
            pad = [copy.deepcopy(value[-1]) for _ in range(target_tn - len(value))]
            return value + pad

        if isinstance(value, tuple):
            aligned = self._align_meta_value(list(value), target_tn)
            return tuple(aligned) if isinstance(aligned, list) else value

        if isinstance(value, torch.Tensor) and value.dim() > 0:
            if value.shape[0] == target_tn:
                return value
            if self.tn_align_mode == 'strict':
                return value
            if value.shape[0] > target_tn:
                return value[:target_tn]
            repeat = target_tn - value.shape[0]
            if value.shape[0] <= 0:
                return value
            pad = value[-1:].expand(repeat, *value.shape[1:])
            return torch.cat([value, pad], dim=0)

>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
        return value

    def _align_img_metas(self, img_metas, target_tn):
        if img_metas is None:
            return None
        if not isinstance(img_metas, list):
            raise TypeError(f'img_metas must be list, got {type(img_metas)}')
        aligned = []
        for meta in img_metas:
            if not isinstance(meta, dict):
<<<<<<< HEAD
                aligned.append(meta)
                continue
=======
                raise TypeError(f'Each meta must be dict, got {type(meta)}')
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
            aligned_meta = {}
            for key, value in meta.items():
                aligned_meta[key] = self._align_meta_value(value, target_tn)
            aligned.append(aligned_meta)
        return aligned

<<<<<<< HEAD
    def _align_mapanything_extra(self, mapanything_extra, target_tn):
        if mapanything_extra is None:
            return None
        if isinstance(mapanything_extra, dict):
            mapanything_extra = [copy.deepcopy(mapanything_extra)]
        if not isinstance(mapanything_extra, list):
            raise TypeError(
                f'mapanything_extra must be dict/list/None, got {type(mapanything_extra)}')

        aligned = []
        for sample_extra in mapanything_extra:
            if sample_extra is None:
                aligned.append(None)
                continue
            if not isinstance(sample_extra, dict):
                raise TypeError(
                    f'Each mapanything_extra item must be dict/None, got {type(sample_extra)}')
            sample_out = copy.deepcopy(sample_extra)
            if 'views' in sample_out:
                views = sample_out['views']
                if isinstance(views, tuple):
                    views = list(views)
                if isinstance(views, list):
                    cur_tn = len(views)
                    if self.tn_align_mode == 'strict':
                        if cur_tn != target_tn:
                            raise ValueError(
                                f'mapanything_extra views TN mismatch: expected {target_tn}, got {cur_tn}')
                    elif self.tn_align_mode == 'truncate_tail':
                        if cur_tn < target_tn:
                            raise ValueError(
                                f'truncate_tail mode requires extra views >= {target_tn}, got {cur_tn}')
                        views = views[:target_tn]
                    else:  # pad_last
                        if cur_tn >= target_tn:
                            views = views[:target_tn]
                        elif cur_tn == 0:
                            views = [dict() for _ in range(target_tn)]
                        else:
                            views = views + [copy.deepcopy(views[-1]) for _ in range(target_tn - cur_tn)]
                sample_out['views'] = views
            aligned.append(sample_out)
        return aligned

    def _slice_img_metas(self, img_metas, start, end, total_views):
=======
    def _align_mapanything_extra(self, mapanything_extra, batch_size, target_tn):
        if mapanything_extra is None:
            return None

        if isinstance(mapanything_extra, dict):
            extras = [copy.deepcopy(mapanything_extra) for _ in range(batch_size)]
        elif isinstance(mapanything_extra, (list, tuple)):
            extras = list(mapanything_extra)
            if len(extras) != batch_size:
                raise ValueError(
                    f'mapanything_extra batch size mismatch: got {len(extras)}, '
                    f'expected {batch_size}')
        else:
            raise TypeError(f'mapanything_extra must be dict/list/None, got {type(mapanything_extra)}')

        out = []
        for sample_extra in extras:
            if sample_extra is None:
                sample_extra = {}
            if not isinstance(sample_extra, dict):
                raise TypeError(f'Each mapanything_extra item must be dict/None, got {type(sample_extra)}')
            sample_out = copy.deepcopy(sample_extra)
            views = sample_out.get('views', None)
            if views is None:
                out.append(sample_out)
                continue
            if not isinstance(views, (list, tuple)):
                raise TypeError('mapanything_extra["views"] must be list/tuple when provided')
            views = list(views)
            if len(views) == target_tn:
                sample_out['views'] = views
                out.append(sample_out)
                continue
            if self.tn_align_mode == 'strict':
                raise ValueError(
                    f'mapanything_extra views TN mismatch: expected {target_tn}, got {len(views)}')
            if self.tn_align_mode == 'truncate_tail':
                if len(views) < target_tn:
                    raise ValueError(
                        f'truncate_tail mode requires extra views >= {target_tn}, got {len(views)}')
                sample_out['views'] = views[:target_tn]
            else:
                if len(views) <= 0:
                    sample_out['views'] = [dict() for _ in range(target_tn)]
                elif len(views) < target_tn:
                    pad = [copy.deepcopy(views[-1]) for _ in range(target_tn - len(views))]
                    sample_out['views'] = views + pad
                else:
                    sample_out['views'] = views[:target_tn]
            out.append(sample_out)
        return out

    def _slice_img_metas(self, img_metas, start, end):
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
        if img_metas is None:
            return None
        sliced = []
        for meta in img_metas:
<<<<<<< HEAD
            if not isinstance(meta, dict):
                sliced.append(meta)
                continue

            chunk_meta = {}
            for key, value in meta.items():
                if isinstance(value, list) and len(value) == total_views:
                    chunk_meta[key] = value[start:end]
                elif isinstance(value, tuple) and len(value) == total_views:
                    chunk_meta[key] = value[start:end]
                elif hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == total_views:
=======
            chunk_meta = {}
            for key, value in meta.items():
                if isinstance(value, list) and len(value) >= end:
                    chunk_meta[key] = value[start:end]
                elif isinstance(value, tuple) and len(value) >= end:
                    chunk_meta[key] = value[start:end]
                elif isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] >= end:
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
                    chunk_meta[key] = value[start:end]
                else:
                    chunk_meta[key] = value
            sliced.append(chunk_meta)
        return sliced

    def _slice_mapanything_extra(self, mapanything_extra, start, end):
        if mapanything_extra is None:
            return None
<<<<<<< HEAD
        sliced = []
        for item in mapanything_extra:
            if item is None:
                sliced.append(None)
                continue
            if not isinstance(item, dict):
                raise TypeError(
                    f'Each mapanything_extra item must be dict/None, got {type(item)}')
            out = copy.deepcopy(item)
            if 'views' in out and isinstance(out['views'], (list, tuple)):
                out['views'] = list(out['views'][start:end])
            sliced.append(out)
        return sliced
=======
        out = []
        for sample_extra in mapanything_extra:
            if sample_extra is None:
                out.append(None)
                continue
            chunk = copy.deepcopy(sample_extra)
            if isinstance(chunk, dict) and isinstance(chunk.get('views', None), list):
                chunk['views'] = chunk['views'][start:end]
            out.append(chunk)
        return out
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)

    def forward(self, img, points=None, img_metas=None, mapanything_extra=None):
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            raise ValueError(
<<<<<<< HEAD
                f'img must be Tensor[B, TN, C, H, W], got type={type(img)} '
                f'shape={getattr(img, "shape", None)}')

        if self.num_views <= 0:
            raise ValueError(f'num_views must be positive, got {self.num_views}')

        expected_tn = self._expected_tn()
        if expected_tn is not None:
            valid_tn = {expected_tn}
            # Online frame-wise extraction may feed one-frame chunks (num_views).
            if self.chunk_by_frame:
                valid_tn.add(self.num_views)
            if img.shape[1] not in valid_tn:
                img = self._align_tn_img(img, expected_tn)
                img_metas = self._align_img_metas(img_metas, expected_tn)
                mapanything_extra = self._align_mapanything_extra(mapanything_extra, expected_tn)

        batch_size, total_views = img.shape[:2]
        if isinstance(mapanything_extra, dict):
            mapanything_extra = [copy.deepcopy(mapanything_extra) for _ in range(batch_size)]
        elif isinstance(mapanything_extra, list) and len(mapanything_extra) == 1 and batch_size > 1:
            mapanything_extra = [copy.deepcopy(mapanything_extra[0]) for _ in range(batch_size)]
        if mapanything_extra is not None:
            if not isinstance(mapanything_extra, list) or len(mapanything_extra) != batch_size:
                raise ValueError(
                    f'mapanything_extra must be None/dict or list with length B={batch_size}, '
                    f'got type={type(mapanything_extra)} len={len(mapanything_extra) if isinstance(mapanything_extra, list) else "N/A"}')

        if total_views % self.num_views != 0:
            raise ValueError(
                f'TN mismatch: total views {total_views} is not divisible by num_views {self.num_views}')
=======
                'img must be Tensor[B, TN, C, H, W], got type='
                f'{type(img)} shape={getattr(img, "shape", None)}')
        if self.num_views <= 0:
            raise ValueError(f'num_views must be positive, got {self.num_views}')

        batch_size, total_views = img.shape[:2]
        expected_tn = self._expected_tn()

        if expected_tn is not None and total_views != expected_tn:
            img = self._align_tn_img(img, expected_tn)
            total_views = int(img.shape[1])
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)

        if img_metas is None:
            img_metas = [{} for _ in range(batch_size)]
        if not isinstance(img_metas, list) or len(img_metas) != batch_size:
<<<<<<< HEAD
            raise ValueError(
                f'img_metas must be list of length B={batch_size}, got {type(img_metas)}')

        if self.chunk_by_frame:
            feats = []
            num_frames = total_views // self.num_views
            for frame_idx in range(num_frames):
                start = frame_idx * self.num_views
                end = start + self.num_views
                chunk_img = img[:, start:end]
                chunk_metas = self._slice_img_metas(img_metas, start, end, total_views)
                chunk_extra = self._slice_mapanything_extra(mapanything_extra, start, end)
                chunk_feat = self.wrapper(
                    chunk_img,
                    points=points,
                    img_metas=chunk_metas,
                    mapanything_extra=chunk_extra)
                if self.strict_shapes and chunk_feat.shape[:2] != (batch_size, self.num_views):
                    raise ValueError(
                        f'Wrapper chunk output shape mismatch at frame {frame_idx}: '
                        f'expected [B={batch_size}, TN_chunk={self.num_views}, ...], '
                        f'got {tuple(chunk_feat.shape)}')
=======
            raise ValueError(f'img_metas must be list of length B={batch_size}, got {type(img_metas)}')
        img_metas = self._align_img_metas(img_metas, total_views)

        mapanything_extra = self._align_mapanything_extra(
            mapanything_extra, batch_size=batch_size, target_tn=total_views)

        output = None
        if self.chunk_by_frame and total_views > self.num_views:
            if total_views % self.num_views != 0:
                raise ValueError(
                    f'TN mismatch: total views {total_views} is not divisible by num_views {self.num_views}')
            feats = []
            for frame_idx, start in enumerate(range(0, total_views, self.num_views)):
                end = start + self.num_views
                chunk_img = img[:, start:end]
                chunk_metas = self._slice_img_metas(img_metas, start, end)
                chunk_extra = self._slice_mapanything_extra(mapanything_extra, start, end)
                chunk_feat = self.wrapper(
                    img=chunk_img,
                    points=points,
                    img_metas=chunk_metas,
                    mapanything_extra=chunk_extra)
                if self.strict_shapes:
                    expect_shape = (batch_size, self.num_views)
                    if chunk_feat.shape[0] != expect_shape[0] or chunk_feat.shape[1] != expect_shape[1]:
                        raise ValueError(
                            f'Wrapper chunk output shape mismatch at frame {frame_idx}: '
                            f'expected [B={batch_size}, TN_chunk={self.num_views}, ...], '
                            f'got {tuple(chunk_feat.shape)}')
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
                feats.append(chunk_feat)
            output = torch.cat(feats, dim=1)
        else:
            output = self.wrapper(
<<<<<<< HEAD
                img,
=======
                img=img,
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
                points=points,
                img_metas=img_metas,
                mapanything_extra=mapanything_extra)

<<<<<<< HEAD
        if self.strict_shapes and output.shape[:2] != (batch_size, total_views):
            raise ValueError(
                f'Wrapper output shape mismatch: expected [B={batch_size}, TN={total_views}, ...], '
                f'got {tuple(output.shape)}')
        if self.strict_shapes and output.device != img.device:
            raise ValueError(
                f'Wrapper output device mismatch: expected {img.device}, got {output.device}')
=======
        if self.strict_shapes:
            if output.shape[0] != batch_size or output.shape[1] != total_views:
                raise ValueError(
                    f'Wrapper output shape mismatch: expected [B={batch_size}, TN={total_views}, ...], '
                    f'got {tuple(output.shape)}')
            if output.device != img.device:
                raise ValueError(
                    f'Wrapper output device mismatch: expected {img.device}, got {output.device}')
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)

        return output.contiguous() if self.expect_contiguous else output
