import copy
import importlib
import os
import sys
import warnings
from contextlib import contextmanager

import torch
from mmengine.model import BaseModule
<<<<<<< HEAD

=======
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
from mmdet3d.registry import MODELS

from .input_adapter import OPUSToMapAnythingInputAdapter
from .output_adapter import MapAnythingOutputAdapter


@contextmanager
def _prepend_sys_path(path):
<<<<<<< HEAD
    if path is None:
        yield
        return

    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f'MapAnything repo_root does not exist: {abs_path}')

    added = False
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)
        added = True
    try:
        yield
=======
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f'MapAnything repo_root does not exist: {abs_path}')
    added = abs_path not in sys.path
    if added:
        sys.path.insert(0, abs_path)
    try:
        yield abs_path
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
    finally:
        if added and abs_path in sys.path:
            sys.path.remove(abs_path)


@MODELS.register_module()
class MapAnythingOPUSEncoder(BaseModule):
    """Internal OPUS wrapper around MapAnything feature encoding."""

    _STRIP_MODULES = (
        'dpt_regressor_head',
        'dense_head',
        'pose_head',
        'scale_head',
        'dense_adaptor',
        'pose_adaptor',
        'scale_adaptor',
    )
    _RANDOM_MASK_DEFAULTS = dict(
<<<<<<< HEAD
        overall_prob=0.9,
        dropout_prob=0.05,
        ray_dirs_prob=0.5,
        depth_prob=0.5,
        cam_prob=0.5,
        sparse_depth_prob=0.5,
=======
        overall_prob=1.0,
        dropout_prob=0.0,
        ray_dirs_prob=0.0,
        depth_prob=0.0,
        cam_prob=0.0,
        sparse_depth_prob=0.0,
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
    )

    def __init__(self,
                 repo_root,
<<<<<<< HEAD
                 mapanything_model_cfg,
                 mapanything_preprocess_cfg=None,
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

        self.repo_root = repo_root
        self.mapanything_model_cfg = dict(mapanything_model_cfg or {})
        self.mapanything_preprocess_cfg = dict(mapanything_preprocess_cfg or {})
        self.freeze = bool(freeze)
        self.enable_random_mask = bool(enable_random_mask)
        self.random_mask_cfg = dict(random_mask_cfg or {})
=======
                 lidar_injection='shared',
                 tn_align_mode='strict',
                 preserve_meta_keys=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.repo_root = os.path.abspath(os.path.expanduser(repo_root))
        if not os.path.isdir(self.repo_root):
            raise FileNotFoundError(f'MapAnything repo_root does not exist: {self.repo_root}')

        self.mapanything_model_cfg = copy.deepcopy(mapanything_model_cfg or {})
        self.mapanything_preprocess_cfg = copy.deepcopy(mapanything_preprocess_cfg or {})
        self.freeze = bool(freeze)
        self.enable_random_mask = bool(enable_random_mask)
        self.random_mask_cfg = copy.deepcopy(random_mask_cfg or {})
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
        self.strip_to_feature_mode = bool(strip_to_feature_mode)
        self.strict_shapes = bool(strict_shapes)
        self.expect_contiguous = bool(expect_contiguous)

<<<<<<< HEAD
        self.input_adapter = OPUSToMapAnythingInputAdapter(
            lidar_injection=lidar_injection,
            tn_align_mode=tn_align_mode,
            preserve_meta_keys=preserve_meta_keys,
        )
        self.output_adapter = MapAnythingOutputAdapter(
            patch_size=int(self.mapanything_preprocess_cfg.get('patch_size', 14)))
=======
        if batch_forward_size is not None and int(batch_forward_size) <= 0:
            raise ValueError(f'batch_forward_size must be positive or None, got {batch_forward_size}')
        self.batch_forward_size = None if batch_forward_size is None else int(batch_forward_size)

        self.input_adapter = OPUSToMapAnythingInputAdapter(
            lidar_injection=lidar_injection,
            tn_align_mode=tn_align_mode,
            preserve_meta_keys=preserve_meta_keys)
        self.output_adapter = MapAnythingOutputAdapter(
            patch_size=self.mapanything_preprocess_cfg.get('patch_size', 14))
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)

        self._build_mapanything_components()
        self._capture_base_geometric_input_cfg()
        self._freeze_model_if_needed()

<<<<<<< HEAD
=======
        encoder = getattr(self.map_model, 'encoder', None)
        out_channels = getattr(encoder, 'enc_embed_dim', None)
        if out_channels is None:
            out_channels = self.mapanything_model_cfg.get('out_channels', 1024)
        self.out_channels = int(out_channels)

>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
    def _capture_base_geometric_input_cfg(self):
        geometric_cfg = getattr(self.map_model, 'geometric_input_config', None)
        if isinstance(geometric_cfg, dict):
            self._base_geometric_input_cfg = copy.deepcopy(geometric_cfg)
        else:
            self._base_geometric_input_cfg = None

<<<<<<< HEAD
    def _sync_geometric_input_cfg_for_mode(self):
        if self._base_geometric_input_cfg is None:
            return

        if self.enable_random_mask and self.training:
            target_cfg = copy.deepcopy(self._base_geometric_input_cfg)
            target_cfg.update(self._RANDOM_MASK_DEFAULTS)
            target_cfg.update(self.random_mask_cfg)
        else:
            target_cfg = self._base_geometric_input_cfg
        self.map_model.geometric_input_config.update(target_cfg)

    def _build_model_from_cfg(self, mapanything_model_cfg):
        cfg = copy.deepcopy(mapanything_model_cfg)
        load_type = cfg.pop('load_type', 'from_pretrained')

        if load_type == 'from_pretrained':
            model_id_or_path = cfg.pop(
                'model_id_or_path',
                cfg.pop('pretrained', None))
            if model_id_or_path is None:
                raise ValueError(
                    'mapanything_model_cfg must provide model_id_or_path (or pretrained) '
                    'when load_type="from_pretrained"')
            from_pretrained_kwargs = dict(cfg.pop('from_pretrained_kwargs', {}))
            model = self._mapanything_cls.from_pretrained(
                model_id_or_path,
                **from_pretrained_kwargs)
        elif load_type == 'init_model_from_config':
            model_name = cfg.pop('model_name', 'mapanything')
            device = cfg.pop('device', 'cpu')
            machine = cfg.pop('machine', 'default')
            model = self._init_model_from_config(
                model_name=model_name,
                device=device,
                machine=machine)
        elif load_type == 'factory':
            factory_module = cfg.pop('factory_module', None)
            factory_fn = cfg.pop('factory_fn', None)
            factory_kwargs = dict(cfg.pop('factory_kwargs', {}))
            if not factory_module or not factory_fn:
                raise ValueError(
                    'load_type="factory" requires factory_module and factory_fn')
            with _prepend_sys_path(self.repo_root):
                module = importlib.import_module(factory_module)
=======
    def _sync_geometric_input_cfg_for_mode(self, mode):
        if self._base_geometric_input_cfg is None:
            return
        if not hasattr(self.map_model, 'geometric_input_config'):
            return

        target_cfg = copy.deepcopy(self._base_geometric_input_cfg)
        if mode and self.enable_random_mask:
            target_cfg.update(self._RANDOM_MASK_DEFAULTS)
            target_cfg.update(self.random_mask_cfg)
        self.map_model.geometric_input_config.update(target_cfg)

    def _build_model_from_cfg(self, cfg):
        cfg = copy.deepcopy(cfg)
        load_type = cfg.pop('load_type', 'from_pretrained')

        if load_type == 'from_pretrained':
            model_id_or_path = cfg.pop('model_id_or_path', cfg.pop('pretrained', None))
            if model_id_or_path is None:
                raise ValueError(
                    'mapanything_model_cfg must provide model_id_or_path (or pretrained) '
                    'when load_type=\"from_pretrained\"')
            from_pretrained_kwargs = cfg.pop('from_pretrained_kwargs', {})
            model = self._mapanything_cls.from_pretrained(
                model_id_or_path, **from_pretrained_kwargs)

        elif load_type == 'init_model_from_config':
            model_name = cfg.pop('model_name', 'mapanything')
            machine = cfg.pop('machine', 'default')
            device = cfg.pop('device', 'cpu')
            model = self._init_model_from_config(
                model_name=model_name, device=device, machine=machine)

        elif load_type == 'factory':
            factory_module = cfg.pop('factory_module', None)
            factory_fn = cfg.pop('factory_fn', None)
            factory_kwargs = cfg.pop('factory_kwargs', {})
            if not factory_module or not factory_fn:
                raise ValueError(
                    'load_type=\"factory\" requires factory_module and factory_fn')
            module = importlib.import_module(factory_module)
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
            if not hasattr(module, factory_fn):
                raise AttributeError(
                    f'Cannot find factory function {factory_fn} in module {factory_module}')
            model = getattr(module, factory_fn)(**factory_kwargs)
<<<<<<< HEAD
        else:
            raise ValueError(
                f'Unsupported mapanything load_type={load_type}. '
                'Use one of ["from_pretrained", "init_model_from_config", "factory"]')
=======

        else:
            raise ValueError(
                f'Unsupported mapanything load_type={load_type}. '
                'Use one of [\"from_pretrained\", \"init_model_from_config\", \"factory\"]')
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)

        if cfg:
            warnings.warn(
                f'Unused keys in mapanything_model_cfg after model build: {sorted(cfg.keys())}',
                stacklevel=2)
        return model

    def _build_mapanything_components(self):
        with _prepend_sys_path(self.repo_root):
            from mapanything.models import MapAnything, init_model_from_config
            from mapanything.utils.image import preprocess_inputs

        self._mapanything_cls = MapAnything
        self._init_model_from_config = init_model_from_config
        self._preprocess_inputs = preprocess_inputs
<<<<<<< HEAD
        self.map_model = self._build_model_from_cfg(self.mapanything_model_cfg)

=======

        self.map_model = self._build_model_from_cfg(self.mapanything_model_cfg)
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
        if self.strip_to_feature_mode:
            self._strip_mapanything_model(self.map_model)

    def _strip_mapanything_model(self, model):
        for module_name in self._STRIP_MODULES:
            if hasattr(model, module_name):
                delattr(model, module_name)

    def _freeze_model_if_needed(self):
        if not self.freeze:
            return
<<<<<<< HEAD
        if hasattr(self.map_model, 'parameters'):
            for parameter in self.map_model.parameters():
                parameter.requires_grad = False
        if hasattr(self.map_model, 'eval'):
            self.map_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze and hasattr(self.map_model, 'eval'):
            # Keep external encoder frozen in eval mode regardless of parent mode.
=======
        for parameter in self.map_model.parameters():
            parameter.requires_grad = False
        self.map_model.eval()

    def train(self, mode=True):
        super().train(mode)
        self._sync_geometric_input_cfg_for_mode(mode)
        if self.freeze:
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
            self.map_model.eval()
        return self

    def _get_model_device(self):
<<<<<<< HEAD
        for parameter in self.map_model.parameters():
            return parameter.device
        for buffer in self.map_model.buffers():
=======
        parameter = next(iter(self.map_model.parameters()), None)
        if parameter is not None:
            return parameter.device
        buffer = next(iter(self.map_model.buffers()), None)
        if buffer is not None:
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
            return buffer.device
        return torch.device('cpu')

    def _ensure_model_device(self, target_device):
        current_device = self._get_model_device()
        if current_device != target_device:
            self.map_model.to(target_device)

    def _move_processed_views_to_model_device(self, processed_views):
<<<<<<< HEAD
        ignore_keys = {'instance', 'idx', 'true_shape', 'data_norm_type'}
        target_device = self._get_model_device()

        def _move_to_device(value):
=======
        target_device = self._get_model_device()

        def _move_to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(target_device, non_blocking=True)
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
            if isinstance(value, dict):
                return {k: _move_to_device(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_move_to_device(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_move_to_device(v) for v in value)
<<<<<<< HEAD
            if hasattr(value, 'to'):
                return value.to(target_device, non_blocking=True)
            return value

        for view in processed_views:
            for key in list(view.keys()):
                if key in ignore_keys:
                    continue
                view[key] = _move_to_device(view[key])
        return processed_views

    def _infer_image_hw(self, processed_views):
        if not processed_views:
            return None
        first = processed_views[0]
        img = first.get('img', None)
        if isinstance(img, torch.Tensor) and img.dim() >= 4:
            return int(img.shape[-2]), int(img.shape[-1])
        return None
=======
            return value

        return [_move_to_device(view) for view in processed_views]

    def _infer_image_hw(self, processed_views):
        first_img = processed_views[0]['img']
        if not isinstance(first_img, torch.Tensor) or first_img.dim() != 4:
            raise ValueError(f'views[0][\"img\"] must be Tensor[B,C,H,W], got {type(first_img)}')
        return int(first_img.shape[2]), int(first_img.shape[3])

    def _collate_view_values(self, key, values):
        if any(v is None for v in values):
            if all(v is None for v in values):
                return None
            raise ValueError(f'Inconsistent None values for key \"{key}\" across batch')

        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(values, dim=0)

        if key == 'data_norm_type':
            return copy.deepcopy(first)

        if isinstance(first, tuple):
            if not all(isinstance(v, tuple) and len(v) == len(first) for v in values):
                raise ValueError(f'Tuple length mismatch for key \"{key}\"')
            return tuple(self._collate_view_values(f'{key}.{i}', [v[i] for v in values])
                         for i in range(len(first)))

        if isinstance(first, dict):
            all_keys = set()
            for item in values:
                all_keys.update(item.keys())
            out = {}
            for sub_key in all_keys:
                sub_values = [item.get(sub_key, None) for item in values]
                out[sub_key] = self._collate_view_values(f'{key}.{sub_key}', sub_values)
            return out

        if isinstance(first, list):
            if all(item == first for item in values):
                return copy.deepcopy(first)
            return [copy.deepcopy(item) for item in values]

        return copy.deepcopy(first)

    def _collate_processed_batch_views(self, processed_batch_views):
        if not processed_batch_views:
            raise ValueError('processed_batch_views must be non-empty')

        total_views = len(processed_batch_views[0])
        for sample_idx, sample_views in enumerate(processed_batch_views):
            if len(sample_views) != total_views:
                raise ValueError(
                    f'Processed sample {sample_idx} has {len(sample_views)} views, '
                    f'expected {total_views}')
            if any(not isinstance(view, dict) for view in sample_views):
                raise TypeError('All processed views must be dict')

        batched_views = []
        for view_idx in range(total_views):
            per_sample_views = [sample_views[view_idx] for sample_views in processed_batch_views]
            ordered_keys = set()
            for view in per_sample_views:
                ordered_keys.update(view.keys())
            batched_view = {}
            for key in ordered_keys:
                values = [view.get(key, None) for view in per_sample_views]
                batched_view[key] = self._collate_view_values(key, values)
            batched_views.append(batched_view)
        return batched_views
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)

    def _run_model(self, processed_views):
        if self.freeze:
            with torch.no_grad():
<<<<<<< HEAD
                return self.map_model.forward(processed_views)
        return self.map_model.forward(processed_views)

    def _get_patch_size(self):
        patch_size = self.mapanything_preprocess_cfg.get('patch_size', None)
        if patch_size is not None:
            return int(patch_size)
        encoder = getattr(self.map_model, 'encoder', None)
        if encoder is not None and hasattr(encoder, 'patch_size'):
            try:
                return int(getattr(encoder, 'patch_size'))
            except Exception:
                pass
        return self.output_adapter.patch_size
=======
                return self.map_model(processed_views)
        return self.map_model(processed_views)

    def _slice_tensor_or_list(self, value, start, end):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value[start:end]
        if isinstance(value, list):
            return value[start:end]
        if isinstance(value, tuple):
            return list(value[start:end])
        return value

    def _forward_single(self, img, points=None, img_metas=None, mapanything_extra=None):
        batch_size, total_views = img.shape[:2]
        batch_views = self.input_adapter(
            img=img,
            points=points,
            img_metas=img_metas,
            mapanything_extra=mapanything_extra)

        processed_batch_views = []
        for sample_views in batch_views:
            processed_batch_views.append(
                self._preprocess_inputs(sample_views, **self.mapanything_preprocess_cfg))

        batched_views = self._collate_processed_batch_views(processed_batch_views)

        self._ensure_model_device(img.device)
        batched_views = self._move_processed_views_to_model_device(batched_views)

        raw_output = self._run_model(batched_views)
        image_hw = self._infer_image_hw(batched_views)
        output = self.output_adapter(
            raw_output,
            batch_size=batch_size,
            total_views=total_views,
            image_hw=image_hw,
            expect_contiguous=self.expect_contiguous)

        if self.strict_shapes:
            if output.shape[0] != batch_size or output.shape[1] != total_views:
                raise ValueError(
                    f'Output shape mismatch: expected [B={batch_size}, TN={total_views}, ...], '
                    f'got {tuple(output.shape)}')
            if output.device != img.device:
                raise ValueError(
                    f'Output device mismatch: expected {img.device}, got {output.device}')
        return output
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)

    def forward(self, img, points=None, img_metas=None, mapanything_extra=None):
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            raise ValueError(
<<<<<<< HEAD
                f'img must be Tensor[B, TN, C, H, W], got type={type(img)} '
                f'shape={getattr(img, "shape", None)}')

        batch_size, total_views = img.shape[:2]
=======
                'img must be Tensor[B, TN, C, H, W], got type='
                f'{type(img)} shape={getattr(img, "shape", None)}')

        batch_size = img.shape[0]
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
        if img_metas is None:
            img_metas = [{} for _ in range(batch_size)]
        if not isinstance(img_metas, list) or len(img_metas) != batch_size:
            raise ValueError(
                f'img_metas must be list of length B={batch_size}, got {type(img_metas)}')

<<<<<<< HEAD
        self._ensure_model_device(img.device)
        self._sync_geometric_input_cfg_for_mode()

        batch_views = self.input_adapter(
            img=img,
            points=points,
            img_metas=img_metas,
            mapanything_extra=mapanything_extra,
        )

        per_sample_feats = []
        patch_size = self._get_patch_size()
        for sample_idx, sample_views in enumerate(batch_views):
            processed_views = self._preprocess_inputs(
                sample_views,
                **self.mapanything_preprocess_cfg)
            processed_views = self._move_processed_views_to_model_device(processed_views)
            raw_output = self._run_model(processed_views)

            sample_image_hw = self._infer_image_hw(processed_views)
            sample_feat = self.output_adapter(
                raw_output,
                batch_size=1,
                total_views=total_views,
                image_hw=sample_image_hw,
                patch_size=patch_size)
            if sample_feat.shape[0] != 1:
                raise ValueError(
                    f'Per-sample feature should have batch dim 1, got {tuple(sample_feat.shape)} '
                    f'at sample {sample_idx}')
            per_sample_feats.append(sample_feat)

        output = torch.cat(per_sample_feats, dim=0)
        if self.strict_shapes and output.shape[:2] != (batch_size, total_views):
            raise ValueError(
                f'Output shape mismatch: expected [B={batch_size}, TN={total_views}, ...], '
                f'got {tuple(output.shape)}')
        if self.strict_shapes and output.device != img.device:
            raise ValueError(
                f'Output device mismatch: expected {img.device}, got {output.device}')

=======
        if self.batch_forward_size is None or batch_size <= self.batch_forward_size:
            return self._forward_single(
                img=img,
                points=points,
                img_metas=img_metas,
                mapanything_extra=mapanything_extra)

        output_chunks = []
        for start in range(0, batch_size, self.batch_forward_size):
            end = min(start + self.batch_forward_size, batch_size)
            chunk_output = self._forward_single(
                img=img[start:end],
                points=self._slice_tensor_or_list(points, start, end),
                img_metas=img_metas[start:end],
                mapanything_extra=self._slice_tensor_or_list(mapanything_extra, start, end))
            output_chunks.append(chunk_output)

        output = torch.cat(output_chunks, dim=0)
        if self.strict_shapes and output.shape[0] != batch_size:
            raise ValueError(
                f'Output batch mismatch: expected B={batch_size}, got {output.shape[0]}')
>>>>>>> b8cc5df (mapanything 融合分支开发完成，提交初始版本。特种融合只包含图像，res+map 直接求和 双线性差值 主要改动包括：)
        return output.contiguous() if self.expect_contiguous else output
