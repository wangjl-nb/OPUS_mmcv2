import copy
import importlib
import os
import sys
import warnings
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from .input_adapter import OPUSToMapAnythingInputAdapter
from .output_adapter import MapAnythingOutputAdapter


@contextmanager
def _prepend_sys_path(path):
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
    finally:
        if added and abs_path in sys.path:
            sys.path.remove(abs_path)


def _find_local_dinov2_repo(explicit_repo_root=None):
    if explicit_repo_root is not None:
        candidate = os.path.abspath(os.path.expanduser(str(explicit_repo_root)))
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(f'Configured local_dinov2_repo does not exist: {candidate}')

    hub_dir = torch.hub.get_dir()
    candidates = (
        os.path.join(hub_dir, 'facebookresearch_dinov2_main'),
        os.path.join(hub_dir, 'facebookresearch_dinov2_master'),
    )
    for repo_dir in candidates:
        if os.path.isdir(repo_dir):
            return repo_dir

    raise FileNotFoundError(
        'Local DINOv2 torch hub repo not found. '
        f'Expected one of: {candidates}')


@contextmanager
def _patch_torch_hub_offline_for_dinov2(dinov2_repo_dir):
    original_torch_hub_load = torch.hub.load

    def _offline_torch_hub_load(repo_or_dir, model, *args, **kwargs):
        repo_key = str(repo_or_dir).strip().lower()
        if repo_key in {
                'facebookresearch/dinov2',
                'facebookresearch/dinov2:main',
                'facebookresearch/dinov2:master'}:
            kwargs.pop('force_reload', None)
            return original_torch_hub_load(
                str(dinov2_repo_dir), model, *args, source='local', **kwargs)
        return original_torch_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = _offline_torch_hub_load
    try:
        yield
    finally:
        torch.hub.load = original_torch_hub_load


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
        overall_prob=0.9,
        dropout_prob=0.05,
        ray_dirs_prob=0.5,
        depth_prob=0.5,
        cam_prob=0.5,
        sparse_depth_prob=0.5,
    )
    _ANYUP_CHECKPOINT_URLS = dict(
        anyup='https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth',
        anyup_multi_backbone='https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth',
    )
    _ANYUP_CHECKPOINT_NAMES = dict(
        anyup='anyup_paper.pth',
        anyup_multi_backbone='anyup_multi_backbone.pth',
    )

    def __init__(self,
                 repo_root,
                 mapanything_model_cfg,
                 mapanything_preprocess_cfg=None,
                 anyup_cfg=None,
                 freeze=True,
                 enable_random_mask=False,
                 random_mask_cfg=None,
                 strip_to_feature_mode=True,
                 strict_shapes=True,
                 expect_contiguous=True,
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
        self.strip_to_feature_mode = bool(strip_to_feature_mode)
        self.strict_shapes = bool(strict_shapes)
        self.expect_contiguous = bool(expect_contiguous)
        self.anyup_cfg = dict(anyup_cfg or {})
        self.anyup_enabled = bool(self.anyup_cfg.get('enabled', False))
        self.anyup_mode = str(self.anyup_cfg.get('mode', 'anyup')).lower()
        if self.anyup_mode not in ('anyup', 'bilinear'):
            raise ValueError(
                f'anyup_cfg.mode must be one of ("anyup", "bilinear"), got {self.anyup_mode}')
        self.anyup_model = None
        self.anyup_freeze = True
        self.anyup_q_chunk_size = None
        self.anyup_pyramid_divisors = [4, 8, 16, 32]
        self.anyup_pyramid_num_levels = 4
        self.anyup_pyramid_downsample_mode = 'area'
        self.anyup_pyramid_align_corners = False
        self.anyup_upsample_output_divisor = 1
        self.anyup_view_batch_size = None
        self.anyup_output_channels = None
        self.anyup_output_proj = None

        self.input_adapter = OPUSToMapAnythingInputAdapter(
            lidar_injection=lidar_injection,
            tn_align_mode=tn_align_mode,
            preserve_meta_keys=preserve_meta_keys,
        )
        self.output_adapter = MapAnythingOutputAdapter(
            patch_size=int(self.mapanything_preprocess_cfg.get('patch_size', 14)))

        self._build_mapanything_components()
        self._capture_base_geometric_input_cfg()
        self._freeze_model_if_needed()
        if self.anyup_enabled:
            self._build_anyup_components()

    def _capture_base_geometric_input_cfg(self):
        geometric_cfg = getattr(self.map_model, 'geometric_input_config', None)
        if isinstance(geometric_cfg, dict):
            self._base_geometric_input_cfg = copy.deepcopy(geometric_cfg)
        else:
            self._base_geometric_input_cfg = None

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
        force_local_torch_hub = bool(cfg.pop('force_local_torch_hub', True))
        local_dinov2_repo = cfg.pop('local_dinov2_repo', None)

        if load_type == 'from_pretrained':
            model_id_or_path = cfg.pop(
                'model_id_or_path',
                cfg.pop('pretrained', None))
            if model_id_or_path is None:
                raise ValueError(
                    'mapanything_model_cfg must provide model_id_or_path (or pretrained) '
                    'when load_type="from_pretrained"')
            from_pretrained_kwargs = dict(cfg.pop('from_pretrained_kwargs', {}))
            use_local_dinov2_patch = force_local_torch_hub or bool(
                from_pretrained_kwargs.get('local_files_only', False))
            if use_local_dinov2_patch:
                dinov2_repo_dir = _find_local_dinov2_repo(local_dinov2_repo)
                patch_ctx = _patch_torch_hub_offline_for_dinov2(dinov2_repo_dir)
            else:
                patch_ctx = nullcontext()
            with patch_ctx:
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
            if not hasattr(module, factory_fn):
                raise AttributeError(
                    f'Cannot find factory function {factory_fn} in module {factory_module}')
            model = getattr(module, factory_fn)(**factory_kwargs)
        else:
            raise ValueError(
                f'Unsupported mapanything load_type={load_type}. '
                'Use one of ["from_pretrained", "init_model_from_config", "factory"]')

        if cfg:
            warnings.warn(
                f'Unused keys in mapanything_model_cfg after model build: {sorted(cfg.keys())}',
                stacklevel=2)
        return model

    def _build_mapanything_components(self):
        with _prepend_sys_path(self.repo_root):
            from mapanything.models import MapAnything, init_model_from_config
            from mapanything.utils.image import preprocess_inputs
            from mapanything.utils.inference import preprocess_input_views_for_inference

        self._mapanything_cls = MapAnything
        self._init_model_from_config = init_model_from_config
        self._preprocess_inputs = preprocess_inputs
        self._preprocess_input_views_for_inference = preprocess_input_views_for_inference
        self.map_model = self._build_model_from_cfg(self.mapanything_model_cfg)

        if self.strip_to_feature_mode:
            self._strip_mapanything_model(self.map_model)

    @staticmethod
    def _ensure_view_tensor(view, key, expected_shape_suffix, sample_idx, view_idx):
        value = view.get(key, None)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f'sample={sample_idx} view={view_idx} missing tensor key "{key}", '
                f'got type={type(value)}')
        if value.dim() != len(expected_shape_suffix):
            raise ValueError(
                f'sample={sample_idx} view={view_idx} key="{key}" dim mismatch: '
                f'expected dim={len(expected_shape_suffix)}, got shape={tuple(value.shape)}')
        if tuple(value.shape) != tuple(expected_shape_suffix):
            raise ValueError(
                f'sample={sample_idx} view={view_idx} key="{key}" shape mismatch: '
                f'expected={tuple(expected_shape_suffix)}, got={tuple(value.shape)}')
        if not torch.isfinite(value).all():
            raise ValueError(
                f'sample={sample_idx} view={view_idx} key="{key}" contains non-finite values')
        return value

    def _validate_preprocessed_views_for_geometry(self, views_before, views_after, sample_idx):
        if len(views_before) != len(views_after):
            raise ValueError(
                f'sample={sample_idx}: view count mismatch after preprocess_input_views_for_inference, '
                f'before={len(views_before)}, after={len(views_after)}')

        for view_idx, (before_view, after_view) in enumerate(zip(views_before, views_after)):
            expects_ray = ('intrinsics' in before_view) or ('ray_directions' in before_view)
            expects_depth = ('depth_z' in before_view)
            expects_pose = ('camera_poses' in before_view)

            image = after_view.get('img', None)
            if not isinstance(image, torch.Tensor) or image.dim() != 4:
                raise ValueError(
                    f'sample={sample_idx} view={view_idx}: processed image must be [1,3,H,W], '
                    f'got type={type(image)} shape={getattr(image, "shape", None)}')
            if image.shape[0] != 1:
                raise ValueError(
                    f'sample={sample_idx} view={view_idx}: processed image batch dim must be 1, '
                    f'got {tuple(image.shape)}')
            h, w = int(image.shape[-2]), int(image.shape[-1])

            if expects_ray or expects_depth:
                self._ensure_view_tensor(
                    after_view,
                    key='ray_directions_cam',
                    expected_shape_suffix=(1, h, w, 3),
                    sample_idx=sample_idx,
                    view_idx=view_idx)

            if expects_depth:
                self._ensure_view_tensor(
                    after_view,
                    key='depth_along_ray',
                    expected_shape_suffix=(1, h, w, 1),
                    sample_idx=sample_idx,
                    view_idx=view_idx)

            if expects_pose:
                self._ensure_view_tensor(
                    after_view,
                    key='camera_pose_quats',
                    expected_shape_suffix=(1, 4),
                    sample_idx=sample_idx,
                    view_idx=view_idx)
                self._ensure_view_tensor(
                    after_view,
                    key='camera_pose_trans',
                    expected_shape_suffix=(1, 3),
                    sample_idx=sample_idx,
                    view_idx=view_idx)

    def _normalize_is_metric_scale(self, processed_views, sample_idx):
        for view_idx, view in enumerate(processed_views):
            if 'is_metric_scale' not in view:
                continue
            value = view['is_metric_scale']
            image = view.get('img', None)
            if not isinstance(image, torch.Tensor) or image.dim() != 4:
                raise ValueError(
                    f'sample={sample_idx} view={view_idx}: expected image tensor [1,3,H,W] '
                    f'before normalizing is_metric_scale, got {type(image)} '
                    f"shape={getattr(image, 'shape', None)}")
            target_device = image.device
            if isinstance(value, torch.Tensor):
                metric = value.to(device=target_device, dtype=torch.bool)
            elif isinstance(value, np.ndarray):
                metric = torch.as_tensor(value, device=target_device, dtype=torch.bool)
            elif isinstance(value, (list, tuple)):
                metric = torch.as_tensor(value, device=target_device, dtype=torch.bool)
            elif isinstance(value, (bool, np.bool_)):
                metric = torch.tensor([bool(value)], device=target_device, dtype=torch.bool)
            else:
                raise TypeError(
                    f'sample={sample_idx} view={view_idx}: unsupported is_metric_scale type {type(value)}')

            metric = metric.reshape(-1)
            if metric.numel() == 0:
                raise ValueError(
                    f'sample={sample_idx} view={view_idx}: is_metric_scale cannot be empty')
            if metric.numel() != 1:
                raise ValueError(
                    f'sample={sample_idx} view={view_idx}: is_metric_scale must contain exactly one value, '
                    f'got shape={tuple(metric.shape)}')
            view['is_metric_scale'] = metric

    def _strip_mapanything_model(self, model):
        for module_name in self._STRIP_MODULES:
            if hasattr(model, module_name):
                delattr(model, module_name)

    def _freeze_model_if_needed(self):
        if not self.freeze:
            return
        if hasattr(self.map_model, 'parameters'):
            for parameter in self.map_model.parameters():
                parameter.requires_grad = False
        if hasattr(self.map_model, 'eval'):
            self.map_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze and hasattr(self.map_model, 'eval'):
            # Keep external encoder frozen in eval mode regardless of parent mode.
            self.map_model.eval()
        if self.anyup_enabled and self.anyup_freeze and self.anyup_model is not None:
            self.anyup_model.eval()
        return self

    def _get_model_device(self):
        for parameter in self.map_model.parameters():
            return parameter.device
        for buffer in self.map_model.buffers():
            return buffer.device
        return torch.device('cpu')

    def _ensure_model_device(self, target_device):
        current_device = self._get_model_device()
        if current_device != target_device:
            self.map_model.to(target_device)

    def _get_anyup_device(self):
        if self.anyup_model is None:
            return torch.device('cpu')
        for parameter in self.anyup_model.parameters():
            return parameter.device
        for buffer in self.anyup_model.buffers():
            return buffer.device
        return torch.device('cpu')

    def _ensure_anyup_device(self, target_device):
        if self.anyup_model is None:
            return
        current_device = self._get_anyup_device()
        if current_device != target_device:
            self.anyup_model.to(target_device)

    def _build_anyup_components(self):
        self._parse_anyup_runtime_cfg()
        if self.anyup_mode == 'bilinear':
            self.anyup_model = None
            return

        anyup_repo_root = self.anyup_cfg.get('repo_root', None)
        if not anyup_repo_root:
            raise ValueError('anyup_cfg.repo_root must be provided when anyup_cfg.enabled=True and mode="anyup"')
        anyup_repo_root = os.path.abspath(os.path.expanduser(anyup_repo_root))
        if not os.path.isdir(anyup_repo_root):
            raise FileNotFoundError(f'AnyUp repo_root does not exist: {anyup_repo_root}')

        variant = str(self.anyup_cfg.get('variant', 'anyup_multi_backbone'))
        if variant not in self._ANYUP_CHECKPOINT_NAMES:
            raise ValueError(
                f'anyup_cfg.variant must be one of {tuple(self._ANYUP_CHECKPOINT_NAMES.keys())}, '
                f'got {variant}')

        checkpoint_path = self.anyup_cfg.get('checkpoint_path', None)
        if not checkpoint_path:
            checkpoint_path = os.path.join(
                anyup_repo_root,
                'checkpoints',
                self._ANYUP_CHECKPOINT_NAMES[variant])
        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        allow_online_download = bool(
            self.anyup_cfg.get('allow_online_download_if_missing', False))

        if not os.path.isfile(checkpoint_path):
            if not allow_online_download:
                raise FileNotFoundError(
                    f'AnyUp checkpoint not found: {checkpoint_path}. '
                    'Set anyup_cfg.allow_online_download_if_missing=True or place checkpoint locally.')
            checkpoint_url = self._ANYUP_CHECKPOINT_URLS.get(variant, None)
            if checkpoint_url is None:
                raise ValueError(f'No checkpoint URL configured for AnyUp variant={variant}')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.hub.download_url_to_file(
                checkpoint_url,
                checkpoint_path,
                progress=True)

        with _prepend_sys_path(anyup_repo_root):
            from anyup.model import AnyUp
            self.anyup_model = AnyUp()
            if bool(self.anyup_cfg.get('use_natten', False)):
                from anyup.layers import setup_cross_attention_block
                self.anyup_model.cross_decode = setup_cross_attention_block(
                    use_natten=True,
                    qk_dim=self.anyup_model.cross_decode.cross_attn.attention.embed_dim,
                    num_heads=4,
                    window_ratio=self.anyup_model.cross_decode.window_ratio,
                    use_params_from=self.anyup_model.cross_decode,
                )

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
            state_dict = state_dict['state_dict']
        incompatible = self.anyup_model.load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(incompatible, 'missing_keys', []))
        unexpected_keys = list(getattr(incompatible, 'unexpected_keys', []))
        if missing_keys or unexpected_keys:
            warnings.warn(
                'AnyUp checkpoint loaded with key mismatch: '
                f'missing={missing_keys}, unexpected={unexpected_keys}',
                stacklevel=2)

        self.anyup_model.eval()
        if self.anyup_freeze:
            for parameter in self.anyup_model.parameters():
                parameter.requires_grad = False

    def _parse_anyup_runtime_cfg(self):
        self.anyup_freeze = bool(self.anyup_cfg.get('freeze', True))
        self.anyup_q_chunk_size = self.anyup_cfg.get('q_chunk_size', None)
        if self.anyup_q_chunk_size is not None:
            self.anyup_q_chunk_size = int(self.anyup_q_chunk_size)
            if self.anyup_q_chunk_size <= 0:
                raise ValueError(
                    f'anyup_cfg.q_chunk_size must be positive or None, got {self.anyup_q_chunk_size}')
        self.anyup_view_batch_size = self.anyup_cfg.get('view_batch_size', None)
        if self.anyup_view_batch_size is not None:
            self.anyup_view_batch_size = int(self.anyup_view_batch_size)
            if self.anyup_view_batch_size <= 0:
                self.anyup_view_batch_size = None
        self.anyup_output_channels = self.anyup_cfg.get('output_channels', None)
        if self.anyup_output_channels is not None:
            self.anyup_output_channels = int(self.anyup_output_channels)
            if self.anyup_output_channels <= 0:
                self.anyup_output_channels = None
        if self.anyup_output_channels is not None:
            output_in_channels = int(self.anyup_cfg.get('output_in_channels', 1024))
            if output_in_channels <= 0:
                raise ValueError(
                    f'anyup_cfg.output_in_channels must be positive, got {output_in_channels}')
            self.anyup_output_proj = nn.Conv2d(
                output_in_channels,
                self.anyup_output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True)

        pyramid_cfg = dict(self.anyup_cfg.get('pyramid', {}))
        output_divisors = pyramid_cfg.get('output_divisors', [4, 8, 16, 32])
        if isinstance(output_divisors, tuple):
            output_divisors = list(output_divisors)
        if not isinstance(output_divisors, list) or len(output_divisors) == 0:
            raise ValueError(
                'anyup_cfg.pyramid.output_divisors must be a non-empty list of integers')
        self.anyup_pyramid_divisors = [int(v) for v in output_divisors]
        if any(v <= 0 for v in self.anyup_pyramid_divisors):
            raise ValueError(
                f'anyup_cfg.pyramid.output_divisors must be positive, got {self.anyup_pyramid_divisors}')
        self.anyup_pyramid_num_levels = int(
            pyramid_cfg.get('num_levels', len(self.anyup_pyramid_divisors)))
        if self.anyup_pyramid_num_levels <= 0:
            raise ValueError(
                f'anyup_cfg.pyramid.num_levels must be positive, got {self.anyup_pyramid_num_levels}')
        if len(self.anyup_pyramid_divisors) < self.anyup_pyramid_num_levels:
            raise ValueError(
                'anyup_cfg.pyramid.output_divisors length must be >= num_levels, '
                f'got divisors={self.anyup_pyramid_divisors}, num_levels={self.anyup_pyramid_num_levels}')
        self.anyup_pyramid_divisors = self.anyup_pyramid_divisors[:self.anyup_pyramid_num_levels]
        self.anyup_pyramid_downsample_mode = str(
            pyramid_cfg.get('downsample_mode', 'area'))
        self.anyup_pyramid_align_corners = bool(
            pyramid_cfg.get('align_corners', False))
        self.anyup_upsample_output_divisor = int(
            self.anyup_cfg.get('upsample_output_divisor', 1))
        if self.anyup_upsample_output_divisor <= 0:
            raise ValueError(
                'anyup_cfg.upsample_output_divisor must be positive, '
                f'got {self.anyup_upsample_output_divisor}')

    def _downsample_anyup_feature(self, feat_2d, target_hw):
        if feat_2d.shape[-2:] == target_hw:
            return feat_2d
        mode = self.anyup_pyramid_downsample_mode
        # CUDA area downsample relies on adaptive_avg_pool2d with int32 indexing.
        # Split the batch if total elements exceed int32 range.
        if mode == 'area' and feat_2d.numel() >= (2**31 - 1):
            per_sample_elems = feat_2d[0].numel()
            max_batch = max((2**31 - 1) // max(per_sample_elems, 1), 1)
            chunks = []
            for start in range(0, feat_2d.shape[0], max_batch):
                end = min(start + max_batch, feat_2d.shape[0])
                chunks.append(F.interpolate(feat_2d[start:end], size=target_hw, mode=mode))
            return torch.cat(chunks, dim=0)
        if mode in ('nearest', 'area', 'nearest-exact'):
            return F.interpolate(feat_2d, size=target_hw, mode=mode)
        return F.interpolate(
            feat_2d,
            size=target_hw,
            mode=mode,
            align_corners=self.anyup_pyramid_align_corners)

    @staticmethod
    def _view_image_from_processed_view(view):
        image = view.get('img', None)
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                'Processed view must contain tensor image under key "img", '
                f'got type={type(image)}')
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4:
            raise ValueError(
                f'Processed view image must have shape [B,3,H,W] or [3,H,W], got {tuple(image.shape)}')
        if image.shape[0] != 1:
            raise ValueError(
                f'AnyUp per-view image batch must be 1, got shape {tuple(image.shape)}')
        return image

    def _compute_anyup_level_sizes(self, height, width):
        sizes = []
        for divisor in self.anyup_pyramid_divisors: 
            out_h = max((int(height) + divisor - 1) // divisor, 1)
            out_w = max((int(width) + divisor - 1) // divisor, 1)
            sizes.append((out_h, out_w))
        return sizes

    def _compute_anyup_upsample_size(self, height, width):
        divisor = int(self.anyup_upsample_output_divisor)
        if divisor <= 1:
            return int(height), int(width)
        out_h = max((int(height) + divisor - 1) // divisor, 1)
        out_w = max((int(width) + divisor - 1) // divisor, 1)
        return out_h, out_w

    def _run_anyup_model(self, hr_image, lr_feat, output_size):
        if self.anyup_mode == 'bilinear':
            if not isinstance(lr_feat, torch.Tensor) or lr_feat.dim() != 4:
                raise ValueError(
                    'Bilinear AnyUp fallback expects lr_feat Tensor[N, C, H, W], '
                    f'got type={type(lr_feat)} shape={getattr(lr_feat, "shape", None)}')
            return F.interpolate(
                lr_feat,
                size=output_size,
                mode='bilinear',
                align_corners=self.anyup_pyramid_align_corners)
        if self.anyup_model is None:
            raise RuntimeError('AnyUp model is not initialized')
        if self.anyup_freeze:
            with torch.no_grad():
                return self.anyup_model(
                    hr_image,
                    lr_feat,
                    output_size=output_size,
                    q_chunk_size=self.anyup_q_chunk_size)
        return self.anyup_model(
            hr_image,
            lr_feat,
            output_size=output_size,
            q_chunk_size=self.anyup_q_chunk_size)

    def _project_anyup_level_feat(self, feat_2d):
        if self.anyup_output_proj is None:
            return feat_2d
        proj_weight = self.anyup_output_proj.weight
        if feat_2d.shape[1] != proj_weight.shape[1]:
            raise ValueError(
                'AnyUp projection channel mismatch: '
                f'feat_channels={feat_2d.shape[1]} vs proj_in_channels={proj_weight.shape[1]}')
        in_dtype = feat_2d.dtype
        if in_dtype != proj_weight.dtype:
            feat_2d = feat_2d.to(dtype=proj_weight.dtype)
        feat_2d = self.anyup_output_proj(feat_2d)
        if feat_2d.dtype != in_dtype:
            feat_2d = feat_2d.to(dtype=in_dtype)
        return feat_2d

    def _iter_anyup_view_chunks(self, view_indices):
        if self.anyup_view_batch_size is None or self.anyup_view_batch_size >= len(view_indices):
            yield list(view_indices)
            return
        for start in range(0, len(view_indices), self.anyup_view_batch_size):
            end = min(start + self.anyup_view_batch_size, len(view_indices))
            yield list(view_indices[start:end])

    def _build_anyup_pyramid_per_sample(self, sample_feat, processed_views, total_views, sample_idx):
        if not isinstance(sample_feat, torch.Tensor) or sample_feat.dim() != 5:
            raise ValueError(
                'sample_feat must be Tensor[1, TN, C, H, W] before AnyUp pyramid conversion, '
                f'got type={type(sample_feat)} shape={getattr(sample_feat, "shape", None)}')
        if sample_feat.shape[0] != 1:
            raise ValueError(
                f'AnyUp per-sample path requires batch dim 1, got {tuple(sample_feat.shape)}')
        if sample_feat.shape[1] != total_views:
            raise ValueError(
                f'AnyUp TN mismatch at sample {sample_idx}: '
                f'feature TN={sample_feat.shape[1]} vs expected {total_views}')
        if len(processed_views) != total_views:
            raise ValueError(
                f'processed_views length mismatch at sample {sample_idx}: '
                f'got {len(processed_views)}, expected {total_views}')

        view_images = []
        view_anyup_target_hws = []
        view_level_sizes = []
        for view_idx in range(total_views):
            hr_image = self._view_image_from_processed_view(processed_views[view_idx])
            image_hw = (int(hr_image.shape[-2]), int(hr_image.shape[-1]))
            target_hw = self._compute_anyup_upsample_size(*image_hw)
            level_sizes = self._compute_anyup_level_sizes(*image_hw)
            if any(level_h > target_hw[0] or level_w > target_hw[1]
                   for level_h, level_w in level_sizes):
                raise ValueError(
                    'AnyUp pyramid level is larger than upsample target. '
                    f'Got image_hw={image_hw}, upsample_output_divisor={self.anyup_upsample_output_divisor}, '
                    f'upsample_target={target_hw}, level_sizes={level_sizes}. '
                    'Increase upsample resolution or use larger output_divisors.')
            view_images.append(hr_image)
            view_anyup_target_hws.append(target_hw)
            view_level_sizes.append(level_sizes)

        if total_views == 0:
            raise RuntimeError(f'AnyUp received empty views at sample {sample_idx}')

        ref_level_sizes = view_level_sizes[0]
        for view_idx in range(1, total_views):
            if view_level_sizes[view_idx] != ref_level_sizes:
                raise ValueError(
                    f'AnyUp pyramid requires same processed view size for stacking, but got '
                    f'{view_level_sizes[view_idx]} vs {ref_level_sizes} at sample {sample_idx}')

        num_levels = len(ref_level_sizes)
        per_level_feats = [None for _ in range(num_levels)]
        hw_groups = {}
        for view_idx, target_hw in enumerate(view_anyup_target_hws):
            hw_groups.setdefault(target_hw, []).append(view_idx)

        # Fixed-size preprocessing is the common path; pre-concatenate once to reduce cat overhead.
        merged_view_images = None
        if len(hw_groups) == 1:
            merged_view_images = torch.cat(view_images, dim=0)

        for target_hw, group_indices in hw_groups.items():
            for chunk_indices in self._iter_anyup_view_chunks(group_indices):
                if merged_view_images is not None:
                    hr_batch = merged_view_images[chunk_indices]
                else:
                    hr_batch = torch.cat([view_images[idx] for idx in chunk_indices], dim=0)
                lr_batch = sample_feat[0, chunk_indices]  # [N_chunk, C, h, w]
                hr_batch_feat = self._run_anyup_model(
                    hr_batch,
                    lr_batch,
                    output_size=target_hw)
                if hr_batch_feat.dim() != 4 or hr_batch_feat.shape[0] != len(chunk_indices):
                    raise ValueError(
                        f'AnyUp batch output shape mismatch at sample {sample_idx}: '
                        f'expected [N={len(chunk_indices)}, C, H, W], got {tuple(hr_batch_feat.shape)}')

                # Vectorized pyramid generation over a chunk: significantly fewer tiny ops.
                for level_idx, size in enumerate(ref_level_sizes):
                    level_feat = self._downsample_anyup_feature(hr_batch_feat, size)
                    level_feat = self._project_anyup_level_feat(level_feat)
                    if per_level_feats[level_idx] is None:
                        per_level_feats[level_idx] = level_feat.new_empty(
                            (total_views, *level_feat.shape[1:]))
                    per_level_feats[level_idx][chunk_indices] = level_feat
                del hr_batch_feat

        if per_level_feats is None:
            raise RuntimeError(f'AnyUp produced no level features at sample {sample_idx}')
        sample_levels = []
        for level_idx, level_views in enumerate(per_level_feats):
            if level_views is None:
                raise RuntimeError(
                    f'AnyUp level {level_idx} is empty at sample {sample_idx}')
            level_tensor = level_views.unsqueeze(0)  # [1, TN, C, H, W]
            sample_levels.append(level_tensor)
        return sample_levels

    def _move_processed_views_to_model_device(self, processed_views):
        ignore_keys = {'instance', 'idx', 'true_shape', 'data_norm_type'}
        target_device = self._get_model_device()

        def _move_to_device(value):
            if isinstance(value, dict):
                return {k: _move_to_device(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_move_to_device(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_move_to_device(v) for v in value)
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

    def _run_model(self, processed_views):
        if self.freeze:
            with torch.no_grad():
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

    def forward(self, img, points=None, img_metas=None, mapanything_extra=None):
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            raise ValueError(
                f'img must be Tensor[B, TN, C, H, W], got type={type(img)} '
                f'shape={getattr(img, "shape", None)}')

        batch_size, total_views = img.shape[:2]
        if img_metas is None:
            img_metas = [{} for _ in range(batch_size)]
        if not isinstance(img_metas, list) or len(img_metas) != batch_size:
            raise ValueError(
                f'img_metas must be list of length B={batch_size}, got {type(img_metas)}')

        self._ensure_model_device(img.device)
        if self.anyup_enabled:
            self._ensure_anyup_device(img.device)
        self._sync_geometric_input_cfg_for_mode()

        batch_views = self.input_adapter(
            img=img,
            points=points,
            img_metas=img_metas,
            mapanything_extra=mapanything_extra,
        )

        per_sample_feats = []
        per_sample_pyramids = []
        patch_size = self._get_patch_size()
        for sample_idx, sample_views in enumerate(batch_views):
            processed_views = self._preprocess_inputs(
                sample_views,
                **self.mapanything_preprocess_cfg)
            processed_views_before_inference = [dict(v) for v in processed_views]
            processed_views = self._preprocess_input_views_for_inference(processed_views)
            self._normalize_is_metric_scale(
                processed_views,
                sample_idx=sample_idx)
            self._validate_preprocessed_views_for_geometry(
                processed_views_before_inference,
                processed_views,
                sample_idx=sample_idx)
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
            if self.anyup_enabled:
                sample_levels = self._build_anyup_pyramid_per_sample(
                    sample_feat,
                    processed_views=processed_views,
                    total_views=total_views,
                    sample_idx=sample_idx)
                per_sample_pyramids.append(sample_levels)
            else:
                per_sample_feats.append(sample_feat)

        if self.anyup_enabled:
            if len(per_sample_pyramids) != batch_size:
                raise RuntimeError(
                    f'AnyUp sample count mismatch: got {len(per_sample_pyramids)}, expected {batch_size}')
            num_levels = len(per_sample_pyramids[0])
            output_levels = []
            for level_idx in range(num_levels):
                level_batch = torch.cat(
                    [sample_levels[level_idx] for sample_levels in per_sample_pyramids],
                    dim=0)
                if self.strict_shapes and level_batch.shape[:2] != (batch_size, total_views):
                    raise ValueError(
                        f'AnyUp output level {level_idx} shape mismatch: '
                        f'expected [B={batch_size}, TN={total_views}, ...], got {tuple(level_batch.shape)}')
                if self.strict_shapes and level_batch.device != img.device:
                    raise ValueError(
                        f'AnyUp output level {level_idx} device mismatch: '
                        f'expected {img.device}, got {level_batch.device}')
                output_levels.append(level_batch.contiguous() if self.expect_contiguous else level_batch)
            return output_levels

        output = torch.cat(per_sample_feats, dim=0)
        if self.strict_shapes and output.shape[:2] != (batch_size, total_views):
            raise ValueError(
                f'Output shape mismatch: expected [B={batch_size}, TN={total_views}, ...], '
                f'got {tuple(output.shape)}')
        if self.strict_shapes and output.device != img.device:
            raise ValueError(
                f'Output device mismatch: expected {img.device}, got {output.device}')

        return output.contiguous() if self.expect_contiguous else output
