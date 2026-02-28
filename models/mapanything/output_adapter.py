import math

import torch


class MapAnythingOutputAdapter:
    """Normalize MapAnything outputs to OPUS tensor format [B, TN, C, H, W]."""

    _DICT_KEYS = (
        'feat',
        'features',
        'image_features',
        'encoder_features',
        'fused_features',
    )

    def __init__(self, patch_size=14):
        self.patch_size = int(patch_size)

    def _extract_from_dict(self, output):
        for key in self._DICT_KEYS:
            if key in output:
                return output[key]
        raise KeyError(
            f'Unsupported output dict keys {list(output.keys())}. '
            f'Expected one of {list(self._DICT_KEYS)}')

    def _tokens_to_map(self, tokens, image_hw, patch_size):
        if not isinstance(tokens, torch.Tensor) or tokens.dim() != 3:
            raise TypeError(f'tokens must be Tensor[B,N,C], got {type(tokens)}')
        if image_hw is None:
            raise ValueError('image_hw is required to restore token features to 2D maps')

        batch_size, num_tokens, channels = tokens.shape
        image_h, image_w = image_hw
        patch_h = max(int(image_h // patch_size), 1)
        patch_w = max(int(image_w // patch_size), 1)
        target_tokens = patch_h * patch_w

        if num_tokens == target_tokens + 1:
            tokens = tokens[:, 1:, :]
            num_tokens -= 1
        if num_tokens != target_tokens:
            root = int(math.sqrt(num_tokens))
            if root * root == num_tokens:
                patch_h = root
                patch_w = root
                target_tokens = num_tokens
            else:
                raise ValueError(
                    f'Cannot map token features to 2D: token_count={num_tokens}, '
                    f'expected around {target_tokens} from image_hw={image_hw} and patch_size={patch_size}')

        feat = tokens.reshape(batch_size, patch_h, patch_w, channels)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return feat

    def _ensure_view_feat(self, feat, image_hw=None, patch_size=None):
        patch_size = self.patch_size if patch_size is None else int(patch_size)
        if not isinstance(feat, torch.Tensor):
            raise TypeError(f'Each feature must be Tensor, got {type(feat)}')

        if feat.dim() == 4:
            return feat
        if feat.dim() == 3:
            # Token output: [B, N, C] -> [B, C, H, W]
            return self._tokens_to_map(feat, image_hw=image_hw, patch_size=patch_size)

        raise ValueError(
            f'Unsupported feature tensor dim={feat.dim()}, expected 3/4/5')

    def _to_b_tn_chw(self,
                     output,
                     batch_size,
                     total_views,
                     image_hw=None,
                     patch_size=None):
        if isinstance(output, dict):
            output = self._extract_from_dict(output)

        if isinstance(output, torch.Tensor):
            if output.dim() == 5:
                if output.shape[0] != batch_size:
                    raise ValueError(
                        f'Output batch mismatch: expected B={batch_size}, got {output.shape[0]}')
                if output.shape[1] != total_views:
                    raise ValueError(
                        f'Output TN mismatch: expected TN={total_views}, got {output.shape[1]}')
                return output

            if output.dim() == 4:
                if output.shape[0] == batch_size * total_views:
                    return output.reshape(batch_size, total_views, *output.shape[1:])
                if total_views == 1 and output.shape[0] == batch_size:
                    return output[:, None, ...]
                raise ValueError(
                    f'Cannot interpret 4D output shape {tuple(output.shape)} '
                    f'as [B={batch_size}, TN={total_views}, C, H, W]')

            if output.dim() == 3:
                view_feat = self._ensure_view_feat(
                    output, image_hw=image_hw, patch_size=patch_size)
                if total_views == 1:
                    return view_feat[:, None, ...]
                raise ValueError(
                    '3D output tensor [B,N,C] only supported when total_views==1 or list output is provided')

            raise ValueError(
                f'Unsupported tensor output dim={output.dim()}, expected 3/4/5')

        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise ValueError('Empty output list is not supported')
            if len(output) == 1:
                return self._to_b_tn_chw(
                    output[0],
                    batch_size=batch_size,
                    total_views=total_views,
                    image_hw=image_hw,
                    patch_size=patch_size)

            if len(output) != total_views:
                raise ValueError(
                    f'List output length mismatch: expected {total_views} views, got {len(output)}')

            view_feats = [
                self._ensure_view_feat(feat, image_hw=image_hw, patch_size=patch_size)
                for feat in output
            ]
            for view_idx, feat in enumerate(view_feats):
                if feat.shape[0] != batch_size:
                    raise ValueError(
                        f'View {view_idx} batch mismatch: expected B={batch_size}, got {feat.shape[0]}')
            return torch.stack(view_feats, dim=1)

        raise TypeError(f'Unsupported output type {type(output)}')

    def __call__(self,
                 output,
                 batch_size,
                 total_views,
                 image_hw=None,
                 patch_size=None):
        feat = self._to_b_tn_chw(
            output,
            batch_size=batch_size,
            total_views=total_views,
            image_hw=image_hw,
            patch_size=patch_size)
        if not feat.is_contiguous():
            feat = feat.contiguous()
        return feat

