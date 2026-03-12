#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import os.path as osp
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_TALK2DINO_ROOT = '/root/wjl/Talk2DINO'
DEFAULT_CONFIG_JSON = (
    '/root/wjl/Talk2DINO/tartanground_label_ae/configs/pca256_talk2dino_reg.json'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Encode one free-form text query into the Talk2DINO PCA256 latent space.')
    parser.add_argument('--text-query', required=True, help='Free-form text query to encode.')
    parser.add_argument('--config-json', default=DEFAULT_CONFIG_JSON,
                        help='Path to Talk2DINO text latent config json.')
    parser.add_argument('--talk2dino-root', default=DEFAULT_TALK2DINO_ROOT,
                        help='Path to Talk2DINO project root.')
    parser.add_argument('--device', default='cpu',
                        help='Torch device for text encoding, default cpu to avoid GPU contention.')
    parser.add_argument('--out-npy', required=True, help='Output path for normalized latent vector .npy')
    parser.add_argument('--out-json', required=True, help='Output path for metadata json')
    return parser.parse_args()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def resolve_device(device_str):
    device = torch.device(device_str)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA requested but unavailable: {device_str}')
    return device


def ensure_parent(path):
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_template_getter(talk2dino_root):
    template_path = osp.join(
        talk2dino_root,
        'src',
        'open_vocabulary_segmentation',
        'datasets',
        'templates.py')
    if not osp.isfile(template_path):
        raise FileNotFoundError(f'Template module not found: {template_path}')
    spec = importlib.util.spec_from_file_location('talk2dino_templates', template_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load templates module from {template_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_template


def get_templates(talk2dino_root, template_set):
    getter = load_template_getter(talk2dino_root)
    try:
        templates = getter(template_set)
    except Exception:
        templates = getter('subset')
    if not templates:
        raise RuntimeError(f'No prompt templates resolved for template_set={template_set}')
    return list(templates)


def load_projection_layer(talk2dino_root):
    if talk2dino_root not in sys.path:
        sys.path.insert(0, talk2dino_root)
    from src.model import ProjectionLayer
    return ProjectionLayer


def load_projection_model(talk2dino_root, config_path, weights_path, device):
    projection_layer = load_projection_layer(talk2dino_root)
    projection = projection_layer.from_config(str(config_path))
    state_dict = torch.load(str(weights_path), map_location='cpu')
    projection.load_state_dict(state_dict, strict=True)
    projection.to(device).eval()
    return projection


def load_clip_model(model_name, device):
    import clip

    model, _ = clip.load(model_name, device=device, jit=False)
    model.eval()
    return model


def encode_text_query(text_query, templates, clip_model, projection, pca_model, device):
    import clip

    prompts = [template.format(text_query) for template in templates]
    tokenized = clip.tokenize(prompts, truncate=True).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tokenized).float().mean(dim=0, keepdim=True)
        proj_feat = projection.project_clip_txt(text_feat)
        proj_feat = F.normalize(proj_feat, dim=-1)
    latent = pca_model.transform(proj_feat.cpu().numpy().astype(np.float32)).astype(np.float32)
    if latent.shape != (1, latent.shape[-1]):
        raise ValueError(f'Unexpected latent shape: {latent.shape}')
    latent_norm_before = float(np.linalg.norm(latent[0]))
    if latent_norm_before <= 1e-8:
        raise RuntimeError('Encoded latent has near-zero norm before normalization.')
    latent = latent / latent_norm_before
    return latent[0], prompts, latent_norm_before


def main():
    args = parse_args()

    config = load_json(args.config_json)
    talk2dino_root = osp.abspath(args.talk2dino_root)
    if not osp.isdir(talk2dino_root):
        raise FileNotFoundError(f'Talk2DINO root not found: {talk2dino_root}')

    device = resolve_device(args.device)
    config_path = osp.abspath(config['projection_config_path'])
    weights_path = osp.abspath(config['projection_weights_path'])
    pca_pickle_path = osp.abspath(config['pca_pickle_path'])

    for path in [config_path, weights_path, pca_pickle_path]:
        if not osp.exists(path):
            raise FileNotFoundError(f'Required Talk2DINO asset not found: {path}')

    templates = get_templates(talk2dino_root, config['template_set'])
    projection = load_projection_model(
        talk2dino_root=talk2dino_root,
        config_path=config_path,
        weights_path=weights_path,
        device=device)
    clip_model = load_clip_model(config['clip_model'], device=device)
    # Some environments pickle the PCA object against numpy._core paths.
    np_core = getattr(np, '_core', None)
    if np_core is None:
        np_core = np.core
    sys.modules.setdefault('numpy._core', np_core)
    np_core_multiarray = getattr(np_core, 'multiarray', None)
    if np_core_multiarray is not None:
        sys.modules.setdefault('numpy._core.multiarray', np_core_multiarray)
    with open(pca_pickle_path, 'rb') as handle:
        pca_model = pickle.load(handle)

    latent, prompts, latent_norm_before = encode_text_query(
        text_query=args.text_query,
        templates=templates,
        clip_model=clip_model,
        projection=projection,
        pca_model=pca_model,
        device=device)

    ensure_parent(args.out_npy)
    ensure_parent(args.out_json)
    np.save(args.out_npy, latent.astype(np.float32))

    meta = dict(
        text_query=args.text_query,
        prompt_count=len(prompts),
        prompts=prompts,
        clip_model=config['clip_model'],
        template_set=config['template_set'],
        config_json=osp.abspath(args.config_json),
        talk2dino_root=talk2dino_root,
        projection_config_path=config_path,
        projection_weights_path=weights_path,
        pca_pickle_path=pca_pickle_path,
        latent_dim=int(latent.shape[0]),
        latent_norm_before_normalize=latent_norm_before,
        latent_norm_after_normalize=float(np.linalg.norm(latent)),
        device=str(device),
    )
    with open(args.out_json, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f'[talk2dino_text] text="{args.text_query}" latent_dim={latent.shape[0]}')
    print(f'[talk2dino_text] npy={osp.abspath(args.out_npy)}')
    print(f'[talk2dino_text] json={osp.abspath(args.out_json)}')


if __name__ == '__main__':
    main()
