#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images, rgb as denorm_rgb


def strip_mapanything_model(model: torch.nn.Module) -> torch.nn.Module:
    modules_to_remove = [
        "dpt_regressor_head",
        "dense_head",
        "pose_head",
        "scale_head",
        "dense_adaptor",
        "pose_adaptor",
        "scale_adaptor",
    ]
    for module_name in modules_to_remove:
        if hasattr(model, module_name):
            delattr(model, module_name)
    return model


def find_local_hf_snapshot(repo_id: str) -> Path:
    hub_cache = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "hub",
        )
    )
    snapshot_root = hub_cache / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshot_root.exists():
        raise FileNotFoundError(f"Local HuggingFace cache not found: {snapshot_root}")

    snapshots = sorted(
        [p for p in snapshot_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for snapshot_dir in snapshots:
        if (snapshot_dir / "config.json").exists() and (
            (snapshot_dir / "model.safetensors").exists()
            or (snapshot_dir / "pytorch_model.bin").exists()
        ):
            return snapshot_dir
    raise FileNotFoundError(f"No valid model snapshot found under: {snapshot_root}")


def find_local_dinov2_repo() -> Path:
    hub_dir = Path(torch.hub.get_dir())
    candidates = [
        hub_dir / "facebookresearch_dinov2_main",
        hub_dir / "facebookresearch_dinov2_master",
    ]
    for repo_dir in candidates:
        if repo_dir.exists():
            return repo_dir
    raise FileNotFoundError(
        "Local DINOv2 torch hub repo not found. Expected one of: "
        + ", ".join(str(x) for x in candidates)
    )


def patch_torch_hub_offline_for_dinov2(dinov2_repo_dir: Path) -> None:
    original_torch_hub_load = torch.hub.load

    def offline_torch_hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            kwargs.pop("force_reload", None)
            return original_torch_hub_load(
                str(dinov2_repo_dir), model, *args, source="local", **kwargs
            )
        return original_torch_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = offline_torch_hub_load


def load_mapanything_model(
    repo_id: str,
    local_model_dir: Optional[str],
    allow_online: bool,
    device: torch.device,
) -> torch.nn.Module:
    if not allow_online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        patch_torch_hub_offline_for_dinov2(find_local_dinov2_repo())

    model_source = None
    if local_model_dir:
        candidate = Path(local_model_dir).expanduser().resolve()
        if candidate.exists():
            model_source = candidate
        else:
            raise FileNotFoundError(f"--mapanything-local-dir not found: {candidate}")
    elif not allow_online:
        model_source = find_local_hf_snapshot(repo_id)

    if model_source is None:
        model = MapAnything.from_pretrained(repo_id, local_files_only=not allow_online)
        source_desc = repo_id
    else:
        model = MapAnything.from_pretrained(
            str(model_source), local_files_only=not allow_online
        )
        source_desc = str(model_source)

    print(f"[Info] MapAnything source: {source_desc}")
    return strip_mapanything_model(model).to(device).eval()


def load_anyup_model(
    anyup_repo: str,
    anyup_variant: str,
    device: torch.device,
    allow_online: bool,
) -> torch.nn.Module:
    repo = str(Path(anyup_repo).expanduser().resolve())
    checkpoint_names = {
        "anyup": "anyup_paper.pth",
        "anyup_multi_backbone": "anyup_multi_backbone.pth",
    }
    if allow_online:
        try:
            model = torch.hub.load(
                repo, anyup_variant, source="local", pretrained=True, device=str(device)
            )
            print(f"[Info] AnyUp variant: {anyup_variant} (pretrained)")
        except Exception as exc:
            print(
                "[Warn] Failed to load pretrained AnyUp checkpoint; "
                f"falling back to random init. Error: {exc}"
            )
            model = torch.hub.load(
                repo, anyup_variant, source="local", pretrained=False, device=str(device)
            )
    else:
        model = torch.hub.load(
            repo, anyup_variant, source="local", pretrained=False, device=str(device)
        )
        ckpt_name = checkpoint_names[anyup_variant]
        ckpt_path = Path(torch.hub.get_dir()) / "checkpoints" / ckpt_name
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"[Info] AnyUp variant: {anyup_variant} (pretrained from local cache)")
        else:
            print(
                f"[Warn] Offline mode and local AnyUp checkpoint missing: {ckpt_path}. "
                "Using randomly initialized AnyUp."
            )
    return model.to(device).eval()


def joint_pca_rgb(
    lr_feat: torch.Tensor, hr_feat: torch.Tensor, max_fit_samples: int = 50000
) -> Tuple[torch.Tensor, torch.Tensor]:
    lr_flat = lr_feat.permute(1, 2, 0).reshape(-1, lr_feat.shape[0]).float().cpu()
    hr_flat = hr_feat.permute(1, 2, 0).reshape(-1, hr_feat.shape[0]).float().cpu()

    n_lr = lr_flat.shape[0]
    n_hr = hr_flat.shape[0]
    n_fit_lr = min(n_lr, max_fit_samples // 2)
    n_fit_hr = min(n_hr, max_fit_samples - n_fit_lr)

    idx_lr = torch.randperm(n_lr)[:n_fit_lr]
    idx_hr = torch.randperm(n_hr)[:n_fit_hr]
    fit = torch.cat([lr_flat[idx_lr], hr_flat[idx_hr]], dim=0)

    mean = fit.mean(dim=0, keepdim=True)
    centered = fit - mean
    q = min(16, centered.shape[0] - 1, centered.shape[1])
    _, _, v = torch.pca_lowrank(centered, q=max(3, q), center=False)
    pcs = v[:, :3]

    proj_lr = (lr_flat - mean) @ pcs
    proj_hr = (hr_flat - mean) @ pcs

    mins = torch.minimum(proj_lr.min(dim=0).values, proj_hr.min(dim=0).values)
    maxs = torch.maximum(proj_lr.max(dim=0).values, proj_hr.max(dim=0).values)
    denom = (maxs - mins).clamp_min(1e-6)

    proj_lr = (proj_lr - mins) / denom
    proj_hr = (proj_hr - mins) / denom

    lr_rgb = proj_lr.reshape(lr_feat.shape[1], lr_feat.shape[2], 3).clamp(0.0, 1.0)
    hr_rgb = proj_hr.reshape(hr_feat.shape[1], hr_feat.shape[2], 3).clamp(0.0, 1.0)
    return lr_rgb, hr_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upsample MapAnything features with AnyUp and visualize before/after PCA-RGB."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/root/wjl/OPUS_mmcv2/third_party/map-anything/test_images/rgb",
        help="Image folder or list path accepted by mapanything.utils.image.load_images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/wjl/anyup/outputs/mapanything_anyup_demo",
        help="Directory to save PCA visualization.",
    )
    parser.add_argument(
        "--mapanything-repo",
        type=str,
        default="facebook/map-anything-apache-v1",
        help="HF repo id for MapAnything.",
    )
    parser.add_argument(
        "--mapanything-local-dir",
        type=str,
        default=None,
        help="Optional local model snapshot directory for offline load.",
    )
    parser.add_argument(
        "--anyup-repo",
        type=str,
        default="/root/wjl/anyup",
        help="Local AnyUp repo path for torch.hub local loading.",
    )
    parser.add_argument(
        "--anyup-variant",
        type=str,
        default="anyup_multi_backbone",
        choices=["anyup", "anyup_multi_backbone"],
        help="Which AnyUp checkpoint entrypoint to use.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=1,
        help="Number of views to pass into MapAnything inference.",
    )
    parser.add_argument(
        "--q-chunk-size",
        type=int,
        default=256,
        help="Query chunk size for AnyUp to reduce memory usage.",
    )
    parser.add_argument(
        "--max-fit-samples",
        type=int,
        default=50000,
        help="Number of sampled pixels to fit PCA.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string. Use 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--allow-online",
        action="store_true",
        help="Allow downloading model weights when local cache is missing.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for PCA sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Info] Device: {device}")

    map_model = load_mapanything_model(
        repo_id=args.mapanything_repo,
        local_model_dir=args.mapanything_local_dir,
        allow_online=args.allow_online,
        device=device,
    )
    upsampler = load_anyup_model(
        args.anyup_repo, args.anyup_variant, device, allow_online=args.allow_online
    )

    views = load_images(args.images_dir)
    if len(views) == 0:
        raise RuntimeError(f"No images found in: {args.images_dir}")

    views = views[: args.num_views]
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        predictions = map_model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=None,
            use_amp=use_amp,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            confidence_percentile=10,
            use_multiview_confidence=False,
        )

        lr_feat = predictions[0][0].to(device)
        hr_image = views[0]["img"].to(device)
        hr_feat = upsampler(
            hr_image,
            lr_feat.unsqueeze(0),
            output_size=hr_image.shape[-2:],
            q_chunk_size=args.q_chunk_size,
        )[0]

    lr_rgb, hr_rgb = joint_pca_rgb(lr_feat, hr_feat, max_fit_samples=args.max_fit_samples)
    input_rgb = denorm_rgb(views[0]["img"][0], views[0]["data_norm_type"][0])

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mapanything_anyup_pca_before_after.png"

    h_lr, w_lr = lr_feat.shape[-2:]
    h_hr, w_hr = hr_feat.shape[-2:]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_rgb)
    axes[0].set_title(f"Input RGB ({h_hr}x{w_hr})")
    axes[0].axis("off")

    axes[1].imshow(lr_rgb.numpy())
    axes[1].set_title(f"Before Upsample (PCA-RGB)\nMapAnything {h_lr}x{w_lr}")
    axes[1].axis("off")

    axes[2].imshow(hr_rgb.numpy())
    axes[2].set_title(f"After Upsample (PCA-RGB)\nAnyUp {h_hr}x{w_hr}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[Info] Feature before upsample: {tuple(lr_feat.shape)}")
    print(f"[Info] Feature after  upsample: {tuple(hr_feat.shape)}")
    print(f"[Info] Saved visualization: {out_path}")


if __name__ == "__main__":
    main()
