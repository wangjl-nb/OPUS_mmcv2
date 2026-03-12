# AGENTS.md - OPUS_mmcv2 Session Onboarding

This file is a fast-start guide for new sessions working in this repository.

## 1) Mandatory Read Order
1. `docs/TASK_WORKORDER_CURRENT.md`
2. `docs/PROJECT_GUIDE.md`
3. `docs/INDEX.md`
4. `docs/INSTALL_opus_gpt.md` (only when environment or launcher behavior needs verification)

If `docs/TASK_WORKORDER_CURRENT.md` is missing, copy from `docs/TASK_WORKORDER_TEMPLATE.md` first.
If a deleted historical doc is mentioned in old notes, prefer the current guide plus direct code inspection.

## 2) Environment and Working Directory
- Repository root: `/root/wjl/OPUS_mmcv2`
- Preferred environment: `opus-mmcv2`
- Basic check:
  - `conda run -n opus-mmcv2 python -c "import mmengine, mmcv, mmdet3d, models, loaders; print('ok')"`

## 3) Main Entry Points
- Distributed training launcher: `dist_train.sh`
- Training entry: `train.py`
- Validation entry: `val.py`
- Core fusion model files:
  - `models/opusv1_fusion/opus.py`
  - `models/opusv1_fusion/opus_head.py`
  - `models/opusv1_fusion/opus_transformer.py`
- Data pipeline files:
  - `loaders/pipelines/loading.py`
  - `loaders/pipelines/pack_occ3d_inputs.py`

## 4) Config Discovery Flow
- Do not assume a single canonical experiment config; this repo keeps multiple task-specific configs over time.
- For any new session:
  1. Start from the user-provided config path if one exists.
  2. Otherwise inspect the most recent active run under `outputs/<ModelType>/...` and read its frozen config copy first.
  3. Only then trace back to the source config in `configs/`.
- When comparing runs, prefer the frozen config inside the output directory over the source config in `configs/`, because source configs may have changed after the run started.

## 5) Run and Resume Conventions
- Standard 8-GPU run:
  - `conda run -n opus-mmcv2 bash dist_train.sh 8 <config>`
- `train.py` behavior:
  - `cfg.batch_size` is treated as global and divided by `world_size` for train/val/test dataloaders.
- Output directory pattern:
  - `outputs/<ModelType>/<config_name>_<YYYY-MM-DD/HH-MM-SS>/`
- Main log file pattern:
  - `outputs/.../<run_id>/<run_id>.log`
- Always inspect the frozen config copy in output directory before debugging mismatches.

## 6) Branch-Specific Constraints (Do Not Regress)
- Keep `LoadMapAnythingExtraFromDepth.filter_depth_by_pcrange` filtering in local LiDAR frame using `sensor2lidar` transform.
- Do not add extra post-`pts_lidar` filtering in depth-points path unless explicitly requested.
- Keep AnyUp offline-safe default in configs:
  - `allow_online_download_if_missing=False`
- Keep robust `mapanything_extra` batch/shape handling:
  - `models/mapanything/input_adapter.py`
  - `models/opusv1_fusion/opus.py`
  - `loaders/pipelines/loading.py`

## 7) Common Pitfalls
- `pc_voxel_size` and `voxel_size` are not interchangeable:
  - `pc_voxel_size`: point voxelization granularity for sparse point branch.
  - `voxel_size`: occupancy grid granularity used by occ mapping/evaluation.
- `sparse_shape` order is `[z, y, x]`, and must match `point_cloud_range` with `pc_voxel_size`.
- Early depth runs can show near-zero mIoU if `model.test_cfg.pts.score_thr` is too high.
- TPV-only mode can trigger DDP unused-parameter errors if unused point conv branches are not handled correctly.
- In pure external-encoder MapAnything paths, keep `ida_aug_conf.final_dim` and `mapanything_preprocess_cfg.size` aligned.
  - With `patch_size=14`, a nominal `640x640` image typically needs a patch-aligned `630x630` effective input size.
- In binary-occupiedness + feature-supervision heads, keep the score mode, positive-only feature supervision, and test-time score threshold logically aligned.
- Single-GPU val/test with `mapanything_extra` can diverge from offline multiframe behavior unless the sweeps loader is forced into offline mode.
- `img_encoder.anyup_cfg.enabled=True` with `mode='bilinear'` means “enable bilinear pyramid adapter”, not “instantiate AnyUp network”.
  - Do not turn `enabled` off unless you also redesign the image feature levels expected by the transformer.

## 8) Visualization And Demo Outputs
- Store generated demo and visualization artifacts under `/root/wjl/OPUS_mmcv2/demos`.
- Before using a visualization script, check whether it applies `mask_camera` by default.
  - Some export paths intentionally crop predictions to camera-visible voxels unless explicitly disabled.

## 9) Validation Checklist for Changes
1. Compile changed python files:
   - `conda run -n opus-mmcv2 python -m py_compile <changed_files>`
2. For config edits, parse config explicitly:
   - `conda run -n opus-mmcv2 python - <<'PY'`
   - `from mmengine.config import Config`
   - `cfg = Config.fromfile('<config_path>')`
   - `print('ok', cfg.model.type)`
   - `PY`
3. If touching data pipeline, run a minimal smoke check on one sample or one transform path.

## 10) Git Hygiene
- Do not revert unrelated user changes.
- Do not add temporary one-off scripts to git unless explicitly requested.
- Keep diffs scoped and traceable to the requested task.

## 11) Recommended Session Loop
1. Confirm the exact objective, config, and target log/run path.
2. Reproduce or inspect with the smallest effective check.
3. Apply minimal code/config changes.
4. Validate in `opus-mmcv2`.
5. Append concise handoff notes to `docs/TASK_WORKORDER_CURRENT.md`.
