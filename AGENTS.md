# AGENTS.md - OPUS_mmcv2 Session Onboarding

This file is a fast-start guide for new sessions working in this repository.

## 1) Mandatory Read Order
1. `docs/TASK_WORKORDER_CURRENT.md`
2. `docs/PROJECT_GUIDE.md`
3. `docs/INDEX.md`
4. `docs/LIDAR_FEATURE_EXTRACTOR_FLOW.md` (only when LiDAR branch details are needed)

If `docs/TASK_WORKORDER_CURRENT.md` is missing, copy from `docs/TASK_WORKORDER_TEMPLATE.md` first.

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

## 4) Active Config Anchors
- Fusion baseline:
  - `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`
- TPV + GT depth points:
  - `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_tpv_lite_depth_gt.py`
- MapAnything office config:
  - `configs/opusv1-fusion_nusc-occ3d/Tartanground_office_res_map_sum_0.5_gts0.1.py`
- Combined TPV + depth + mapanything:
  - `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_150e_tpv_gt-depth_bilinear.py`
- Flat standalone version of the combined config:
  - `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_150e_tpv_gt-depth_bilinear_flat.py`

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

## 7) Common Pitfalls
- `pc_voxel_size` and `voxel_size` are not interchangeable:
  - `pc_voxel_size`: point voxelization granularity for sparse point branch.
  - `voxel_size`: occupancy grid granularity used by occ mapping/evaluation.
- `sparse_shape` order is `[z, y, x]`, and must match `point_cloud_range` with `pc_voxel_size`.
- Early depth runs can show near-zero mIoU if `model.test_cfg.pts.score_thr` is too high.
- TPV-only mode can trigger DDP unused-parameter errors if unused point conv branches are not handled correctly.

## 8) Validation Checklist for Changes
1. Compile changed python files:
   - `conda run -n opus-mmcv2 python -m py_compile <changed_files>`
2. For config edits, parse config explicitly:
   - `conda run -n opus-mmcv2 python - <<'PY'`
   - `from mmengine.config import Config`
   - `cfg = Config.fromfile('<config_path>')`
   - `print('ok', cfg.model.type)`
   - `PY`
3. If touching data pipeline, run a minimal smoke check on one sample or one transform path.

## 9) Git Hygiene
- Do not revert unrelated user changes.
- Do not add temporary one-off scripts to git unless explicitly requested.
- Keep diffs scoped and traceable to the requested task.

## 10) Recommended Session Loop
1. Confirm the exact objective, config, and target log/run path.
2. Reproduce or inspect with the smallest effective check.
3. Apply minimal code/config changes.
4. Validate in `opus-mmcv2`.
5. Append concise handoff notes to `docs/TASK_WORKORDER_CURRENT.md`.
