# Task Workorder

## Metadata
- Workorder ID: `WO-2026-02-28-001`
- Created Date: `2026-02-28`
- Last Updated: `2026-03-10 (binary-occ + PCA128 feature supervision office run launched)`
- Owner Session: `Codex CLI session`
- Status: `in_progress`
- Priority: `high`

## Request Snapshot
- User Original Request:
  - Make `PROJECT_GUIDE.md` generic (not tied to one run/task).
  - Provide a blank workorder template.
  - Create a current-session workorder file from that template for task-specific info.
  - Future sessions should read task workorder + project guide first.
  - Add an ablation config for Fusion: remove LiDAR features while keeping LiDAR-based query init.
  - Add another Fusion ablation config: remove both LiDAR init and LiDAR features.
- Success Criteria:
  - Generic `PROJECT_GUIDE.md` with fast handoff orientation.
  - `TASK_WORKORDER_TEMPLATE.md` exists and reusable.
  - `TASK_WORKORDER_CURRENT.md` exists and is ready for ongoing task notes.
- Non-goals:
  - No model code change.
  - No training run/restart.

## Context Snapshot
- Related Config(s):
  - `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`
  - `configs/opusv1-fusion_nusc-occ3d/tartanground-t_r50_640x640_9f_occ3d_100e.py`
  - `configs/opusv1-fusion_nusc-occ3d/Tartanground_indoor.py`
- Related Log(s):
  - No new run required in this step.
- Related Data Path(s):
  - N/A
- Related Code/File(s):
  - `PROJECT_GUIDE.md`
  - `TASK_WORKORDER_TEMPLATE.md`
  - `TASK_WORKORDER_CURRENT.md`

## Analysis Notes
- Observed Symptoms:
  - Existing guide mixed generic architecture with task/run-specific records.
  - Depth-point run (`.../14-47-02`) underperformed LiDAR baseline severely (`occ/mIoU` near 0.x vs 20+ at same epoch range).
- Hypotheses:
  - Separating "project knowledge" and "task workorder" improves new-session startup speed.
  - Depth history frames were transformed with static `sensor2lidar` (no ego-motion compensation), causing major temporal misalignment in point branch.
  - Depth-to-camera axis convention likely mismatched LiDAR frame expectation in current pipeline (`tartanair_ned` vs actual camera extrinsics convention).
- Risks / Assumptions:
  - Assume future sessions will keep `TASK_WORKORDER_CURRENT.md` updated after each task step.
  - Even after fixing history ego-motion, depth pseudo-points may still lag LiDAR due modality/domain differences.
  - Early-epoch (`epoch=5`) zero mIoU is not sufficient to rule in/out geometry fixes; need at least `epoch=10` comparison.

## Plan
1. Create blank workorder template.
2. Create current workorder file from template.
3. Keep `PROJECT_GUIDE.md` focused on reusable project-level knowledge.

## Execution Log
- `[2026-02-28]` Added reusable workorder template -> done.
- `[2026-02-28]` Created current workorder for task-specific tracking -> done.
- `[2026-02-28]` Updated project guide toward generic handoff workflow -> done.
- `[2026-02-28]` Added `drop_lidar_feat` switch in `OPUSV1Fusion` (train/test paths) -> done.
- `[2026-02-28]` Added fusion ablation config that enables the switch and keeps base init logic -> done.
- `[2026-02-28]` Added second fusion ablation config (drop LiDAR feat + disable LiDAR init) -> done.
- `[2026-02-28]` Fixed DDP unused-parameter issue in ablation: changed zeroing to graph-connected `pts_feats * 0.0` -> done.
- `[2026-02-28]` Synced project guide with ablation entry mapping and usage notes -> done.
- `[2026-02-28]` Implemented `scripts/data/add_depth_paths_to_ann.py` for incremental depth metadata injection in ann pkl -> done.
- `[2026-02-28]` Generated `train_with_depth.pkl` / `val_with_depth.pkl` / `test_with_depth.pkl` and reports under `/root/wjl/tartanground_demo/` -> done.
- `[2026-02-28]` Idempotence check passed (`train_with_depth.pkl` -> `train_with_depth_pass2.pkl`: added=0, existed>0) -> done.
- `[2026-02-28]` Added `LoadPointsFromMultiViewDepth` transform for depth->pseudo-LiDAR points with history sweeps support -> done.
- `[2026-02-28]` Updated Tartanground dataset adapter to expose current-frame `cams` metadata to pipeline -> done.
- `[2026-02-28]` Added config-level point source switch file (`lidar`/`depth`) and default depth-mode ann routing -> done.
- `[2026-02-28]` Re-generated `*_with_depth.pkl` using `opus-mmcv2` python to avoid cross-env pickle incompatibility -> done.
- `[2026-02-28]` Implemented depth history ego-motion compensation in `LoadPointsFromMultiViewDepth` (history dynamic extrinsics; current frame unchanged) -> done.
- `[2026-02-28]` Added dynamic extrinsics fallback policy (`static|skip|raise`, default `static`) for backward compatibility -> done.
- `[2026-02-28]` Compared bad depth run (`14-47-02`) vs LiDAR baseline (`2026-02-11/14-01-29`): confirmed huge mIoU gap and isolated run used static history transforms -> done.
- `[2026-03-01]` Fixed merged-code regressions in `OPUSV1Fusion.forward_train`: removed missing `_need_pts_branch` dependency and restored safe `pts_feats` path -> done.
- `[2026-03-01]` Added missing `_maybe_drop_lidar_feat` implementation and wired `drop_lidar_feat` attribute in constructor -> done.
- `[2026-03-01]` Ran 8-GPU depth experiment with NED convention and val interval=5; observed `epoch5 mIoU=0.00` -> done.
- `[2026-03-02]` Audited depth-vs-LiDAR geometry/range on 80 sampled train frames; generated reports in `outputs/analysis/` -> done.
- `[2026-03-02]` Ran 8-GPU opencv-convention controlled experiment (`5ep`) and resumed to `10ep`; observed major metric lift by epoch 10 vs NED run -> done.
- `[2026-03-02]` Switched demo base config dataset paths from `TartanGround_Indoor` to `data/tartanground_demo` (`dataset_root`, `occ_root`) so depth switch config resolves existing `*_with_depth.pkl` -> done.
- `[2026-03-02]` Fixed training crash `Input TN mismatch: expected 48, got 54` by syncing `img_encoder.num_frames` with global `num_frames` in demo base config -> done.
- `[2026-03-02]` Aligned demo 9f config class mapping to office dataset taxonomy (`occ_names/rare_classes/cls_weights`) to resolve class-count mismatch vs dataset labels -> done.
- `[2026-03-02]` Reset demo 9f base hyper-params to match prior opencv depth experiment (`num_query=9600`, `num_refines=[1,4,16,32,64,128]`, `voxel_size=0.05`, `batch_size=16`, `img_pad size_divisor=32`, `rare_weights/tail_weight=12`) while keeping `val_interval=5` -> done.
- `[2026-03-02]` Reproduced all-zero validation using single-GPU offline eval on `epoch_5.pth` with `score_thr=0.3` -> done.
- `[2026-03-02]` Ran threshold ablation on same checkpoint: `score_thr=0.0` gives non-zero metrics (`mIoU=1.53`, `IoU=48.65`) -> done.
- `[2026-03-02]` Applied depth-config hotfix: override `model.test_cfg.pts.score_thr=0.0` in `tartanground_demo_r50_640x640_9f_100e_depth_points_switch.py` -> done.
- `[2026-03-02]` Confirmed running depth job reached `Epoch(val)[80]` with `occ/mIoU=26.04`, `occ/IoU=48.72`, then entered `Epoch(train)[81]` -> done.
- `[2026-03-02]` Stopped old 8-GPU depth run per user request (`launcher pid=863448`) -> done.
- `[2026-03-02]` First restart attempt failed in background shell (`ModuleNotFoundError: mmengine`), root cause = missing conda activation -> done.
- `[2026-03-02]` Relaunched 8-GPU depth run with explicit `conda activate opus-mmcv2` -> in_progress.
- `[2026-03-03]` Added `TPVLiteEncoder` and TPV tri-plane sampling path (`XY/XZ/YZ`) for OPUSV1Fusion; kept depth->pseudo-points data chain unchanged -> done.
- `[2026-03-03]` Wired TPV branch through detector/head/transformer (`tpv_feats` end-to-end) with explicit mutual-exclusion guard (`use_pts_sampling` vs `use_tpv_sampling`) -> done.
- `[2026-03-03]` Added TPV experiment config `..._tpv_lite_depth.py` (`enable_tpv_feature_branch=True`, `enable_pts_feature_branch=False`, transformer TPV sampling on) -> done.
- `[2026-03-03]` Added TPV-only unit tests (`tests/test_tpv_lite.py`) and passed all targeted checks (`6/6`) -> done.
- `[2026-03-03]` Reworked TPV path to reuse `SparseEncoder` 3D middle features (`return_middle_feats=True`) and replaced point aggregation TPV with 3D FPN-style upsample + skip residual fusion -> done.
- `[2026-03-03]` Fixed TPV-DDP crash (`Expected to have finished reduction`) by freezing `pts_middle_encoder.conv_out` in TPV-only mode (that branch is unused in TPV path) -> done.
- `[2026-03-03]` Restarted 8-GPU TPV training after fix; run is stable through `Epoch(train)[1][20/27]` without unused-parameter error -> in_progress.

## Changes Made
- `TASK_WORKORDER_TEMPLATE.md`: added blank reusable workorder structure.
- `TASK_WORKORDER_CURRENT.md`: initialized current task workorder from template and filled current request context.
- `PROJECT_GUIDE.md`: switched to generic quick-start + architecture + session triage style.
- `models/opusv1_fusion/opus.py`: added `drop_lidar_feat` switch and zero-out hook for `pts_feats`.
- `models/opusv1_fusion/opus.py`: updated zero-out to graph-connected path (`pts_feats * 0.0`) for DDP stability.
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_ablation_drop_lidar_feat.py`: new ablation config.
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_ablation_drop_lidar_feat_no_lidar_init.py`: new ablation config (also sets `init_pos_lidar=None`).
- `PROJECT_GUIDE.md`: added guidance for Fusion ablation (drop LiDAR features, keep init).
- `.gitignore`: unignored `scripts/data/add_depth_paths_to_ann.py`.
- `scripts/data/add_depth_paths_to_ann.py`: added ann pkl depth-path incremental updater (cams + cam_sweeps, report json, strict/skip flags, atomic write).
- `loaders/pipelines/loading.py`: added `LoadPointsFromMultiViewDepth` transform (`[x,y,z,intensity,time]`, depth png/npy loading, cam->lidar transform, max point cap).
- `loaders/pipelines/loading.py`: added history dynamic extrinsics path (`sensor2global -> current lidar`) with options `history_dynamic_extrinsics=True` and `dynamic_extrinsics_fallback='static'`.
- `models/opusv1_fusion/opus.py`: fixed training-time regressions after branch merge (`_need_pts_branch` call removed, `_maybe_drop_lidar_feat` restored, `drop_lidar_feat` state wired).
- `loaders/tartanground_occ3d_dataset.py`: `get_data_info()` now exposes `cams` field for downstream depth loader lookup.
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_depth_points_switch.py`: new config with `point_input_source` switch and depth-mode pkl routing.
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_depth_points_switch.py`: updated `depth_points_cfg.coord_convention` from `tartanair_ned` to `opencv` (2026-03-02).
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`: updated `dataset_root` / `occ_root` to demo dataset paths (`/root/wjl/OPUS_mmcv2/data/tartanground_demo/` and `.../gts/`).
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`: fixed frame-count consistency (`num_frames=9`, `img_encoder.num_frames=num_frames`) to avoid TN shape mismatch in `MapAnythingOccEncoder`.
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`: aligned class-related params with office reference (`occ_names` 79 classes, `rare_classes`, `cls_weights`) and kept `empty_label=len(occ_names)` auto-consistent.
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`: restored prior opencv-depth experiment capacity/weighting settings, but preserved `train_cfg.val_interval=5` for denser validation prints.
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_depth_points_switch.py`: override `model.test_cfg.pts.score_thr=0.0` to avoid early-epoch over-pruning to empty predictions.
- `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat.py`: rewired `prototype_npz_path` / `prototype_bridge_path` from the stale `/root/wjl/Talk2DINO/...` location to local `data/tartanground_demo/office79_prototypes_pca256.npz` and `data/tartanground_demo/office79_bridge.json`.
- `/root/wjl/Occ-3D/data_preprocess/tartanground_to_occ3d_pkl.py`: GT generation now keeps legacy `mask_camera` and additionally writes per-camera visibility metadata as `mask_camera_bits` (`uint8` bitfield) plus `camera_names` into each `labels.npz`.
- `scripts/export_gt_viewmask_ply.py`: exports one sample's GT visible voxels into `1view/3view/6view` PLYs using `mask_camera_bits`, copies the corresponding current-frame images, and writes per-case metadata plus a run summary.
- `PROJECT_GUIDE.md`: documented depth point-source switch and query-init compatibility.
- `outputs/analysis/tartanground_demo_depth_points_switch_opencv_5ep.py`: analysis-only config copy with `coord_convention='opencv'` for controlled comparison.
- `models/lidar_encoder/tpv_lite_encoder.py`: replaced with sparse-3D TPV encoder (`high+skip` input, 3D conv refine, FPN-style upsample + residual fusion, then `XY/XZ/YZ` projection).
- `models/lidar_encoder/__init__.py`: exported `TPVLiteEncoder`.
- `models/opusv1_fusion/opus_sampling.py`: added `sampling_tpv_feats(...)` for tri-plane feature sampling + optional query-conditioned plane fusion.
- `models/opusv1_fusion/opus_transformer.py`: added TPV sampling switches (`use_tpv_sampling`, `tpv_fusion_mode`), `tpv_feats` routing, and sampling-mode mutual-exclusion checks.
- `models/opusv1_fusion/opus_head.py`: `forward(..., tpv_feats=None)` and TPV forwarding to transformer.
- `models/opusv1_fusion/opus.py`: TPV extraction now uses sparse-3D middle features (`_extract_sparse_3d_feat_for_tpv`), and TPV mode keeps `pts_voxel_encoder/pts_middle_encoder` trainable while freezing only unused 2D BEV head path.
- `models/opusv1_fusion/opus.py`: TPV-only mode now additionally freezes `pts_middle_encoder.conv_out` to avoid DDP unused-parameter reduction errors.
- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_tpv_lite_depth.py`: TPV config now enforces `pts_middle_encoder.return_middle_feats=True` and switches TPV encoder args to sparse-3D mode.
- `tests/test_tpv_lite.py`: TPV-only unit tests (encoder shape/empty, TPV sampling shape+axis mapping, TPV transformer smoke, mode exclusion).

- `[2026-03-06]` Converted `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_150e_tpv_gt-depth_bilinear_flat.py` from `ResNet/FPN + MapAnything weighted fusion` to pure `MapAnything` external-image-encoder path (`use_external_img_encoder=True`, `img_backbone=None`, `img_neck=None`, `img_feature_fusion=None`) -> done.
- `[2026-03-06]` Kept `img_encoder.anyup_cfg.enabled=True` but clarified semantics with `mode='bilinear'`: this enables the bilinear pyramid adapter only and does **not** instantiate or run AnyUp network weights -> done.
- `[2026-03-06]` Fixed hidden image-size mismatch in the office flat config: aligned `ida_aug_conf.final_dim` and `mapanything_preprocess_cfg.size` to `630x630` while preserving original source `H/W=640` and `patch_size=14` compatibility -> done.
- `[2026-03-06]` Synced `AGENTS.md` and `docs/PROJECT_GUIDE.md` with current office flat-config semantics, the `640 -> 630` alignment issue, and the `enabled=True + mode='bilinear'` interpretation -> done.
- `[2026-03-06]` Re-synced markdown entrypoints after deleting `docs/README.md` and `docs/LIDAR_FEATURE_EXTRACTOR_FLOW.md`: updated `AGENTS.md` mandatory read order and cleaned stale `docs/INDEX.md` references -> done.
- `[2026-03-07]` Added and launched `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_150e_tpv_lidar_gt-depth_bilinear_flat.py` as the LiDAR-point ablation counterpart of the pure-MapAnything TPV office flat config; only the point branch was switched to `LoadPointsFromFile + LoadPointsFromMultiSweeps`, while `mapanything_extra` still uses GT depth from `_with_depth` splits -> done.
- `[2026-03-09]` Visualized the best TPV+Depth office checkpoint (`epoch_130.pth`, best `mIoU=34.14`) with `scripts/compare_multiframe_and_query_vis.py`; exported preview PLYs for `history=0` and `history=5` under `/root/wjl/OPUS_mmcv2/demo_outputs_tpv_depth_best_e130/` -> done.
- `[2026-03-09]` Fixed `scripts/visualize_query_trajectory_ply.py` to support current TPV-only `OPUSV1Fusion` configs by forwarding `mapanything_extra`, using `_extract_pts_feat_for_head`, and passing `tpv_feats` into `pts_bbox_head`; reran query trajectory export successfully -> done.
- `[2026-03-09]` Generated a fixed-index multiframe visualization for sample indices `0,10,20,30,40` (tokens `000000/000100/000200/000300/000400`) to approximate a stride-10 temporal window while preserving each sample's internal 9-frame model input. Outputs saved under `/root/wjl/OPUS_mmcv2/demo_outputs_tpv_depth_idx_0_10_20_30_40/` with `history_0`, `history_4`, and query-trajectory exports -> done.
- `[2026-03-09]` Clarified multiframe export semantics: `history=4` on five explicit indices yields five progressively accumulated demos (`1/2/3/4/5` sample aggregation), and the final sample in `history_4` corresponds to the full aggregation of `0,10,20,30,40` -> done.
- `[2026-03-10]` Replaced the old office semantic-feature branch design (`79-way semantic cls + 128-d feature`) with `binary occupiedness score + 128-d prototype-supervised feature` in `models/opusv1_fusion/opus_transformer.py` / `models/opusv1_fusion/opus_head.py` -> done.
- `[2026-03-10]` Added binary occupancy target construction with distance-threshold ignore band (`pos=0.10`, `neg=0.20`) and switched feature supervision to positive queries only -> done.
- `[2026-03-10]` Added single-process val/test compatibility fixes for the binary-occ office config: `LoadMapAnythingExtraFromDepth` now tolerates filename/img count mismatch in online multiframe eval, and the new test pipeline forces sweeps loader offline mode -> done.
- `[2026-03-10]` Deleted obsolete config `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_150e_tpv_gt-depth_bilinear_pca128_feat_flat.py`; added replacement config `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_150e_tpv_gt-depth_binary-occ_pca128_feat_flat.py` -> done.
- `[2026-03-10]` Smoke validation passed in `opus-mmcv2`: config parse, python compile, CUDA one-sample `loss()` (`loss_occ`, `loss_pts`, `loss_feat`) and CUDA one-sample `predict()` with the new config -> done.
- `[2026-03-10]` Launched 8-GPU office training with `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_150e_tpv_gt-depth_binary-occ_pca128_feat_flat.py`; run dir: `outputs/OPUSV1Fusion/TT_Office_mapanything_640x640_9f_150e_tpv_gt-depth_binary-occ_pca128_feat_flat_2026-03-10/18-53-34/` -> in_progress.
- `[2026-03-11]` Added `--disable-camera-mask` to the multiframe PLY export path (`scripts/inference_demo_multiframe.py` + compare wrappers) so masked/no-mask point-cloud demos can be generated from the same checkpoint without touching evaluation defaults -> done.
- `[2026-03-11]` Standardized demo output location to `/root/wjl/OPUS_mmcv2/demos`; generated masked and no-mask visualization outputs for the binary-occ `epoch_130` checkpoint under that root -> done.
- `[2026-03-11]` Generalized `/root/wjl/Talk2DINO/tartanground_label_ae` prototype export pipeline to support PCA256 office prototypes (`office79_prototypes_pca256.npz`, field `latent_256`) while keeping existing PCA128 artifacts compatible -> done.
- `[2026-03-11]` Upgraded `models/opusv1_fusion/opus_head.py` semantic supervision from pure regression to an open-vocab-friendly mixed loss: cosine regression + prototype CE + hard-negative margin, plus weak-positive semantic supervision and tail-class feature reweighting -> done.
- `[2026-03-11]` Added new config `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat.py` with `num_query=9600`, `total_epochs=100`, `context_ratio=0.5`, `feature_dims=256`, and PCA256 prototype wiring -> done.
- `[2026-03-11]` CUDA smoke validation passed for the new PCA256 mixloss config: one-sample `loss()` and one-sample `predict()` both succeeded in `opus-mmcv2` -> done.
- `[2026-03-11]` First launch of the PCA256 mixloss run failed immediately in AMP due to `hard-negative margin` using `-1e6` sentinel on `float16` tensors (`RuntimeError: value cannot be converted to type at::Half without overflow`) -> done.
- `[2026-03-11]` Fixed the AMP overflow by replacing the hard negative sentinel with `torch.finfo(dtype).min` in `models/opusv1_fusion/opus_head.py`, re-ran CUDA smoke successfully, and relaunched 8-GPU training -> done.
- `[2026-03-11]` Active run: `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat.py`; run dir: `outputs/OPUSV1Fusion/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_2026-03-11/11-08-09/` -> in_progress.
- `[2026-03-11]` Generalized `/root/wjl/Talk2DINO/tartanground_label_ae/scripts/train_tartanground_text_ae.py` to support latent-field naming by dimension and PCA-only prototype export; added `/root/wjl/Talk2DINO/tartanground_label_ae/configs/pca256_talk2dino_reg.json` and exported `office79_prototypes_pca256.npz` with field `latent_256` -> done.
- `[2026-03-11]` Added integrated office config `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat.py` (`num_query=9600`, `100e`, `PCA256`, cosine+CE+margin mixed semantic loss, strong/weak semantic positives, `context_ratio=0.5`) -> done.
- `[2026-03-11]` Launched 8-GPU run for the integrated `pca256_mixloss` office config; run dir: `outputs/OPUSV1Fusion/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_2026-03-11/11-00-11/` -> in_progress.
- `[2026-03-12]` Repointed `prototype_npz_path` / `prototype_bridge_path` in `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat.py` from `/root/wjl/Talk2DINO/...` to the user-provided local assets under `data/tartanground_demo/` -> done.
- `[2026-03-12]` Re-validated the updated office `pca256_mixloss` config in `opus-mmcv2`: `py_compile`, config parse, train-dataset sample build (`img (54, 3, 630, 630)`, `points (~5.6e5, 5)`, `views=54`), and full model build all succeeded -> done.
- `[2026-03-12]` Launched fresh 8-GPU training for the updated local-prototype office config; active run dir: `outputs/OPUSV1Fusion/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_2026-03-12/15-43-48/` -> in_progress.
- `[2026-03-12]` Extended `/root/wjl/Occ-3D/data_preprocess/tartanground_to_occ3d_pkl.py` so new `labels.npz` files keep legacy `mask_camera` and additionally store per-camera visibility as `mask_camera_bits` plus `camera_names` for future dynamic-view GT masking -> done.
- `[2026-03-12]` Smoke-generated one sample to `/tmp/tg_occ3d_camvisbits_smoke/` and verified `mask_camera == (mask_camera_bits != 0)` while `semantics/mask_lidar/mask_camera` stayed identical to the old `gts_0.1` output for the same sample -> done.
- `[2026-03-12]` Added `scripts/export_gt_viewmask_ply.py` to export a sample's GT visible voxels as 1-view / 3-view / 6-view PLYs from `mask_camera_bits`, and to copy the matching current-frame camera images into per-case folders -> done.
- `[2026-03-12]` Ran the new GT-viewmask export on the smoke GT sample (`train.pkl` sample `0`, token `CarWelding_Data_diff_P1003_000000`) and wrote results under `demos/gt_viewmask_train_000000_CarWelding_camvisbits_smoke/` -> done.

## Validation
- Command / Check:
  - `/root/miniconda3/envs/opus-mmcv2/bin/python -m py_compile /root/wjl/Occ-3D/data_preprocess/tartanground_to_occ3d_pkl.py`
  - `/root/miniconda3/envs/opus-mmcv2/bin/python /root/wjl/Occ-3D/data_preprocess/tartanground_to_occ3d_pkl.py --data-root /root/wjl/OPUS_mmcv2/data/TartanGround_Indoor --pkls train --out-root /tmp/tg_occ3d_camvisbits_smoke --seg-rgbs /root/wjl/OPUS_mmcv2/data/TartanGround_Indoor/seg_rgbs.txt --seg-label-map /mnt/data/datasets/TartanGround_unified/seg_label_map.json --target-label-map /mnt/data/datasets/TartanGround_unified/seg_label.json --pcd-template "/mnt/data/datasets/TartanGround_unified/{scene_name}/{scene_name}_sem.pcd" --num-workers 1 --chunk-size 1 --log-every 1 --vox 0.1 --limit 1`
  - Smoke GT output check on `/tmp/tg_occ3d_camvisbits_smoke/.../labels.npz` -> new keys `mask_camera_bits` and `camera_names` present; `mask_camera == (mask_camera_bits != 0)`; old `semantics/mask_lidar/mask_camera` match the baseline sample under `gts_0.1`.
  - File existence and section sanity check.
  - `python -m py_compile models/opusv1_fusion/opus.py`
  - `python -m py_compile configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_ablation_drop_lidar_feat.py`
  - `python -m py_compile configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_ablation_drop_lidar_feat_no_lidar_init.py`
  - `python3 -m py_compile scripts/data/add_depth_paths_to_ann.py`
  - `python3 scripts/data/add_depth_paths_to_ann.py --in-pkl /root/wjl/tartanground_demo/train.pkl --out-pkl /root/wjl/tartanground_demo/train_with_depth.pkl --report-json /root/wjl/tartanground_demo/train_with_depth.depth_report.json`
  - `python3 scripts/data/add_depth_paths_to_ann.py --in-pkl /root/wjl/tartanground_demo/val.pkl --out-pkl /root/wjl/tartanground_demo/val_with_depth.pkl --report-json /root/wjl/tartanground_demo/val_with_depth.depth_report.json`
  - `python3 scripts/data/add_depth_paths_to_ann.py --in-pkl /root/wjl/tartanground_demo/test.pkl --out-pkl /root/wjl/tartanground_demo/test_with_depth.pkl --report-json /root/wjl/tartanground_demo/test_with_depth.depth_report.json`
  - `python3 scripts/data/add_depth_paths_to_ann.py --in-pkl /root/wjl/tartanground_demo/train_with_depth.pkl --out-pkl /root/wjl/tartanground_demo/train_with_depth_pass2.pkl --report-json /root/wjl/tartanground_demo/train_with_depth_pass2.depth_report.json`
  - `/root/miniconda3/envs/opus-mmcv2/bin/python scripts/data/add_depth_paths_to_ann.py --in-pkl /root/wjl/tartanground_demo/train.pkl --out-pkl /root/wjl/tartanground_demo/train_with_depth.pkl --report-json /root/wjl/tartanground_demo/train_with_depth.depth_report.json`
  - `/root/miniconda3/envs/opus-mmcv2/bin/python scripts/data/add_depth_paths_to_ann.py --in-pkl /root/wjl/tartanground_demo/val.pkl --out-pkl /root/wjl/tartanground_demo/val_with_depth.pkl --report-json /root/wjl/tartanground_demo/val_with_depth.depth_report.json`
  - `/root/miniconda3/envs/opus-mmcv2/bin/python scripts/data/add_depth_paths_to_ann.py --in-pkl /root/wjl/tartanground_demo/test.pkl --out-pkl /root/wjl/tartanground_demo/test_with_depth.pkl --report-json /root/wjl/tartanground_demo/test_with_depth.depth_report.json`
  - `python3 -m py_compile loaders/pipelines/loading.py loaders/tartanground_occ3d_dataset.py configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_depth_points_switch.py`
  - `LoadPointsFromMultiViewDepth` direct smoke test on one sample -> points shape `(447660, 5)`, `time==0` and `time>0` both present.
  - `python -m py_compile loaders/pipelines/loading.py` (after ego-motion fix)
  - `python -m py_compile models/opusv1_fusion/opus.py` (after merge-regression fixes)
  - `python -m py_compile configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_depth_points_switch.py` (after switching to `coord_convention='opencv'`)
  - `python -m py_compile configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py` (after frame-count consistency fix)
  - `/root/miniconda3/envs/opus-mmcv2/bin/python - <<'PY' ... Config.fromfile('...depth_points_switch.py') ... print(num_frames, img_encoder.num_frames) ... PY` -> both are `9`
  - `/root/miniconda3/envs/opus-mmcv2/bin/python - <<'PY' ... Config.fromfile('...depth_points_switch.py') ... print(len(occ_names), len(cls_weights), empty_label, num_classes) ... PY` -> `79, 79, 79, 79`
  - `/root/miniconda3/envs/opus-mmcv2/bin/python - <<'PY' ... Config.fromfile('...depth_points_switch.py') ... print(num_query, num_refines, voxel_size, size_divisor, rare_weights, tail_weight, batch_size, val_interval) ... PY` -> `9600`, `[1,4,16,32,64,128]`, `[0.05,0.05,0.05]`, `32`, `12`, `12`, `16`, `5`
  - Dynamic-vs-static geometric checks (`opus-mmcv2` env): current frame static==dynamic (trans error 0); history dynamic aligns to LiDAR sweep translation at numerical-noise level.
  - Depth/LiDAR range audit (80 sampled frames):
    - `outputs/analysis/depth_lidar_range_check_20260302_093619.json`
    - `outputs/analysis/depth_value_audit_20260302_093717.json`
    - `outputs/analysis/depth_vs_lidar_multiframe_range_20260302_093941.json`
  - Controlled training runs (8 GPU):
    - NED convention run (`2026-03-01/11-11-02`) with `val_interval=5`: `epoch5 occ/mIoU=0.0000`.
    - Opencv convention run (`2026-03-02/09-40-36`) with `val_interval=5`: `epoch5 occ/mIoU=0.0000`.
    - Opencv resume to 10 epochs (`resume_from=.../epoch_5.pth`): `epoch10 occ/mIoU=1.9000`, `occ/IoU=43.5100`.
  - Run-log comparison:
    - bad depth run (`14-47-02`) at epoch 40: `occ/mIoU: 0.54`
    - LiDAR baseline (`2026-02-11/14-01-29`) at epoch 40: `occ/mIoU: 21.33`
    - NED (ego-motion fixed) run at epoch 10 (`17-01-00`): `occ/mIoU: 0.20`, `occ/IoU: 5.80`
    - Opencv controlled run at epoch 10 (`09-40-36` resumed): `occ/mIoU: 1.90`, `occ/IoU: 43.51`
    - backup code in bad depth run confirms static transform path: `_camera_to_lidar` uses only `sensor2lidar_*`.
  - Axis mapping check from dataset extrinsics (`CAM_FRONT`, row-vector convention):
    - camera `+x -> lidar +X`
    - camera `+z -> lidar +Y` (forward)
    - camera `+y -> lidar -Z` (up)
    - This is consistent with OpenCV camera frame (`x right, y down, z forward`) and user-provided LiDAR frame (`x right, y front, z up`).
  - Threshold ablation on same checkpoint (`2026-03-02/11-27-29/epoch_5.pth`, single-GPU offline val):
    - `score_thr=0.3` -> `occ/mIoU=0.0000`, `occ/IoU=0.0000`
    - `score_thr=0.0` -> `occ/mIoU=1.5300`, `occ/IoU=48.6500`
  - Dataset consistency check:
    - `val.pkl` and `val_with_depth.pkl` both contain 108 samples with 108 unique tokens (not a split-size mismatch issue).
  - Epoch-80 summary extraction:
    - source log: `outputs/OPUSV1Fusion/tartanground_demo_r50_640x640_9f_100e_depth_points_switch_2026-03-02/13-23-17/20260302_132317/20260302_132317.log`
    - metric line: `Epoch(val)[80] occ/mIoU=26.04, occ/IoU=48.72`
  - Restart checks:
    - failed boot log: `outputs/analysis/train_depth_switch_8gpu_restart_20260302_193710.log` (`ModuleNotFoundError: mmengine`)
    - active boot log: `outputs/analysis/train_depth_switch_8gpu_restart_20260302_193737.log`
    - active worker pids observed: `1016007-1016014` (`train.py --config ...depth_points_switch.py`)
  - TPV-Lite targeted checks (`2026-03-03`):
  - `/root/miniconda3/envs/opus-mmcv2/bin/python -m py_compile models/lidar_encoder/tpv_lite_encoder.py models/lidar_encoder/__init__.py models/opusv1_fusion/opus.py models/opusv1_fusion/opus_head.py models/opusv1_fusion/opus_sampling.py models/opusv1_fusion/opus_transformer.py configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_tpv_lite_depth.py tests/test_tpv_lite.py` -> passed.
  - `/root/miniconda3/envs/opus-mmcv2/bin/python -m unittest discover -s tests -p 'test_tpv_lite.py'` -> `Ran 6 tests ... OK`.
  - `/root/miniconda3/envs/opus-mmcv2/bin/python - <<'PY' ... Config.fromfile('...tpv_lite_depth.py') ... PY` -> TPV flags resolved as expected (`enable_tpv_feature_branch=True`, `use_tpv_sampling=True`, `use_pts_sampling=False`).
  - 8-GPU run #1 (after sparse-TPV refactor): reached `Epoch(train)[1][1/27]` then failed with DDP unused params indices `60,61,62` (`pts_middle_encoder.conv_out.*`).
  - Applied fix: freeze `pts_middle_encoder.conv_out` in TPV-only mode.
  - 8-GPU run #2 (after fix): reached `Epoch(train)[1][20/27]` with normal loss logs and no unused-parameter crash.
- Result:
  - Workorder/template files present and readable.
  - Python syntax checks passed for modified model and new config.
  - `train/val/test` depth-path injected pkl generated successfully; each report shows missing_file_count=0 and depth_exists_ratio=1.0.
  - Idempotence re-run passed for train split (added_count=0, existed_count=24864).
  - `opus-mmcv2` environment can now load regenerated `*_with_depth.pkl` normally (no `numpy._core` unpickle error).
  - Depth history dynamic extrinsics fix is in place and validated locally.
  - Primary diagnostic conclusion: historical ego-motion misalignment was one confirmed issue, but coordinate convention mismatch is an additional major factor.
  - Evidence: with only `coord_convention` switched from `tartanair_ned` to `opencv`, `epoch10` metrics improved from `mIoU 0.20 / IoU 5.80` to `mIoU 1.90 / IoU 43.51`.
  - Additional root cause isolated: early-stage depth models produce lower confidence; `score_thr=0.3` can prune all predictions and force val metrics to 0.0 even when geometry path is functioning.
  - Depth-value audit indicates raw depth files are generally valid (no non-finite/negative), with ~2.46% far outliers `>30m` that are clipped by current `depth_max=30`.
  - Remaining gap to LiDAR baseline is still significant; next isolations should target density/sampling policy and long-horizon convergence (20/40 epoch).
  - Latest completed full run snapshot before restart: `Epoch 80` validation was stable non-zero (`mIoU 26.04`, `IoU 48.72`).
  - TPV-Lite code path now compiles and passes targeted unit tests without launching full training.
  - Current PCA256 mixloss best checkpoint by validation `occ/mIoU` is `epoch_90.pth` from `outputs/OPUSV1Fusion/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_2026-03-11/11-08-09/` (`mIoU=28.74`, `IoU=66.57`).
  - Free-text voxel-feature similarity export is now available through `scripts/encode_text_query_talk2dino.py` + `scripts/export_text_query_similarity_pointcloud.py` with OPUS inference in `opus-mmcv2` and text latent generation in `talk2dino`.
  - Formal `chair` demo export for `val` sample `0` was generated under `demos/text_query_similarity/val_TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_epoch_90_chair_2026-03-12_10-01-08/`.
  - Updated text-similarity point-cloud colormap from red/white to red/blue so low-activation voxels remain visible.
  - Formal `table` demo export for `val` sample `0` was generated under `demos/text_query_similarity/val_TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_epoch_90_table_2026-03-12_10-05-39/`.
  - Open-vocab probe `worktable` (near-synonym of `table`, not an exact Office-79 raw label) was exported under `demos/text_query_similarity/val_TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_epoch_90_worktable_2026-03-12_10-09-29/`; versus `table` on the same sample, common-voxel Pearson correlation was `0.882` and top-5k high-response voxel IoU was `0.489`.
  - Closed-set control `desk` (exact Office-79 raw label) was exported under `demos/text_query_similarity/val_TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_epoch_90_desk_2026-03-12_10-11-43/`; similarity stats on the same sample were `min=-0.212`, `max=0.883`, `mean=0.124`.
  - Frame/sample index `40` probe `monitor` was exported under `demos/text_query_similarity/val_TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_epoch_90_monitor_2026-03-12_10-15-46/`; token `Office_Data_omni_P0001_000400`, voxel count `261967`, similarity stats `min=-0.208`, `max=0.987`, `mean=-0.006`.
  - Frame/sample index `20` probe `monitor` was exported under `demos/text_query_similarity/val_TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_epoch_90_monitor_2026-03-12_10-18-39/`; token `Office_Data_omni_P0001_000200`, voxel count `245189`, similarity stats `min=-0.221`, `max=0.943`, `mean=-0.003`.
  - Frame/sample index `50` probe `monitor` was exported under `demos/text_query_similarity/val_TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_epoch_90_monitor_2026-03-12_10-21-56/`; token `Office_Data_omni_P0001_000500`, voxel count `362541`, similarity stats `min=-0.226`, `max=0.991`, `mean=0.0005`.
  - Same frame/sample index `50` probe `display` was exported under `demos/text_query_similarity/val_TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_epoch_90_display_2026-03-12_10-24-29/`; token `Office_Data_omni_P0001_000500`, voxel count `363091`, similarity stats `min=-0.196`, `max=0.416`, `mean=0.098`, and common-voxel Pearson correlation vs `monitor` on the same sample was `0.588`.
  - Same frame/sample index `50` probe `screen` was exported under `demos/text_query_similarity/val_TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_epoch_90_screen_2026-03-12_10-28-38/`; token `Office_Data_omni_P0001_000500`, voxel count `363282`, similarity stats `min=-0.230`, `max=0.727`, `mean=0.124`, with common-voxel Pearson correlation `0.760` vs `display` and `0.821` vs `monitor`.

## Pending / Handoff
- Next Steps (ordered):
  1. Start TPV-Lite depth training/eval with `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_tpv_lite_depth.py` once GPU window is confirmed.
  2. After each new user request, append concise updates to this file.
  3. Keep `PROJECT_GUIDE.md` generic; task details stay here.
  4. Continue opencv-convention depth run to `20/40` epochs and compare against NED and LiDAR baselines at matched epochs.
  5. Add one ablation on `sample_stride` / `max_points_total` to test depth point density impact.
  6. Default free-text visualization smoke command:
     - `conda run -n opus-mmcv2 python scripts/export_text_query_similarity_pointcloud.py --config outputs/OPUSV1Fusion/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_2026-03-11/11-08-09/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat.py --weights outputs/OPUSV1Fusion/TT_Office_mapanything_640x640_9f_100e_tpv_gt-depth_binary-occ_pca256_mixloss_flat_2026-03-11/11-08-09/epoch_90.pth --split val --sample-indices 0 --text-query chair --disable-camera-mask`
- Questions for User:
  - None.
- Next session read order:
  1. `TASK_WORKORDER_CURRENT.md`
  2. `PROJECT_GUIDE.md`
