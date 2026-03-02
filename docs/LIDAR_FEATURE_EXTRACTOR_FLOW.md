# OPUSV1Fusion LiDAR Feature Extractor Flow (Train-Centric, with Inference Notes)

This document explains the **active LiDAR feature extraction path** used by OPUS fusion configs, with a concrete shape derivation for:

- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`

It focuses on:

1. End-to-end LiDAR branch structure.
2. Input/output tensor contracts.
3. Per-stage spatial resolution and channel evolution.
4. How LiDAR BEV features are consumed by the fusion transformer.

---

## 1) Scope and Ground Truth Files

Primary code path:

- `models/opusv1_fusion/opus.py`
- `models/opusv1_fusion/opus_head.py`
- `models/opusv1_fusion/opus_transformer.py`
- `models/opusv1_fusion/opus_sampling.py`
- `loaders/pipelines/loading.py`
- `loaders/pipelines/pack_occ3d_inputs.py`

Config anchor:

- `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`

Upstream dependencies used by this path:

- `mmdet3d.models.middle_encoders.SparseEncoder`
- `mmdet3d.models.backbones.SECOND`
- `mmdet3d.models.necks.SECONDFPN`

---

## 2) High-Level Pipeline (LiDAR branch)

`points (N x 5)`
-> `hard voxelization`
-> `voxel encoder (HardSimpleVFE)`
-> `sparse middle encoder (SparseEncoder)`
-> `2D BEV backbone (SECOND)`
-> `BEV neck (SECONDFPN)`
-> `final_conv (3x3, 512->256)`
-> `pts_feats_for_head: [B, 256, H_bev, W_bev]`
-> `sampling_pts_feats(...)` in each decoder layer
-> fused into query updates.

---

## 3) Config Snapshot Used for Shape Derivation

From `tartanground_demo_r50_640x640_9f_100e.py`:

| Key | Value |
|---|---|
| `point_cloud_range` | `[-20, -20, -3, 20, 20, 5]` |
| `pc_voxel_size` | `[0.05, 0.05, 0.05]` |
| `pts_voxel_layer.max_num_points` | `10` |
| `pts_middle_encoder.type` | `SparseEncoder` |
| `pts_middle_encoder.sparse_shape` | `[161, 800, 800]` (z,y,x) |
| `pts_middle_encoder.output_channels` | `128` |
| `pts_middle_encoder.encoder_channels` | `((16,16,32),(32,32,64),(64,64,128),(128,128))` |
| `pts_middle_encoder.encoder_paddings` | `((0,0,1),(0,0,1),(0,0,[0,1,1]),(0,0))` |
| `pts_middle_encoder.block_type` | `basicblock` |
| `pts_backbone.type` | `SECOND` |
| `pts_backbone.in_channels` | `1152` |
| `pts_backbone.out_channels` | `[128, 256]` |
| `pts_backbone.layer_strides` | `[1, 2]` |
| `pts_neck.type` | `SECONDFPN` |
| `pts_neck.in_channels` | `[128, 256]` |
| `pts_neck.out_channels` | `[256, 256]` |
| `pts_neck.upsample_strides` | `[1, 2]` |
| `OPUSV1Fusion.final_conv` | `Conv2d(512 -> 256, k3,s1,p1)` |
| Transformer | `embed_dims=256, num_groups=4, num_points=4, num_frames=9` |

---

## 4) Input Contract Before Feature Extraction

### 4.1 Data pipeline to `inputs['points']`

| Stage | Code | Input | Output | Notes |
|---|---|---|---|---|
| Base point load | `LoadPointsFromFile` | raw lidar file | `BasePoints[N0,5]` | uses `load_dim=5, use_dim=5` in config |
| Multi-sweep merge | `LoadPointsFromMultiSweeps` | keyframe + sweeps | `BasePoints[N,5]` | sets current frame `time=0`; sweep time as `ts - sweep_ts`; applies sweep `sensor2lidar` transform |
| Occupancy frame conversion | `LiDARToOccSpace` | `BasePoints[N,5]` in lidar frame | `BasePoints[N,5]` in occ frame | transforms xyz only, keeps extra dims (intensity/time) |
| Tensor packing | `PackOcc3DInputs` | `BasePoints` | `inputs['points']` tensor | model receives tensor/list tensor |

Notes:

- Feature dims are expected to be 5 (`x,y,z,intensity,time`) for this config.
- `PointsRangeFilter` runs after `LiDARToOccSpace`, so filtering is in occupancy coordinate space.

---

## 5) Training Main Chain: Detailed Tensor/Resolution Table

Notation:

- `B`: batch size
- `N_i`: number of points in sample `i`
- `V`: number of non-empty voxels across batch
- `Pmax`: max points per voxel (`10`)

### 5.1 Module-level chain (`OPUSV1Fusion.extract_pts_feat`)

| Step | Code entry | Input tensor | Output tensor | Shape detail |
|---|---|---|---|---|
| 1 | `voxelize(points)` | list of `B` point tensors | `(voxels, num_points, coors)` | `voxels: [V, Pmax, 5]`, `num_points: [V]`, `coors: [V,4]` (`batch,z,y,x`) |
| 2 | `pts_voxel_encoder(...)` (`HardSimpleVFE`) | `(voxels, num_points, coors)` | `voxel_features` | `voxel_features: [V, 5]` |
| 3 | `pts_middle_encoder(...)` (`SparseEncoder`) | `(voxel_features, coors, batch_size)` | BEV dense feature | `x: [B, 1152, 100, 100]` |
| 4 | `pts_backbone(x)` (`SECOND`) | `[B,1152,100,100]` | tuple multi-scale | `x1: [B,128,100,100]`, `x2: [B,256,50,50]` |
| 5 | `pts_neck((x1,x2))` (`SECONDFPN`) | 2 levels | list with fused BEV | `pts_feats[0]: [B,512,100,100]` |
| 6 | `final_conv(pts_feats[0])` | `[B,512,100,100]` | LiDAR feature for head | `pts_feats_for_head: [B,256,100,100]` |

### 5.2 Where this happens in code

- Extraction path: `models/opusv1_fusion/opus.py::extract_pts_feat`
- Head input conversion: `models/opusv1_fusion/opus.py::_extract_pts_feat_for_head`
- Training callsite: `models/opusv1_fusion/opus.py::forward_train`

---

## 6) Spatial Resolution Derivation (Why 100x100 and 1152 channels)

### 6.1 Base occupancy lattice from config

Given:

- x range: `[-20, 20]`, voxel size `0.05` -> `Nx = 40 / 0.05 = 800`
- y range: `[-20, 20]`, voxel size `0.05` -> `Ny = 800`
- z range: `[-3, 5]`, voxel size `0.05` -> `Nz = 8 / 0.05 = 160`

Configured sparse shape is `[161,800,800]` (z,y,x), matching one extra z bin convention.

### 6.2 SparseEncoder stage-by-stage downsampling

`SparseEncoder` uses `block_type='basicblock'`. In this mode, each stage's last block (except final stage) performs a stride-2 sparse conv.

| Sparse stage | Key op | Output channels | Output (z,y,x) |
|---|---|---:|---|
| Input | sparse tensor init | 5 | `(161, 800, 800)` |
| conv_input | SubMConv3d (no downsample) | 16 | `(161, 800, 800)` |
| encoder stage1 | last block stride2, padding=1 | 32 | `(81, 400, 400)` |
| encoder stage2 | last block stride2, padding=1 | 64 | `(41, 200, 200)` |
| encoder stage3 | last block stride2, padding=[0,1,1] | 128 | `(20, 100, 100)` |
| encoder stage4 | no stride2 | 128 | `(20, 100, 100)` |
| conv_out | kernel=(3,1,1), stride=(2,1,1) | 128 | `(9, 100, 100)` |

Then `SparseEncoder.forward` converts sparse output to dense:

- dense shape: `[B, 128, 9, 100, 100]`
- flatten `C*D`: `[B, 128*9, 100, 100] = [B, 1152, 100, 100]`

This exactly matches `pts_backbone.in_channels=1152` in config.

### 6.3 SECOND + SECONDFPN

| Module | Input | Output |
|---|---|---|
| `SECOND` stage1 (stride1) | `[B,1152,100,100]` | `[B,128,100,100]` |
| `SECOND` stage2 (stride2) | `[B,128,100,100]` | `[B,256,50,50]` |
| `SECONDFPN` upsample+concat | `([B,128,100,100], [B,256,50,50])` | `[B,512,100,100]` |
| `final_conv` | `[B,512,100,100]` | `[B,256,100,100]` |

---

## 7) How LiDAR Features Are Consumed in the Fusion Transformer

### 7.1 Pre-sampling reshape

In decoder forward (`OPUSTransformerDecoder.forward`):

- Input `pts_feats_for_head`: `[B,256,100,100]`
- Group split by `num_groups=4`:
  - per-group channel `Cg = 256 / 4 = 64`
  - reshaped to `[B*4, 64, 100, 100]`

### 7.2 Point-wise BEV sampling

In `sampling_pts_feats(...)`:

- sampling points input: `[B,Q,G,P,3]`
- transform `occ -> lidar` with `occ2lidar`
- normalize xy into `[-1,1]`
- `grid_sample` over `pts_feats` gives:
  - sampled point feature: `[B,Q,G,P,Cg]`

For current config:

- `Q=4800`, `G=4`, `P=4`, `Cg=64`
- sampled LiDAR feature: `[B,4800,4,4,64]`

### 7.3 Fusion with image sampled features

In each decoder layer:

- image sampled feature: `[B,Q,G,T*P,Cg]` where `T=9` -> `36` points
- LiDAR sampled feature: `[B,Q,G,P,Cg]` -> `4` points
- concatenated along sample-point axis: `[B,Q,G,(T+1)*P,Cg] = [B,Q,G,40,64]`
- fed into `AdaptiveMixing` to update query features.

---

## 8) Train vs Inference: LiDAR Branch Differences

| Path | LiDAR feature extraction | Drop-lidar switch behavior |
|---|---|---|
| Train (`forward_train`) | computes `pts_feats_for_head` | applies `_maybe_drop_lidar_feat` |
| Test offline (`simple_test_offline`) | computes `pts_feats_for_head` | **does not** apply `_maybe_drop_lidar_feat` |
| Test online (`simple_test_online`) | computes `pts_feats_for_head` | applies `_maybe_drop_lidar_feat` |

Implication:

- If `drop_lidar_feat=True`, train and online-test behavior match; offline-test currently differs.

---

## 9) Switch Matrix (What Each Switch Affects)

| Switch | Location | Affects LiDAR BEV extractor? | Affects query init from raw points? |
|---|---|---|---|
| `enable_pts_feature_branch` | `OPUSV1Fusion` ctor | Yes (can disable branch entirely) | No |
| `drop_lidar_feat` | `OPUSV1Fusion._maybe_drop_lidar_feat` | Yes (zeroes `pts_feats` before head, where called) | No |
| `transformer.use_pts_sampling` | `OPUSV1FusionTransformer` | Uses/ignores `pts_feats` in decoder sampling | No |
| `init_pos_lidar` | `OPUSV1FusionHead` | No | Yes (controls query init from points) |
| `query_init_mix` | `train_cfg.pts` | No | Yes (LiDAR FPS/random mix for init queries) |

Key separation:

- **Raw `points` path** (query init) and **`pts_feats` path** (decoder feature sampling) are related but not the same.

---

## 10) Non-Active Alternative Module

Repo also contains:

- `models/lidar_encoder/sparse_encoder4x.py` (`SparseEncoder8x`)

This is **not** the active path in the current OPUSV1Fusion config above (which uses mmdet3d `SparseEncoder` + `SECOND` + `SECONDFPN`).

---

## 11) Code Index (Quick Jump)

- LiDAR extraction entry:
  - `models/opusv1_fusion/opus.py` (`_extract_pts_feat_for_head`, `extract_pts_feat`, `voxelize`)
- Train/test callsites:
  - `models/opusv1_fusion/opus.py` (`forward_train`, `simple_test_offline`, `simple_test_online`)
- Query init from points:
  - `models/opusv1_fusion/opus_head.py` (`get_init_position`)
- Fusion transformer and sampling:
  - `models/opusv1_fusion/opus_transformer.py` (`OPUSTransformerDecoder`, `OPUSSampling`)
  - `models/opusv1_fusion/opus_sampling.py` (`sampling_pts_feats`, `sampling_4d`)
- Data pipeline points path:
  - `loaders/pipelines/loading.py` (`LoadPointsFromMultiSweeps`, `LiDARToOccSpace`)
  - `loaders/pipelines/pack_occ3d_inputs.py` (`PackOcc3DInputs`)
- Config anchor:
  - `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`

