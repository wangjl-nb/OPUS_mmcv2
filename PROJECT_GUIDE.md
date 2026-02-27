# OPUS_mmcv2 Project Guide (Model Logic)

## Repo layout (high level)
- `configs/`: model & dataset configs (OPUSV1/OPUSV2, fusion variants)
- `models/`: core model implementations (OPUSV1, OPUSV1Fusion, heads, transformers)
- `loaders/`: datasets + pipelines + metrics
- `train.py`, `val.py`: training/validation entry points

## Model overview
OPUS is an occupancy prediction framework using query points + transformer-style refinement. There are two main variants:
- **OPUSV1 (camera-only)**: image backbone/neck → transformer head → sparse occupancy.
- **OPUSV1Fusion (camera + LiDAR)**: image branch + point branch → fusion head → sparse occupancy.

Key files:
- OPUSV1 detector: `models/opusv1/opus.py`
- OPUSV1 head: `models/opusv1/opus_head.py`
- OPUSV1 transformer: `models/opusv1/opus_transformer.py`
- OPUSV1 fusion detector: `models/opusv1_fusion/opus.py`
- OPUSV1 fusion head: `models/opusv1_fusion/opus_head.py`

---

## OPUSV1 forward (camera-only)

### Training forward (call chain)
`train.py` → `OPUSV1.loss` → `OPUSV1.forward_train` → `OPUSV1.extract_feat` → `OPUSV1Head.forward` → `OPUSV1Transformer.forward` → `OPUSV1Head.loss`

### Tensor shapes (core)
- Input image tensor: `img` is `[B, N, C, H, W]` where:
  - `B` batch size
  - `N` total views across frames
  - `C=3`
- `extract_feat` flattens to `[B*N, C, H, W]`, applies augment & norm, and re-groups to multi-level features:
  - `img_feats`: list of FPN levels, each `[B, N, C_l, H_l, W_l]`
- Head output:
  - `all_cls_scores`: list of `L` layers, each `[B, Q, P_l, C]`
  - `all_refine_pts`: list of `L` layers, each `[B, Q, P_l, 3]` (encoded coords)
  - `Q=num_query`, `P_l=num_refines[l]`, `C=num_classes`

### Data/meta updates
- `extract_feat` updates `img_metas[b]['img_shape']`, `ori_shape`, and `input_shape` based on **actual** resized/padded input.
- `img_metas` must contain `ego2img` and `ego2occ` for projection in the transformer.

### Inference forward
- `simple_test_offline`: runs full feature extraction for all frames at once.
- `simple_test_online`: frame-by-frame feature extraction with **cache** keyed by image filename, then reorganizes to the same shape expected by the head.

---

## OPUSV1Fusion forward (camera + LiDAR)

### Training forward (call chain)
`train.py` → `OPUSV1Fusion.loss` → `OPUSV1Fusion.forward_train` →
- image branch: `extract_img_feat` (same as OPUSV1)
- point branch: `voxelize → pts_voxel_encoder → pts_middle_encoder → pts_backbone → pts_neck → final_conv`
→ `OPUSV1FusionHead.forward` → `OPUSV1FusionHead.loss`

### Key differences vs OPUSV1
- Additional input: `points` (LiDAR point clouds).
- Point features are projected to a 2D map (`final_conv`) and fused in the head.
- Head can initialize query points using LiDAR (see next section).

---

## Query initialization logic (重点)

### OPUSV1 (camera-only)
- Query points are **learned embeddings**:
  - `OPUSV1Head._init_layers`: `nn.Embedding(num_query, 3)`
  - Initialized **uniformly** in `[0, 1]` (encoded space).

### OPUSV1Fusion
- Query points can be initialized with LiDAR points (`init_pos_lidar`):
  - `None`: use learned embeddings (same as OPUSV1).
  - `all` or `curr`: use LiDAR points + random fill.
- `query_init_mix` controls **FPS + random** mixing:
  - `lidar_ratio`, `random_ratio` decide split
  - `random_mode` controls random sampling strategy
- Uses FPS (`furthest_point_sample`) + optional uniform sampling in `pc_range`.
 - 具体实现要点（`models/opusv1_fusion/opus_head.py`）：
   - `get_init_position` 在 `torch.no_grad()` 下执行，初始化不会参与反向传播。
   - `init_pos_lidar='curr'` 且点云含时间戳时，会先筛选当前帧（`pts[:, -1] == 0`）。
   - `num_lidar/num_random` 由 `_resolve_query_init_mix` 决定：若 `query_init_mix.enabled=False`，则 **默认全用 LiDAR**（`num_lidar=num_query`）。
   - LiDAR 采样：`_sample_lidar_queries` 先做 FPS，若点数不足则 **重复采样** 或 **回退到 pc_range 均匀采样**。
   - Random 采样：`_sample_random_queries` 默认“先点云后均匀”，点云不够再用 `pc_range` 补齐。
   - 最终若 `mixed_points` 数量 ≠ `num_query`，会 **截断或补齐**，确保 query 数严格一致。
   - 初始化点最终通过 `encode_points` 映射到 `[0,1]` 归一化坐标（基于 `pc_range`）。

---

## Transformer + 4D sampling

### Transformer pipeline
- `OPUSV1Transformer` → `OPUSTransformerDecoder` → per-layer `OPUSTransformerDecoderLayer`.
- Each layer predicts:
  - `cls_score`: `[B, Q, P_l, C]`
  - `refine_pts`: `[B, Q, P_l, 3]`
- `num_refines` may differ by layer.

### OPUSSampling (关键逻辑)
Located in `models/opusv1/opus_transformer.py`.
- Inputs:
  - `query_points` (encoded), `query_feat`
  - `mlvl_feats` (multi-scale image feats)
  - `occ2img` projection matrix
- `occ2img` is built in `OPUSV1Transformer.forward` as `ego2img @ occ2ego`, where `occ2ego = inverse(ego2occ)` from `img_metas`.
- Steps:
  1. Decode `query_points` to xyz.
  2. Predict `sampling_offset` and `scale_weights` from `query_feat`.
  3. Build spatio-temporal sampling points: `[B, Q, T, G, P, 3]`.
  4. Call `sampling_4d` (in `models/opusv1/opus_sampling.py`).

### sampling_4d constraints
- Uses `occ2img` and **global** `image_h/w` from `img_metas[0]['img_shape'][0]`.
- `num_views` must match the real number of views; it is pulled from transformer.
- Only **one valid view** is kept per sampled point (argmax over valid view mask).

**Implication:** dynamic view counts or per-view resolutions require changes here.

---

## Occupancy conversion (get_occ)
In `OPUSV1Head.get_occ`:
1. Take final layer `refine_pts` + `cls_scores`.
2. Decode points and filter by distance + score.
3. Concatenate coords and class scores, pass through `Voxelization` to build sparse voxels.
4. If `padding=True`, apply 3D dilation/erosion to fill holes.
5. Output:
   - `sem_pred`: predicted class per voxel
   - `occ_loc`: voxel locations

---

## Loss design & GT pairing

### GT construction
- `get_sparse_voxels` converts dense voxel grid to sparse points via:
  - `pc_range` + `voxel_size`
  - filters by `empty_label`
  - `mask_camera` selects camera-visible voxels

### KNN pairing
- Two-direction pairing:
  - **pred → gt**: get nearest GT for each predicted point
  - **gt → pred**: get nearest pred for each GT

### Losses
- **Classification:** FocalLoss (`use_sigmoid=True`)
  - **重要：** `cls_weights` 长度必须等于 `num_classes`，否则会触发 shape mismatch。
- **Regression:** SmoothL1/L1 for point alignment

### Fusion head enhancements
OPUSV1Fusion head adds optional training strategies (via `train_cfg`):
- `tail_focus`: EMA tail class emphasis
- `hard_mining`: pred→gt hard pair mining
- `gt_balance`: per-class GT subsampling / repetition

---

## OPUSV1 vs OPUSV1Fusion — key differences
- **Inputs**:
  - OPUSV1: `img`
  - OPUSV1Fusion: `img` + `points`
- **Query init**:
  - OPUSV1: learned embeddings only
  - OPUSV1Fusion: optional LiDAR-based initialization
- **Head features**:
  - OPUSV1: only image features
  - OPUSV1Fusion: image + point features
- **Training策略**:
  - OPUSV1Fusion adds tail-focus / hard-mining / gt-balance hooks

---

## 常见改造点（改造入口建议）

### 1) 替换图像编码器
- 配置入口:
  - OPUSV1 / OPUSV1Fusion: `img_backbone`, `img_neck`
  - 若使用外部 encoder（如 fusion 版自定义 encoder），确保输出格式与 head 期望一致。
- 注意点:
  - 输出多层特征数需与 `num_levels` 对齐。
  - 通道数需与 `img_neck` / `transformer` 兼容。

### 2) 动态视角 / 动态分辨率
- 关键限制点:
  - `OPUSSampling` 使用 **单一** `image_h/w = img_metas[0][img_shape][0]`。
  - `sampling_4d` 假设 **所有视角同尺寸**，且 `num_views` 固定。
- 改造建议:
  - 让 `img_metas` 支持 per-view `img_shape`。
  - 修改 `sampling_4d` 使其按 view 使用对应的 `H/W`。
  - 确保 `ego2img` / `ego2occ` 与 view 数一致。

### 3) Query 采样策略
- OPUSV1Fusion 中 `query_init_mix` 是改造入口：
  - FPS vs random 比例可控制 query 覆盖范围
  - `random_mode` 可切换 uniform / point-first 行为

### 4) Loss/GT 设计调整
- 可在 `train_cfg` 增删 tail_focus / hard_mining / gt_balance。
- 修改 `get_sparse_voxels` 可影响 GT 分布与稀疏化策略。

---

## 常见踩坑/排查记录

- **mask_camera 全 0 → 空 GT**：
  - 现象：`get_sparse_voxels` 只保留 camera-visible voxels 后，某些样本 GT 为空，导致 KNN/loss 报错或 loss 为 NaN。
  - 建议：启用 `hard_camera_mask` 时增加 **fallback**（如 mask 全 0 时退回非空体素），或在 `loss_single` 中跳过空 GT。
  - 现象记录：当前数据集 `train.pkl` 统计到约 **13.73%** 样本 `mask_camera` 全 0。

- **KNN invalid configuration / CUDA error**：
  - 常见原因：`num_refines` 或 `num_query` 过大，预测点数量爆炸。
  - 建议：降低 `num_refines`（如 128→32），或结合 `hard_camera_mask`/GT 下采样减少 KNN 输入规模。

- **`cls_weights` 维度不匹配**：
  - 现象：FocalLoss shape mismatch。
  - 建议：确保 `len(cls_weights) == num_classes`。

- **`grad_norm` 为 NaN**：
  - 常见原因：`sampling_4d` 投影数值不稳定（AMP 下 `occ2img @ points` 溢出、除以近零深度、或 `num_views=1` 时 `i_view/(N-1)` 产生 NaN/Inf）。
  - 解决：在 `models/opusv1/opus_sampling.py` 使用 **fp32 投影 + nan_to_num + clamp**，并对 `num_views=1` 做除零保护。
  - 仍偶发时：可用 `SafeAmpOptimWrapper` 开启 `sanitize_nonfinite_grads=True`，并适当降低 `loss_scale`。

- **`ModuleNotFoundError: numpy._core`（PKL 与 NumPy 版本不匹配）**：
  - 常见原因：`*.pkl` 用 **NumPy 2.x** 序列化（pickle 里引用 `numpy._core.*`），但训练环境是 **NumPy 1.x**。
  - 解决：用 NumPy 1.x 重新生成 PKL（可用兼容 unpickler 将 `numpy._core` 映射到 `numpy.core`），或升级训练环境 NumPy 到 2.x（需确认依赖兼容）。

---

## Consistency checklist (doc only)
- 文档不包含任何数据路径 / 训练命令；允许保留关键踩坑记录与统计。
- 所有函数名、文件名、关键字段与代码一致。
