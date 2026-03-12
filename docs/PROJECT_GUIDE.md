# OPUS_mmcv2 Project Guide (Model Logic)

## 文档定位
- 本文档只保留**项目级通用知识**（结构、调用链、关键开关、常见排查）。
- 任务级进展、实验结论、临时决策统一记录在 `TASK_WORKORDER_CURRENT.md`。
- 新 session 建议先读：`TASK_WORKORDER_CURRENT.md` -> `PROJECT_GUIDE.md`。

## 文档导航
- 快速定位：Repo layout / 需求到代码入口映射 / 1 分钟排查模板
- 模型主链：OPUSV1 / OPUSV1Fusion forward、query 初始化、4D/TPV sampling
- 训练核心：GT 构造、KNN 配对、loss 字段释义
- 专项主题：Depth points 切换、MapAnything 融合、Fusion 消融、TPV-Lite
- 收尾检查：常见踩坑与 consistency checklist

## 可用的python环境
```bash
conda activate opus-mmcv2
```

## Repo layout (high level)
- `configs/`: model & dataset configs (OPUSV1/OPUSV2, fusion variants)
- `models/`: core model implementations (OPUSV1, OPUSV1Fusion, heads, transformers)
- `loaders/`: datasets + pipelines + metrics
- `scripts/data/`: dataset metadata utility scripts (ann-file maintenance)
- `train.py`, `val.py`: training/validation entry points

## Office baseline config
- Canonical Office baseline source config:
  - `configs/opusv1-fusion_nusc-occ3d/TT_Office_baseline.py`
- Historical Office run/output directories may still use older long experiment names.
  - Treat the short config above as the source-of-truth baseline and the long names as archived run labels only.
- Recommended Office config naming going forward:
  - baseline: `TT_Office_baseline.py`
  - single-topic variant: `TT_Office_<topic>.py`
  - combined variant: `TT_Office_<topic1>_<topic2>.py`
  - Do not encode stable baseline attributes like input size, TPV, GT-depth, binary-occ, or PCA256 in new filenames unless they are the experimental variable.

## Dataset ann metadata tooling (generic)
- Incremental depth metadata injection script:
  - `scripts/data/add_depth_paths_to_ann.py`
  - Purpose: add optional `depth_path` into `infos[*].cams[*]` and `infos[*].cam_sweeps[*][*]` without touching original pkl.
  - Supports:
    - `--skip-if-exists` (default true, only fill missing keys)
    - `--strict-exists` (fail on missing mapped depth files)
    - `--report-json` (emit summary + missing samples)
  - Typical usage:
    - `python3 scripts/data/add_depth_paths_to_ann.py --in-pkl <src>.pkl --out-pkl <dst>_with_depth.pkl --report-json <dst>.depth_report.json`

## Point-source switch (LiDAR / Depth)
- New depth-to-points transform:
  - `loaders/pipelines/loading.py` -> `LoadPointsFromMultiViewDepth`
  - Converts selected multi-view depth maps into pseudo-LiDAR points (`[x, y, z, intensity, time]`).
- Query init compatibility:
  - Keep `init_pos_lidar='curr'` unchanged.
  - Depth mode sets current-frame points to `time=0`, history sweeps `time>0`, so current-only query init still works.
- Config switch example:
  - `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_depth_points_switch.py`
  - `point_input_source = 'depth'` or `'lidar'`

## TPV-Lite feature branch (Fusion)
- Purpose:
  - Keep depth->pseudo-points input path, but replace 2D BEV point-feature sampling with tri-plane TPV sampling (`XY/XZ/YZ`) so Z-axis structure is retained.
  - Reuse sparse 3D backbone features from `SparseEncoder` and convert to TPV after a lightweight 3D FPN-style upsample + skip residual fusion.
  - In TPV-only mode (`enable_pts_feature_branch=False`), freeze `pts_middle_encoder.conv_out` to avoid DDP unused-parameter reduction errors.
- Main code:
  - Encoder: `models/lidar_encoder/tpv_lite_encoder.py` (`TPVLiteEncoder`)
  - Sampling: `models/opusv1_fusion/opus_sampling.py` (`sampling_tpv_feats`)
  - Routing: `models/opusv1_fusion/opus.py` / `models/opusv1_fusion/opus_head.py` / `models/opusv1_fusion/opus_transformer.py`
- Config entry:
  - `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e_tpv_lite_depth_gt.py`
  - Key switches:
    - `model.enable_tpv_feature_branch=True`
    - `model.enable_pts_feature_branch=False`
    - `model.pts_middle_encoder.return_middle_feats=True`
    - `transformer.use_tpv_sampling=True`
    - `transformer.use_pts_sampling=False`

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

## Session quick-start（通用）
- 新 session 默认读取顺序：
  1. `TASK_WORKORDER_CURRENT.md`（任务上下文）
  2. `PROJECT_GUIDE.md`（项目通用知识）
- 若 `TASK_WORKORDER_CURRENT.md` 不存在，先从 `TASK_WORKORDER_TEMPLATE.md` 复制一份再开始记录。
- 先看当前 run 的完整日志：`outputs/<exp_name>/<date_or_tag>/<run_id>/<run_id>.log`
- 再看同目录冻结配置副本：`outputs/<exp_name>/<date_or_tag>/<config_name>.py`
- 定位 loss 逻辑优先读：`models/opusv1_fusion/opus_head.py`（`loss_single` 与 `loss`）
- 核对开关优先关注：`init_pos_lidar`、`query_init_mix`、`tail_focus`、`hard_mining`、`gt_balance`

## 新 session 需求定位（建议流程）
1. **先把需求归类**（只选一个主类）：
   - 训练效果问题（loss/收敛/精度）
   - 模型结构改造（backbone/head/transformer）
   - 数据与标注问题（dataset/pipeline/GT 构造）
   - 推理结果问题（get_occ/阈值/后处理）
   - 速度或显存问题（num_query/num_refines/采样）
2. **从配置到代码双向定位**：
   - 先从 config 找字段（如 `train_cfg.pts.*`、`pts_bbox_head.*`）
   - 再跳到对应实现（head/transformer/dataset）
3. **只读最小必要文件**（避免上下文过载）：
   - 先读入口函数，再按调用链下钻
4. **输出结论时固定模板**：
   - 现象 -> 代码位置 -> 原因假设 -> 验证方式 -> 修改建议

## 需求到代码入口映射（最常用）
- 改训练 loss/匹配/权重：`models/opusv1_fusion/opus_head.py`
- 改 query 初始化策略：`models/opusv1_fusion/opus_head.py`（`get_init_position` 相关）
- 做 Fusion 消融（去 LiDAR 特征但保留 LiDAR 初始化）：`models/opusv1_fusion/opus.py`（`drop_lidar_feat`）
- 改 transformer 采样/投影：`models/opusv1/opus_transformer.py`, `models/opusv1/opus_sampling.py`
- 改 detector 主流程：`models/opusv1/opus.py`, `models/opusv1_fusion/opus.py`
- 改数据读取与评估：`loaders/`（dataset + pipeline + metrics）
- 改实验行为（超参/开关）：`configs/`

## 项目架构速览（文字版）
- **配置层**：`configs/` 决定模块拼装与训练/测试开关
- **模型层**：
  - detector（组织多模态输入与训练/推理流程）
  - head（query refinement、target matching、loss）
  - transformer/sampling（跨视角时空采样与特征聚合）
- **数据层**：`loaders/` 提供样本、变换、GT、评估协议
- **运行层**：`train.py` / `val.py` 负责 runner、hook、日志与 checkpoint

## 1 分钟排查模板（新 session 直接复用）
1. **任务一句话**：我要解决什么问题？验收标准是什么？
2. **配置定位**：本次实际运行用的 config 是哪个？核心开关值是什么？
3. **日志定位**：问题首次出现在哪个 epoch/iter？伴随哪些指标异常（`loss`/`grad_norm`/memory）？
4. **代码定位**：该问题主链路在哪个文件和函数（detector/head/transformer/dataset）？
5. **最小验证**：先做一个最小改动验证假设（只改 1~2 个开关或 1 处逻辑）。
6. **结果记录**：保留“现象 -> 改动 -> 结果 -> 下一步”四行结论，便于接力。

## Demo / Visualization 约定
- 所有新的 demo、PLY、query 可视化结果统一放在 `/root/wjl/OPUS_mmcv2/demos`。
- 导出预测 PLY 时，要先确认脚本是否默认应用 `mask_camera`。
  - 当前多帧导出脚本默认会按 `mask_camera` 裁掉预测体素。
  - 若需要查看完整预测点云，应显式关闭该裁剪开关，而不是只改评估配置。

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
   - `num_lidar/num_random` 由 `_resolve_query_init_mix` 决定：若 `query_init_mix.enabled=False`，在走 LiDAR 初始化分支时会 **默认全用 LiDAR**（`num_lidar=num_query`）。
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
  - `cls_weights` 会在代码里自动 pad/truncate 到 `num_classes`（`models/opusv1_fusion/opus_head.py`），但建议配置里直接对齐，避免隐式权重错误。
- **Regression:** SmoothL1/L1 for point alignment

### Fusion head enhancements
OPUSV1Fusion head adds optional training strategies (via `train_cfg`):
- `tail_focus`: EMA tail class emphasis
- `hard_mining`: pred→gt hard pair mining
- `gt_balance`: per-class GT subsampling / repetition

### 训练日志 loss 字段对照（通用）
- `loss`: 总损失（mmengine 会把所有带 `loss` 的项求和打印）
- `init_loss_pts`: 初始 query 点的回归损失（仅当 `init_pos_lidar=None` 时存在）
- `loss_cls` / `loss_pts`: 最后一层 decoder（第 6 层）损失
- `d0.loss_cls` ~ `d4.loss_cls`: 前 5 层 decoder 的分类辅助损失
- `d0.loss_pts` ~ `d4.loss_pts`: 前 5 层 decoder 的回归辅助损失

> 当前头部的总损失结构可写成：  
> `loss = init_loss_pts + Σ_{k∈{d0,d1,d2,d3,d4,final}} (loss_cls_k + loss_pts_k)`  
> 其中 `final` 对应日志里的 `loss_cls/loss_pts`。  
> 按打印精度回算时，`loss` 与分解结果通常只会有极小舍入误差。

### `loss_cls` / `loss_pts` 在代码里的真实含义
- `loss_cls`（`loss_single`）：
  - FocalLoss，输入是每个预测点的类别分数与 KNN 匹配得到的 GT 类别。
  - 样本权重 = `cls_weights × 距离权重`，若开启 `tail_focus`，尾部类会再乘 `tail_weight`。
- `loss_pts`（`loss_single`）由两部分相加：
  - **gt→pred**：所有 GT 点到其最近预测点的回归损失（带 `gt_pts_weights`，包含空体素/尾类加权）
  - **pred→gt**：预测点到最近 GT 点的回归损失（可经 `hard_mining` 保留 top-k 难样本）
  - 两部分都调用 `loss_pts` 配置（例如 `SmoothL1Loss(beta=0.2, loss_weight=0.5)`），再相加输出

### loss 曲线通用解读建议
- 看趋势优先按 epoch 均值，不要被单个 iter 尖峰误导
- `loss_cls` 比 `loss_pts` 更容易抖动，尤其在 tail reweight / focal 下属于常见现象
- `loss_pts` 若长期不降，优先检查 GT 稀疏化与匹配（`get_sparse_voxels`、KNN 配对、hard_mining）
- 出现偶发大尖峰时，联动看 `grad_norm`、batch 数据分布、以及 tail_focus/hard_mining 参数是否过激

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

### 5) Fusion LiDAR 特征消融（保留初始化）
- 推荐做法：保持 `input_modality.use_lidar=True` 与 points pipeline 不变，只将 `model.drop_lidar_feat=True`。
- 含义：LiDAR points 仍可用于 `init_pos_lidar` 初始化 query，但进入 head 的 `pts_feats` 被置零。
- 优点：对比更干净（只消融 LiDAR feature contribution，不破坏 query 初始化路径）。
- 实现注意：置零要用“保留计算图”的方式（如 `pts_feats * 0.0`），不要直接 `torch.zeros_like` 覆盖，否则 DDP 可能报 unused-parameter/reduction 错误。
- 常用配置（当前分支未单独保留 ablation 配置文件，建议基于主配置开关）：
  - 仅去 LiDAR 特征（保留 LiDAR 初始化）：基于 `configs/opusv1-fusion_nusc-occ3d/tartanground_demo_r50_640x640_9f_100e.py`，设置 `model.drop_lidar_feat=True`
  - 同时去 LiDAR 初始化 + LiDAR 特征：在上一项基础上，同时关闭 LiDAR 初始化相关开关

---

## MapAnything 外部分支（当前 pure bilinear path）

### 当前办公室 flat 配置语义
- 当前组合配置以 `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_150e_tpv_gt-depth_bilinear_flat.py` 为准。
- 图像路径走 **纯外部编码器**：
  - `model.use_external_img_encoder=True`
  - `img_backbone=None`
  - `img_neck=None`
  - `img_feature_fusion=None`
- 含义：不再走 `ResNet/FPN` 主分支，也不再做“FPN + MapAnything weighted sum”；直接使用 `MapAnythingOccEncoder` 产出的多层图像特征。
- 组合语义可以概括为：`MapAnything only + bilinear pyramid + GT-depth pseudo points + TPV`。

### 模块职责（通用结构）
- `models/mapanything/input_adapter.py`  
  将 OPUS 的 `img[B,TN,C,H,W] + img_metas + optional points/mapanything_extra` 转成 MapAnything 的 `views(list[dict])` 输入格式（单视角 HWC 图像 + 视角元数据）。
- `models/mapanything/opus_mapanything_wrapper.py`  
  负责 MapAnything 模型加载（支持 `from_pretrained`）、预处理（`preprocess_inputs`）、前向与设备/冻结控制，以及 bilinear pyramid 适配。
- `models/mapanything/output_adapter.py`  
  将 MapAnything 输出统一为 OPUS 可消费的张量格式：`[B, TN, C, Hf, Wf]`。
- `models/backbones/mapanything_occ_encoder.py`  
  作为可注册 backbone（`type='MapAnythingOccEncoder'`），提供 frame chunk 前向、TN 对齐和 OPUS 接口桥接。
- `models/opusv1_fusion/opus.py`（detector 融合点）  
  当 `use_external_img_encoder=True` 时，图像路径会直接走外部编码器返回多层特征，绕过内部 `ResNet/FPN` 主分支。

### Bilinear pyramid 模式与 `anyup_cfg` 语义
- 当前配置中的 `img_encoder.anyup_cfg.enabled=True` 只是启用“金字塔适配路径”，不是启用 AnyUp 网络本体。
- 真正决定是否实例化 AnyUp 模型的是 `img_encoder.anyup_cfg.mode`：
  - `mode='anyup'`：加载 AnyUp checkpoint，走 AnyUp 网络前向。
  - `mode='bilinear'`：不创建 AnyUp 模型，改为使用 `F.interpolate(...)` 做上采样与金字塔构建。
- 因为 OPUS transformer 仍然期望 `num_levels=4`，所以当前 pure-MapAnything 配置必须保留 `enabled=True`，让 wrapper 把单层 MapAnything 特征扩成 4 层 bilinear pyramid。
- 当前 bilinear pyramid 还会通过 `output_channels=256` 把各层投影到 OPUS `embed_dims`，以匹配 transformer 输入通道。

### 分辨率约束与 `640 -> 630` 问题
- MapAnything 主干使用 `patch_size=14`，`fixed_size` 预处理会把输入宽高向下对齐到 14 的整数倍。
- 这意味着如果你把 `mapanything_preprocess_cfg.size=(640,640)`，MapAnything 实际看到的不是 `640x640`，而是 `630x630`。
- 旧问题的根源是：
  - OPUS 主链 `ida_aug_conf.final_dim=640`，因此 `img_metas['img_shape']` 按 `640` 记录；
  - MapAnything `fixed_size + patch_size=14` 实际会落到 `630`；
  - 两条分支的几何/采样分辨率不一致，容易造成投影归一化错位。
- 当前修正后的建议是：
  - 原始输入图仍按 `H=640, W=640` 记录；
  - `ida_aug_conf.final_dim=(630,630)`；
  - `mapanything_preprocess_cfg.size=(630,630)`；
  - 两条链路都使用同一 patch-aligned 最终尺寸。

### 调试 checklist（当前 pure-MapAnything 配置）
1. **branch**：确认 `use_external_img_encoder=True` 且 `img_feature_fusion=None`；否则你不是在 pure-MapAnything 路径上。
2. **mode**：确认 `anyup_cfg.enabled=True` 且 `mode='bilinear'`；这表示启用 bilinear pyramid，不表示启用 AnyUp 网络。
3. **resolution**：始终同步修改 `ida_aug_conf.final_dim` 与 `mapanything_preprocess_cfg.size`；对当前 `640` 原图与 `patch_size=14`，推荐两者都设为 `630`。
4. **meta**：`img_metas['img_shape'/'pad_shape'/'input_shape']` 必须与实际最终图像尺寸一致，否则 `sampling_4d` 投影会错位。
5. **shape**：630 输入下，raw MapAnything token map 恢复后应接近 `45x45`；随后 bilinear pyramid 应给出 4 层图像特征供 `transformer.num_levels=4` 使用。
6. **legacy notes**：旧的 `616 + size_divisor=8 + ResNet/FPN weighted fusion` 方案属于历史融合路径，不再适用于当前办公室 flat 配置。

---

## Binary-Occ + PCA128 语义特征监督（办公室当前实验）

### 当前配置
- 当前 feature-learning 主实验以 `configs/opusv1-fusion_nusc-occ3d/TT_Office_mapanything_640x640_9f_150e_tpv_gt-depth_binary-occ_pca128_feat_flat.py` 为准。
- 该配置替代了已删除的旧版 `..._pca128_feat_flat.py`（旧版是“79 类语义 cls + 128-d feature”并行监督）。

### 设计目标
- 将“有没有东西”和“是什么类别”拆开：
  - `occ score` 只负责 occupiedness/query gating；
  - `128-d feat` 只负责语义 prototype 对齐与最终类别解码。
- 这样可以避免旧版中 `cls head` 和 `feat head` 同时学习语义、但最终只信 feature 分支的目标冲突。

### 训练逻辑
- `models/opusv1_fusion/opus_transformer.py`
  - `transformer.score_mode='binary_occ'` 时，decoder 输出单通道 `occ score`，不再输出 79 类语义 logits。
  - `feat_branch` 保留，继续从同一个 `query_feat` 输出 `128-d` semantic feature。
- `models/opusv1_fusion/opus_head.py`
  - `loss_occ`：根据预测点到最近 GT 非空体素的距离构造二值 occupiedness target。
  - `occ_target_cfg.pos_dist_thr=0.10`：判为正样本。
  - `occ_target_cfg.neg_dist_thr=0.20`：判为负样本。
  - 中间带 `(0.10, 0.20)` 忽略，不参与 `loss_occ`。
  - `loss_feat`：只对正样本计算，目标来自 `office79_prototypes_ae128.npz` 中的 `latent_128` prototype。
  - 当前默认权重：
    - `loss_occ` focal weight = `1.0`
    - `loss_feat` cosine = `2.0`
    - `loss_feat` mse = `0.1`

### 推理解码
- 先用 `occ score` 对 refined points 做筛选，再聚合到 voxel。
- 聚合后的 voxel 继续用二值 `occ score` 做保留。
- 仅对保留下来的 voxel，用 `128-d voxel feature` 与 prototype bank 做 cosine similarity。
- `argmax(similarity)` 输出最终语义类别。

### 调试 checklist（binary-occ feature 配置）
1. **score mode**：确认 `transformer.score_mode='binary_occ'` 且 `occ_out_channels=1`。
2. **feature supervision**：确认 `feature_supervision.only_positive=True`，否则 feature loss 会被负样本污染。
3. **thresholds**：确认 `occ_target_cfg.pos_dist_thr <= neg_dist_thr`；默认 `0.10 / 0.20`。
4. **test gating**：确认 `model.test_cfg.pts.score_thr=0.1`；过高会在 early epoch 误删 feature queries。
5. **single-GPU val/test**：测试管道必须让多帧图像走 offline path；当前配置已设置 `LoadMultiViewImageFromMultiSweeps(..., force_offline=True)`。
6. **mapanything_extra**：`loaders/pipelines/loading.py` 当前已兼容 online val/test 场景下 `filename` 长于 `img` 的 batch；如果后续修改这段逻辑，不要回退这个兼容性修复。

### 综合升级方向（PCA256 + mixed semantic loss）
- 若 `binary-occ + PCA128 regression` 已经把 `IoU` 拉高但 `mIoU` 仍然偏低，优先考虑：
  - 把文本 prototype 从 `PCA128` 升到 `PCA256`
  - 将 semantic supervision 从纯 `cosine/mse regression` 改为混合损失：
    - `cosine prototype alignment`
    - `prototype CE`
    - `hard-negative margin`
  - 同时扩大 semantic 正样本覆盖（强正样本 + 弱正样本）并增加尾类权重
- 这条路线的目的不是重新训练一个闭集分类头，而是在保留开放词表语义几何的前提下提高当前数据集的类别可分性。

---

## 常见踩坑/排查记录

- **mask_camera 全 0 → 空 GT**：
  - 现象：`get_sparse_voxels` 只保留 camera-visible voxels 后，某些样本 GT 为空，导致 KNN/loss 报错或 loss 为 NaN。
  - 建议：启用 `hard_camera_mask` 时增加 **fallback**（如 mask 全 0 时退回非空体素），或在 `loss_single` 中跳过空 GT。
  - 经验：多数据集都会出现该问题，建议在数据检查脚本里统计 `mask_camera` 全 0 比例。

- **KNN invalid configuration / CUDA error**：
  - 常见原因：`num_refines` 或 `num_query` 过大，预测点数量爆炸。
  - 建议：降低 `num_refines`（如 128→32），或结合 `hard_camera_mask`/GT 下采样减少 KNN 输入规模。

- **`cls_weights` 维度不匹配**：
  - 现象：虽然当前代码会自动 pad/truncate，但会出现“训练能跑、权重却和预期不一致”的隐性问题。
  - 建议：仍然手动保证 `len(cls_weights) == num_classes`，并与 `occ_names` 顺序逐项对齐。

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
