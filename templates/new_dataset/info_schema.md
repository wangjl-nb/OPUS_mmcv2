# ann_file / GT Schema (Occ3D)

This document describes the keys required by current training/eval pipelines.

## 1) ann_file format

Your `train.pkl` / `val.pkl` / `test.pkl` should be one of:

- `{'infos': list[dict]}` (recommended)
- `{'data_list': list[dict]}`
- `list[dict]`

Each sample `info` is shown below.

## 2) Required keys in each `info`

### 2.1 Identity and time

- `token` (str)
  - Unique sample id.
  - Used by GT path template, e.g. `{token}`.
- `scene_name` (str)
  - Scene id/name.
  - Used by GT path template, e.g. `{scene_name}`.
- `timestamp` (int or float)
  - Timestamp in microseconds.

### 2.2 Ego / lidar pose

- `ego2global_translation` (array-like, shape `[3]`)
- `ego2global_rotation` (array-like, shape `[4]`, quaternion in `[w, x, y, z]`)
- `lidar2ego_translation` (array-like, shape `[3]`)
- `lidar2ego_rotation` (array-like, shape `[4]`, quaternion in `[w, x, y, z]`)

### 2.3 Cameras

- `cams` (dict)
  - key: camera name (`CAM_FRONT`, ...)
  - value: camera dict with required keys:
    - `data_path` (str): image path
    - `timestamp` (int/float): microseconds
    - `sensor2global_translation` (array-like, `[3]`)
    - `sensor2global_rotation` (array-like, `[3, 3]` rotation matrix)
    - `cam_intrinsic` (array-like, `[3, 3]`)

### 2.4 Camera sweeps

- `cam_sweeps` (list)
  - each element is `dict[cam_name -> cam_dict]`
  - `cam_dict` has the same fields as in `cams`

### 2.5 Lidar

- `lidar_path` (str)
- `lidar_points` (dict)
  - required field: `lidar_path` (str)
- `lidar_sweeps` (list)
  - each element has:
    - `data_path` (str)
    - `timestamp` (int/float, microseconds)
    - `sensor2lidar_rotation` (array-like, `[3, 3]`)
    - `sensor2lidar_translation` (array-like, `[3]`)

### 2.6 Optional keys (for occupancy task)

- `scene_token` (str)
- `lidar_token` (str)

These are needed if you use `LoadOccupancyFromFile` / `OccupancyMetric` path templates with `{scene_token}` and `{lidar_token}`.

## 3) GT file schema (`labels.npz`)

For Occ3D loader (`LoadOcc3DFromFile`), one GT file should contain:

- `semantics`: `np.ndarray[int]`, shape `[W, H, Z]`
- `mask_camera`: `np.ndarray[bool or uint8]`, shape `[W, H, Z]`
- `mask_lidar`: `np.ndarray[bool or uint8]`, shape `[W, H, Z]`

Notes:

- Three arrays must have exactly the same shape.
- `semantics` label ids must match your config class mapping.
- `empty_label` in config must match your dataset definition.

## 4) Keys emitted by `get_data_info()` (adapter output)

Your dataset class should produce these keys for current fusion pipeline:

- Core/meta:
  - `sample_idx` (int)
  - `sample_token` (str)
  - `scene_name` (str)
  - `timestamp` (float, seconds)
- Geometry:
  - `ego2lidar` (`[4, 4]`)
  - `ego2obj` (`[4, 4]`)
  - `ego2occ` (`[4, 4]`)
  - `ego2global_translation`, `ego2global_rotation`
  - `lidar2ego_translation`, `lidar2ego_rotation`
- Camera branch:
  - `img_filename` (list[str])
  - `img_timestamp` (list[float])
  - `ego2img` (list[`[4, 4]`])
  - `cam_sweeps` (dict with `prev` and `next` lists)
  - `cam_types` (list[str])
  - `num_views` (int)
- Lidar branch:
  - `pts_filename` (str)
  - `lidar_points` (dict with `lidar_path`)
  - `lidar_sweeps` (dict with `prev` and `next` lists)

## 5) Frequent mistakes

- Quaternion order is wrong (`[x, y, z, w]` instead of `[w, x, y, z]`).
- `sensor2global_rotation` is quaternion, but code expects a `[3, 3]` matrix in `cams`.
- Missing `lidar_points.lidar_path` (LoadPointsFromFile reads this key).
- `cam_sweeps` entries missing some camera keys.
- GT path template placeholders do not exist in `info` (e.g. template uses `{token}` but info has no `token`).
