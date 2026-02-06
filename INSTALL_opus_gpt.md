# OPUS `opus_gpt` Environment Setup

This document records a tested environment setup that can successfully start:

```bash
bash dist_train.sh 8 /root/wjl/OPUS_mmcv2/configs/opusv1-fusion_nusc-occ3d/tartanground-t_r50_640x640_8f_nusc-occ3d_100e.py
```

## 1) Prerequisites

- Linux + NVIDIA GPU
- `nvidia-smi` works
- CUDA Toolkit 12.1 available (`/usr/local/cuda/bin/nvcc --version`)
- Conda installed

## 2) Create Conda Environment

```bash
conda create -n opus_gpt python=3.10 -y
conda activate opus_gpt
python -m pip install -U pip
```

## 3) Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

## 4) Install OpenMMLab Core Packages

```bash
pip install openmim
mim install mmengine==0.10.7
mim install mmdet==3.3.0
mim install mmdet3d==1.4.0
```

## 5) Install Sparse/Graph CUDA Dependencies

```bash
pip install spconv-cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

## 6) Install Project Runtime Dependencies

```bash
pip install \
  numpy==1.24.1 \
  packaging==24.1 \
  matplotlib==3.5.3 \
  opencv-python==4.8.1.78 \
  einops==0.8.0 \
  fvcore==0.1.5.post20221221 \
  prettytable==3.10.2 \
  psutil==7.2.2 \
  regex==2024.7.24 \
  scipy==1.14.0 \
  termcolor==2.4.0 \
  pyturbojpeg==1.8.2 \
  onnx==1.16.2 \
  rich==13.4.2
```

## 7) Install MMCV With Ops (Required)

This project needs `mmcv.ops`. Install from the repo-local `./mmcv` source with CUDA ops enabled.

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export MMCV_WITH_OPS=1
export FORCE_CUDA=1

python -m pip install --force-reinstall --no-deps --no-build-isolation ./mmcv
```

## 8) Build Project CUDA Extensions

```bash
cd models/csrc
python setup.py build_ext --inplace
cd ../..
```

## 9) Verify Environment

```bash
python - <<'PY'
import torch, torchvision, mmcv, mmengine, mmdet, mmdet3d
import mmcv.ops
import models, loaders
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('mmcv:', mmcv.__version__)
print('mmengine:', mmengine.__version__)
print('mmdet:', mmdet.__version__)
print('mmdet3d:', mmdet3d.__version__)
print('Import check passed.')
PY
```

Expected tested versions:

- `torch 2.2.0+cu121`
- `torchvision 0.17.0+cu121`
- `mmcv 2.1.0` (with ops)
- `mmengine 0.10.7`
- `mmdet 3.3.0`
- `mmdet3d 1.4.0`
- `numpy 1.24.1`

## 10) Start Training

```bash
bash dist_train.sh 8 /root/wjl/OPUS_mmcv2/configs/opusv1-fusion_nusc-occ3d/tartanground-t_r50_640x640_8f_nusc-occ3d_100e.py
```

## 11) Optional Smoke Test (Only Check Startup)

```bash
timeout 180 bash dist_train.sh 8 /root/wjl/OPUS_mmcv2/configs/opusv1-fusion_nusc-occ3d/tartanground-t_r50_640x640_8f_nusc-occ3d_100e.py
```

If logs show MMEngine config printing and distributed workers start, environment setup is correct. `timeout` stopping the job is expected.

## Troubleshooting

- `ModuleNotFoundError: No module named 'mmcv._ext'`
  - MMCV was installed without ops. Re-run step 7.
- `MMCV==2.2.0 is incompatible`
  - Use repo-local MMCV 2.1.0 build from step 7.
- NumPy 2.x related ABI warnings
  - Ensure `numpy==1.24.1` and `opencv-python==4.8.1.78`.
