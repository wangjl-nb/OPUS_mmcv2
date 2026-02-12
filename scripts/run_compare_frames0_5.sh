#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${REPO_ROOT}/configs/opusv1-fusion_nusc-occ3d/tartanground-t_r50_640x640_8f_nusc-occ3d_100e.py}"
WEIGHTS_PATH="${2:-/root/wjl/OPUS_mmcv2/outputs/OPUSV1Fusion/tartanground-t_r50_640x640_8f_nusc-occ3d_100e_2026-02-09/20-37-48/epoch_100.pth}"
SAVE_DIR="${3:-${REPO_ROOT}/outputs}"
MAX_SAMPLES="${4:-5}"

python "${REPO_ROOT}/scripts/compare_multiframe_and_query_vis.py" \
  --config "${CONFIG_PATH}" \
  --weights "${WEIGHTS_PATH}" \
  --split val \
  --history-a 0 \
  --history-b 5 \
  --compare-batch-size 1 \
  --compare-num-workers 4 \
  --max-samples "${MAX_SAMPLES}" \
  --output-format ply \
  --max-voxels 3000000000 \
  --compare-save-dir "${SAVE_DIR}" \
  --vis-save-dir "${SAVE_DIR}"
