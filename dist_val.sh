#!/usr/bin/env bash
GPUS=$1
CONFIG=$2
WEIGHT=$3
MASTER_PORT=${MASTER_PORT:-${4:-29500}}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VAL_PY="${SCRIPT_DIR}/val.py"

python3 -m torch.distributed.run \
  --master_port "${MASTER_PORT}" \
  --nproc_per_node "${GPUS}" \
  "${VAL_PY}" \
  --config "${CONFIG}" \
  --weights "${WEIGHT}"
