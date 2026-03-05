#!/usr/bin/env bash
GPUS=$1
CONFIG=$2
OPUS_DISABLE_MSMV_CUDA=${OPUS_DISABLE_MSMV_CUDA:-1}
OPUS_RUN_TIMESTAMP=${OPUS_RUN_TIMESTAMP:-$(date +%Y-%m-%d/%H-%M-%S)}
PYTHONWARNINGS="ignore:torch.utils.checkpoint:UserWarning,ignore:The torch.cuda.*DtypeTensor constructors are no longer recommended.:UserWarning" \
OPUS_DEBUG_FINITE=1 \
OPUS_DISABLE_MSMV_CUDA=$OPUS_DISABLE_MSMV_CUDA \
OPUS_RUN_TIMESTAMP=$OPUS_RUN_TIMESTAMP \
python3 -m torch.distributed.run --nproc_per_node $GPUS train.py --config $CONFIG ${@:3}
