#!/usr/bin/env bash
GPUS=$1
CONFIG=$2
PYTHONWARNINGS="ignore:torch.utils.checkpoint:UserWarning,ignore:The torch.cuda.*DtypeTensor constructors are no longer recommended.:UserWarning" \
OPUS_DEBUG_FINITE=1 python3 -m torch.distributed.run --nproc_per_node $GPUS train.py --config $CONFIG ${@:3}
