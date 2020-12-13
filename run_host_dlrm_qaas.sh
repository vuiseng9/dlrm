#!/usr/bin/env bash

AUTOX_ROOT=/workspace/autox
DLRM_ROOT=/workspace/dlrm

export PYTHONPATH=$DLRM_ROOT:$AUTOX_ROOT:$PYTHONPATH

export config=$AUTOX_ROOT/autox/server/qaas/cfg/dlrm_kaggle_autoq_qaas.json
export qenv_name=dlrm
export data=$DLRM_ROOT/input
export weight=$DLRM_ROOT/ckpt/criteo_kaggle.pt

export GPU_ID=0

cd $AUTOX_ROOT/autox/server/ && \
CUDA_VISIBLE_DEVICES=$GPU_ID python3 qaas/manage.py \
    run \
    --no-reload \
    --host=0.0.0.0
# (optional)    --port=5005
