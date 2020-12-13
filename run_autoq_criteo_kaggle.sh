#!/usr/bin/env bash
export PYTHONPATH=/path/to/autox

CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch_auto_nncf.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot=13-512-256-64-16 \
--arch-mlp-top=512-256-1 \
--data-generation=dataset \
--data-set=kaggle \
--raw-data-file=./input/train.txt \
--processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=128 \
--print-freq=1 \
--print-time \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--use-gpu \
--dataset-multiprocessing \
--test-freq=1 \
--load-model=ckpt/criteo_kaggle.pt \
--inference-only \
--nncf_config=dlrm_kaggle_autoq_nncfcfg.json \
--log-dir=/tmp/autoq-dlrm-runs
