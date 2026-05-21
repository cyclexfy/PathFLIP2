#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

CKPT_ID=${CKPT_ID:-pathflip}
ARGS_PATH=${ARGS_PATH:-outputs/pathflip/lightning_logs/version_0/hparams.yaml}
CKPT_PATH=${CKPT_PATH:-outputs/pathflip/checkpoint/pytorch_model.bin}
EVAL_SCRIPT=${EVAL_SCRIPT:-src.eval.eval_zero_shot_classification}
BATCH_SIZE=${BATCH_SIZE:-1}
DATASETS=${DATASETS:-CPTAC_NSCLC}

if [ ! -f "$ARGS_PATH" ]; then
    echo "Args config file not found: $ARGS_PATH"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "Checkpoint file not found: $CKPT_PATH"
    exit 1
fi

python -m "$EVAL_SCRIPT" \
    --args_path "$ARGS_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --ckpt_id "$CKPT_ID" \
    --datasets $DATASETS \
    --batch_size "$BATCH_SIZE"
