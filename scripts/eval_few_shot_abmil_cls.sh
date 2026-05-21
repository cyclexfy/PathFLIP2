#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ARGS_PATH=${ARGS_PATH:-outputs/pathflip/lightning_logs/version_0/hparams.yaml}
CKPT_PATH=${CKPT_PATH:-outputs/pathflip/checkpoint/pytorch_model.bin}
DATASETS=${DATASETS:-CPTAC_NSCLC}

python -m src.eval.eval_few_shot_abmil_classification \
    --args_path "$ARGS_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --datasets $DATASETS \
    --k_shot "${K_SHOT:-16}" \
    --num_runs "${NUM_RUNS:-10}" \
    --seed "${SEED:-777}" \
    --max_iter "${MAX_ITER:-300}" \
    --abmil_hidden_dim "${ABMIL_HIDDEN_DIM:-128}" \
    --abmil_lr "${ABMIL_LR:-1e-4}" \
    --accum_steps "${ACCUM_STEPS:-4}"
