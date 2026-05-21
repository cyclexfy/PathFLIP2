#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

python -m src.eval.eval_retrieval \
    --args_path "${ARGS_PATH:-outputs/pathflip/lightning_logs/version_0/hparams.yaml}" \
    --ckpt_path "${CKPT_PATH:-outputs/pathflip/checkpoint/pytorch_model.bin}" \
    --test_data_path "${TEST_DATA_PATH:-datasets/SlideBench-Caption-TCGA-plus.json}" \
    --ckpt_id "${CKPT_ID:-pathflip_retrieval}"
