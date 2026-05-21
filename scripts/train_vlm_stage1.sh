#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -m src.tools.train_vlm \
    --devices "${DEVICES:-0}" \
    --max_epochs "${MAX_EPOCHS:-5}" \
    --results_dir "${RESULTS_DIR:-outputs}" \
    --align_model_ckpt_path "${ALIGN_MODEL_CKPT_PATH:-}" \
    --filename pathflip_vlm_stage1 \
    --freeze_llm \
    --freeze_image_encoder \
    --freeze_text_encoder \
    --weight_decay 0 \
    --init_lr 1e-3 \
    --min_lr 5e-6 \
    --warmup_lr 1e-6 \
    --save_by epoch \
    --save_interval 5
