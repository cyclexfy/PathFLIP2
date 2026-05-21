#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -m src.tools.train_vlm \
    --devices "${DEVICES:-0}" \
    --max_epochs "${MAX_EPOCHS:-2}" \
    --results_dir "${RESULTS_DIR:-outputs}" \
    --align_model_ckpt_path "${ALIGN_MODEL_CKPT_PATH:-}" \
    --stage1_ckpt_path "${STAGE1_CKPT_PATH:-}" \
    --filename pathflip_vlm_stage2 \
    --train_data_path \
        datasets/SlideInstruct_train_stage1_caption_valid.json \
        datasets/SlideInstruct_train_stage2_vqa_valid.json \
    --llm_lora \
    --save_by epoch \
    --save_interval 1 \
    --caption_repeat 1
