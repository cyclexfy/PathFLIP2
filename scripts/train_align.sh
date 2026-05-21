#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -m src.tools.train_align \
    --devices "${DEVICES:-0}" \
    --max_epochs "${MAX_EPOCHS:-50}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --log_every_n_steps 15 \
    --results_dir "${RESULTS_DIR:-outputs}" \
    --contrast_loss siglip \
    --accumulate_grad_batches 1 \
    --eval_retrieval_on_epoch \
    --text_encoder_use_lora \
    --text_pooling_type mean \
    --use_soft_topk \
    --save_every_n_epochs 25
