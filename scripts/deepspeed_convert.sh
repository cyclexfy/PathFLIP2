#!/bin/bash

if [ -z "$ORIGINAL_CKPT_PATH" ] || [ -z "$MERGED_CKPT_PATH" ]; then
    echo "Set ORIGINAL_CKPT_PATH and MERGED_CKPT_PATH before running this script."
    exit 1
fi

python -m deepspeed.utils.zero_to_fp32 \
    "$ORIGINAL_CKPT_PATH" \
    "$MERGED_CKPT_PATH"
