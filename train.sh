#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Fresh Arabic comparison. Use the previous encoder learning-rate scale
# with the loss/config fixes and the existing data, tokenizer and batch settings.
python asr_finetune_with_speechhints.py \
  --train-only \
  --language ug-CN \
  --tokenizer-mode custom \
  --tokenizer-vocab-size 2048 \
  --encoder-lr-scale 1.0 \
  --lr 0.1 \
  --warmup-steps 100 \
  --seed 42 \
  --epochs 50 \
  --max-duration 70 \
  --batch-duration 1920 \
  --fused-batch-size 64 \
  --train-workers 32 \
  --validation-workers 16 \
  --validation-batch-size 16 \
  --log-every-n-steps 100 \
  --run-name uyghur-arabic-v2-enc10 \
  --wandb --wandb-project nemotron-uyghur \
  "$@"
