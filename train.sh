python asr_finetune_with_speechhints.py \
  --train-only \
  --language ug-CN \
  --tokenizer-mode custom \
  --epochs 50 \
  --max-duration 70 \
  --batch-duration 1920 \
  --fused-batch-size 64 \
  --train-workers 32 \
  --validation-workers 16 \
  --validation-batch-size 16 \
  --log-every-n-steps 100 \
  --run-name uyghur-96gb-fused8 \
  --wandb --wandb-project nemotron-uyghur

