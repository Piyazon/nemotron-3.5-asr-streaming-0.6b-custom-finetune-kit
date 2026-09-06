python asr_finetune_with_speechhints.py \
  --train-only \
  --language ug-CN \
  --tokenizer-mode custom \
  --epochs 20 \
  --max-duration 70 \
  --batch-duration 720 \
  --fused-batch-size 8 \
  --train-workers 16 \
  --validation-workers 8 \
  --validation-batch-size 16 \
  --log-every-n-steps 100 \
  --run-name uyghur-96gb-fused8

  