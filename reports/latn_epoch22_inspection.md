# Latin epoch-22 NeMo archive inspection

Inspected on 2026-09-07, read-only, with the local Nemo Python environment.

Source: `/Users/pi/Downloads/nemotron-3.5-asr-streaming-uyghur-latn-best-epoch22-wer9.59.nemo`

## Main finding

The archive contains a **13,087-piece multilingual tokenizer**, with the saved directory name `universal_merged_tokenizer_latest_40`. It does not contain the custom 2,048-piece Uyghur tokenizer used by the current Arabic training recipe.

The embedded SentencePiece vocabulary matches `joint.vocabulary` exactly, including order. The decoder embedding and joint output both have 13,088 rows: 13,087 text tokens plus the RNN-T blank. These are properties of the actual saved model, not merely training configuration defaults.

This changes the interpretation of the Latin/Arabic comparison: it involves different vocabularies and output structures as well as different scripts. It does not establish that 2,048 tokens are intrinsically insufficient, or prove which pretrained decoder weights were retained when the Latin run began.

## Verified model structure

| Property | Saved model |
| --- | --- |
| Model class | `EncDecRNNTBPEModelWithPrompt` |
| Saved NeMo version | `2.8.0rc0` |
| Weight/bias elements | 637,997,088, approximately 638 million |
| Encoder | 24 Conformer layers; model dimension 1,024; 8 attention heads |
| Subsampling | `dw_striding`, factor 8; causal downsampling |
| Attention context options | `[[56, 3], [56, 0], [56, 6], [56, 13]]`, `chunked_limited` |
| Decoder | 2-layer LSTM; hidden size 640 |
| Joint | Hidden size 640; ReLU; 13,088 output classes including blank |
| Language prompt inputs | 128 |
| Audio frontend | 16 kHz; 128 features; 25 ms window; 10 ms hop; FFT 512 |
| Tokenizer | SentencePiece Unigram; 13,087 pieces; byte fallback disabled |
| Tokenizer normalization | `nmt_nfkc` |
| Saved decoding strategy | `greedy_batch`; maximum 10 symbols per step |

The config calls its tokenizer type `bpe`; the embedded SentencePiece model itself specifies Unigram. The NeMo tokenizer category and SentencePiece segmentation algorithm are distinct here.

All 657 state-dict tensors are stored as FP32. This does **not** identify training precision: mixed-precision training can still save FP32 weights. The total tensor-element count is 638,030,384, including 33,296 preprocessing buffer elements.

## Saved configuration: historical use is unverified

These values are present in `model_config.yaml`. They must not be treated as a recovered fine-tuning command: the archive has no training provenance, all dataset manifest paths are null, and no Uyghur prompt mapping is saved. The config may retain defaults from a base model or an export process.

| Configuration field | Saved value |
| --- | --- |
| Optimizer | AdamW |
| Adam betas | `(0.9, 0.98)` |
| Weight decay | `0.001` |
| Noam LR scale, `optim.lr` | `0.5` |
| Scheduler | `NoamAnnealing` |
| Scheduler model dimension | `1024` |
| Warmup | `10000` steps |
| Minimum LR | `1e-6` |
| Training batch duration | `200` seconds |
| Training duration filter | `0.1` to `20` seconds |
| Training workers | `8` |
| Fused RNN-T batch size | `2` |
| Validation batch size / workers | `2` / `2` |
| SpecAugment | 2 frequency masks, width 27; 10 time masks, width 0.05 |
| Encoder / decoder / joint dropout | `0.1` / `0.2` / `0.2` |
| Loss selector | `default` |
| `warprnnt_numba_kwargs` | `fastemit_lambda=0.005`, `clamp=-1.0` |

If the saved Noam configuration was actually used, its nominal peak LR would be `0.5 / sqrt(1024 * 10000) = 0.00015625`. The `0.5` value is a scheduler scale, not a constant AdamW learning rate. Neither this historical schedule nor the actual runtime loss backend is verified by the archive.

## What cannot be recovered from this archive

`model_weights.ckpt` is a plain state dict, inspected with `torch.load(weights_only=True)` inside `FakeTensorMode`, without allocating the full tensor payload. It is not a Lightning training checkpoint.

It contains no optimizer state, scheduler state, epoch, global step, callbacks, or training hyperparameters. There is also no `custom_finetune` provenance in the model config. Consequently the archive cannot establish:

- The exact fine-tuning command, actual LR schedule, encoder LR multiplier, frozen layers, training precision, gradient accumulation, seed, or hardware.
- The training/validation dataset versions or normalization used for WER.
- The training/inference prompt selected for Uyghur. No Uyghur entry appears in any saved prompt dictionary, and no default language is saved.
- The epoch and WER in the filename; `epoch22` and `wer9.59` are filename labels, not independently recorded metrics in the archive.

The current repository's continuation path requires the requested language prompt to exist in the checkpoint. This archive lacks `ug-CN`, so an unchanged `--init-from-nemo ... --language ug-CN` would fail that check. The original prompt choice needs to be recovered before prescribing equivalent continuation settings.

The original Lightning `.ckpt`, run configuration, or launch logs would be needed to recover those missing training details. Local Git history alone does not prove which code or command produced this archive.

## Token coverage check on the current small dataset

This diagnostic uses the 837 current `small_dataset/final` transcripts and their UMSC Latin conversions. These are **not** the historical training or validation corpus. Both rows below use this old archive's same 13,087-piece tokenizer; the Arabic row does not measure the current custom Arabic tokenizer.

| Input script | Tokens per whitespace-delimited word | Unknown-token fraction |
| --- | --- | --- |
| UMSC Latin | 4.637 | 36 / 31,593 = 0.114% |
| Arabic | 6.602 | 7,761 / 44,978 = 17.255% |

All observed Latin unknown tokens were punctuation: `«`, `»`, and `:`. Tested standard Uyghur Latin letters, including `é`, `ö`, and `ü`, were covered. Several Uyghur Arabic letters are unsupported by this old tokenizer. These token-coverage measurements are not ASR error rates and do not explain the WER difference on their own.

The source `.nemo`, training code, and dataset were not modified. Small extracted artifacts remain in `/private/tmp/uyghur-latn-epoch22-inspection/`.
