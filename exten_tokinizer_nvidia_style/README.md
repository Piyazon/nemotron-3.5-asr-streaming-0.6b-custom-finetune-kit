# NVIDIA-style tokenizer extension for Uyghur

This separate recipe follows NVIDIA's [new-language tokenizer extension
notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-extend-tokenizer-to-newlang-ft-acoustic-model.ipynb):
train a new-language Unigram tokenizer, merge its missing pieces into the
pretrained multilingual tokenizer, and fine-tune the prompted RNNT model.
The default new-language vocabulary is **2,048 before merging**. It is
configurable and is not the final merged size. Small corpora can produce fewer
than the requested number of pieces.

The original tokenizer's piece IDs and normalizer are retained. Duplicate normal
pieces receive the higher score, matching NVIDIA's merge, so segmentation can
change even when IDs do not. New special tokens retain their SentencePiece
types, and vocabulary text files are written in token-ID order. The new
`<ug-CN>` output token is separate from the language prompt slot.

`ug-CN` is assigned the first unused slot across the base model's saved prompt
dictionaries. Existing languages and aliases retain their slots. That mapping
is saved in the model and all dataset configurations for subsequent inference.
If `ug-CN` already exists, its slot is reused. No prompt dimension is added.

## Run

Training needs Linux and an NVIDIA CUDA GPU, using the parent repository's
dependencies. From the repository root:

```bash
python -m pip install -r exten_tokinizer_nvidia_style/requirements.txt

# If the pretrained .nemo is not already present:
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/nemotron-3.5-asr-streaming-0.6b', local_dir='pretrained_model')"

# Prepare Common Voice only; rerun if the old manifests included THUYG-20:
python prepare_hf_uyghur_fast.py --format flac --max-duration 39.99

# Prepare tokenizer/manifests, allocate ug-CN, and train:
bash exten_tokinizer_nvidia_style/train.sh
```

The defaults read `custom_asr_data/train_manifest.json` and
`custom_asr_data/test_manifest.json`. The latter is used as **validation** for
checkpoint selection; use a separate untouched test set for final quality
claims. Original manifests, audio, and the parent training scripts are not
modified. Audio paths must exist on the machine running preparation.

Use explicit paths and a fresh output directory when needed:

```bash
python exten_tokinizer_nvidia_style/train.py \
  --base-model pretrained_model/nemotron-3.5-asr-streaming-0.6b.nemo \
  --train-manifest custom_asr_data/train_manifest.json \
  --validation-manifest custom_asr_data/test_manifest.json \
  --language ug-CN \
  --tokenizer-vocab-size 2048 \
  --output-dir exten_tokinizer_nvidia_style/runs/uyghur-merged-2048
```

Add `--prepare-only` to build and inspect the tokenizer, prompt allocation and
resolved training configuration without loading model weights or requiring
CUDA. This needs only `sentencepiece`, `protobuf`, and `omegaconf`. Subsequent
training with the same arguments reuses the verified prepared assets.

`--prompt-index N` optionally selects a specific unused slot. Occupied slots,
inconsistent saved mappings, missing audio, conflicting manifest languages,
and overlapping train/validation audio paths are rejected.

## NVIDIA training parameters

The notebook's overrides are in `nvidia_recipe.yaml`; its referenced NeMo YAML
is vendored under `upstream/`. These defaults are independent of the parent's
`train.sh`, including its larger GPU batching settings.

| Setting | Value |
| --- | --- |
| Devices | 1 |
| Maximum optimizer steps | 2,000 |
| Maximum epochs | Unlimited, subject to the step cap |
| Training batches per epoch | Capped at 200 |
| Gradient accumulation | 4 batches |
| Training audio budget per batch | 200 seconds |
| Optimizer | AdamW; betas 0.9 / 0.98; weight decay 0.001 |
| Noam scale / dimension | 2.0 / 1,024 |
| Warmup | 2,000 optimizer steps |
| Minimum LR | 0.000001 |
| Gradient clipping | 0.5 |
| Precision | `bf16` (Lightning's mixed BF16 alias) |
| Training metric interval | 100 optimizer steps |
| Validation epochs | Every 20 epochs |
| Validation interval within those epochs | Half an epoch (`0.5`) |
| Training duration filter | 0.1–39.99 seconds |
| Training workers | 8 |
| Validation batch size / workers | 2 / 2 |
| SpecAugment | 2 frequency masks, width 27; 10 time masks, width 0.05 |

The Noam scale `2.0` is **not** a raw constant AdamW learning rate. Its nominal
peak is `2 / sqrt(1024 * 2000)`, approximately **0.00140**. The tutorial stops at
the warmup boundary; this is its short demonstration recipe, not a converged
Uyghur training schedule. Change `nvidia_recipe.yaml` explicitly for a later
schedule experiment. Optional batch and worker flags are described below.

The acoustic architecture and initial encoder weights come from the `.nemo`.
As in the tutorial, changing vocabulary rebuilds the RNNT prediction decoder
and joint network. The encoder remains trainable; there is no freezing or
separate encoder learning rate. Keeping multilingual tokens does not guarantee
retaining the original languages' ASR accuracy.

The adaptations to make this work with the repository's data are ordinary
untarred audio manifests, a persistent `ug-CN` mapping, and explicit `langID`
conditioning for both training and validation. The notebook's literal-period
tagging is applied to copied transcripts (`. <ug-CN>`); sentences without a
period receive no appended tag. Existing matching tags are not duplicated.
The new tokenizer uses NeMo's Unigram builder defaults, including complete
character coverage and case-folding, without enabling byte fallback. The
merged tokenizer keeps the base tokenizer's normalization and byte settings.

## Try larger batches on a 96 GB GPU

The defaults still match NVIDIA's recipe. After updating the scripts on the
training machine, start a separate experiment from the repository root:

```bash
bash exten_tokinizer_nvidia_style/train.sh \
  --fused-batch-size 8 \
  --validation-batch-size 8 \
  --output-dir runs/ug-CN-fused8
```

`train.sh` changes into this folder, so this output path is under
`exten_tokinizer_nvidia_style/runs/`. `--fused-batch-size` sets the number of clips
processed together by the RNNT joint/loss. It is applied to the live joint after
vocabulary replacement, and saved in the model configuration. This setting
keeps the training audio batch and gradient accumulation unchanged. See
[NeMo's batch-splitting explanation](https://docs.nvidia.com/nemo/speech/nightly/asr/configs.html#effect-of-fused-batch-step).

Compare training steps per second and peak memory across short and long clips.
A larger internal batch may improve throughput but needs measurement; higher
memory occupancy alone does not establish a speedup. If 8 runs out of memory,
try 4. If it improves speed with ample peak-memory headroom, try 16 in another
run. No GPU speedup or memory requirement is guaranteed by this configuration.

Other optional flags:

| Flag | Effect when supplied | Default when omitted |
| --- | --- | --- |
| `--batch-duration 400` | Raises the dynamic training audio budget | 200 seconds |
| `--fused-batch-size 8` | Raises the internal RNNT batch and enables fused loss/WER | Base `.nemo` setting |
| `--train-workers 16` | Sets training loader processes | 8 |
| `--validation-workers 4` | Sets validation loader processes | 2 |
| `--validation-batch-size 8` | Sets validation clips per batch | 2 |

Increase workers only if loading data is limiting throughput. Raising the audio
budget changes how much data contributes to each optimizer update; accumulation
stays at 4 and the learning-rate schedule is unchanged. It is a separate training
experiment. The duration sampler uses quadratic duration weighting, so the
budget is not an exact total of raw audio seconds.

These flags take effect on a new process launch. The command starts fresh from
`--base-model`; it does not modify a running job or resume its optimizer state.

## Weights & Biases

Enable W&B on the training server after logging into your account:

```bash
wandb login
bash exten_tokinizer_nvidia_style/train.sh \
  --wandb --wandb-project nemotron-uyghur \
  --wandb-name ug-CN-unigram2048 \
  --output-dir runs/ug-CN-wandb
```

You can combine these flags with the batch overrides above. Add
`--wandb-entity YOUR_TEAM` for a team project. Without `--wandb-project`, the
project defaults to `WANDB_PROJECT` or `nemotron-asr-finetune`. Without
`--wandb-name`, the display name is the output directory name. Each launch gets
a fresh W&B run ID, independent of NeMo's local `test` version directory.

W&B receives the metrics emitted by NeMo through Lightning: `train_loss`,
`training_batch_wer`, actual `learning_rate`, `val_wer`, step/epoch progress,
and NeMo's step timings. W&B also collects system metrics, including GPU
utilization and memory when supported on the training server. Run configuration
includes optimizer/scheduler, batching, language/prompt mapping, vocabulary sizes,
sample counts, tokenizer checksums, and aggregate unknown-token coverage. A small
`run-metadata` artifact contains `training_recipe.yaml` and
`preparation_summary.json`. The final summary records completion/failure status,
completed optimizer steps, and best validation WER when available.

NVIDIA's logging and validation intervals are unchanged: training metrics are
logged every 100 optimizer steps, and validation runs halfway through and at the
end of every 20th epoch. Validation WER appears only after validation runs.
`compute_eval_loss` is false in this recipe, so there is no validation-loss curve.
The model checkpoints, audio, transcripts, tokenizer binaries, and prediction
console output stay local. TensorBoard remains enabled.

For a server without network access, add `--wandb-offline` alongside `--wandb`.
The launcher prints the exact `wandb sync ...` command to upload that run later.
The run ID, local W&B directory, and dashboard URL (online runs) are also saved
in `wandb_run.json` inside the checkpoint/log directory. `--prepare-only` records
your W&B settings in the recipe without starting a W&B run.

These flags apply on the next training launch; they do not attach to an already
running job. Use a fresh output directory, since this launcher does not resume
optimizer state. W&B is optional and disabled when `--wandb` is omitted.

## Outputs and inference

By default, outputs are under `exten_tokinizer_nvidia_style/runs/ug-CN-nvidia/`:

- `assets/<fingerprint>/`: copied tagged manifests, training-only text,
  base/new/merged tokenizer files, token counts and coverage diagnostics,
  plus `training_prompt<N>.yaml` with the resolved recipe.
- `checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming/test/`:
  TensorBoard logs, NeMo's validation-selected checkpoints, saved training
  configurations, and the explicit `nemotron-extended-final.nemo` export.

An existing training directory is rejected. Use a new `--output-dir` for another
run. This launcher starts fresh from `--base-model`; it does not resume optimizer
state. Prepared assets are reusable, and their checksums are verified before use.

Transcribe with the saved `ug-CN` prompt on your CUDA machine:

```bash
python exten_tokinizer_nvidia_style/transcribe.py \
  --model exten_tokinizer_nvidia_style/runs/ug-CN-nvidia/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming/test/nemotron-extended-final.nemo \
  --language ug-CN path/to/audio.wav
```

This convenience command uses `model.transcribe()`. For a streaming benchmark,
use NeMo's `speech_to_text_cache_aware_streaming_infer.py` with the exported model,
`target_lang=ug-CN`, `att_context_size=[56,3]`, `decoder_type=rnnt`,
`pad_and_drop_preencoded=true`, and `batch_size=8`, as in the NVIDIA notebook.
Use identical tag stripping and scoring normalization for model comparisons.

CPU regression checks:

```bash
python -m unittest exten_tokinizer_nvidia_style.test_recipe exten_tokinizer_nvidia_style.test_tracking -v
```
