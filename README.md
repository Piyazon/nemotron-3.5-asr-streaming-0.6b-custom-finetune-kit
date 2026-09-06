# Nemotron 3.5 ASR Fine-Tuning Kit

Self-contained scripts to **fine-tune** the [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) model on your own speech data, then **evaluate** and **benchmark** the result.

Code has been adapted from: https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb 

Everything else (NeMo, PyTorch, Lightning, librosa, etc.) comes from pip-installable libraries — nothing is bundled here.

```
finetune-kit/
├── asr_finetune_with_speechhints.py   # main pipeline: convert → manifest → train → evaluate
├── requirements.txt                   # pip dependencies
├── README.md                          # this file
└── bench/                             # optional: post-training benchmarking (requires CrispASR)
    ├── benchmark-nemotron.sh          # full benchmark: convert + compare all models
    ├── compare-nemotron.sh            # lighter comparison only
    ├── convert-nemo-to-gguf.sh        # batch .nemo → GGUF converter
    └── convert-single-nemo-to-gguf.sh # single-file .nemo → GGUF converter
```

---

## Quick Start

```bash
# 0. System packages
sudo apt-get install -y ffmpeg sox libsndfile1 libsox-fmt-mp3

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download pretrained model (one-time, ~1.5 GB)
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('nvidia/nemotron-3.5-asr-streaming-0.6b', local_dir='pretrained_model')"

# 3. Place your training data in traintestset/ (see [Data Preparation](#data-preparation))

# 4. Run the full pipeline (English remains the default)
python asr_finetune_with_speechhints.py
```

This project is intended to run on a **Linux machine with an NVIDIA CUDA GPU**.
macOS is suitable for editing the repository, but not for running this training
pipeline.

### Fine-tune an unsupported language (Uyghur example)

Use a locale for the new language when creating manifests and training:

```bash
python asr_finetune_with_speechhints.py --language ug-CN
```

With the default `--tokenizer-mode auto`, the training step:

1. Reads text from `train_manifest.json` only (the held-out test text is not used).
2. Detects that `ug-CN` is absent from the pretrained prompt dictionary, or that
   the pretrained tokenizer produces unknown tokens.
3. Builds a Unicode-aware SentencePiece BPE tokenizer with byte fallback under
   `custom_asr_data/tokenizers/`.
4. Installs the vocabulary with NeMo's `change_vocabulary()` and allocates the
   first unused language-prompt slot.
5. Fine-tunes every sample with explicit `ug-CN` language conditioning.

The generated tokenizer is content-addressed: rerunning with unchanged training
text reuses it, while changed transcripts or vocabulary size produce a new one.
The exact tokenizer directory and prompt index are also recorded in every new
Lightning checkpoint, so checkpoint export and testing no longer have to guess
which same-sized tokenizer belongs to a checkpoint.

For the two Hugging Face Uyghur datasets, prepare and train with:

```bash
python prepare_hf_uyghur_fast.py --format flac
python asr_finetune_with_speechhints.py \
  --train-only \
  --language ug-CN \
  --tokenizer-mode custom \
  --run-name uyghur-v2
```

The preparation script combines `piyazon/cv-corpus-ug-24-latn` and
`piyazon/thuyg20-datasets`. It uses the Arabic-script `sentence` column from both
repositories, sets the language to `ug-CN`, and exports the original clips
without concatenation. It merges their training splits into
`custom_asr_data/train_manifest.json` and their held-out splits into
`custom_asr_data/test_manifest.json`, preferring `validation` over `test` when
both exist. If a source has neither, it creates a seeded 98/2 row split for that
source. The resulting manifests replace previous preparation results after both
sources have exported successfully. Audio is stored separately by repository,
dataset fingerprint and split, preventing row-index collisions and stale reuse.

The default maximum duration is 40 seconds. To include longer recordings, pass
the same `--max-duration` value to preparation and training, for example `70`.
The export log reports skipped clips and their reasons. Loading the repositories
uses your existing Hugging Face login or `HF_TOKEN` when access is required.

---

## Data Preparation

Place your data in a `traintestset/` directory alongside this script (dataset 1 and 2 can have same speaker or different speaker, in here I have separated them as I have recorded from different books.):

```
finetune-kit/                          (this directory)
├── asr_finetune_with_speechhints.py
├── traintestset/                      (your data — create this)
│   ├── p1/                            dataset 1
│   │   ├── 1.wav                      audio file (any format ffmpeg understands)
│   │   ├── 2.wav
│   │   └── transcript.csv             pipe-delimited transcripts
│   ├── p2/                            dataset 2
│   │   ├── 1.wav
│   │   └── transcript.csv
│   └── p3/                            ...
├── pretrained_model/                  (downloaded in step 2)
│   └── nemotron-3.5-asr-streaming-0.6b.nemo
```

### `transcript.csv` Format

No header row. Each line: `<filename>|<transcription text>`

```
1.wav|Whenever we read about a scientific breakthrough...
2.wav|When we walk into a library, we are surrounded...
3.wav|When we go on the Internet, we can read millions...
```

- **Column 1:** WAV filename (must match an actual file in the same `pN/` directory)
- **Column 2:** Ground-truth transcript text
- Delimiter: `|` (pipe)
- Audio can be any sample rate / channel count — the pipeline converts to mono 16 kHz automatically

---

## Running the Pipeline

### Full run (all steps)

```bash
python asr_finetune_with_speechhints.py
```

Runs all 4 steps: convert audio → build manifests → fine-tune → evaluate.

### Step-by-step

```bash
# Step 1 only: convert audio to mono 16 kHz
python asr_finetune_with_speechhints.py --convert-only

# Steps 1–2: convert + build JSON manifests
python asr_finetune_with_speechhints.py --manifest-only

# Step 3 only: fine-tune (requires manifests from step 2)
python asr_finetune_with_speechhints.py --train-only

# Step 4 only: evaluate trained model on test set
python asr_finetune_with_speechhints.py --evaluate

# Optional: apply speech-hint normalization to manifests
python asr_finetune_with_speechhints.py --apply-speechhints
```

### Hyperparameters

Convergence depends on the language, tokenizer reset, and amount of speech data.

```bash
python asr_finetune_with_speechhints.py --epochs 50 --lr 0.1
```

`--lr` has NVIDIA's Noam-scheduler meaning: it is a scale factor, not the raw
AdamW learning rate. With the defaults (`0.1`, `d_model=1024`, 100 warmup
steps), the effective peak learning rate is approximately `3.1e-4`.

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 20 | Maximum training epochs (for full training) |
| `--lr` | 0.1 | Noam learning-rate scale factor (not raw AdamW LR) |
| `--encoder-lr-scale` | 1.0 | Encoder learning rate relative to the decoder/joint rate; use 0.1 to experiment with slower acoustic updates |
| `--seed` | 42 | Seed for model initialization and data loading; GPU kernels can still be nondeterministic |
| `--warmup-steps` | 100 | Noam linear warmup steps |
| `--noam-d-model` | 1024 | Model dimension used by Noam scaling |
| `--max-duration` | 40 | Maximum train/validation clip duration in seconds |
| `--batch-duration` | 720 | Approximate audio seconds per dynamic training batch; starting profile for RTX PRO 6000 96 GB |
| `--train-workers` | 16 | Training data-loader processes; adjust for host CPU/RAM |
| `--validation-workers` | 8 | Validation data-loader processes |
| `--validation-batch-size` | 16 | Validation clips per batch |
| `--fused-batch-size` | 4 | Clips per internal RNNT joint/loss batch |
| `--log-every-n-steps` | 100 | Training metric logging interval in optimizer steps |
| `--run-name` | none | Optional checkpoint subdirectory to keep retrains separate |
| `--language` | `en-US` for new manifests | Locale used in manifests and prompt conditioning, e.g. `ug-CN` |
| `--tokenizer-mode` | `auto` | Choose `auto`, `base`, or `custom` |
| `--tokenizer-vocab-size` | 2048 | Requested generated BPE size (minimum 512) |
| `--prompt-index` | first unused | Optional explicit unused prompt slot |

Tokenizer modes:

- `auto` generates a tokenizer if the locale is not in the model prompt
  dictionary or if any training transcript produces `<unk>` with the base
  tokenizer.
- `base` keeps the pretrained tokenizer. Use this only when it can represent
  the new language's writing system.
- `custom` always builds or reuses a tokenizer from the training transcripts.

For a step-by-step unsupported-language run:

```bash
python asr_finetune_with_speechhints.py --manifest-only --language ug-CN
python asr_finetune_with_speechhints.py --train-only --language ug-CN
python asr_finetune_with_speechhints.py --evaluate --language ug-CN
```

---

## How the Pipeline Works

### Step 1 — Audio Conversion
Scans `traintestset/p*/` for all audio files, converts to **mono 16 kHz WAV** via ffmpeg. Output: `custom_asr_data/wavs/`. Idempotent (skips already-converted files).

### Step 2 — Manifest Building
Reads each `transcript.csv`, matches WAVs, computes durations via ffprobe. Shuffles and splits 80/20 into:

- `custom_asr_data/train_manifest.json` — training set
- `custom_asr_data/test_manifest.json` — test set

Each line contains `audio_filepath`, `duration`, `text`, `language`, `lang`, and
`target_lang`. The language fields use the locale supplied with `--language`.

### Step 3 — Fine-Tuning
Loads the pretrained model (`EncDecRNNTBPEModelWithPrompt`) and fine-tunes via
PyTorch Lightning. For an unsupported language, it generates and installs the
training-text tokenizer before data loaders and optimization are created. It
also adds the locale to an unused prompt slot and uses fixed `langID` prompting
during training and validation.

Changing the vocabulary retains the pretrained acoustic encoder, but NeMo
reinitializes the RNNT prediction decoder and joint output network. A very small
speech dataset may therefore be insufficient. A target-language-only custom
tokenizer creates a specialized checkpoint and does not preserve output ability
for all original languages.

| Setting | Value |
|---------|-------|
| Optimizer | AdamW + Noam (scale=0.1, d_model=1024, warmup=100, weight_decay=0.001) |
| Precision | BF16 mixed |
| Gradient clipping | 5.0 |
| Max clip duration | 40s |
| Batch duration | 720s (starting profile for RTX PRO 6000 96 GB) |
| RNNT internal batch | 4 clips, with fused joint/loss/WER enabled |
| Data loading | 16 training workers, 8 validation workers, pinned host memory |
| Validation | Batch size 16; WER decoding only (RNNT validation loss disabled) |
| Training logging | Every 100 optimizer steps |

The explicit Noam scheduler is important. The model uses a dynamic Lhotse
sampler, so this kit bypasses NeMo's automatic step-count calculation, but it
does not bypass LR scheduling. Feeding raw `0.1` directly to AdamW is incorrect
for this recipe.

The 96 GB defaults are a starting configuration, not a measured maximum. For the
combined Uyghur dataset with clips up to about 65 seconds, start a new run with:

```bash
python asr_finetune_with_speechhints.py --train-only \
  --language ug-CN --tokenizer-mode custom --max-duration 70 \
  --batch-duration 720 --fused-batch-size 4 \
  --train-workers 16 --validation-workers 8 --validation-batch-size 16 \
  --log-every-n-steps 100 --run-name uyghur-96gb
```

The duration budget controls the encoder batch. The internal RNNT batch controls
how many clips the joint/loss processes together; it can constrain throughput
even when the encoder batch is large. The script applies the internal size to
both the live module and its saved config. See [NeMo's batch splitting
documentation](https://docs.nvidia.com/nemo/speech/nightly/asr/configs.html#effect-of-fused-batch-step).

Measure peak memory and audio processed per second across short and long clips.
If there is ample headroom, try `--fused-batch-size 8` first, or increase the
duration budget from 720 to 960 in a separate trial. Increasing workers helps
only when data loading is the bottleneck and host CPU/RAM are available. For an
out-of-memory error during the joint/loss, lower the internal batch to 2 or 1;
for encoder memory pressure, lower the duration budget to 480 or 240. If validation
runs out of memory, lower `--validation-batch-size` to 8 or 4. Keep the 70-second
clip limit if those long recordings should remain in the dataset.

Larger training batches also change the number of optimizer updates per epoch
and the amount of audio seen during the step-based Noam warmup. Compare validation
WER when changing batch sizes; higher GPU utilization alone does not establish
better training quality. BF16 precision, learning-rate settings, model dimensions
and tokenizer vocabulary size are not changed by this throughput profile.

The logging interval controls training metrics. Reference/prediction examples
may still appear in bursts from RNNT sub-batches and during validation. Changes
take effect on the next process launch. This training entry point starts from
the pretrained model; rerunning it does not resume the currently running job.

**Saved checkpoints:**

| File | Meaning |
|------|---------|
| `nemotron-asr-best1-wer-X.nemo` | Lowest validation WER |
| `nemotron-asr-best2-wer-X.nemo` | 2nd best |
| `nemotron-asr-best3-wer-X.nemo` | 3rd best |
| `nemotron-asr-finetuned.nemo` | Final epoch (regardless of WER) |

TensorBoard logs → `checkpoints/tb_logs/`.
The training run also saves `training_config.yaml`, including the seed and
encoder learning-rate multiplier. Its startup log reports median and 95th
percentile training clip durations so you can compare training coverage with
recordings that fail in deployment.

### Step 4 — Evaluation
Loads the **best validation-WER export** from the selected run and transcribes
every evaluation sample with the manifest's language prompt. A full pipeline run
evaluates the exact best export it just trained. Use `--run-name` to select a
specific run, or `--checkpoint` to evaluate a particular `.nemo` archive. Without
either, evaluation selects best1 from the most recently exported run.

Evaluation computes corpus **WER**, **CER**, word substitutions, deletions and
insertions using JiWER. It writes `summary.json` and `transcriptions.jsonl` under
`<checkpoint-directory>/evaluation/<model-name>/<manifest-name>/`, or under
`--report-dir`. Repeating evaluation at the same location replaces those reports.
Each recording includes its reference, hypothesis, duration, deletion rate and
deleted text spans. The summary also groups results into under 10s, 10–20s,
20–40s and 40s or longer. Rates are computed from total edit counts, rather than
averaging recording-level rates.

Scoring normalizes Unicode to NFC and collapses whitespace. It retains case
and punctuation, and CER includes spaces. Deleted spans are text alignments;
they do not identify audio timestamps or prove that the model emitted blanks.

```bash
python asr_finetune_with_speechhints.py --evaluate --language ug-CN --run-name uyghur-v2

# Compare a particular model on accurately transcribed problem recordings.
python asr_finetune_with_speechhints.py --evaluate \
  --checkpoint /path/to/model.nemo \
  --eval-manifest /path/to/problem_recordings.jsonl \
  --report-dir /path/to/evaluation-report
```

`test_checkpoint.py` and `export_checkpoint_to_nemo.py` also default to the best
validation-WER `.ckpt`, recovered from Lightning's saved callback metadata.
Use `--run-name uyghur-v2` to restrict selection to one run, `--selection latest`
to test the newest retained weights, or `--checkpoint /path/to/model.ckpt` to
choose explicitly. A missing best checkpoint or missing selection metadata
requires an explicit selection; it does not silently fall back to another model.

### Investigating skipped speech

First evaluate the existing best model and inspect recordings with high deletion
rates. Compare the same audio in NeMo and in the deployed runtime, with the same
checkpoint and `ug-CN` prompt. For a missing passage, also try a separate crop
that includes some surrounding speech.

For the next controlled training experiment, a smaller encoder learning rate
lets the randomly initialized decoder/joint learn faster relative to the
pretrained acoustic encoder:

```bash
python asr_finetune_with_speechhints.py --train-only \
  --language ug-CN --tokenizer-mode custom \
  --encoder-lr-scale 0.1 --seed 42 --run-name uyghur-slower-encoder
```

Compare this with `--encoder-lr-scale 1.0` using the same data, seed and epoch
budget in a separate run. The existing default remains 1.0. With the default
Noam settings, scale 0.1 gives an encoder peak LR of approximately `3.1e-5` and
a decoder/joint peak LR of `3.1e-4`. Both groups use the existing Noam schedule;
the encoder continues training throughout. This is an experiment, not a
confirmed remedy for omissions.

Before extending training, listen to samples while reading the exact manifest
transcripts. Check that all spoken phrases are transcribed and that augmentations
preserve the audio/text pairing. Keep all variants of a source recording in the
same split; the preparation script's random row split does not ensure this.
Use validation speech representative of deployment speakers, noise and duration,
and keep a separate final test set: this kit currently uses `test_manifest.json`
for checkpoint selection, so that manifest serves as validation data. Review both
WER and deletion rate; reducing deletions while greatly increasing insertions is
not an improvement.

Regression checks for selection, scoring and optimizer grouping can run without
NeMo or a GPU:

```bash
python -m pip install 'jiwer>=3.1,<5'
python -m unittest discover -s tests -v
```

---

## Post-Training: Benchmarking (`bench/`)

The `bench/` scripts compare fine-tuned checkpoints against the pretrained base model using [CrispASR](https://github.com/k2-fsa/crispasr). These are **optional** — the Python script's built-in evaluation (step 4) already gives you WER.

### Additional dependency
- **[CrispASR](https://github.com/k2-fsa/crispasr)** — must be cloned and built separately. The scripts expect:
  - Binary at `<your-path>/CrispASR/build/bin/crispasr` (or in `$PATH`)
  - Converter at `<your-path>/CrispASR/models/convert-nemotron-to-gguf.py`

Set envvars to override defaults:
```bash
export CRISPASR_BIN=/path/to/crispasr
export CONVERT_SCRIPT=/path/to/convert-nemotron-to-gguf.py
```

### Scripts

| Script | What it does |
|--------|-------------|
| `bench/benchmark-nemotron.sh` | Full run: convert best-N checkpoints to Q8_0 GGUF, benchmark all models against test set, print WER table + ranking + diffs, persist GGUFs |
| `bench/compare-nemotron.sh` | Same comparison but lighter (no persistence) |
| `bench/convert-nemo-to-gguf.sh --all [q4_k\|q8_0]` | Batch convert all best-N `.nemo` files to GGUF |
| `bench/convert-nemo-to-gguf.sh <file.nemo> [q4_k\|q8_0]` | Convert a single `.nemo` file |
| `bench/convert-single-nemo-to-gguf.sh <file.nemo>` | Minimal single-file converter |

### Typical workflow after training

```bash
# 1. Train (produces .nemo checkpoints)
python asr_finetune_with_speechhints.py --train-only

# 2. Benchmark all models (requires CrispASR)
export CRISPASR_BIN=/path/to/crispasr
export CONVERT_SCRIPT=/path/to/convert-nemotron-to-gguf.py
bash bench/benchmark-nemotron.sh
```

---

## Directory Layout After a Full Run

```
finetune-kit/
├── asr_finetune_with_speechhints.py
├── requirements.txt
├── README.md
├── bench/
│   └── ...
├── traintestset/           (your input data)
├── pretrained_model/       (pretrained .nemo)
├── custom_asr_data/        (auto-generated)
│   ├── wavs/               converted mono 16kHz audio
│   ├── tokenizers/          generated, content-addressed BPE tokenizers
│   ├── train_manifest.json
│   └── test_manifest.json
├── checkpoints/            (auto-generated)
│   ├── FastConformer-Transducer-BPE-Prompt-Streaming/test/
│   │   ├── nemotron-asr-best1-wer-X.nemo
│   │   ├── nemotron-asr-best2-wer-X.nemo
│   │   ├── nemotron-asr-best3-wer-X.nemo
│   │   ├── nemotron-asr-finetuned.nemo
│   │   └── *.ckpt
│   └── tb_logs/            TensorBoard logs
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Lower `--batch-duration` from 720 to 480 or 240, and/or `--fused-batch-size` from 4 to 2 or 1; for validation, lower `--validation-batch-size` from 16 to 8 or 4 |
| Numba CUDA compile error during RNNT loss | Reinstall the compatible CUDA target and NumPy constraint: `pip install --upgrade --force-reinstall "numpy>=1.26,<2.5" "numba-cuda[cu12]"` |
| `Pretrained model not found` | Run the download command in [Quick Start](#quick-start) step 2 |
| `No transcript.csv in pX, skipping` | Each speaker dir needs a `transcript.csv` (pipe-delimited, no header) |
| `Converted file pX_Y.wav not found` | Filenames in `transcript.csv` must match actual files; run `--convert-only` first |
| `--language ... does not match manifest language` | Rebuild with `--manifest-only --language <locale>`, or use the locale already stored in the manifests |
| No free prompt slot | Omit a conflicting `--prompt-index`; automatic allocation uses the first unused slot |
| Poor new-language output after a short run | A custom vocabulary resets the RNNT decoder/joint; add more clean transcribed speech and validate on held-out data |
| Bench scripts: `conversion script not found` | Set `CONVERT_SCRIPT` envvar to point to your CrispASR conversion script |

## Reference
Nemotron: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
Tutorials: https://github.com/nvidia-riva/tutorials/tree/main
CrispASR: https://github.com/CrispStrobe/CrispASR
