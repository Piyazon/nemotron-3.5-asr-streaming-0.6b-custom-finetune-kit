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
python asr_finetune_with_speechhints.py --epochs 50 --lr 0.00005
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 20 | Maximum training epochs (for full training) |
| `--lr` | 0.1 | Learning rate (AdamW) |
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
| Optimizer | AdamW (lr=0.1, weight_decay=0.001) |
| Precision | BF16 mixed |
| Gradient clipping | 5.0 |
| Max clip duration | 40s |
| Batch duration | 100s |

**Saved checkpoints:**

| File | Meaning |
|------|---------|
| `nemotron-asr-best1-wer-X.nemo` | Lowest validation WER |
| `nemotron-asr-best2-wer-X.nemo` | 2nd best |
| `nemotron-asr-best3-wer-X.nemo` | 3rd best |
| `nemotron-asr-finetuned.nemo` | Final epoch (regardless of WER) |

TensorBoard logs → `checkpoints/tb_logs/`.

### Step 4 — Evaluation
Loads the latest checkpoint, transcribes every test sample with the manifest's
language prompt, and computes corpus **WER** (Word Error Rate) using NeMo's
built-in metric. It prints sample transcriptions for inspection.

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
| `CUDA out of memory` | Reduce `batch_duration` in the script (e.g., 100 → 50) |
| `PTX version mismatch` during RNNT loss | Pin `numba<0.60` for CUDA driver ≤ 12.4: `pip install "numba<0.60"` |
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
