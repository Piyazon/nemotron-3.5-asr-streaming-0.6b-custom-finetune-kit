#!/usr/bin/env python3
"""
Fine-Tune Nemotron 3.5 ASR Streaming Model with Custom Dataset & Speech Hints
==============================================================================

Combines:
  - Dataset preparation (audio conversion + manifest building) from the
    customize workflow (traintestset/  ->  custom_asr_data/)
  - Fine-tuning of nemotron-3.5-asr-streaming-0.6b using NeMo Python API
    (from asr-finetune-nemotron-3.5-asr-streaming-prompt notebook)
  - Optional speech-hint grammar post-processing (from
    asr-customize-speechhints) for inverse text normalization on transcripts

Uses the installed NeMo package directly -- no cloned repo needed.

Usage:
    python asr_finetune_with_speechhints.py                     # full pipeline
    python asr_finetune_with_speechhints.py --convert-only       # step 1 only
    python asr_finetune_with_speechhints.py --manifest-only      # steps 1-2
    python asr_finetune_with_speechhints.py --train-only         # step 3 only
    python asr_finetune_with_speechhints.py --evaluate           # step 4 only
    python asr_finetune_with_speechhints.py --language ug-CN     # unsupported language
    python asr_finetune_with_speechhints.py --apply-speechhints  # post-process transcripts

Prerequisites:
    - GPU with CUDA
    - NeMo toolkit installed (nemo_toolkit[asr])
    - ffmpeg, sox installed
    
pip uninstall nemo_toolkit -y
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import unicodedata

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # tutorials/

# Raw speaker data (p1/, p2/, ..., each with *.wav + transcript.csv)
RAW_DATA_DIR = os.path.join(DATA_DIR, "traintestset")

# Output: converted WAVs + JSON manifests
CUSTOM_DATA_DIR = os.path.join(DATA_DIR, "custom_asr_data")
CONVERTED_WAVS_DIR = os.path.join(CUSTOM_DATA_DIR, "wavs")

# Tokenizers generated from the training transcripts.  A content fingerprint
# is included in each tokenizer directory so changed transcripts/options never
# silently reuse a stale tokenizer.
TOKENIZER_ROOT_DIR = os.path.join(CUSTOM_DATA_DIR, "tokenizers")

DEFAULT_LANGUAGE = "en-US"
DEFAULT_TOKENIZER_VOCAB_SIZE = 2048
DEFAULT_MAX_DURATION = 60.0
DEFAULT_BATCH_DURATION = 240.0
DEFAULT_TRAIN_WORKERS = 8
DEFAULT_VALIDATION_WORKERS = 8
DEFAULT_VALIDATION_BATCH_SIZE = 4
DEFAULT_WARMUP_STEPS = 100
DEFAULT_NOAM_D_MODEL = 1024

# Pretrained model from HuggingFace
PRETRAINED_MODEL = os.path.join(
    DATA_DIR, "pretrained_model", "nemotron-3.5-asr-streaming-0.6b.nemo"
)

# Checkpoint output
CHECKPOINT_DIR = os.path.join(
    DATA_DIR, "checkpoints", "FastConformer-Transducer-BPE-Prompt-Streaming", "test"
)

# Dataset split
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Speech Hints (from asr-customize-speechhints.ipynb)
# ---------------------------------------------------------------------------
# Available grammar symbols for inverse text normalization:
#   $OOV_NUMERIC_SEQUENCE, $OOV_ALPHA_SEQUENCE, $OOV_ALPHA_NUMERIC_SEQUENCE
#   $FULLPHONENUM, $POSTALCODE, $OOV_CLASS_ORDINAL, $OOV_CLASS_NUMERIC
#   $PERCENT, $TIME, $MONEY, $MONTH, $DAY
#
# When --apply-speechhints is used, we try to normalize transcript text
# using the speech_hint library.  If the library is not installed, we fall
# back gracefully (transcripts are kept as-is).

SPEECH_HINT_RULES = [
    # Phone numbers like "one eight hundred five five five four oh oh one"
    (r"\d[\d\s]{9,}\d", "$FULLPHONENUM"),
    # Percentages
    (r"\d+\s*percent", "$PERCENT"),
    # Time expressions
    (r"\d{1,2}:\d{2}", "$TIME"),
    # Money
    (r"\$\d+[\.,]?\d*", "$MONEY"),
]

try:
    from speech_hint import apply_hint  # type: ignore
    SPEECH_HINT_AVAILABLE = True
except ImportError:
    SPEECH_HINT_AVAILABLE = False


def normalize_with_speech_hints(text: str) -> str:
    """Apply speech-hint grammars to normalize a transcript string.

    This is the Python-side equivalent of what the asr-customize-speechhints
    notebook demonstrates with FST-based grammars.  Each rule attempts an
    inverse-text-normalization pass; if apply_hint raises, we skip that rule.
    """
    if not SPEECH_HINT_AVAILABLE:
        return text

    for _, grammar in SPEECH_HINT_RULES:
        try:
            text = apply_hint(text, grammar)
        except Exception:
            pass  # Grammar did not match; move on
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)
    os.makedirs(CONVERTED_WAVS_DIR, exist_ok=True)
    os.makedirs(TOKENIZER_ROOT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def log(msg: str):
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)


def read_manifest_entries(manifest_path: str) -> list[dict]:
    """Read and minimally validate a NeMo JSON-lines manifest."""
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {manifest_path} at line {line_num}: {exc}"
                ) from exc
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Manifest entry must be a JSON object in {manifest_path} "
                    f"at line {line_num}"
                )
            if not str(entry.get("text", "")).strip():
                raise ValueError(
                    f"Missing transcript text in {manifest_path} at line {line_num}"
                )
            entries.append(entry)
    if not entries:
        raise ValueError(f"Manifest contains no usable entries: {manifest_path}")
    return entries


def resolve_manifest_language(manifest_path: str, requested: str | None = None) -> str:
    """Return the single language in a manifest and validate a CLI override."""
    if requested is not None:
        requested = requested.strip()
    languages = set()
    for entry in read_manifest_entries(manifest_path):
        language = (
            entry.get("language")
            or entry.get("lang")
            or entry.get("target_lang")
        )
        if language:
            languages.add(str(language))

    if not languages:
        raise ValueError(
            f"No language field found in {manifest_path}. Rebuild it with --manifest-only."
        )
    if len(languages) != 1:
        raise ValueError(
            "This script currently expects one language per training run, but "
            f"{manifest_path} contains: {sorted(languages)}"
        )

    manifest_language = next(iter(languages))
    if requested and requested != manifest_language:
        raise ValueError(
            f"--language={requested!r} does not match manifest language "
            f"{manifest_language!r}. Rebuild manifests with --manifest-only "
            "or use the manifest language."
        )
    return manifest_language


def tokenizer_unknown_rate(tokenizer, texts: list[str]) -> tuple[int, int, float]:
    """Measure how often the base tokenizer emits its unknown-token ID."""
    unk_id = getattr(tokenizer, "unk_id", None)
    if unk_id is None or unk_id < 0:
        return 0, 0, 0.0

    unknown = 0
    total = 0
    for text in texts:
        token_ids = tokenizer.text_to_ids(text)
        total += len(token_ids)
        unknown += sum(token_id == unk_id for token_id in token_ids)
    rate = unknown / total if total else 0.0
    return unknown, total, rate


def manifest_duration_summary(manifest_path: str) -> dict[str, float | int]:
    """Summarize duration coverage without decoding any audio."""
    entries = read_manifest_entries(manifest_path)
    durations = []
    for line_num, entry in enumerate(entries, 1):
        try:
            duration = float(entry["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid duration in {manifest_path} at entry {line_num}"
            ) from exc
        if duration <= 0:
            raise ValueError(
                f"Duration must be positive in {manifest_path} at entry {line_num}"
            )
        durations.append(duration)

    return {
        "samples": len(durations),
        "hours": sum(durations) / 3600.0,
        "maximum": max(durations),
    }


def build_custom_tokenizer(
    train_manifest: str,
    language: str,
    vocab_size: int,
) -> str:
    """Build or reuse a SentencePiece BPE tokenizer from training text only."""
    if vocab_size < 512:
        raise ValueError(
            "--tokenizer-vocab-size must be at least 512 because byte fallback "
            "reserves 256 byte tokens in addition to the language vocabulary."
        )

    texts = [entry["text"].strip() for entry in read_manifest_entries(train_manifest)]
    fingerprint_input = json.dumps(
        {
            "language": language,
            "vocab_size": vocab_size,
            "tokenizer_type": "bpe",
            "character_coverage": 1.0,
            "byte_fallback": True,
            "texts": texts,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_input).hexdigest()[:12]
    safe_language = re.sub(r"[^A-Za-z0-9_.-]+", "_", language).strip("._") or "language"
    tokenizer_dir = os.path.join(
        TOKENIZER_ROOT_DIR,
        safe_language,
        f"bpe_v{vocab_size}_{fingerprint}",
    )
    required_files = ("tokenizer.model", "tokenizer.vocab", "vocab.txt")
    if all(os.path.isfile(os.path.join(tokenizer_dir, name)) for name in required_files):
        log(f"Reusing generated tokenizer: {tokenizer_dir}")
        return tokenizer_dir

    os.makedirs(tokenizer_dir, exist_ok=True)
    corpus_path = os.path.join(tokenizer_dir, "train_text.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace("\n", " ").strip())
            f.write("\n")

    # Use NeMo's tokenizer builder so the directory contains every artifact
    # expected by ASRBPEMixin/change_vocabulary and later .nemo serialization.
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model

    log(
        f"Generating {language} SentencePiece BPE tokenizer from "
        f"{len(texts)} training transcripts (requested vocab={vocab_size}) ..."
    )
    create_spt_model(
        data_file=corpus_path,
        vocab_size=vocab_size,
        sample_size=-1,
        do_lower_case=False,
        tokenizer_type="bpe",
        output_dir=tokenizer_dir,
        character_coverage=1.0,
        byte_fallback=True,
        split_by_unicode_script=False,
        remove_extra_whitespaces=False,
    )

    missing = [
        name for name in required_files
        if not os.path.isfile(os.path.join(tokenizer_dir, name))
    ]
    if missing:
        raise RuntimeError(
            f"Tokenizer generation did not create required files {missing} in {tokenizer_dir}"
        )

    metadata = {
        "language": language,
        "requested_vocab_size": vocab_size,
        "source_manifest": os.path.abspath(train_manifest),
        "training_transcripts": len(texts),
        "fingerprint": fingerprint,
        "byte_fallback": True,
        "character_coverage": 1.0,
    }
    with open(os.path.join(tokenizer_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    log(f"Generated tokenizer: {tokenizer_dir}")
    return tokenizer_dir


# ---------------------------------------------------------------------------
# Step 1  -  Convert audio (stereo 48kHz -> mono 16kHz)
# ---------------------------------------------------------------------------
def convert_audio():
    """Convert all WAV files from raw speaker dirs to mono 16kHz.

    Reads *.wav from traintestset/p{N}/, produces p{N}_{file}.wav in
    custom_asr_data/wavs/.
    """
    ensure_dirs()
    log(f"Scanning for WAV files in {RAW_DATA_DIR} ...")

    converted = 0
    skipped = 0
    speaker_dirs = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "p*")))

    for spk_dir in speaker_dirs:
        if not os.path.isdir(spk_dir):
            continue
        spk_name = os.path.basename(spk_dir)
        for wav_path in glob.glob(os.path.join(spk_dir, "*.wav")):
            basename = os.path.basename(wav_path)
            out_name = f"{spk_name}_{basename}"
            out_path = os.path.join(CONVERTED_WAVS_DIR, out_name)

            if os.path.exists(out_path):
                skipped += 1
                continue

            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", wav_path,
                    "-ac", "1",        # mono
                    "-ar", "16000",    # 16 kHz (ASR model expectation)
                    "-loglevel", "error",
                    out_path,
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                warn(f"ffmpeg failed for {wav_path}: {result.stderr.strip()}")
                continue

            converted += 1

    log(f"Converted: {converted} | Skipped (already done): {skipped}")
    log(f"Output directory: {CONVERTED_WAVS_DIR}")


# ---------------------------------------------------------------------------
# Step 2  -  Build NeMo JSON manifests
# ---------------------------------------------------------------------------
def build_manifests(
    apply_speech_hints: bool = False,
    language: str = DEFAULT_LANGUAGE,
):
    """Read transcript.csv from each speaker dir and produce train/test
    JSON manifests in the format expected by NeMo's data loader.

    Each manifest line is a JSON object:
        {"audio_filepath": "...", "duration": ..., "text": "...",
         "language": "<locale>", "lang": "<locale>",
         "target_lang": "<locale>"}

    Optionally applies speech-hint grammars to normalize transcript text.
    """
    if not language or not language.strip():
        raise ValueError("language must be a non-empty locale such as 'ug-CN'")
    language = language.strip()
    ensure_dirs()
    random.seed(RANDOM_SEED)

    log(f"Building manifests for language {language!r} ...")
    all_entries = []
    speaker_dirs = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "p*")))

    for spk_dir in speaker_dirs:
        if not os.path.isdir(spk_dir):
            continue
        csv_path = os.path.join(spk_dir, "transcript.csv")
        if not os.path.exists(csv_path):
            warn(f"No transcript.csv in {spk_dir}, skipping.")
            continue

        spk_name = os.path.basename(spk_dir)

        # utf-8-sig accepts both normal UTF-8 and files with a BOM, which is
        # common when multilingual CSV files have been saved by spreadsheet apps.
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or "|" not in line:
                    continue

                wav_name, text = line.split("|", maxsplit=1)
                wav_name = wav_name.strip()
                text = unicodedata.normalize("NFC", text.strip())

                # Optional speech-hint normalization on transcripts
                if apply_speech_hints:
                    text = normalize_with_speech_hints(text)

                converted_name = f"{spk_name}_{wav_name}"
                wav_path = os.path.join(CONVERTED_WAVS_DIR, converted_name)

                if not os.path.exists(wav_path):
                    warn(f"Converted file {converted_name} not found (line {line_num})")
                    continue

                # Duration via ffprobe
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet",
                        "-show_entries", "format=duration",
                        "-of", "csv=p=0", wav_path,
                    ],
                    capture_output=True, text=True,
                )
                duration = float(result.stdout.strip())

                all_entries.append({
                    "audio_filepath": os.path.abspath(wav_path),
                    "duration": round(duration, 4),
                    "text": text,
                    # `language` becomes Cut.supervisions[0].language in Lhotse
                    # and selects the model's language-prompt index.
                    "language": language,
                    "lang": language,
                    "target_lang": language,
                })

    if len(all_entries) < 2:
        raise ValueError(
            "At least two valid transcript/audio pairs are required to create "
            "non-empty train and test manifests."
        )

    # Shuffle and split
    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * TRAIN_SPLIT)
    train_entries = all_entries[:split_idx]
    test_entries = all_entries[split_idx:]

    # Write manifests (JSONL format)
    train_manifest = os.path.join(CUSTOM_DATA_DIR, "train_manifest.json")
    test_manifest = os.path.join(CUSTOM_DATA_DIR, "test_manifest.json")

    for path, entries in [(train_manifest, train_entries),
                          (test_manifest, test_entries)]:
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

    total_duration = sum(e["duration"] for e in all_entries)
    log(f"Total samples : {len(all_entries)}")
    log(f"Total duration: {total_duration:.1f}s ({total_duration / 3600:.2f} hrs)")
    log(f"Train samples : {len(train_entries)} -> {train_manifest}")
    log(f"Test  samples : {len(test_entries)} -> {test_manifest}")

    # Show sample entries
    log("\n--- Sample manifest entries ---")
    for entry in all_entries[:3]:
        name = os.path.basename(entry["audio_filepath"])
        txt = entry["text"][:80]
        if len(entry["text"]) > 80:
            txt += "..."
        log(f"  {name} | dur={entry['duration']:.1f}s | {txt}")

    return train_manifest, test_manifest


# ---------------------------------------------------------------------------
# Step 3  -  Fine-tune with NeMo Python API (no cloned repo needed)
# ---------------------------------------------------------------------------
def run_training(train_manifest: str, test_manifest: str, epochs: int = 20,
                 lr: float = 0.1, language: str = DEFAULT_LANGUAGE,
                 tokenizer_mode: str = "auto",
                 tokenizer_vocab_size: int = DEFAULT_TOKENIZER_VOCAB_SIZE,
                 prompt_index: int | None = None,
                 max_duration: float = DEFAULT_MAX_DURATION,
                 batch_duration: float = DEFAULT_BATCH_DURATION,
                 warmup_steps: int = DEFAULT_WARMUP_STEPS,
                 noam_d_model: int = DEFAULT_NOAM_D_MODEL,
                 run_name: str | None = None,
                 train_workers: int = DEFAULT_TRAIN_WORKERS,
                 validation_workers: int = DEFAULT_VALIDATION_WORKERS,
                 validation_batch_size: int = DEFAULT_VALIDATION_BATCH_SIZE):
    """Fine-tune the pretrained model using NeMo's Python API directly.

    Loads EncDecRNNTBPEModelWithPrompt from the .nemo file, updates data
    and optimizer configs, then trains with pytorch_lightning.Trainer.
    """
    import warnings
    warnings.filterwarnings("ignore")

    language = language.strip()
    if tokenizer_mode not in {"auto", "base", "custom"}:
        raise ValueError(
            "tokenizer_mode must be one of: 'auto', 'base', or 'custom'"
        )
    if lr <= 0:
        raise ValueError("--lr must be positive")
    if max_duration <= 0 or batch_duration <= 0:
        raise ValueError("--max-duration and --batch-duration must be positive")
    if warmup_steps <= 0:
        raise ValueError("--warmup-steps must be positive for the Noam scheduler")
    if noam_d_model <= 0:
        raise ValueError("--noam-d-model must be positive")
    if train_workers < 0 or validation_workers < 0:
        raise ValueError("Data-loader worker counts cannot be negative")
    if validation_batch_size <= 0:
        raise ValueError("--validation-batch-size must be positive")
    if run_name is not None:
        run_name = run_name.strip()
        if not run_name or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
            raise ValueError(
                "--run-name may contain only letters, digits, dot, underscore, and dash"
            )
    checkpoint_dir = (
        os.path.join(CHECKPOINT_DIR, run_name) if run_name else CHECKPOINT_DIR
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    resolve_manifest_language(train_manifest, requested=language)
    resolve_manifest_language(test_manifest, requested=language)

    train_summary = manifest_duration_summary(train_manifest)
    validation_summary = manifest_duration_summary(test_manifest)
    log(
        "Training manifest: "
        f"{train_summary['samples']:,} samples, {train_summary['hours']:.2f} h, "
        f"max={train_summary['maximum']:.2f}s"
    )
    log(
        "Validation manifest: "
        f"{validation_summary['samples']:,} samples, "
        f"max={validation_summary['maximum']:.2f}s"
    )
    log(
        "RTX 5090 data profile: "
        f"batch_duration={batch_duration:g}s, train_workers={train_workers}, "
        f"validation_batch_size={validation_batch_size}, "
        f"validation_workers={validation_workers}, pinned_memory=True"
    )
    if train_summary["maximum"] > max_duration:
        warn(
            f"Training entries longer than --max-duration={max_duration:g}s will "
            "be filtered by the data loader."
        )
    if validation_summary["maximum"] > max_duration:
        warn(
            f"Validation entries longer than --max-duration={max_duration:g}s "
            "will be filtered by the data loader."
        )
    if not os.path.exists(PRETRAINED_MODEL):
        print(f"\n[ERROR] Pretrained model not found: {PRETRAINED_MODEL}")
        print("Download with:")
        print(
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('nvidia/nemotron-3.5-asr-streaming-0.6b', "
            f"local_dir='{os.path.dirname(PRETRAINED_MODEL)}')\""
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Import NeMo model class and Lightning Trainer
    # ------------------------------------------------------------------
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
    from omegaconf import OmegaConf, open_dict
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger

    # ------------------------------------------------------------------
    # Load pretrained model
    # ------------------------------------------------------------------
    model_size = os.path.getsize(PRETRAINED_MODEL) / 1024 ** 3
    log(f"Loading pretrained model: {PRETRAINED_MODEL} ({model_size:.1f} GB)")

    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=PRETRAINED_MODEL,
        map_location="cpu",
    )
    log("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Add an unsupported language prompt and choose/build its tokenizer
    # ------------------------------------------------------------------
    train_entries = read_manifest_entries(train_manifest)
    train_texts = [entry["text"] for entry in train_entries]

    prompt_cfg = model.cfg.model_defaults.get("prompt_dictionary", {})
    if not prompt_cfg:
        prompt_dictionary = {}
    elif OmegaConf.is_config(prompt_cfg):
        prompt_dictionary = OmegaConf.to_container(prompt_cfg, resolve=True)
    else:
        prompt_dictionary = dict(prompt_cfg)
    prompt_was_known = language in prompt_dictionary

    unknown, token_count, unknown_rate = tokenizer_unknown_rate(
        model.tokenizer, train_texts
    )
    log(
        "Base-tokenizer coverage: "
        f"{unknown} unknown token(s) out of {token_count} "
        f"({unknown_rate:.2%})"
    )

    use_custom_tokenizer = tokenizer_mode == "custom" or (
        tokenizer_mode == "auto" and (not prompt_was_known or unknown > 0)
    )
    if use_custom_tokenizer:
        reasons = []
        if not prompt_was_known:
            reasons.append(f"{language!r} is absent from the model prompt dictionary")
        if unknown > 0:
            reasons.append("the base tokenizer emits unknown tokens")
        if tokenizer_mode == "custom":
            reasons.append("--tokenizer-mode=custom was requested")
        log("Custom tokenizer selected: " + "; ".join(reasons))
        tokenizer_dir = build_custom_tokenizer(
            train_manifest=train_manifest,
            language=language,
            vocab_size=tokenizer_vocab_size,
        )
        model.change_vocabulary(
            new_tokenizer_dir=tokenizer_dir,
            new_tokenizer_type="bpe",
        )
        log(
            "Installed generated tokenizer. NeMo reinitialized the RNNT "
            "decoder/joint for the new vocabulary; the acoustic encoder was retained."
        )
        active_tokenizer_dir = os.path.abspath(tokenizer_dir)
    else:
        log(
            "Keeping the pretrained tokenizer "
            f"(--tokenizer-mode={tokenizer_mode}, language already representable)."
        )
        active_tokenizer_dir = None

    num_prompts = int(
        model.cfg.get(
            "num_prompts",
            model.cfg.model_defaults.get("num_prompts", 128),
        )
    )
    if prompt_was_known:
        selected_prompt_index = int(prompt_dictionary[language])
        if prompt_index is not None and prompt_index != selected_prompt_index:
            raise ValueError(
                f"Language {language!r} already uses prompt index "
                f"{selected_prompt_index}; --prompt-index={prompt_index} conflicts."
            )
        log(f"Using existing prompt for {language!r}: index {selected_prompt_index}")
    else:
        used_indices = {int(value) for value in prompt_dictionary.values()}
        if prompt_index is not None:
            if not 0 <= prompt_index < num_prompts:
                raise ValueError(
                    f"--prompt-index must be in [0, {num_prompts - 1}], got {prompt_index}"
                )
            if prompt_index in used_indices:
                raise ValueError(
                    f"Prompt index {prompt_index} is already assigned; omit "
                    "--prompt-index to allocate the first unused slot automatically."
                )
            selected_prompt_index = prompt_index
        else:
            selected_prompt_index = next(
                (index for index in range(num_prompts) if index not in used_indices),
                None,
            )
            if selected_prompt_index is None:
                raise RuntimeError(
                    f"All {num_prompts} prompt slots are assigned; no slot is "
                    f"available for unsupported language {language!r}."
                )
        prompt_dictionary[language] = selected_prompt_index
        with open_dict(model.cfg.model_defaults):
            model.cfg.model_defaults.prompt_dictionary = OmegaConf.create(
                prompt_dictionary
            )
        log(
            f"Added unsupported language prompt {language!r} at unused "
            f"index {selected_prompt_index}."
        )

    # ------------------------------------------------------------------
    # Update data configs
    # ------------------------------------------------------------------
    # Keep lhotse (required for prompt indices in this model).
    OmegaConf.set_struct(model.cfg.train_ds, False)
    model.cfg.train_ds.manifest_filepath = train_manifest
    model.cfg.train_ds.is_tarred = False
    model.cfg.train_ds.shuffle = True
    model.cfg.train_ds.num_workers = train_workers
    model.cfg.train_ds.pin_memory = True
    model.cfg.train_ds.max_duration = max_duration
    model.cfg.train_ds.batch_duration = batch_duration
    model.cfg.train_ds.initialize_prompt_feature = True
    model.cfg.train_ds.prompt_dictionary = OmegaConf.create(prompt_dictionary)
    model.cfg.train_ds.num_prompts = num_prompts
    model.cfg.train_ds.default_prompt_mode = "langID"
    model.cfg.train_ds.unified_auto_ratio = 0.0
    model.cfg.train_ds.default_lang = language
    OmegaConf.set_struct(model.cfg.train_ds, True)

    OmegaConf.set_struct(model.cfg.validation_ds, False)
    model.cfg.validation_ds.manifest_filepath = test_manifest
    model.cfg.validation_ds.is_tarred = False
    model.cfg.validation_ds.num_workers = validation_workers
    model.cfg.validation_ds.pin_memory = True
    # Validation avoids gradient storage, so a larger batch remains practical.
    model.cfg.validation_ds.batch_size = validation_batch_size
    model.cfg.validation_ds.max_duration = max_duration
    model.cfg.validation_ds.initialize_prompt_feature = True
    model.cfg.validation_ds.prompt_dictionary = OmegaConf.create(prompt_dictionary)
    model.cfg.validation_ds.num_prompts = num_prompts
    model.cfg.validation_ds.default_prompt_mode = "langID"
    model.cfg.validation_ds.unified_auto_ratio = 0.0
    model.cfg.validation_ds.default_lang = language
    OmegaConf.set_struct(model.cfg.validation_ds, True)

    # ------------------------------------------------------------------
    # Record enough provenance to reconstruct custom-vocabulary checkpoints
    # without guessing which generated tokenizer was used.
    with open_dict(model.cfg):
        # Checkpoint selection monitors val_wer, so skip the memory-intensive
        # RNNT validation loss for the larger validation batch.
        model.cfg.compute_eval_loss = False
        model.cfg.custom_finetune = OmegaConf.create(
            {
                "language": language,
                "tokenizer_mode": tokenizer_mode,
                "tokenizer_dir": active_tokenizer_dir,
                "tokenizer_vocab_size": tokenizer_vocab_size,
                "prompt_index": selected_prompt_index,
                "train_manifest": os.path.abspath(train_manifest),
                "validation_manifest": os.path.abspath(test_manifest),
                "max_duration": max_duration,
                "batch_duration": batch_duration,
                "train_workers": train_workers,
                "validation_workers": validation_workers,
                "validation_batch_size": validation_batch_size,
                "run_name": run_name,
            }
        )

    # ------------------------------------------------------------------
    # Update optimizer config (from NVIDIA's Nemotron tutorial)
    # ------------------------------------------------------------------
    OmegaConf.set_struct(model.cfg.optim, False)
    model.cfg.optim.name = "adamw"
    model.cfg.optim.lr = lr
    model.cfg.optim.weight_decay = 0.001
    if model.cfg.optim.get("sched") is None:
        model.cfg.optim.sched = OmegaConf.create({})
    OmegaConf.set_struct(model.cfg.optim.sched, False)
    model.cfg.optim.sched.name = "NoamAnnealing"
    model.cfg.optim.sched.warmup_steps = warmup_steps
    model.cfg.optim.sched.d_model = noam_d_model
    OmegaConf.set_struct(model.cfg.optim.sched, True)
    OmegaConf.set_struct(model.cfg.optim, True)

    # ------------------------------------------------------------------
    # Set up data loaders on the model
    # ------------------------------------------------------------------
    log("Setting up training and validation data ...")
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    train_dl = model.train_dataloader()
    val_dl = model.val_dataloader()
    try:
        log(f"Training batches: {len(train_dl)}")
    except TypeError:
        log("Training batches: dynamic (bucketing sampler)")
    try:
        log(f"Validation batches: {len(val_dl)}")
    except TypeError:
        log("Validation batches: dynamic (bucketing sampler)")

    # ------------------------------------------------------------------
    # Avoid NeMo's automatic step-count calculation, which expects a normal
    # DataLoader.batch_size and fails with this model's dynamic Lhotse sampler.
    # Crucially, retain NVIDIA's Noam schedule: optim.lr=0.1 is a scale factor,
    # not a raw AdamW learning rate.
    # ------------------------------------------------------------------
    import torch as _torch
    from nemo.core.optim.lr_scheduler import NoamAnnealing

    # RTX 5090/Blackwell benefits from Tensor Core paths for any float32
    # operations that remain around the BF16 mixed-precision training graph.
    _torch.set_float32_matmul_precision("high")
    if _torch.cuda.is_available():
        _torch.backends.cuda.matmul.allow_tf32 = True
        _torch.backends.cudnn.allow_tf32 = True

    def _custom_configure_optimizers(self):
        optim_cfg = self.cfg.optim
        betas = tuple(optim_cfg.get("betas", (0.9, 0.98)))
        optimizer = _torch.optim.AdamW(
            self.parameters(),
            lr=float(optim_cfg.lr),
            betas=betas,
            weight_decay=float(optim_cfg.weight_decay),
        )
        scheduler = NoamAnnealing(
            optimizer,
            d_model=noam_d_model,
            warmup_steps=warmup_steps,
        )
        # NeMo expects _optimizer to be set for training_step logging
        self._optimizer = optimizer
        peak_lr = float(optim_cfg.lr) / (noam_d_model * warmup_steps) ** 0.5
        log(
            f"Optimizer: AdamW, Noam scale={optim_cfg.lr}, "
            f"warmup={warmup_steps}, d_model={noam_d_model}, "
            f"effective peak LR~{peak_lr:.3g}, wd={optim_cfg.weight_decay}"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    model.configure_optimizers = _custom_configure_optimizers.__get__(
        model, type(model)
    )
    log("Configured AdamW with an explicit step-wise Noam scheduler")
    log("Optimizer configured.")

    # ------------------------------------------------------------------
    # Training callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="nemotron-asr-finetuned-{epoch:02d}-{global_step}",
        save_top_k=3,
        monitor="val_wer",
        mode="min",
        every_n_epochs=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(DATA_DIR, "checkpoints", "tb_logs"),
        name="nemotron-asr-finetune",
    )

    # ------------------------------------------------------------------
    # PyTorch Lightning Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        devices=1,
        max_epochs=epochs,
        precision="bf16-mixed",
        callbacks=[checkpoint_cb, lr_monitor],
        logger=tb_logger,
        accumulate_grad_batches=1,
        gradient_clip_val=5.0,
        log_every_n_steps=10,
        val_check_interval=1.0,  # validate once per epoch
        enable_progress_bar=True,
        benchmark=True,
    )

    # ------------------------------------------------------------------
    # Train!
    # ------------------------------------------------------------------
    log(f"\nStarting fine-tuning ({epochs} epochs, Noam scale={lr}) ...\n")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # ------------------------------------------------------------------
    # Convert top-3 best checkpoints (lowest WER) to .nemo
    # ------------------------------------------------------------------
    # ModelCheckpoint.best_k_models is a dict {ckpt_path: monitor_score}
    best_k = checkpoint_cb.best_k_models     # type: dict[str, float]

    if best_k:
        import torch as _torch

        # Sort ascending (mode="min" → lowest WER first)
        sorted_best = sorted(best_k.items(), key=lambda kv: kv[1])
        log(f"\nConverting {len(sorted_best)} best checkpoint(s) to .nemo ...")

        # Save the training model's final state so we can restore it after
        # swapping in each best-ckpt's weights.  We reuse `model` because
        # building a fresh EncDecRNNTBPEModelWithPrompt from cfg fails
        # (tokenizer artifact registration needs nemo_file_folder).
        final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for rank, (ckpt_path, wer_val) in enumerate(sorted_best, start=1):
            wer_str = f"{wer_val:.4f}" if wer_val is not None else "wer-unknown"
            nemo_name = f"nemotron-asr-best{rank}-wer-{wer_str}.nemo"
            nemo_path = os.path.join(checkpoint_dir, nemo_name)

            log(f"  Best #{rank} (WER={wer_str}): loading {os.path.basename(ckpt_path)} ...")

            # .ckpt files are Lightning checkpoints (ZIP/PK), not NeMo archives
            # (tar.gz).  Load the state_dict and swap it into the existing model.
            ckpt_data = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt_data["state_dict"]

            # Lightning prefixes keys with "model." — strip if present
            if any(k.startswith("model.") for k in sd):
                sd = {k[len("model."):]: v for k, v in sd.items()}
            model.load_state_dict(sd)

            model.save_to(nemo_path)
            log(f"    -> Saved: {nemo_path}")

        # Restore the final-epoch weights back into `model`
        model.load_state_dict(final_state)
    else:
        log("\nNo best checkpoints found to convert.")

    # ------------------------------------------------------------------
    # Save final .nemo checkpoint (last epoch, regardless of WER)
    # ------------------------------------------------------------------
    final_nemo = os.path.join(checkpoint_dir, "nemotron-asr-finetuned.nemo")
    model.save_to(final_nemo)
    log(f"\nTraining complete!")
    log(f"Final model saved to: {final_nemo}")
    log(f"Best checkpoints in:  {checkpoint_dir}")


# ---------------------------------------------------------------------------
# Step 4  -  Evaluate (CER / WER) using NeMo Python API
# ---------------------------------------------------------------------------
def find_nemo_checkpoint() -> str | None:
    """Search common output locations for the latest .nemo file."""
    search_bases = [CHECKPOINT_DIR, os.path.join(DATA_DIR, "checkpoints")]

    for base in search_bases:
        matches = glob.glob(os.path.join(base, "**", "*.nemo"), recursive=True)
        # Exclude the pretrained model
        matches = [m for m in matches if "pretrained_model" not in m]
        if matches:
            return max(matches, key=os.path.getmtime)  # latest
    return None


def run_evaluation(test_manifest: str, language: str | None = None):
    """Run prompt-conditioned inference on the test manifest and compute WER."""
    import warnings
    warnings.filterwarnings("ignore")

    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
    from lightning.pytorch import Trainer
    from nemo.collections.asr.metrics.wer import word_error_rate

    language = resolve_manifest_language(test_manifest, language)

    nemo_file = find_nemo_checkpoint()
    if not nemo_file:
        print(f"\n[ERROR] No .nemo checkpoint found.")
        print(f"Searched under: {CHECKPOINT_DIR} and {DATA_DIR}/checkpoints/")
        sys.exit(1)

    log(f"Evaluating checkpoint: {nemo_file}")

    dummy_trainer = Trainer(devices=1, accelerator="gpu")
    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=nemo_file,
        map_location="cpu",
        trainer=dummy_trainer,
    )
    model.trainer = dummy_trainer
    model.eval()
    model.cuda()

    prompt_dictionary = model.cfg.model_defaults.get("prompt_dictionary", {})
    if language not in prompt_dictionary:
        raise ValueError(
            f"Checkpoint has no prompt for manifest language {language!r}. "
            "Evaluate a checkpoint produced by this fine-tuning script."
        )

    entries = read_manifest_entries(test_manifest)

    log(f"Transcribing {len(entries)} samples ...")
    hyps = []
    refs = []

    for i, entry in enumerate(entries):
        audio_path = entry["audio_filepath"]
        ref_text = entry["text"]
        refs.append(ref_text)

        result = model.transcribe(
            audio=[audio_path],
            batch_size=1,
            target_lang=language,
            verbose=False,
        )
        hypothesis = result[0]
        hyp_text = (
            hypothesis
            if isinstance(hypothesis, str)
            else str(getattr(hypothesis, "text", hypothesis))
        ).strip()
        hyps.append(hyp_text)

        if (i + 1) % 5 == 0 or i == len(entries) - 1:
            log(f"  Transcribed {i+1}/{len(entries)}")

    # NeMo's string helper computes corpus-level WER.  Keep case because the
    # checkpoint is trained for native punctuation and capitalization.
    wer = word_error_rate(hypotheses=hyps, references=refs)
    log(f"\n{'=' * 50}")
    log(f"Evaluation Results")
    log(f"{'=' * 50}")
    log(f"Samples : {len(refs)}")
    log(f"WER     : {wer:.2%}")

    # Show a few examples
    log(f"\n--- Sample transcriptions ---")
    for i in range(min(5, len(refs))):
        log(f"  Ref : {refs[i]}")
        log(f"  Hyp : {hyps[i]}")
        log()


# ---------------------------------------------------------------------------
# Extra  -  Apply speech hints to existing manifests
# ---------------------------------------------------------------------------
def apply_speechhints_to_manifests():
    """Post-process already-built manifests by normalizing the text field
    with speech-hint grammars.

    Reads train_manifest.json and test_manifest.json from custom_asr_data/,
    rewrites them in-place with normalized text.
    """
    if not SPEECH_HINT_AVAILABLE:
        log("speech_hint library not installed; installing it now ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "speech-hints"],
            check=True,
        )
        from speech_hint import apply_hint  # type: ignore
        SPEECH_HINT_AVAILABLE = True

    for manifest_name in ["train_manifest.json", "test_manifest.json"]:
        manifest_path = os.path.join(CUSTOM_DATA_DIR, manifest_name)
        if not os.path.exists(manifest_path):
            warn(f"{manifest_name} not found; skipping.")
            continue

        entries = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    entry["text"] = normalize_with_speech_hints(entry["text"])
                    entries.append(entry)

        with open(manifest_path, "w") as f:
            for entry in entries:
                json.dump(entry, f)
                f.write("\n")

        log(f"Normalized {len(entries)} entries in {manifest_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Nemotron 3.5 ASR Streaming on custom dataset "
            "with optional speech-hint normalization"
        )
    )
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert audio files (step 1)")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Convert audio + build manifests (steps 1-2)")
    parser.add_argument("--train-only", action="store_true",
                        help="Only run fine-tuning (step 3)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Only evaluate trained model (step 4)")
    parser.add_argument("--apply-speechhints", action="store_true",
                        help="Post-process manifests with speech-hint grammars")
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Locale written to manifests and used for prompt conditioning "
            f"(for example ug-CN; default for new manifests: {DEFAULT_LANGUAGE})"
        ),
    )
    parser.add_argument(
        "--tokenizer-mode",
        choices=("auto", "base", "custom"),
        default="auto",
        help=(
            "auto: generate a tokenizer when the language is unsupported or "
            "the base tokenizer emits <unk>; base: always keep the pretrained "
            "tokenizer; custom: always generate from training text (default: auto)"
        ),
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=DEFAULT_TOKENIZER_VOCAB_SIZE,
        help=(
            "Requested BPE vocabulary size for a generated tokenizer "
            f"(default: {DEFAULT_TOKENIZER_VOCAB_SIZE}, minimum: 512)"
        ),
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=None,
        help=(
            "Unused prompt index for a language absent from the model; "
            "by default the first unused slot is selected"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20,
                        help="Max training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help=(
                            "Noam learning-rate scale factor, not the raw AdamW LR "
                            "(default: 0.1, per NVIDIA tutorial)"
                        ))
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=f"Noam linear warmup steps (default: {DEFAULT_WARMUP_STEPS})",
    )
    parser.add_argument(
        "--noam-d-model",
        type=int,
        default=DEFAULT_NOAM_D_MODEL,
        help=f"Noam model dimension (default: {DEFAULT_NOAM_D_MODEL})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help=(
            "Maximum train/validation clip duration in seconds "
            f"(default: {DEFAULT_MAX_DURATION:g})"
        ),
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=DEFAULT_BATCH_DURATION,
        help=(
            "Approximate total audio seconds per dynamic training batch "
            f"(default: {DEFAULT_BATCH_DURATION:g})"
        ),
    )
    parser.add_argument(
        "--train-workers",
        type=int,
        default=DEFAULT_TRAIN_WORKERS,
        help=(
            "Training data-loader processes "
            f"(default: {DEFAULT_TRAIN_WORKERS} for a 32 GB RTX 5090 host)"
        ),
    )
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=DEFAULT_VALIDATION_WORKERS,
        help=(
            "Validation data-loader processes "
            f"(default: {DEFAULT_VALIDATION_WORKERS})"
        ),
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=DEFAULT_VALIDATION_BATCH_SIZE,
        help=(
            "Validation clips per batch "
            f"(default: {DEFAULT_VALIDATION_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional safe subdirectory under the checkpoint directory, for "
            "example uyghur-v2; prevents mixing a retrain with old files"
        ),
    )
    args = parser.parse_args()

    if sys.platform == "darwin":
        parser.error(
            "This pipeline is intentionally disabled on macOS. Run it on a "
            "Linux host with an NVIDIA CUDA GPU; use macOS only to edit the files."
        )

    if args.apply_speechhints:
        apply_speechhints_to_manifests()
        return

    if not any([args.convert_only, args.manifest_only,
                args.train_only, args.evaluate]):
        # ---- Full pipeline ----
        log("=" * 70)
        log("Nemotron 3.5 ASR Streaming — Full Fine-Tuning Pipeline")
        log("=" * 70)

        log("\n>>> Step 1: Converting audio ...")
        convert_audio()

        log("\n>>> Step 2: Building manifests ...")
        manifest_language = (args.language or DEFAULT_LANGUAGE).strip()
        train_manifest, test_manifest = build_manifests(
            language=manifest_language
        )

        log("\n>>> Step 3: Fine-tuning model ...")
        run_training(
            train_manifest,
            test_manifest,
            epochs=args.epochs,
            lr=args.lr,
            language=manifest_language,
            tokenizer_mode=args.tokenizer_mode,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
            prompt_index=args.prompt_index,
            max_duration=args.max_duration,
            batch_duration=args.batch_duration,
            warmup_steps=args.warmup_steps,
            noam_d_model=args.noam_d_model,
            run_name=args.run_name,
            train_workers=args.train_workers,
            validation_workers=args.validation_workers,
            validation_batch_size=args.validation_batch_size,
        )

        log("\n>>> Step 4: Evaluating model ...")
        run_evaluation(test_manifest, language=manifest_language)

        log("\n" + "=" * 70)
        log("Pipeline complete!")
        log("=" * 70)
        return

    if args.convert_only:
        convert_audio()
        return

    if args.manifest_only:
        convert_audio()
        build_manifests(language=args.language or DEFAULT_LANGUAGE)
        return

    if args.train_only:
        train_manifest = os.path.join(CUSTOM_DATA_DIR, "train_manifest.json")
        test_manifest = os.path.join(CUSTOM_DATA_DIR, "test_manifest.json")
        for m in [train_manifest, test_manifest]:
            if not os.path.exists(m):
                print(f"[ERROR] {m} not found. Run --manifest-only first.")
                sys.exit(1)
        manifest_language = resolve_manifest_language(
            train_manifest, requested=args.language
        )
        resolve_manifest_language(test_manifest, requested=manifest_language)
        run_training(
            train_manifest,
            test_manifest,
            epochs=args.epochs,
            lr=args.lr,
            language=manifest_language,
            tokenizer_mode=args.tokenizer_mode,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
            prompt_index=args.prompt_index,
            max_duration=args.max_duration,
            batch_duration=args.batch_duration,
            warmup_steps=args.warmup_steps,
            noam_d_model=args.noam_d_model,
            run_name=args.run_name,
            train_workers=args.train_workers,
            validation_workers=args.validation_workers,
            validation_batch_size=args.validation_batch_size,
        )
        return

    if args.evaluate:
        test_manifest = os.path.join(CUSTOM_DATA_DIR, "test_manifest.json")
        if not os.path.exists(test_manifest):
            print("[ERROR] test_manifest.json not found. Run training first.")
            sys.exit(1)
        run_evaluation(test_manifest, language=args.language)
        return


if __name__ == "__main__":
    main()
