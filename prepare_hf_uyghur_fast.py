#!/usr/bin/env python3

# Prevent each worker from spawning lots of BLAS threads
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from datasets import load_dataset, Audio


DATASET_ID = "piyazon/cv-corpus-ug-24-augment"
TEXT_COLUMN = "sentence"

LANGUAGE = "ug-CN"
SAMPLE_RATE = 16000

MIN_DURATION = 0.2
MAX_DURATION = 40.0


def normalize_text(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def decode_audio(audio):
    """
    Support current HuggingFace/torchcodec AudioDecoder
    and older datasets audio dictionaries.
    """

    # Older datasets
    if isinstance(audio, dict):
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        return arr, sr

    # Current datasets + torchcodec
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()

        arr = samples.data

        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        else:
            arr = np.asarray(arr)

        sr = int(samples.sample_rate)

        return arr.astype(np.float32, copy=False), sr

    raise TypeError(f"Unknown audio type: {type(audio)}")


def make_mono(audio):
    audio = np.asarray(audio)

    if audio.ndim == 1:
        return audio

    # torchcodec normally returns [channels, samples]
    if audio.shape[0] <= 8:
        return audio.mean(axis=0)

    return audio.mean(axis=1)


def process_batch(
    batch,
    indices,
    split_name,
    audio_dir,
    output_format,
):
    """
    Runs inside Dataset.map() worker processes.
    """

    manifest_lines = []
    errors = []

    split_dir = Path(audio_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    extension = "wav" if output_format == "wav" else "flac"

    for audio_obj, text_raw, idx in zip(
        batch["audio"],
        batch[TEXT_COLUMN],
        indices,
    ):
        try:
            text = normalize_text(text_raw)

            if not text:
                manifest_lines.append("")
                errors.append("empty text")
                continue

            audio, sr = decode_audio(audio_obj)
            audio = make_mono(audio)

            # We cast Audio to 16 kHz before map(), so this should
            # already be 16000.
            if sr != SAMPLE_RATE:
                manifest_lines.append("")
                errors.append(f"bad sample rate: {sr}")
                continue

            duration = len(audio) / SAMPLE_RATE

            if duration < MIN_DURATION or duration > MAX_DURATION:
                manifest_lines.append("")
                errors.append(f"duration: {duration:.2f}")
                continue

            filename = f"{idx:08d}.{extension}"
            output_path = split_dir / filename

            # Allows safe resume
            if not output_path.exists():

                if output_format == "wav":
                    # FAST: no compression
                    sf.write(
                        output_path,
                        audio,
                        SAMPLE_RATE,
                        subtype="PCM_16",
                    )

                else:
                    # Smaller but slower
                    sf.write(
                        output_path,
                        audio,
                        SAMPLE_RATE,
                        format="FLAC",
                        subtype="PCM_16",
                    )

            entry = {
                "audio_filepath": str(output_path.resolve()),
                "duration": round(duration, 4),

                # Native Uyghur transcript from the `sentence` column
                "text": text,

                # Custom Uyghur prompt allocated by the fine-tuning script
                "language": LANGUAGE,
                "lang": LANGUAGE,
                "target_lang": LANGUAGE,
            }

            manifest_lines.append(
                json.dumps(entry, ensure_ascii=False)
            )

            errors.append("")

        except Exception as e:
            manifest_lines.append("")
            errors.append(str(e)[:200])

    return {
        "manifest_line": manifest_lines,
        "error": errors,
    }


def export_split(
    ds,
    split_name,
    manifest_path,
    audio_dir,
    workers,
    batch_size,
    output_format,
):

    print()
    print("=" * 70)
    print(f"PARALLEL EXPORT: {split_name}")
    print("=" * 70)
    print(f"Samples : {len(ds):,}")
    print(f"Workers : {workers}")
    print(f"Batch   : {batch_size}")
    print(f"Format  : {output_format.upper()}")
    print(f"Text    : {TEXT_COLUMN}")
    print(f"Lang    : {LANGUAGE}")
    print()

    processed = ds.map(
        process_batch,

        batched=True,
        batch_size=batch_size,

        with_indices=True,

        num_proc=workers,

        fn_kwargs={
            "split_name": split_name,
            "audio_dir": str(audio_dir),
            "output_format": output_format,
        },

        # Don't copy all original HF columns into temporary dataset
        remove_columns=ds.column_names,

        # Important because this map() has file-writing side effects
        load_from_cache_file=False,

        desc=f"Export {split_name}",
    )

    written = 0
    skipped = 0
    total_seconds = 0.0

    with open(manifest_path, "w", encoding="utf-8") as fout:

        for line, error in zip(
            processed["manifest_line"],
            processed["error"],
        ):
            if not line:
                skipped += 1
                continue

            fout.write(line + "\n")

            obj = json.loads(line)

            total_seconds += obj["duration"]
            written += 1

    print()
    print(f"{split_name} complete")
    print(f"  written : {written:,}")
    print(f"  skipped : {skipped:,}")
    print(f"  hours   : {total_seconds / 3600:.2f}")
    print(f"  manifest: {manifest_path}")

    return written, skipped, total_seconds


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--format",
        choices=["wav", "flac"],
        default="wav",
        help="wav = fastest, flac = smaller",
    )

    args = parser.parse_args()

    output_dir = Path("custom_asr_data")
    audio_dir = output_dir / "audio"

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = output_dir / "train_manifest.json"
    test_manifest = output_dir / "test_manifest.json"

    print("=" * 70)
    print("Nemotron 3.5 Uyghur FAST Dataset Preparation")
    print("=" * 70)

    print(f"Dataset : {DATASET_ID}")
    print(f"Text    : {TEXT_COLUMN}")
    print(f"Prompt  : {LANGUAGE}")
    print(f"Workers : {args.workers}")
    print(f"Format  : {args.format}")
    print()

    print("Loading dataset...")

    dataset = load_dataset(DATASET_ID)

    print()
    print("Available splits:")

    for name, split in dataset.items():
        print(f"  {name}: {len(split):,}")

    print()
    print("Casting audio to 16 kHz...")

    # Resampling happens lazily inside parallel map workers.
    for split_name in dataset.keys():
        dataset[split_name] = dataset[split_name].cast_column(
            "audio",
            Audio(
                sampling_rate=SAMPLE_RATE,
                num_channels=1,
            ),
        )

    if "train" not in dataset:
        raise RuntimeError("Dataset has no train split")

    train_ds = dataset["train"]

    if "validation" in dataset:
        test_ds = dataset["validation"]

    elif "test" in dataset:
        test_ds = dataset["test"]

    else:
        print("No validation/test split. Creating 98/2 split.")

        tmp = train_ds.train_test_split(
            test_size=0.02,
            seed=42,
        )

        train_ds = tmp["train"]
        test_ds = tmp["test"]

    train_stats = export_split(
        train_ds,
        "train",
        train_manifest,
        audio_dir,
        args.workers,
        args.batch_size,
        args.format,
    )

    test_stats = export_split(
        test_ds,
        "test",
        test_manifest,
        audio_dir,
        args.workers,
        args.batch_size,
        args.format,
    )

    print()
    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)

    print(
        f"train: {train_stats[0]:,} entries "
        f"/ {train_stats[2] / 3600:.2f} hours"
    )

    print(
        f"test : {test_stats[0]:,} entries "
        f"/ {test_stats[2] / 3600:.2f} hours"
    )

    print()
    print(train_manifest.resolve())
    print(test_manifest.resolve())

    print()
    print("First training example:")

    with open(train_manifest, encoding="utf-8") as f:
        print(f.readline().strip())


if __name__ == "__main__":
    main()
