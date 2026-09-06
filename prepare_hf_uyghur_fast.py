#!/usr/bin/env python3

# Prevent each worker from spawning lots of BLAS threads
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from collections import Counter
import json
import math
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path

DATASET_IDS = (
    "piyazon/cv-corpus-ug-24-latn",
    "piyazon/thuyg20-datasets",
)
# Both repositories provide Arabic-script Uyghur in `sentence`.
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
    import numpy as np

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
    import numpy as np

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
    dataset_id,
    max_duration=MAX_DURATION,
):
    """
    Runs inside Dataset.map() worker processes.
    """
    import soundfile as sf

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

            if duration < MIN_DURATION or duration > max_duration:
                manifest_lines.append("")
                errors.append("outside duration limits")
                continue

            filename = f"{idx:08d}.{extension}"
            output_path = split_dir / filename

            # Allows safe resume
            if not output_path.exists():
                # Publish only complete files, so an interrupted write is not
                # mistaken for a finished clip on the next run.
                with tempfile.NamedTemporaryFile(
                    dir=split_dir, prefix=f".{idx:08d}.", suffix=".tmp", delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                try:
                    sf.write(
                        temporary_path,
                        audio,
                        SAMPLE_RATE,
                        format=output_format.upper(),
                        subtype="PCM_16",
                    )
                    temporary_path.replace(output_path)
                finally:
                    temporary_path.unlink(missing_ok=True)

            entry = {
                "audio_filepath": str(output_path.resolve()),
                "duration": round(duration, 4),

                # Arabic-script Uyghur; sentence_latn is intentionally unused.
                "text": text,

                # Custom Uyghur prompt allocated by the training script.
                "language": LANGUAGE,
                "lang": LANGUAGE,
                "target_lang": LANGUAGE,
                "source_dataset": dataset_id,
                "source_split": split_name,
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
    dataset_id,
    max_duration=MAX_DURATION,
):

    print()
    print("=" * 70)
    print(f"PARALLEL EXPORT: {dataset_id} / {split_name}")
    print("=" * 70)
    print(f"Samples : {len(ds):,}")
    print(f"Workers : {workers}")
    print(f"Batch   : {batch_size}")
    print(f"Format  : {output_format.upper()}")
    print(f"Text    : {TEXT_COLUMN}")
    print(f"Lang    : {LANGUAGE}")
    print(f"Max dur : {max_duration:g}s")
    print()

    # Namespacing separates equal row indices from different repositories.
    # Fingerprinting also avoids reusing old audio if rows/revisions change.
    source_audio_dir = (
        Path(audio_dir) / dataset_id.replace("/", "__") / ds._fingerprint
    )

    processed = ds.map(
        process_batch,

        batched=True,
        batch_size=batch_size,

        with_indices=True,

        num_proc=workers,

        fn_kwargs={
            "split_name": split_name,
            "audio_dir": str(source_audio_dir),
            "output_format": output_format,
            "dataset_id": dataset_id,
            "max_duration": max_duration,
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
    skip_reasons = Counter()

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fout:

        for line, error in zip(
            processed["manifest_line"],
            processed["error"],
        ):
            if not line:
                skipped += 1
                skip_reasons[error or "unknown"] += 1
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
    for reason, count in skip_reasons.most_common(10):
        print(f"  skipped {count:,}: {reason}")

    if not written:
        raise RuntimeError(f"No usable audio exported from {dataset_id}/{split_name}")

    return written, skipped, total_seconds


def select_splits(dataset, dataset_id):
    """Preserve each source's held-out split before combining repositories."""
    if "train" not in dataset:
        raise RuntimeError(f"{dataset_id}: dataset has no train split")
    if "validation" in dataset:
        selected = {"train": dataset["train"], "validation": dataset["validation"]}
    elif "test" in dataset:
        selected = {"train": dataset["train"], "test": dataset["test"]}
    else:
        print(f"{dataset_id}: no validation/test split; creating a seeded 98/2 row split.")
        selected = dataset["train"].train_test_split(test_size=0.02, seed=42)

    for split_name, split in selected.items():
        missing = {"audio", TEXT_COLUMN} - set(split.column_names)
        if missing:
            raise ValueError(f"{dataset_id}/{split_name}: missing required columns {sorted(missing)}")
        if not len(split):
            raise ValueError(f"{dataset_id}/{split_name}: split is empty")
    return selected


def merge_manifests(source_paths, destination):
    """Replace a combined manifest only after all its source exports succeeded."""
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            for source in source_paths:
                with open(source, encoding="utf-8") as input_file:
                    shutil.copyfileobj(input_file, output)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    parser.add_argument(
        "--max-duration", type=float, default=MAX_DURATION,
        help=f"Maximum clip duration in seconds; match the training setting (default: {MAX_DURATION:g})",
    )

    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be positive")
    if not math.isfinite(args.max_duration) or args.max_duration < MIN_DURATION:
        parser.error(f"--max-duration must be finite and at least {MIN_DURATION}")

    from datasets import load_dataset, Audio

    output_dir = Path("custom_asr_data")
    audio_dir = output_dir / "audio"

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = output_dir / "train_manifest.json"
    test_manifest = output_dir / "test_manifest.json"

    print("=" * 70)
    print("Nemotron 3.5 Arabic-Script Uyghur FAST Dataset Preparation")
    print("=" * 70)

    print(f"Datasets: {', '.join(DATASET_IDS)}")
    print(f"Text    : {TEXT_COLUMN}")
    print(f"Prompt  : {LANGUAGE}")
    print(f"Workers : {args.workers}")
    print(f"Format  : {args.format}")
    print()

    manifests = {"train": [], "test": []}
    totals = {"train": [0, 0, 0.0], "test": [0, 0, 0.0]}
    for dataset_id in DATASET_IDS:
        print(f"\nLoading {dataset_id}...")
        dataset = load_dataset(dataset_id)
        print("Available splits:")
        for name, split in dataset.items():
            print(f"  {name}: {len(split):,}")

        for split_name, ds in select_splits(dataset, dataset_id).items():
            # Resampling happens lazily inside parallel map workers.
            ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, num_channels=1))
            role = "train" if split_name == "train" else "test"
            part_manifest = (
                output_dir / "source_manifests" / dataset_id.replace("/", "__")
                / f"{split_name}.jsonl"
            )
            stats = export_split(
                ds, split_name, part_manifest, audio_dir,
                args.workers, args.batch_size, args.format,
                dataset_id, args.max_duration,
            )
            manifests[role].append(part_manifest)
            totals[role] = [a + b for a, b in zip(totals[role], stats)]
        del dataset

    # Keep the current training manifests intact if either source fails to load
    # or export. Each final manifest contains both repositories, once each.
    merge_manifests(manifests["train"], train_manifest)
    merge_manifests(manifests["test"], test_manifest)
    train_stats, test_stats = totals["train"], totals["test"]

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
